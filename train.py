from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Subset

from data import RBPS, dense_labels_to_intervals, load_splits, make_loader
from evaluate import evaluate_model, load_checkpoint
from rsec import RSEC, RSECConfig, fixed_order_nll


PROJECT_ROOT = Path(__file__).resolve().parent

EXCLUDED_TRAINING_IDS = {
    "EIF4A3": {"train": ["hsa_circ_0001678"]},
    "IGF2BP2": {
        "train": ["hsa_circ_0002577", "hsa_circ_0034970", "hsa_circ_0047707", "hsa_circ_0053429"],
    },
    "IGF2BP3": {"train": ["hsa_circ_0002811"], "val": ["hsa_circ_0032884"]},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def event_checkpoint_better(
    candidate: dict[str, float], best: dict[str, float] | None
) -> bool:
    if best is None:
        return True
    ranking = (
        ("event_ap_avg_0_3_0_7", 1),
        ("event_f1_iou_0_5", 1),
        ("count_mae", -1),
    )
    for key, direction in ranking:
        delta = direction * (float(candidate[key]) - float(best[key]))
        if delta > 1e-12:
            return True
        if delta < -1e-12:
            return False
    return False


def event_only_metrics(values: dict[str, Any] | None) -> dict[str, Any] | None:
    if values is None:
        return None
    prefixes = ("event_", "count_", "mean_")
    return {
        key: value
        for key, value in values.items()
        if key.startswith(prefixes) or key == "sequence_count"
    }


def event_only_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                key: value
                for key, value in row.items()
                if not key.startswith("val_")
                or key.startswith(("val_event_", "val_count_", "val_mean_"))
                or key == "val_sequence_count"
            }
        )
    return output


def train(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text())
    rbp_config = config["datasets"][args.rbp]
    training = config["training"]
    runtime = config["runtime"]
    seed = int(training["seed"])
    seed_everything(seed)
    device = torch.device(args.device)
    data_root = Path(args.data_root) if args.data_root else PROJECT_ROOT / config["data_root"]
    split_path = PROJECT_ROOT / rbp_config["split"]
    splits, computed_hash, _ = load_splits(data_root, args.rbp, split_path)
    excluded_ids = EXCLUDED_TRAINING_IDS.get(args.rbp, {})
    for split, identifiers in excluded_ids.items():
        subset = splits[split]
        excluded = set(identifiers)
        indices = [index for index in subset.indices if subset.dataset.ids[index] not in excluded]
        if len(subset) - len(indices) != len(excluded):
            raise RuntimeError(f"{args.rbp}: excluded IDs missing from the original {split} split")
        splits[split] = Subset(subset.dataset, indices)

    model_config = dict(config["model"])
    if isinstance(model_config.get("dilations"), list):
        model_config["dilations"] = ",".join(map(str, model_config["dilations"]))
    model = RSEC(RSECConfig(**model_config)).to(device)
    if args.init_checkpoint:
        checkpoint = load_checkpoint(args.init_checkpoint, device)
        if checkpoint.get("model_name") != "rsec_null_span_fixed_order":
            raise RuntimeError("initialization checkpoint is not the full RSEC model")
        if checkpoint.get("split_hash") != computed_hash:
            raise RuntimeError("initialization checkpoint was trained with a different split")
        if checkpoint.get("excluded_training_ids", {}) != excluded_ids:
            raise RuntimeError("initialization checkpoint uses different training or validation exclusions")
        model.load_state_dict(checkpoint["model_state"], strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    amp = bool(runtime["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_loader = make_loader(
        splits["train"],
        int(training["batch_size"]),
        True,
        seed,
        int(runtime["num_workers"]),
        device.type == "cuda",
    )
    validation_loader = make_loader(
        splits["val"],
        int(training["eval_batch_size"]),
        False,
        seed,
        int(runtime["num_workers"]),
        device.type == "cuda",
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output / "best_event_checkpoint.pt"
    last_checkpoint_path = output / "last_checkpoint.pt"
    rows: list[dict[str, Any]] = []
    best_event_metrics: dict[str, float] | None = None
    best_event_primary = -float("inf")
    event_patience = 0
    minimum_delta = float(training["early_stopping_min_delta"])
    patience_limit = int(training["early_stopping_patience"])
    start_epoch = 1

    if args.resume_from_checkpoint and last_checkpoint_path.is_file():
        resume = load_checkpoint(last_checkpoint_path, device)
        if resume.get("checkpoint_type") != "training_resume":
            raise RuntimeError("last checkpoint is not a resumable training checkpoint")
        if resume.get("model_name") != "rsec_null_span_fixed_order":
            raise RuntimeError("last checkpoint belongs to a different model")
        if resume.get("split_hash") != computed_hash:
            raise RuntimeError("last checkpoint was trained with a different split")
        if resume.get("excluded_training_ids", {}) != excluded_ids:
            raise RuntimeError("last checkpoint uses different training or validation exclusions")
        if resume.get("split_strategy") != config["split_protocol"]["strategy"]:
            raise RuntimeError("last checkpoint uses an incompatible split strategy")
        model.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        scaler.load_state_dict(resume["scaler_state"])
        rows = event_only_rows(list(resume["rows"]))
        best_event_metrics = event_only_metrics(resume.get("best_event_metrics"))
        best_event_primary = float(resume["best_event_primary"])
        event_patience = int(resume["event_patience"])
        start_epoch = int(resume["epoch"]) + 1
        restore_rng_state(resume["rng_state"])
        print(f"resumed {args.rbp} from epoch {start_epoch - 1}")
    elif args.resume_from_checkpoint:
        print(f"no last checkpoint found for {args.rbp}; starting from epoch 1")

    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for features, labels, mask, _, _ in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            gold = dense_labels_to_intervals(labels, mask)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                outputs = model(
                    features,
                    mask,
                    gold["gold_starts"],
                    gold["gold_ends"],
                    gold["gold_event_mask"],
                )
                objective = fixed_order_nll(outputs, gold["gold_event_counts"])
            scaler.scale(objective["loss"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(objective["loss"].detach()))

        validation_metrics, _ = evaluate_model(
            model,
            validation_loader,
            device,
            protocol="native",
            threshold=0.0,
            amp=amp,
        )
        if event_checkpoint_better(validation_metrics, best_event_metrics):
            best_event_metrics = dict(validation_metrics)
            payload = {
                "model_name": "rsec_null_span_fixed_order",
                "display_name": "RSEC",
                "model_config": asdict(model.cfg),
                "seed": seed,
                "split_hash": computed_hash,
                "excluded_training_ids": excluded_ids,
                "split_strategy": config["split_protocol"]["strategy"],
                "implementation_version": 1,
                "selection_implementation_version": 2,
                "selection_protocol": "event_protocol",
                "selection_primary_metric": "event_ap_avg_0_3_0_7",
                "selected_epoch": epoch,
                "training_objective": "fixed_left_to_right_unified_null_span_nll",
                "canonical_gold_order": "left_to_right_by_start_then_end",
                "model_state": model.state_dict(),
                "val_metrics": validation_metrics,
            }
            atomic_save(payload, best_checkpoint_path)

        if (
            validation_metrics["event_ap_avg_0_3_0_7"]
            > best_event_primary + minimum_delta
        ):
            best_event_primary = validation_metrics["event_ap_avg_0_3_0_7"]
            event_patience = 0
        else:
            event_patience += 1

        row = {
            "epoch": epoch,
            "train_joint_nll": float(np.mean(losses)),
            **{f"val_{key}": value for key, value in validation_metrics.items()},
            "event_patience": event_patience,
        }
        rows.append(row)
        write_rows(output / "training_log.csv", rows)
        atomic_save(
            {
                "checkpoint_type": "training_resume",
                "model_name": "rsec_null_span_fixed_order",
                "model_config": asdict(model.cfg),
                "epoch": epoch,
                "seed": seed,
                "split_hash": computed_hash,
                "excluded_training_ids": excluded_ids,
                "split_strategy": config["split_protocol"]["strategy"],
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "rows": rows,
                "best_event_metrics": best_event_metrics,
                "best_event_primary": best_event_primary,
                "event_patience": event_patience,
                "rng_state": capture_rng_state(),
            },
            last_checkpoint_path,
        )
        print(
            f"epoch={epoch} train_nll={row['train_joint_nll']:.6f} "
            f"val_event_ap={validation_metrics['event_ap_avg_0_3_0_7']:.6f} "
            f"val_f1={validation_metrics['event_f1_iou_0_5']:.6f}"
        )
        if event_patience >= patience_limit:
            break

    summary = {
        "rbp": args.rbp,
        "split_hash": computed_hash,
        "excluded_training_ids": excluded_ids,
        "checkpoint": str(best_checkpoint_path),
        "epochs_run": len(rows),
        "best_validation_metrics": best_event_metrics,
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train the full fixed-order RSEC model on a predefined experimental split.')
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--rbp", required=True, choices=RBPS)
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
