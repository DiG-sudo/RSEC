"""Checkpoint loading and Native/SeqCal evaluation for the full RSEC model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from data import (
    RBPS,
    dense_labels_to_intervals,
    labels_to_event_rows,
    load_splits,
    make_loader,
)
from metrics import evaluate_predictions
from rsec import RSEC, RSECConfig, decode_from_memory


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model_from_checkpoint(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[RSEC, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint.get("model_name") != "rsec_null_span_fixed_order":
        raise RuntimeError(f"not a full fixed-order RSEC checkpoint: {checkpoint_path}")
    expected_objective = "fixed_left_to_right_unified_null_span_nll"
    if checkpoint.get("training_objective") != expected_objective:
        raise RuntimeError(f"unexpected training objective: {checkpoint.get('training_objective')}")
    model = RSEC(RSECConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def seqcal_prefix(
    events: list[tuple[int, int]], confidences: list[float], threshold: float
) -> tuple[list[tuple[int, int]], list[float]]:
    keep = 0
    for confidence in confidences:
        if confidence < threshold:
            break
        keep += 1
    return events[:keep], confidences[:keep]


@torch.inference_mode()
def evaluate_model(
    model: RSEC,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    protocol: str,
    threshold: float,
    amp: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if protocol not in {"native", "seqcal"}:
        raise ValueError("protocol must be native or seqcal")
    gold_rows: list[list[tuple[int, int]]] = []
    predicted_rows: list[list[tuple[int, int]]] = []
    confidence_rows: list[list[float]] = []
    records: list[dict[str, Any]] = []

    for features, labels, mask, lengths, identifiers in loader:
        features = features.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        targets = labels_to_event_rows(labels, mask)
        gold = dense_labels_to_intervals(labels, mask)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            outputs = model(
                features,
                mask,
                gold["gold_starts"],
                gold["gold_ends"],
                gold["gold_event_mask"],
            )
            decoded = decode_from_memory(model, outputs["memory"], mask)
        for index, identifier in enumerate(identifiers):
            native_events = list(decoded["generation_events"][index])
            native_confidences = list(decoded["generation_confidences"][index])
            if protocol == "seqcal":
                events, confidences = seqcal_prefix(
                    native_events, native_confidences, threshold
                )
                termination = (
                    "confidence_below_threshold"
                    if len(events) < len(native_events)
                    else decoded["termination_reason"][index]
                )
            else:
                events, confidences = native_events, native_confidences
                termination = decoded["termination_reason"][index]
            gold_rows.append(targets[index])
            predicted_rows.append(events)
            confidence_rows.append(confidences)
            records.append(
                {
                    "id": identifier,
                    "length": int(lengths[index]),
                    "gold_events": [list(event) for event in targets[index]],
                    "predicted_events": [
                        {"start": start, "end": end, "confidence": confidence}
                        for (start, end), confidence in zip(events, confidences)
                    ],
                    "generation_events": [
                        {"start": start, "end": end, "confidence": confidence}
                        for (start, end), confidence in zip(
                            native_events, native_confidences
                        )
                    ],
                    "termination_reason": termination,
                }
            )

    metrics = evaluate_predictions(
        gold_rows,
        predicted_rows,
        confidence_rows,
    )
    metrics["sequence_count"] = len(records)
    return metrics, records


def configured_rbp(
    config: dict[str, Any],
    rbp: str,
    data_root_override: str | None = None,
) -> tuple[dict[str, torch.utils.data.Subset], str, dict[str, Any]]:
    entry = config["datasets"][rbp]
    data_root = (
        Path(data_root_override)
        if data_root_override is not None
        else resolve_project_path(config["data_root"])
    )
    subsets, computed_hash, _ = load_splits(
        data_root, rbp, resolve_project_path(entry["split"])
    )
    
    return subsets, computed_hash, entry


def evaluate_rbp(
    config: dict[str, Any],
    rbp: str,
    device: torch.device,
    split: str,
    protocol: str,
    batch_size: int,
    data_root_override: str | None,
) -> dict[str, Any]:
    subsets, computed_hash, entry = configured_rbp(config, rbp, data_root_override)
    if "checkpoint" not in entry or "checkpoint_sha256" not in entry:
        raise RuntimeError(f"{rbp}: no fixed-split checkpoint is configured yet")
    checkpoint_path = resolve_project_path(entry["checkpoint"])
    actual_checksum = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if actual_checksum != entry["checkpoint_sha256"]:
        raise RuntimeError(f"{rbp}: checkpoint checksum mismatch")
    model, checkpoint = build_model_from_checkpoint(checkpoint_path, device)
    if checkpoint.get("split_hash") != computed_hash:
        raise RuntimeError(f"{rbp}: checkpoint was trained with a different split")
    if checkpoint.get("split_strategy") != config["split_protocol"]["strategy"]:
        raise RuntimeError(f"{rbp}: checkpoint split strategy is incompatible")
    loader = make_loader(
        subsets[split],
        batch_size=batch_size,
        shuffle=False,
        seed=int(config["training"]["seed"]),
        num_workers=int(config["runtime"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    if protocol == "seqcal" and "seqcal_threshold" not in entry:
        raise RuntimeError(f"{rbp}: no validation-selected SeqCal threshold is configured")
    metrics, predictions = evaluate_model(
        model,
        loader,
        device,
        protocol,
        float(entry.get("seqcal_threshold", 0.0)),
        amp=bool(config["runtime"]["amp"]),
    )
    return {
        "rbp": rbp,
        "split": split,
        "protocol": protocol,
        "seqcal_threshold": (
            float(entry["seqcal_threshold"]) if protocol == "seqcal" else None
        ),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "selected_epoch": int(checkpoint["selected_epoch"]),
        "split_hash": computed_hash,
        "metrics": metrics,
        "predictions": predictions,
    }


def verify_all_checkpoints(config: dict[str, Any], device: torch.device) -> None:
    for rbp in RBPS:
        entry = config["datasets"][rbp]
        _, computed_hash, _ = configured_rbp(config, rbp)
        if "checkpoint" not in entry or "checkpoint_sha256" not in entry:
            raise RuntimeError(f"{rbp}: no fixed-split checkpoint is configured yet")
        path = resolve_project_path(entry["checkpoint"])
        actual_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_checksum != entry["checkpoint_sha256"]:
            raise RuntimeError(f"{rbp}: checkpoint checksum mismatch")
        model, checkpoint = build_model_from_checkpoint(path, device)
        del model
        if checkpoint.get("split_hash") != computed_hash:
            raise RuntimeError(f"{rbp}: checkpoint was trained with a different split")
        if checkpoint.get("split_strategy") != config["split_protocol"]["strategy"]:
            raise RuntimeError(f"{rbp}: checkpoint split strategy mismatch")
        print(f"{rbp}: checkpoint OK (epoch {checkpoint['selected_epoch']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--rbp", choices=(*RBPS, "all"), default="all")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--protocol", choices=("native", "seqcal"), default="seqcal")
    parser.add_argument("--data-root", help="Override the configured circGMST data directory")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output")
    parser.add_argument("--checkpoints-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    if args.checkpoints_only:
        verify_all_checkpoints(config, device)
        return
    rbps = RBPS if args.rbp == "all" else (args.rbp,)
    results = [
        evaluate_rbp(
            config,
            rbp,
            device,
            args.split,
            args.protocol,
            args.batch_size,
            args.data_root,
        )
        for rbp in rbps
    ]
    output = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT
        / config["split_protocol"]["output_root"]
        / "evaluation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, allow_nan=True))
    for result in results:
        values = result["metrics"]
        print(
            f"{result['rbp']}: F1@0.5={values['event_f1_iou_0_5']:.6f} "
            f"F1@0.7={values['event_f1_iou_0_7']:.6f} "
            f"CountMAE={values['count_mae']:.6f}"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
