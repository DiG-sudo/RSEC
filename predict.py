from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from data import RBPS, one_hot, read_fasta, sequence_id
from evaluate import build_model_from_checkpoint, seqcal_prefix
from rsec import decode


PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_sequences(path: Path, limit: int | None) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for header, sequence in read_fasta(path):
        identifier = sequence_id(header)
        if not identifier:
            raise RuntimeError("FASTA contains an empty sequence identifier")
        if identifier in seen:
            raise RuntimeError(f"duplicate FASTA sequence identifier: {identifier}")
        sequence = sequence.upper().replace("U", "T")
        if not sequence:
            raise RuntimeError(f"empty sequence: {identifier}")
        seen.add(identifier)
        records.append((identifier, sequence))
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise RuntimeError(f"no FASTA records found in {path}")
    return records


def make_batch(records: list[tuple[str, str]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(len(sequence) for _, sequence in records)
    features = torch.zeros(len(records), 4, maximum, dtype=torch.float32, device=device)
    mask = torch.zeros(len(records), maximum, dtype=torch.bool, device=device)
    for index, (_, sequence) in enumerate(records):
        encoded = torch.from_numpy(one_hot(sequence)).to(device)
        features[index, :, : encoded.size(1)] = encoded
        mask[index, : encoded.size(1)] = True
    return features, mask


@torch.inference_mode()
def predict(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(Path(args.config).read_text())
    entry = config["datasets"][args.rbp]
    checkpoint_path = project_path(entry["checkpoint"])
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_sha256 != entry["checkpoint_sha256"]:
        raise RuntimeError(f"{args.rbp}: checkpoint checksum mismatch")

    device = torch.device(args.device)
    model, checkpoint = build_model_from_checkpoint(checkpoint_path, device)
    if checkpoint.get("split_hash") != entry["split_hash"]:
        raise RuntimeError(f"{args.rbp}: checkpoint split hash does not match config.yaml")
    if checkpoint.get("split_strategy") != config["split_protocol"]["strategy"]:
        raise RuntimeError(f"{args.rbp}: checkpoint split strategy mismatch")

    records = load_sequences(Path(args.input), args.limit)
    threshold = float(entry["seqcal_threshold"])
    predictions: list[dict[str, Any]] = []
    for offset in range(0, len(records), args.batch_size):
        batch = records[offset : offset + args.batch_size]
        features, mask = make_batch(batch, device)
        decoded = decode(model, features, mask)
        for index, (identifier, sequence) in enumerate(batch):
            native_events = list(decoded["generation_events"][index])
            native_scores = list(decoded["generation_confidences"][index])
            if args.protocol == "seqcal":
                events, scores = seqcal_prefix(native_events, native_scores, threshold)
                termination = (
                    "confidence_below_threshold"
                    if len(events) < len(native_events)
                    else decoded["termination_reason"][index]
                )
            else:
                events, scores = native_events, native_scores
                termination = decoded["termination_reason"][index]
            ordered = sorted(zip(events, scores), key=lambda item: item[0])
            predictions.append(
                {
                    "id": identifier,
                    "length": len(sequence),
                    "predicted_events": [
                        {"start": int(start), "end": int(end), "confidence": float(score)}
                        for (start, end), score in ordered
                    ],
                    "termination_reason": termination,
                }
            )

    return {
        "model": "RSEC",
        "rbp": args.rbp,
        "protocol": args.protocol,
        "coordinate_convention": "zero-based inclusive [start, end]",
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": checkpoint_sha256,
        "training_split_hash": checkpoint["split_hash"],
        "seqcal_threshold": threshold if args.protocol == "seqcal" else None,
        "predictions": predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a paper RSEC checkpoint on an arbitrary nucleotide FASTA file.')
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--rbp", required=True, choices=RBPS)
    parser.add_argument("--input", required=True, help="Unlabelled nucleotide FASTA")
    parser.add_argument("--output", required=True, help="Output JSON")
    parser.add_argument("--protocol", choices=("native", "seqcal"), default="seqcal")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Optional record limit for a smoke test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    payload = predict(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"wrote {len(payload['predictions'])} predictions to {output}")


if __name__ == "__main__":
    main()
