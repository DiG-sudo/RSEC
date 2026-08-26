"""Dataset parsing, predefined split validation, and batching for RSEC."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


RBPS = ("EIF4A3", "EWSR1", "FXR1", "FXR2", "IGF2BP2", "IGF2BP3", "MOV10")
FIXED_SPLIT_STRATEGY = "sequence_level_fixed_holdout_60_20_20"
FIXED_SPLIT_SEED = 42
FIXED_VALIDATION_SEED = FIXED_SPLIT_SEED


def read_fasta(path: str | Path) -> Iterable[tuple[str, str]]:
    header: str | None = None
    buffer: list[str] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(buffer)
            header, buffer = line[1:], []
        else:
            buffer.append(line.strip())
    if header is not None:
        yield header, "".join(buffer)


def sequence_id(header: str) -> str:
    return header.lstrip(">").split()[0].split("|")[0] if header.strip() else ""


def parse_site_header(header: str) -> tuple[str, int, int] | None:
    header = header.lstrip(">")
    parts = header.split()
    if len(parts) >= 3:
        try:
            return parts[0].split("|")[0], int(parts[-2]), int(parts[-1])
        except ValueError:
            pass
    integers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?![A-Za-z])", header)
    if parts and len(integers) >= 2:
        return parts[0].split("|")[0], int(integers[-2]), int(integers[-1])
    return None


def read_sites(path: str | Path) -> list[tuple[str, int, int]]:
    parsed = [parse_site_header(header) for header, _ in read_fasta(path)]
    return [item for item in parsed if item is not None]


def one_hot(sequence: str) -> np.ndarray:
    encoded = np.zeros((4, len(sequence)), dtype=np.float32)
    alphabet = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
    for index, nucleotide in enumerate(sequence.upper()):
        channel = alphabet.get(nucleotide)
        if channel is not None:
            encoded[channel, index] = 1.0
    return encoded


class SequenceDataset(Dataset):
    def __init__(
        self, sequences: dict[str, str], labels: dict[str, np.ndarray], identifiers: Sequence[str]
    ):
        self.sequences = sequences
        self.labels = labels
        self.ids = list(identifiers)
        # Match the authoritative training runner's --cache-sample-tensors
        # behavior: encode each sequence once before DataLoader workers start.
        self._feature_cache = [
            torch.from_numpy(one_hot(self.sequences[identifier]))
            for identifier in self.ids
        ]
        self._label_cache = [
            torch.from_numpy(self.labels[identifier]) for identifier in self.ids
        ]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        identifier = self.ids[index]
        return self._feature_cache[index], self._label_cache[index], identifier


def collate_sequences(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    lengths = torch.tensor([features.size(1) for features, _, _ in batch], dtype=torch.long)
    maximum = int(lengths.max())
    features = torch.zeros(len(batch), 4, maximum)
    labels = torch.zeros(len(batch), maximum)
    mask = torch.zeros(len(batch), maximum, dtype=torch.bool)
    for index, (item_features, item_labels, _) in enumerate(batch):
        length = item_features.size(1)
        features[index, :, :length] = item_features
        labels[index, :length] = item_labels
        mask[index, :length] = True
    return features, labels, mask, lengths, [identifier for _, _, identifier in batch]


def _sample_length(dataset: Dataset, index: int) -> int:
    if isinstance(dataset, Subset):
        return _sample_length(dataset.dataset, dataset.indices[index])
    if not isinstance(dataset, SequenceDataset):
        raise TypeError("unsupported dataset type")
    return len(dataset.sequences[dataset.ids[index]])


def _batch_indices(
    dataset: Dataset, batch_size: int, shuffle: bool, seed: int
) -> list[list[int]]:
    rng = np.random.RandomState(seed)
    edges = (500, 1000, 2000, 4000, 6000)
    groups: dict[float, list[int]] = {edge: [] for edge in (*edges, float("inf"))}
    for index in range(len(dataset)):
        edge = next(
            (edge for edge in edges if _sample_length(dataset, index) <= edge), float("inf")
        )
        groups[edge].append(index)
    batches: list[list[int]] = []
    for edge in (*edges, float("inf")):
        group = groups[edge]
        if shuffle:
            rng.shuffle(group)
        group.sort(key=lambda index: _sample_length(dataset, index))
        batches.extend(group[index : index + batch_size] for index in range(0, len(group), batch_size))
    if shuffle:
        rng.shuffle(batches)
    return batches


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=_batch_indices(dataset, batch_size, shuffle, seed),
        collate_fn=collate_sequences,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _split_payload(manifest: dict) -> dict:
    """Canonical payload for a new split hash.

    Protocol metadata are validated separately. The hash deliberately
    identifies the final ordered train/validation/test membership only, so
    every model using the same RBP receives the same value.
    """
    return {
        "train_ids": list(manifest["train_ids"]),
        "val_ids": list(manifest["val_ids"]),
        "test_ids": list(manifest["test_ids"]),
    }


def split_hash(manifest: dict) -> str:
    payload = _split_payload(manifest)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def load_annotated_sequences(
    data_root: str | Path, rbp: str
) -> tuple[dict[str, str], dict[str, np.ndarray], list[str]]:
    """Merge the former train/test annotations into one sequence-level pool."""
    if rbp not in RBPS:
        raise ValueError(f"unsupported RBP: {rbp}")
    directory = Path(data_root) / rbp
    required = [
        directory / f"{rbp}_seq.fasta",
        directory / f"{rbp}_train.fasta",
        directory / f"{rbp}_test.fasta",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing circGMST files. See data/README.md. Missing: " + ", ".join(missing)
        )
    sequences = {sequence_id(header): sequence for header, sequence in read_fasta(required[0])}
    if "" in sequences:
        raise RuntimeError(f"{rbp}: empty sequence ID in sequence FASTA")
    # All available sequence IDs enter the new split. The old train/test files
    # are used only as annotation sources and confer no split identity.
    identifiers = sorted(sequences)
    labels = {key: np.zeros(len(sequences[key]), dtype=np.float32) for key in identifiers}
    for key, start, end in [*read_sites(required[1]), *read_sites(required[2])]:
        if key not in labels:
            raise RuntimeError(f"{rbp}: annotation references unknown sequence ID {key}")
        if start < end:
            labels[key][max(0, start - 1) : min(len(labels[key]), end)] = 1.0
    return sequences, labels, identifiers


def load_splits(
    data_root: str | Path,
    rbp: str,
    manifest_path: str | Path,
) -> tuple[dict[str, Subset], str, dict]:
    """Load the predefined fixed split; incompatible manifests are rejected."""
    sequences, labels, all_ids = load_annotated_sequences(data_root, rbp)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("split_strategy") != FIXED_SPLIT_STRATEGY:
        raise RuntimeError(
            f"incompatible split strategy: expected {FIXED_SPLIT_STRATEGY!r}, "
            f"found {manifest.get('split_strategy', manifest.get('strategy'))!r}"
        )
    if manifest.get("rbp") != rbp:
        raise RuntimeError(f"manifest RBP {manifest.get('rbp')!r} does not match {rbp!r}")

    if int(manifest.get("split_seed", -1)) != FIXED_SPLIT_SEED:
        raise RuntimeError(
            f"manifest split_seed must be {FIXED_SPLIT_SEED}"
        )

    if int(manifest.get("validation_seed", -1)) != FIXED_VALIDATION_SEED:
        raise RuntimeError(
            f"manifest validation_seed must be {FIXED_VALIDATION_SEED}"
        )
    payload = _split_payload(manifest)
    if any(
        payload[f"{name}_ids"] != sorted(payload[f"{name}_ids"])
        for name in ("train", "val", "test")
    ):
        raise RuntimeError("manifest ID lists must use canonical lexicographic ordering")
    flattened = payload["train_ids"] + payload["val_ids"] + payload["test_ids"]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("split manifest contains overlapping or duplicate IDs")
    if set(flattened) != set(all_ids):
        missing = sorted(set(all_ids) - set(flattened))
        extra = sorted(set(flattened) - set(all_ids))
        raise RuntimeError(
            f"split manifest does not cover all sequence IDs; missing={missing[:3]}, extra={extra[:3]}"
        )
    counts = {name: len(payload[f"{name}_ids"]) for name in ("train", "val", "test")}
    if int(manifest.get("total_sequence_count", -1)) != len(all_ids):
        raise RuntimeError("manifest total_sequence_count is incorrect")
    for name, count in counts.items():
        if int(manifest.get(f"{name}_count", -1)) != count:
            raise RuntimeError(f"manifest {name}_count is incorrect")
        expected_ratio = count / len(all_ids)
        if abs(float(manifest.get(f"{name}_ratio", -1)) - expected_ratio) > 1e-12:
            raise RuntimeError(f"manifest {name}_ratio is incorrect")
    if counts["val"] != int(round(0.20 * len(all_ids))):
        raise RuntimeError("validation size does not equal round(0.20 * total)")
    if counts["test"] != -(-len(all_ids) // 5):
        raise RuntimeError("test size does not equal ceil(0.20 * total)")
    computed_hash = split_hash(manifest)
    if manifest.get("split_hash") != computed_hash:
        raise RuntimeError("split manifest hash does not match its contents")

    dataset = SequenceDataset(sequences, labels, all_ids)
    lookup = {identifier: index for index, identifier in enumerate(all_ids)}
    subsets = {
        name: Subset(dataset, [lookup[key] for key in payload[f"{name}_ids"]])
        for name in ("train", "val", "test")
    }
    return subsets, computed_hash, manifest


def dense_labels_to_intervals(
    labels: torch.Tensor, sequence_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Convert maximal positive runs into inclusive start/end event targets."""
    valid = (labels >= 0.5) & sequence_mask.bool()
    padded = torch.nn.functional.pad(valid, (1, 1), value=False)
    start_edges = (~padded[:, :-1]) & padded[:, 1:]
    end_edges = padded[:, :-1] & (~padded[:, 1:])
    counts = start_edges.sum(-1).long()
    capacity = int(counts.max().item()) if labels.size(0) else 0
    starts = torch.zeros(labels.size(0), capacity, dtype=torch.long, device=labels.device)
    ends = torch.zeros_like(starts)
    event_mask = torch.zeros(labels.size(0), capacity, dtype=torch.bool, device=labels.device)
    for row in range(labels.size(0)):
        count = int(counts[row])
        starts[row, :count] = start_edges[row].nonzero(as_tuple=False).flatten()
        ends[row, :count] = end_edges[row].nonzero(as_tuple=False).flatten() - 1
        event_mask[row, :count] = True
    return {
        "gold_starts": starts,
        "gold_ends": ends,
        "gold_event_mask": event_mask,
        "gold_event_counts": counts,
    }


def labels_to_event_rows(labels: torch.Tensor, mask: torch.Tensor) -> list[list[tuple[int, int]]]:
    targets = dense_labels_to_intervals(labels, mask)
    rows: list[list[tuple[int, int]]] = []
    for index, count in enumerate(targets["gold_event_counts"].tolist()):
        rows.append(
            [
                (
                    int(targets["gold_starts"][index, event]),
                    int(targets["gold_ends"][index, event]),
                )
                for event in range(count)
            ]
        )
    return rows

