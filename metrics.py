from __future__ import annotations

from typing import Sequence

import numpy as np


Event = tuple[int, int]
EventRows = Sequence[Sequence[Event]]


def interval_iou(left: Event, right: Event) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]) + 1)
    union = left[1] - left[0] + 1 + right[1] - right[0] + 1 - intersection
    return intersection / union


def exact_event_matching(
    gold: Sequence[Event], predicted: Sequence[Event], threshold: float
) -> list[tuple[int, int, float]]:
    best: tuple[int, float, list[tuple[int, int, float]]] = (-1, -1.0, [])

    def search(
        gold_index: int, used_predictions: set[int], matches: list[tuple[int, int, float]]
    ) -> None:
        nonlocal best
        if gold_index == len(gold):
            candidate = (len(matches), sum(item[2] for item in matches), matches.copy())
            if candidate[:2] > best[:2]:
                best = candidate
            return
        search(gold_index + 1, used_predictions, matches)
        for predicted_index, event in enumerate(predicted):
            if predicted_index in used_predictions:
                continue
            overlap = interval_iou(gold[gold_index], event)
            if overlap >= threshold:
                used_predictions.add(predicted_index)
                matches.append((gold_index, predicted_index, overlap))
                search(gold_index + 1, used_predictions, matches)
                matches.pop()
                used_predictions.remove(predicted_index)

    search(0, set(), [])
    return best[2]


def event_average_precision(
    gold_rows: EventRows,
    predicted_rows: EventRows,
    score_rows: Sequence[Sequence[float]],
    iou_threshold: float,
) -> float:
    total_gold = sum(len(row) for row in gold_rows)
    if not total_gold:
        return float("nan")
    proposals = [
        (float(score), sequence_index, event)
        for sequence_index, (events, scores) in enumerate(zip(predicted_rows, score_rows))
        for event, score in sorted(
            zip(events, scores), key=lambda item: (item[0][0], item[0][1])
        )
    ]
    order = np.argsort(-np.asarray([item[0] for item in proposals]), kind="mergesort")
    matched = [set() for _ in gold_rows]
    true_positives = 0
    precision_sum = 0.0
    for rank, proposal_index in enumerate(order, 1):
        _, sequence_index, event = proposals[int(proposal_index)]
        candidates = [
            (interval_iou(event, target), target_index)
            for target_index, target in enumerate(gold_rows[sequence_index])
            if target_index not in matched[sequence_index]
        ]
        if candidates:
            best_iou, best_target = max(candidates)
            if best_iou >= iou_threshold:
                matched[sequence_index].add(best_target)
                true_positives += 1
                precision_sum += true_positives / rank
    return precision_sum / total_gold


def event_metrics(gold_rows: EventRows, predicted_rows: EventRows) -> dict[str, float]:
    total_gold = sum(len(row) for row in gold_rows)
    total_predicted = sum(len(row) for row in predicted_rows)
    result: dict[str, float] = {}
    for threshold in (0.3, 0.5, 0.7):
        matched = sum(
            len(exact_event_matching(gold, predicted, threshold))
            for gold, predicted in zip(gold_rows, predicted_rows)
        )
        precision = matched / max(total_predicted, 1)
        recall = matched / max(total_gold, 1)
        suffix = str(threshold).replace(".", "_")
        result[f"event_precision_iou_{suffix}"] = precision
        result[f"event_recall_iou_{suffix}"] = recall
        result[f"event_f1_iou_{suffix}"] = (
            2 * precision * recall / max(precision + recall, 1e-12)
        )
    gold_counts = np.asarray([len(row) for row in gold_rows])
    predicted_counts = np.asarray([len(row) for row in predicted_rows])
    result["count_mae"] = float(np.abs(gold_counts - predicted_counts).mean())
    result["count_exact_accuracy"] = float((gold_counts == predicted_counts).mean())
    result["mean_gold_event_count"] = float(gold_counts.mean())
    result["mean_predicted_event_count"] = float(predicted_counts.mean())
    return result


def evaluate_predictions(
    gold_rows: EventRows,
    predicted_rows: EventRows,
    score_rows: Sequence[Sequence[float]],
) -> dict[str, float]:
    result = event_metrics(gold_rows, predicted_rows)
    average_precisions = {
        f"event_ap_iou_{str(threshold).replace('.', '_')}": event_average_precision(
            gold_rows, predicted_rows, score_rows, threshold
        )
        for threshold in (0.3, 0.4, 0.5, 0.6, 0.7)
    }
    result.update(average_precisions)
    result["event_ap_avg_0_3_0_7"] = float(np.mean(list(average_precisions.values())))
    return result
