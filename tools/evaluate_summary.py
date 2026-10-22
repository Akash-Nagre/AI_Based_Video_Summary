"""Evaluate a coverage report against human annotations (stdlib only).

Usage: python tools/evaluate_summary.py annotations.json coverage_report.json
Topic ids in the annotations must be mapped to discovered topic ids during review.
"""
import json
import sys
from pathlib import Path


def overlap(a, b):
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union else 0.0


def maximum_iou_matches(gold, predicted, threshold=0.5):
    """Return a one-to-one maximum-cardinality match at the IoU threshold."""
    adjacency = []
    for reference in gold:
        candidates = [
            (idx, overlap(reference, candidate))
            for idx, candidate in enumerate(predicted)
        ]
        adjacency.append([
            idx for idx, iou in sorted(candidates, key=lambda pair: (-pair[1], pair[0]))
            if iou >= threshold
        ])

    predicted_to_gold = [-1] * len(predicted)

    def augment(gold_idx, seen):
        for pred_idx in adjacency[gold_idx]:
            if pred_idx in seen:
                continue
            seen.add(pred_idx)
            previous = predicted_to_gold[pred_idx]
            if previous == -1 or augment(previous, seen):
                predicted_to_gold[pred_idx] = gold_idx
                return True
        return False

    for gold_idx in range(len(gold)):
        augment(gold_idx, set())

    pairs = []
    for pred_idx, gold_idx in enumerate(predicted_to_gold):
        if gold_idx >= 0:
            pairs.append({
                "reference_index": gold_idx,
                "prediction_index": pred_idx,
                "temporal_iou": overlap(gold[gold_idx], predicted[pred_idx]),
            })
    return pairs


def evaluate(reference, report):
    selected = [[float(x["start"]), float(x["end"])] for x in report.get("selected_clips", [])]
    gold = [[float(x[0]), float(x[1])] for x in reference.get("essential_intervals", [])]
    matches = maximum_iou_matches(gold, selected, threshold=0.5)
    matched = len(matches)
    recall = matched / len(gold) if gold else 0.0
    p = matched / len(selected) if selected else 0.0
    f1 = 2*p*recall/(p+recall) if p+recall else 0.0
    refs = {str(t["topic_id"]): float(t.get("importance", 1.0)) for t in reference.get("topics", [])}
    found = {str(t["topic_id"]): float(t.get("coverage_score", 0)) for t in report.get("topics", [])}
    total_weight = sum(refs.values())
    weighted = (
        sum(weight * found.get(tid, 0.0) for tid, weight in refs.items()) / total_weight
        if total_weight and found else None
    )
    duration = float(report.get("actual_summary_duration_seconds", 0))
    target = float(report.get("target_summary_duration_seconds", 0))
    redundancy = sum(1 for i, a in enumerate(selected) for b in selected[i+1:] if overlap(a, b) > 0)
    return {"topic_coverage_weighted": weighted, "segment_precision_iou_0.5": p,
        "segment_recall_iou_0.5": recall, "segment_f1_iou_0.5": f1,
        "one_to_one_interval_matches": matches,
        "duration_seconds": duration, "target_duration_seconds": target,
        "duration_adherence": (duration <= target + 1.0) if target else None,
        "overlapping_clip_pairs": redundancy}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    ref, report = (json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[1:])
    print(json.dumps(evaluate(ref, report), indent=2))
