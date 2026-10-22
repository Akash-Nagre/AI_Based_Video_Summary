"""Create fixed-budget video summaries for simple selector baselines.

The baseline candidates come from one saved full-pipeline run, so all methods
use the same source, transcript segmentation, 25% duration budget, and codecs.
The uniform baseline uses evenly spaced source windows. Transcript-only and
vision-only use a 0/1 duration-knapsack over the saved candidate clips, with
their respective normalized score as utility per second. This isolates the
selection signal; it does not compare different ASR systems.

Usage:
  python tools/run_comparative_baselines.py data/jobs/b2221dc7 data/eval_baselines
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.highlights.clip_extractor import extract_clip
from modules.highlights.clip_merger import merge_clips


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def duration_knapsack(rows: list[dict], score_key: str, budget: float) -> list[dict]:
    """Maximize summed score×duration under budget using 10-ms units."""
    unit = 100
    capacity = int(math.floor(budget * unit + 1e-8))
    costs = [int(math.ceil((r["end"] - r["start"]) * unit - 1e-8)) for r in rows]
    values = [(float(r[score_key]) * (r["end"] - r["start"])) for r in rows]
    dp = np.zeros(capacity + 1, dtype=np.float64)
    take = np.zeros((len(rows), capacity + 1), dtype=np.uint8)
    for i, (cost, value) in enumerate(zip(costs, values)):
        if cost > capacity:
            continue
        previous = dp[:-cost].copy() if cost else dp.copy()
        candidate = previous + value
        better = candidate > dp[cost:] + 1e-12
        dp[cost:][better] = candidate[better]
        take[i, cost:][better] = 1
    chosen = []
    remaining = capacity
    for i in range(len(rows) - 1, -1, -1):
        if take[i, remaining]:
            chosen.append(rows[i])
            remaining -= costs[i]
    chosen.reverse()
    # Keep a strict wall-clock budget despite rounding of candidate intervals.
    while sum(r["end"] - r["start"] for r in chosen) > budget + 1e-8:
        chosen.pop()
    return chosen


def make_uniform(video_duration: float, budget: float, count: int) -> list[dict]:
    window = budget / count
    last_start = max(0.0, video_duration - window)
    starts = np.linspace(0.0, last_start, count)
    return [
        {"start": float(start), "end": float(start + window), "text": ""}
        for start in starts
    ]


def render_method(source: Path, method_dir: Path, intervals: list[dict]) -> Path:
    clips_dir = method_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    paths = [clips_dir / f"clip_{i:03d}.mp4" for i in range(len(intervals))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(extract_clip, str(source), row["start"], row["end"], str(path))
            for row, path in zip(intervals, paths)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    output = method_dir / "summary.mp4"
    merge_clips([str(p) for p in paths], str(output))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    job = args.job_dir.resolve()
    out = args.output_dir.resolve()
    existing_complete = out.exists() and all(
        (out / method / "summary.mp4").exists()
        for method in ("uniform", "transcript_only", "vision_only")
    )
    if out.exists() and any(out.iterdir()) and not existing_complete:
        raise SystemExit(f"Output directory contains an incomplete prior run: {out}")
    out.mkdir(parents=True, exist_ok=True)

    source = job / "original.mp4"
    if not source.exists():
        raise SystemExit(f"Missing source video: {source}")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    report = read_json(job / "coverage_report.json")
    rows = read_json(job / "scores.json")
    target = float(report["target_summary_duration_seconds"])
    video_duration = float(report["video_duration_seconds"])
    proposed = report["selected_clips"]
    clip_count = len(proposed)

    methods = {
        "uniform": make_uniform(video_duration, target, clip_count),
        "transcript_only": duration_knapsack(rows, "text_score", target),
        "vision_only": duration_knapsack(rows, "visual_score", target),
    }
    manifest = {
        "source_sha256": source_hash,
        "source_job": str(job),
        "video_duration_seconds": video_duration,
        "target_fraction": report["target_summary_fraction"],
        "target_duration_seconds": target,
        "proposed_run_summary": str(job / "summary.mp4"),
        "proposed_selected_intervals": [
            {"start": c["start"], "end": c["end"], "topic_id": c.get("topic_id")}
            for c in proposed
        ],
        "baseline_definitions": {
            "uniform": "Evenly spaced, equal-duration source windows; count matches proposed system.",
            "transcript_only": "Duration-knapsack over the same candidate clips, utility = transcript score × clip duration.",
            "vision_only": "Duration-knapsack over the same candidate clips, utility = visual activity score × clip duration.",
        },
        "methods": {},
        "human_evaluation": "Pending human reference intervals and reviewer ratings.",
    }
    for name, selected in methods.items():
        seconds = sum(float(r["end"]) - float(r["start"]) for r in selected)
        method_dir = out / name
        method_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "method": name,
            "selected_duration_seconds": seconds,
            "target_duration_seconds": target,
            "candidate_count": len(selected),
            "selected_intervals": [
                {"start": float(r["start"]), "end": float(r["end"]), "text": r.get("text", ""), "topic_id": r.get("topic_id")}
                for r in selected
            ],
        }
        (method_dir / "selection.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        evaluation_report = {
            "video_duration_seconds": video_duration,
            "target_summary_fraction": report["target_summary_fraction"],
            "target_summary_duration_seconds": target,
            "actual_summary_duration_seconds": seconds,
            "selected_clips": [
                {"start": float(r["start"]), "end": float(r["end"]), "text": r.get("text", "")}
                for r in selected
            ],
            "topics": [],
        }
        (method_dir / "coverage_report.json").write_text(json.dumps(evaluation_report, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_path = method_dir / "summary.mp4"
        if not summary_path.exists():
            render_method(source, method_dir, selected)
        entry["summary_video"] = str(summary_path)
        manifest["methods"][name] = entry
        (out / "baseline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
