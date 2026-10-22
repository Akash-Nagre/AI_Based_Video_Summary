# Human evaluation protocol

Create one annotation JSON per source video, review the full source, and map topic IDs to reference topics. Keep the same annotation and target duration when comparing selector versions.

```json
{
  "video_id": "example-video",
  "target_summary_fraction": 0.25,
  "topics": [
    {"topic_id": 0, "label": "reviewer-written topic", "importance": 0.8,
     "supporting_intervals": [[12.0, 48.0]], "essential_intervals": [[24.0, 34.0]],
     "adequately_represented": true}
  ],
  "essential_intervals": [[24.0, 34.0], [130.0, 148.0]],
  "optional_intervals": [[50.0, 58.0]],
  "human_coherence_rating_1_to_5": 4,
  "factual_consistency_rating_1_to_5": 4
}
```

`tools/evaluate_summary.py` computes segment precision/recall/F1 using one-to-one maximum-cardinality interval matching at temporal IoU ≥ 0.5, duration adherence, and overlapping clip pairs. Each reference interval and each selected clip can count in at most one match, which prevents duplicate clips from inflating recall. Importance-weighted topic coverage is reported only when reference topic IDs have been mapped to topic entries in the system report; it is not available for the uniform, transcript-only, or vision-only baseline reports. Interval F1 does not assess whether the words inside a clip explain a topic. Human ratings and factual consistency require reviewer entry and are not inferred by the script.

For a valid before/after comparison, use identical videos, annotations, summary fraction, model/cache configuration, and the same time budget. Record the commit, environment, model IDs, total/stage timings, and the complete report. Compare ablations by disabling one signal at a time; do not interpret a higher metric as an improvement without inspecting representative failures.

## Saved selector baseline experiment

On 2026-10-22, uniform, transcript-only, and vision-only comparison videos were generated from the same 909.8-second Hindi thermodynamics source used by the default batch-1 run. The target budget was 227.45 seconds. The default method selected 227.34 seconds in 21 clips. Uniform selected 227.45 seconds in 21 evenly spaced equal-duration windows. Transcript-only selected 227.42 seconds in 18 of the shared 57 candidate clips; vision-only selected 227.36 seconds in 20 candidates. The single-signal variants maximize normalized score multiplied by clip duration with a 10-ms duration-knapsack and do not apply topic-coverage rewards. All three baseline MP4s were fully decoded by FFmpeg and use H.264/AAC. These are budget and output-validity results; human reference intervals and reviewer ratings are still pending.

The output videos, interval reports, blind review packet, and source hash are in `data/evaluation_baselines_20261022/`. Recreate the selector baselines with:

```powershell
python tools/run_comparative_baselines.py data/jobs/b2221dc7 data/evaluation_baselines_20261022
```
