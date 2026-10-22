# Summarization performance benchmark

## Test input and hardware

Runs used the same 909.8-second H.264 input at a 25% summary-duration budget. The Faster-Whisper model was `small`, running on an NVIDIA GeForce RTX 4050 6 GB through CTranslate2. The baseline used beam size 3 and standard non-batched inference. The optimized batch run kept the model, beam size, compute type (`float16`), visual sampling rate (0.5 fps), selection settings, output codec, and duration budget; it changed only the inference batch size to 4.

## Measured times

| Run | Transcription | Visual analysis | Clip extraction | Total | Cache state |
| --- | ---: | ---: | ---: | ---: | --- |
| Existing baseline | 847.66s | 96.12s | 144.70s | 1,127.53s (18m48s) | Prior run; timings saved with the job |
| Batch size 4 | 56.74s | 115.48s | 85.33s | 329.68s (5m30s) | Audio extraction reused; transcript and visual analysis recomputed; embedding model loaded cold |
| Batch size 4, CUDA overlap | 65.93s | 56.96s visual-stage wait (129.36s feature decode) | 90.69s | 259.40s (4m19s) | Transcript and visual features recomputed; audio and embeddings reused |
| Batch size 4, warm repeat | 0.04s | 0.01s | 95.91s | 141.41s (2m21s) | Audio, transcript, embeddings, and visual scores reused |

The baseline-to-batch-4 run was 70.8% faster on this input. The parallel run overlapped visual decoding with transcription and left 56.96 seconds of visual work after transcription and embedding work finished. The warm run is dominated by rendering the selected clips again.

The video decoded 27,294 frames and sampled 455 at 0.5 fps. The reported scene-change count is a visual-change estimate using a normalized grayscale-change threshold of 0.25; it is not semantic scene recognition.

## Quality and output checks

The baseline discovered 13 transcript topics; batch size 4 discovered 7. The baseline and batch-4 transcripts had 11.4% exact word-sequence similarity. Both coverage reports show 100% importance-weighted coverage, but that metric is computed over each run's own discovered topics, so it cannot establish that batch size 4 retained the baseline's topics. Both runs reported zero high-similarity clip pairs.

The baseline summary duration was 227.12 seconds; batch size 4 selected 225.91 seconds against a 227.45-second budget. The generated MP4 has H.264 video (226.07s) and AAC audio (226.08s), a 10ms stream-duration difference. FFmpeg decoded the entire output without errors.

Batch size 4 meets the under-nine-minute speed target on this machine, but its transcript and topic groupings differ substantially. It is therefore an opt-in benchmark setting; defaults remain batch size 1, beam size 3, and float16 on CUDA. The coverage metric is not human-verified factual completeness, so review the batch-4 result before using it for quality-sensitive output.

Other speed/precision experiments also remain opt-in. Beam size 1 transcribed the full source in 558.6 seconds versus 847.7 seconds at beam size 3, but its exact word-sequence similarity to the saved baseline transcript was only 4.7%. On a 30-second excerpt, CUDA `int8_float16` took 31.3 seconds versus 46.6 seconds for `float16`, but transcript similarity was 49.6%; this short excerpt is not enough to approve the precision change for production.

## Original bottlenecks

The baseline spent 847.66 seconds transcribing, 144.70 seconds encoding clips, and 96.12 seconds scanning video frames. The low-risk changes were a faster H.264 preset, reuse of source-fingerprinted intermediates, a local-first embedding model load, and bounded CUDA/video overlap. Batch inference is the largest measured speed change and carries the quality caveat above.

Benchmark artifacts are in `data/jobs/optimization_batch4_20261021`, `data/jobs/optimization_batch4_warm_20261021`, and `data/jobs/optimization_parallel_batch4_20261021`.

## Quality-preserving follow-up: bounded parallel clip rendering

The saved 12-minute run `data/jobs/b2221dc7` took 725.30 seconds (12m05s). Its active stages were transcription 579.33s, clip rendering 86.97s, embedding 24.96s, audio extraction 7.28s, and selection 3.54s. The `visual_feature_decoding_seconds` value of 129.36s in that run came from the visual-feature cache's original profiling metadata; it was not time spent during that run (`visual_scoring_seconds` was 0.047s). The visual cache therefore was already avoiding a full rescan.

Clip rendering used to invoke FFmpeg sequentially for each selected interval. Since each selected clip can be encoded independently after selection, a bounded pool of two encoders was measured on the same 21 intervals from the same 909.8-second source. Codec, preset, CRF, timestamps, audio mapping, and selection were unchanged.

| Measurement | Sequential | Two encoders | Change |
| --- | ---: | ---: | ---: |
| Clip rendering stage | 86.97s | 43.43s | 50.1% faster; 43.53s saved |
| Full pipeline, cache warm | — | 66.44s | Measured repeat run, including output rendering |
| Estimated cold full run | 725.30s | 681.76s (11m22s) | Estimated by replacing only the measured rendering stage; about 6.0% faster |

The complete warm rerun rediscovered 19 topics and selected exactly the same 21 timestamp/topic intervals as `b2221dc7`. Importance-weighted coverage remained 0.972155 and selected duration remained 227.34s against the same 227.45s budget. The generated H.264/AAC summary decoded fully with FFmpeg; stream durations were 227.667s and 227.683s (16.6ms apart). The benchmark outputs are under `data/jobs/parallel_render_eval` (stage-only experiment) and `data/jobs/optimization_parallel_render_20261021` (complete warm rerun).

The main cold-run time remains accurate transcription: 579.33s in this run, over 80% of the optimized cold-run estimate. Faster decoding settings (beam size 1, batch size 4, or reduced precision) changed transcripts/topics substantially in the prior experiments, so they remain opt-in and were not enabled for this quality-preserving change. Repeating a source with unchanged settings reuses its transcript, embeddings, and visual features; the measured warm pipeline took 66.44s, mostly rendering the final clips.

## Batch-size-2 quality check (not enabled)

To check whether a less aggressive inference setting could approach the requested one-third runtime, the same source was processed with `WHISPER_BATCH_SIZE=2`, keeping model `small`, beam size 3, and CUDA float16. Transcription took 36.28s and the whole run took 105.17s; audio and visual features were cached, while this batch-2 transcript and its embeddings were newly computed. The speed was high, but the result failed the quality-preservation check and is not enabled.

Against the default batch-1 transcript, the batch-2 transcript had a normalized word-token edit divergence of 95.8% and 31.5% unique-token recall (comparison against another model output, not ground truth). Dynamic discovery produced 5 topics versus 19, and 15 clips versus 21. Average best temporal IoU between each default clip and its closest batch-2 clip was 0.136. Both runs' internal coverage metrics were high, but they covered substantially different discovered content. This batch-2 output is therefore not evidence of preserved summary quality, despite meeting the runtime target.
