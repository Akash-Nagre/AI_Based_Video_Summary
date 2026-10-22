# AI Video Summarization System

Local video summarization with Whisper transcription, semantic text embeddings, transcript-derived topic discovery, visual activity features, duration-budgeted clip selection, and FFmpeg rendering.

## Run locally

Requirements: Python 3.11+, FFmpeg and FFprobe on `PATH`, and enough disk space for source video, extracted audio, and generated clips. First transcription and embedding runs may download model weights; core processing does not need an API key. CPU inference is supported but model size and hardware affect processing time. Set `WHISPER_MODEL` to `tiny`, `base`, `small`, `medium`, or `large-v3`; default is `small`. CUDA is attempted automatically, with CPU int8 fallback.

The result page can show all available transcript text from the clips in the generated summary video, in clip order and in the detected source language. It uses the transcript already produced during video processing, so there is no separate text-generation model download. Speech-recognition errors can still affect the text, especially when the source audio is unclear.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```

The website and API start together at `http://127.0.0.1:8000`; `/docs` lists the API endpoints. Open the home page to upload a video and create a summary. The API also supports `POST /upload`, `POST /summarize/{job_id}?target_summary_fraction=0.25` (accepted range: 0.15–0.40), `GET /status/{job_id}`, `GET /result/{job_id}`, `GET /coverage/{job_id}`, and `GET /download/{job_id}`. Generated files and JSON artifacts are stored under `data/jobs/{job_id}`.

## Run with Docker

Docker packages Python, the Python dependencies, and FFmpeg in a container. The other PC needs Docker Desktop (or Docker Engine with Compose), but does not need a separate Python or FFmpeg installation.

1. Share a zip of the project source, including `Dockerfile`, `compose.yaml`, `.dockerignore`, `backend/`, `modules/`, `frontend/`, and `requirements.txt`. Exclude `.venv/`, `.git/`, generated videos, and the contents of `data/`.
2. On the other PC, install and start Docker Desktop, extract the zip, and open PowerShell in the extracted project folder.
3. Build and start the app:

   ```powershell
   docker compose up --build
   ```

4. Open `http://localhost:8000` in a browser on that PC. Stop the app with `Ctrl+C` in the PowerShell window; run `docker compose down` later if needed.

The first build downloads the base image and Python packages. The first summary may also download AI model weights from the internet; Docker keeps those weights in a persistent volume for later runs. Uploaded videos and results stay in the project folder's `data/` directory. The default container setup uses CPU processing; NVIDIA GPU access needs additional Docker GPU configuration. The Compose service uses one application worker because job state is currently stored in memory.

### Let someone else use the website

The address above is local to your computer. To let another person access the app, deploy this FastAPI service on a server they can reach and share its HTTPS address. The server needs Python dependencies, FFmpeg/FFprobe, sufficient storage and memory, and model weights; video processing happens on that server. The current job registry is in memory, so keep a single application process running while jobs are active. Before making the service public, add user authentication, upload and storage limits, and a cleanup policy for uploaded videos and generated results. The app currently has no login or automatic deletion.

## Performance and cache settings

Each run writes stage timings to `timings.json`, resource snapshots to `telemetry.json`, and structured stage events to the backend log. Frame counts, audio duration, transcript and candidate counts, estimated visual-change count, topic count, and render times are included when available. `data/cache` stores atomic, content-fingerprinted audio, transcript, and visual-score intermediates. Cache keys include source bytes and processing settings; changing the source or the transcription/visual configuration creates a fresh entry.

Optional environment settings (defaults preserve the current analysis settings):

| Setting | Default | Purpose |
| --- | --- | --- |
| `WHISPER_MODEL` | `small` | Faster-Whisper model name |
| `WHISPER_BEAM_SIZE` | `3` | Decode beam size; smaller values can be faster and may change recognition |
| `WHISPER_BATCH_SIZE` | `1` | Faster-Whisper batched inference; values above 1 are experimental and can change transcript context |
| `WHISPER_CPU_THREADS` | up to `8` | CPU threads used by Faster-Whisper's CPU fallback |
| `WHISPER_CUDA_COMPUTE_TYPE` | `float16` | CUDA inference precision; benchmark alternatives before changing |
| `WHISPER_CPU_COMPUTE_TYPE` | `int8` | CPU fallback precision |
| `VIDEO_SAMPLE_FPS` | `0.5` | Visual activity sampling rate |
| `VIDEO_DECODE_THREADS` | `4` | FFmpeg-backed OpenCV video decode threads |
| `VIDEO_ENCODER_PRESET` | `veryfast` | H.264 clip encoding speed preset |
| `VIDEO_ENCODER_CRF` | `20` | H.264 quality setting (lower is higher quality/larger files) |
| `MAX_PARALLEL_CLIP_ENCODERS` | `2` (maximum `4`) | Bounded parallel rendering of already-selected clips; does not change clip selection or encoding quality |
| `MAX_CONCURRENT_JOBS` | `1` | Bound simultaneous CPU/GPU-heavy summarization jobs |

Profiling does not itself establish a performance gain. Compare `timings.json` on the same source and hardware, separating the first content-cache miss from repeat runs. Embedding weights are checked locally first; a first run still downloads them if they are not already installed.

Measured before-and-after runs, output checks, and the batch-inference quality comparison are documented in [`docs/performance_optimization_benchmark.md`](docs/performance_optimization_benchmark.md).

## Selection and report

Candidate clips are built from timestamped transcript chunks. Embeddings are grouped dynamically by semantic similarity into discovered topic clusters; there is no fixed subject vocabulary. Selection balances the existing segment score with semantic diversity and newly covered topics, and stops at the requested fraction of source duration. Selected clips are returned in chronological order. The selector can return less than the target when candidates are insufficient; the target is a budget, not a promise to pad the summary.

`coverage_report.json` is generated per job and included in `/result/{job_id}`; it can also be fetched from `/coverage/{job_id}`. It records discovered topic labels and evidence text, source and selected timestamps, target and actual summary duration, and topic coverage gaps. Topic labels are representative source excerpts, not human-validated topic names. Current visual features capture visual activity/scene changes; they do not provide general object or action recognition. Silent or untranscribable videos use a clearly marked visual-activity-only path and produce no semantic topics or transcript summary.

Topic coverage is the importance-weighted sum of each topic's selected evidence duration divided by an estimated useful duration, capped at 1 per topic. The estimate is based on the strongest segment and available topic duration. This is a useful allocation diagnostic, not a factual completeness metric. `coverage_status` distinguishes adequate (≥0.75), partial, and omitted topics. The text view contains speech from the selected highlight clips only, without timestamps.

## Evaluation

The human annotation schema and metric definitions are in [`docs/evaluation.md`](docs/evaluation.md). Add annotations after reviewing a generated report, mapping discovered `topic_id` values to human judgments. Then run:

```powershell
python tools/evaluate_summary.py path\to\annotations.json data\jobs\JOB_ID\coverage_report.json
```

This reports topic coverage, interval precision/recall/F1 at temporal IoU 0.5, duration adherence, and clip overlap. No accuracy claims are made without reviewed annotations. Existing `tests/` scripts include real-model demonstrations that require external data; focused selector tests can be run with `python -m pytest tests/test_coverage_selector.py` after installing requirements and pytest (`python -m pip install pytest`).

## Tests

Run the focused automated regression tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_cache\perf-test-tmp tests/test_coverage_selector.py tests/test_performance_profile.py
```

Several older `tests/test_*.py` files are interactive demo scripts that execute during pytest collection and require the missing `data/input` sample video and `data/output` artifacts. The full `python -m pytest` collection therefore fails until those scripts are converted into ordinary tests. Full real-model inference additionally requires model weights and a supported sample video.
