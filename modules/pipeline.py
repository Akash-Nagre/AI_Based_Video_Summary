
import json
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path

from modules.audio.whisper_transcriber import transcribe_audio
from modules.nlp.transcript_cleaner import clean_transcript
from modules.nlp.segmenter import segment_transcript
from modules.nlp.importance_scorer import (
    load_embedding_model,
    generate_embeddings,
    calculate_text_scores,
)
from modules.nlp.temporal_scorer import calculate_temporal_scores
from modules.nlp.topic_model import discover_topics, assess_topic_coverage
from modules.vision.visual_features import extract_visual_features, score_visual_features
from modules.multimodal.highlight_selector import (
    calculate_base_scores,
    select_highlights,
)
from modules.highlights.clip_extractor import extract_clip
from modules.highlights.clip_merger import merge_clips
from faster_whisper import __version__ as FASTER_WHISPER_VERSION

logger = logging.getLogger("video_pipeline")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
_previous_system_times = None
_visual_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-visual")


def _fingerprint(path: Path) -> str:
    """Content fingerprint used to invalidate reusable per-job intermediates."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    _save_json(temp_path, data)
    os.replace(temp_path, path)


def _copy_atomic(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + f".{uuid.uuid4().hex}.tmp")
    shutil.copyfile(source, temp_path)
    os.replace(temp_path, destination)


def _cached_visual_features(video_path: Path, cache_path: Path, sample_fps: float):
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return (np.asarray(cached["times"], dtype=float),
                np.asarray(cached["changes"], dtype=float),
                cached.get("profiling", {}))
    profiling = {}
    started = time.perf_counter()
    times, changes = extract_visual_features(str(video_path), sample_fps, profiling)
    profiling["visual_feature_decoding_seconds"] = time.perf_counter() - started
    _write_json_atomic(cache_path, {"times": times.tolist(), "changes": changes.tolist(),
                                    "profiling": profiling})
    return times, changes, profiling


def _system_snapshot():
    """Best-effort process/system telemetry; optional dependencies stay optional."""
    snapshot = {"cpu_percent": None, "ram_used_bytes": None,
                "process_rss_bytes": None, "gpu_available": False,
                "gpu_utilization_percent": None,
                "gpu_memory_used_bytes": None,
                "gpu_memory_total_bytes": None,
                "gpu_memory_allocated_bytes": None, "gpu_memory_reserved_bytes": None}
    try:
        import psutil
        process = psutil.Process()
        snapshot["cpu_percent"] = psutil.cpu_percent(interval=None)
        snapshot["ram_used_bytes"] = psutil.virtual_memory().used
        snapshot["process_rss_bytes"] = process.memory_info().rss
    except Exception:
        pass
    if os.name == "nt" and (snapshot["cpu_percent"] is None or snapshot["ram_used_bytes"] is None):
        try:
            import ctypes
            from ctypes import wintypes

            class _FileTime(ctypes.Structure):
                _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [("length", wintypes.DWORD), ("memory_load", wintypes.DWORD),
                            ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_extended_virtual", ctypes.c_ulonglong)]

            memory = _MemoryStatus()
            memory.length = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                snapshot["ram_used_bytes"] = memory.total_phys - memory.avail_phys

            idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                to_int = lambda value: (value.high << 32) | value.low
                current = (to_int(idle), to_int(kernel), to_int(user))
                global _previous_system_times
                if _previous_system_times is not None:
                    delta_idle = current[0] - _previous_system_times[0]
                    delta_total = (current[1] - _previous_system_times[1]) + (current[2] - _previous_system_times[2])
                    if delta_total > 0:
                        snapshot["cpu_percent"] = max(0.0, min(100.0, 100.0 * (delta_total - delta_idle) / delta_total))
                _previous_system_times = current

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD),
                            ("peak_working_set_size", ctypes.c_size_t), ("working_set_size", ctypes.c_size_t),
                            ("quota_peak_paged_pool_usage", ctypes.c_size_t), ("quota_paged_pool_usage", ctypes.c_size_t),
                            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t), ("quota_non_paged_pool_usage", ctypes.c_size_t),
                            ("pagefile_usage", ctypes.c_size_t), ("peak_pagefile_usage", ctypes.c_size_t)]
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                snapshot["process_rss_bytes"] = counters.working_set_size
        except Exception:
            pass
    try:
        torch = sys.modules.get("torch")
        if torch is None:
            raise RuntimeError("torch has not been loaded")
        if torch.cuda.is_available():
            snapshot["gpu_available"] = True
            snapshot["gpu_memory_allocated_bytes"] = torch.cuda.memory_allocated()
            snapshot["gpu_memory_reserved_bytes"] = torch.cuda.memory_reserved()
    except Exception:
        pass
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            snapshot["gpu_available"] = True
            result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                     "--format=csv,noheader,nounits"],
                                    check=True, capture_output=True, text=True, timeout=2)
            utilization, memory_mib, total_mib = result.stdout.strip().splitlines()[0].split(",")
            snapshot["gpu_utilization_percent"] = float(utilization.strip())
            snapshot["gpu_memory_used_bytes"] = int(memory_mib.strip()) * 1024 * 1024
            snapshot["gpu_memory_total_bytes"] = int(total_mib.strip()) * 1024 * 1024
    except Exception:
        pass
    return snapshot


def _record_stage(timings, telemetry, name, started):
    elapsed = time.perf_counter() - started
    timings[name] = elapsed
    telemetry[name] = _system_snapshot()
    logger.info(json.dumps({"event": "pipeline_stage", "stage": name,
                            "seconds": round(elapsed, 4),
                            "resources": telemetry[name]}, separators=(",", ":")))
    return elapsed


def extract_audio(video_path: str, audio_path: str):
    """Extract mono, 16 kHz audio using FFmpeg."""
    output_file = Path(audio_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(output_file)
    ]

    subprocess.run(command, check=True)
    return str(output_file)


def _save_json(path: Path, data):
    """Write JSON output safely with readable formatting."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


def process_video(
    video_path,
    output_dir,
    number_of_highlights=None,
    target_summary_fraction=0.25,
):
    """
    Process a video and generate a highlight summary.

    Returns transcript, candidate segments, selected segments,
    clip paths, final video path, and stage timing information.
    """
    print("\n=== AI VIDEO SUMMARIZATION PIPELINE ===\n")

    total_start = time.perf_counter()
    timings = {}
    telemetry = {}

    video_file = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(__file__).resolve().parents[1] / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    audio_path = output_dir / "audio.wav"
    transcript_path = output_dir / "transcript.json"
    segments_path = output_dir / "segments.json"
    scores_path = output_dir / "scores.json"
    timings_path = output_dir / "timings.json"
    highlights_dir = output_dir / "highlights"
    highlights_dir.mkdir(parents=True, exist_ok=True)
    final_output = output_dir / "summary.mp4"
    report_path = output_dir / "coverage_report.json"
    cache_path = output_dir / "intermediates.json"
    cache_manifest = {}
    if cache_path.exists():
        try:
            cache_manifest = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache_manifest = {}
    if not 0.15 <= float(target_summary_fraction) <= 0.40:
        raise ValueError("target_summary_fraction must be between 0.15 and 0.40")

    stage_start = time.perf_counter()
    source_fingerprint = _fingerprint(video_file)
    timings["source_fingerprint_seconds"] = time.perf_counter() - stage_start
    cache_valid = cache_manifest.get("source_sha256") == source_fingerprint
    model_name = os.getenv("WHISPER_MODEL", "small")
    try:
        import ctranslate2
        whisper_device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        whisper_device = "auto"
    model_config = {"name": model_name, "beam_size": max(1, int(os.getenv("WHISPER_BEAM_SIZE", "3"))),
                    "batch_size": max(1, int(os.getenv("WHISPER_BATCH_SIZE", "1"))), "word_timestamps": True,
                    "vad_filter": True, "condition_on_previous_text": False, "task": "transcribe",
                    "cpu_threads": os.getenv("WHISPER_CPU_THREADS", "auto"),
                    "cuda_compute_type": os.getenv("WHISPER_CUDA_COMPUTE_TYPE", "float16"),
                    "cpu_compute_type": os.getenv("WHISPER_CPU_COMPUTE_TYPE", "int8"),
                    "faster_whisper_version": FASTER_WHISPER_VERSION,
                    "transcription_cache_version": 1,
                    "device": whisper_device}
    visual_sample_fps = float(os.getenv("VIDEO_SAMPLE_FPS", "0.5"))
    visual_decode_threads = max(1, int(os.getenv("VIDEO_DECODE_THREADS", "4")))
    visual_features_key = hashlib.sha256(json.dumps(
        {"source_sha256": source_fingerprint, "visual_cache_version": 1,
         "sample_fps": visual_sample_fps, "decode_threads": visual_decode_threads},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    visual_features_cache_path = cache_dir / "visual_features" / f"{visual_features_key}.json"
    visual_prefetch_future = None

    # 1. Check/extract audio. Silent video uses explicit visual-only fallback.
    stage_start = time.perf_counter()
    print("1. Extracting audio...")
    audio_cache_path = cache_dir / "audio" / f"{source_fingerprint}_mono16k-v1.wav"
    has_audio = cache_manifest.get("has_audio") if cache_valid else None
    if has_audio is None:
        has_audio = _has_audio(video_file)
    transcript = {"language": "unknown", "text": "", "segments": []}
    visual_only = not has_audio
    if has_audio:
        audio_cached = (cache_valid and cache_manifest.get("audio_extraction") == "mono16k-v1"
                        and audio_path.is_file())
        audio_cache_hit = bool(audio_cached)
        if not audio_cached:
            cached_audio_valid = False
            if audio_cache_path.is_file():
                try:
                    cached_audio_valid = _probe_duration(audio_cache_path) > 0
                except Exception:
                    cached_audio_valid = False
            if cached_audio_valid:
                _copy_atomic(audio_cache_path, audio_path)
                audio_cache_hit = True
            else:
                temp_audio = audio_path.with_name(f"audio.{uuid.uuid4().hex}.tmp.wav")
                extract_audio(str(video_file), str(temp_audio))
                os.replace(temp_audio, audio_path)
                try:
                    _copy_atomic(audio_path, audio_cache_path)
                except OSError:
                    logger.exception("Could not populate reusable audio cache")

    _record_stage(timings, telemetry, "audio_extraction_seconds", stage_start)
    timings["audio_cache_hit"] = audio_cache_hit if has_audio else False
    cache_manifest.update({"source_sha256": source_fingerprint, "has_audio": has_audio})
    if has_audio:
        cache_manifest["audio_extraction"] = "mono16k-v1"
    _write_json_atomic(cache_path, cache_manifest)
    print("Audio extraction complete.")

    # 2. Speech-to-text
    stage_start = time.perf_counter()
    print("2. Transcribing audio with Faster-Whisper...")

    config_key = hashlib.sha256(json.dumps(model_config, sort_keys=True).encode()).hexdigest()
    transcript_cache_path = cache_dir / "transcripts" / f"{source_fingerprint}_{config_key}.json"
    transcript_cached = (cache_valid and cache_manifest.get("transcription_config") == model_config
                         and transcript_path.is_file())
    if has_audio and not transcript_cached and transcript_cache_path.is_file():
        try:
            _copy_atomic(transcript_cache_path, transcript_path)
            transcript_cached = True
        except OSError:
            logger.exception("Could not read reusable transcript cache")
    if has_audio and transcript_cached:
        try:
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            transcript_cached = False
    if has_audio and not transcript_cached:
        if whisper_device == "cuda" and not visual_features_cache_path.is_file():
            visual_prefetch_future = _visual_executor.submit(
                _cached_visual_features, video_file, visual_features_cache_path, visual_sample_fps)
        transcript = transcribe_audio(str(audio_path), model_name=model_name, output_path=str(transcript_path))
        try:
            _copy_atomic(transcript_path, transcript_cache_path)
        except OSError:
            logger.exception("Could not populate reusable transcript cache")
    else:
        if not has_audio:
            _save_json(transcript_path, transcript)

    _record_stage(timings, telemetry, "transcription_seconds", stage_start)
    timings["transcript_cache_hit"] = bool(transcript_cached)
    timings["visual_prefetch_started"] = visual_prefetch_future is not None
    timings["audio_duration_seconds"] = _probe_duration(audio_path) if has_audio and audio_path.exists() else 0.0
    timings["transcription_segment_count"] = len(transcript.get("segments", []))
    if has_audio:
        cache_manifest["transcription_config"] = model_config
        _write_json_atomic(cache_path, cache_manifest)
    print(f"Detected language: {transcript.get('language', 'unknown')}")

    # 3. Clean transcript
    stage_start = time.perf_counter()
    print("3. Cleaning transcript...")

    cleaned_segments = clean_transcript(
        transcript["segments"]
    )

    _record_stage(timings, telemetry, "transcript_cleaning_seconds", stage_start)
    print(f"Cleaned segments: {len(cleaned_segments)}")

    # 4. Create candidate segments
    stage_start = time.perf_counter()
    print("4. Creating context-aware candidate segments...")

    candidate_segments = segment_transcript(
        cleaned_segments,
        max_duration=20.0
    )

    _record_stage(timings, telemetry, "segmentation_seconds", stage_start)
    timings["candidate_segment_count"] = len(candidate_segments)
    print(f"Candidate segments: {len(candidate_segments)}")

    _save_json(segments_path, candidate_segments)

    # Handle videos with no usable transcript.
    if not candidate_segments:
        visual_only = True
        duration = _probe_duration(video_file)
        candidate_segments = [{"start": start, "end": min(duration, start + 12.0),
            "text": "[No usable speech transcript; visual-only interval]"}
            for start in np.arange(0.0, duration, 12.0) if min(duration, start + 12.0) > start]
        if not candidate_segments:
            raise ValueError("Video duration is too short or unreadable.")

    # 5. Generate embeddings
    stage_start = time.perf_counter()
    print("5. Generating semantic embeddings...")

    embedding_cache_key = hashlib.sha256(json.dumps(
        {"source_sha256": source_fingerprint, "model": "paraphrase-multilingual-MiniLM-L12-v2",
         "texts": [str(s.get("text", "")).strip() for s in candidate_segments],
         "embedding_cache_version": 1},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    embedding_cache_path = cache_dir / "embeddings" / f"{embedding_cache_key}.json"
    embedding_cache_hit = False
    if visual_only:
        embedding_model = None
        embeddings = np.zeros((len(candidate_segments), 384), dtype=float)
        embedding_method = "not_applicable_visual_only"
    elif embedding_cache_path.is_file():
        try:
            cached_embedding = json.loads(embedding_cache_path.read_text(encoding="utf-8"))
            embeddings = np.asarray(cached_embedding["embeddings"], dtype=np.float32)
            if embeddings.ndim != 2 or len(embeddings) != len(candidate_segments):
                raise ValueError("Cached embedding dimensions do not match transcript")
            embedding_model = None
            embedding_method = cached_embedding["method"]
            embedding_cache_hit = True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            embedding_cache_path.unlink(missing_ok=True)
            embedding_model = load_embedding_model()
            embeddings = generate_embeddings(candidate_segments, embedding_model)
            embedding_method = "lexical_hash_fallback" if getattr(embedding_model, "embedding_fallback", False) else "sentence_transformer"
            _write_json_atomic(embedding_cache_path, {"method": embedding_method, "embeddings": embeddings.tolist()})
    else:
        embedding_model = load_embedding_model()
        embeddings = generate_embeddings(candidate_segments, embedding_model)
        embedding_method = "lexical_hash_fallback" if getattr(embedding_model, "embedding_fallback", False) else "sentence_transformer"
        _write_json_atomic(embedding_cache_path, {"method": embedding_method, "embeddings": embeddings.tolist()})

    _record_stage(timings, telemetry, "embedding_seconds", stage_start)
    timings["embedding_cache_hit"] = embedding_cache_hit

    # 6. Calculate text importance
    stage_start = time.perf_counter()
    print("6. Scoring educational content...")

    text_scores = calculate_text_scores(
        embeddings,
        segments=candidate_segments if not visual_only else None
    )

    _record_stage(timings, telemetry, "text_scoring_seconds", stage_start)

    # 7. Calculate temporal scores
    stage_start = time.perf_counter()
    print("7. Calculating temporal scores...")

    temporal_scores = np.asarray(
        calculate_temporal_scores(candidate_segments),
        dtype=float
    )

    _record_stage(timings, telemetry, "temporal_scoring_seconds", stage_start)

    # 8. Calculate visual scores
    stage_start = time.perf_counter()
    print("8. Calculating visual scores...")

    visual_available = True
    visual_profile = {}
    visual_cache_hit = False
    visual_cache_key = hashlib.sha256(json.dumps(
        {"source_sha256": source_fingerprint, "visual_cache_version": 1,
         "sample_fps": visual_sample_fps,
         "decode_threads": visual_decode_threads,
         "intervals": [[float(s["start"]), float(s["end"])] for s in candidate_segments]},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    try:
        visual_cache_path = cache_dir / "visual" / f"{visual_cache_key}.json"
        if visual_cache_path.is_file():
            visual_cache = json.loads(visual_cache_path.read_text(encoding="utf-8"))
            visual_scores = np.asarray(visual_cache["scores"], dtype=float)
            visual_profile.update(visual_cache.get("profiling", {}))
            visual_cache_hit = True
        else:
            if visual_prefetch_future is not None:
                visual_times, visual_changes, visual_profile = visual_prefetch_future.result()
            else:
                visual_times, visual_changes, visual_profile = _cached_visual_features(
                    video_file, visual_features_cache_path, visual_sample_fps)
            visual_scores = score_visual_features(visual_times, visual_changes, candidate_segments)
            _write_json_atomic(visual_cache_path, {"scores": visual_scores.tolist(),
                                                   "profiling": visual_profile})
        _write_json_atomic(output_dir / "visual_scores.json", {"scores": visual_scores.tolist(),
                                                                 "profiling": visual_profile})
    except Exception as exc:
        # Missing visual signal is unknown evidence; don't treat it as negative.
        visual_scores = np.zeros(len(candidate_segments), dtype=float)
        visual_available = False
        visual_error = str(exc)

    if visual_only and not visual_available:
        raise ValueError("No usable speech transcript or visual analysis is available for this video.")

    _record_stage(timings, telemetry, "visual_scoring_seconds", stage_start)
    timings.update(visual_profile)
    timings["scene_change_count"] = visual_profile.get("estimated_scene_change_count", 0)
    timings["scene_change_count_threshold"] = 0.25
    timings["visual_change_samples"] = visual_profile.get("video_scene_change_samples", 0)
    timings["visual_cache_hit"] = visual_cache_hit

    # Validate score lengths before ranking.
    expected = len(candidate_segments)

    if not (
        len(text_scores)
        == len(visual_scores)
        == len(temporal_scores)
        == expected
    ):
        raise ValueError(
            "The number of text, visual, temporal scores and "
            "candidate segments must match."
        )

    # 9. Combine scores and select highlights
    stage_start = time.perf_counter()
    print("9. Selecting important segments...")

    if visual_only and visual_available:
        base_scores = calculate_base_scores(text_scores, visual_scores, temporal_scores,
            text_weight=0.0, visual_weight=0.82, temporal_weight=0.18)
    elif visual_available:
        base_scores = calculate_base_scores(
        text_scores,
        visual_scores,
        temporal_scores,
        text_weight=0.70,
        visual_weight=0.15,
        temporal_weight=0.15
        )
    else:
        base_scores = calculate_base_scores(text_scores, temporal_scores, visual_scores,
            text_weight=0.82, visual_weight=0.0, temporal_weight=0.18)
    topic_stage_start = time.perf_counter()
    discovered_topics, topic_ids = discover_topics(candidate_segments, embeddings, text_scores) if not visual_only else ([], [])
    for index, segment in enumerate(candidate_segments):
        segment["base_score"] = float(base_scores[index])
        if topic_ids:
            segment["topic_id"] = int(topic_ids[index])
    minimum_topic_seconds = {}
    for topic in discovered_topics:
        indices = topic["segment_indices"]
        best_i = max(indices, key=lambda i: base_scores[i])
        best_duration = candidate_segments[best_i]["end"] - candidate_segments[best_i]["start"]
        minimum_topic_seconds[topic["topic_id"]] = max(1.0, min(float(best_duration), 0.65 * topic["duration_seconds"]))

    timings["topic_discovery_seconds"] = time.perf_counter() - topic_stage_start

    selected_segments = select_highlights(
        candidate_segments,
        base_scores,
        embeddings,
        number_of_highlights=number_of_highlights or len(candidate_segments),
        diversity_weight=0.35,
        minimum_gap=15.0,
        max_duration_seconds=max(1.0, _probe_duration(video_file) * target_summary_fraction),
        topic_ids=topic_ids if topic_ids else None,
        topic_importance={t["topic_id"]: t["importance_share"] for t in discovered_topics},
        minimum_topic_seconds=minimum_topic_seconds,
    )

    timings["topic_count"] = len(discovered_topics)
    _record_stage(timings, telemetry, "selection_seconds", stage_start)

    score_records = []

    for index, segment in enumerate(candidate_segments):
        score_records.append({
            "segment_id": index,
            "topic_id": segment.get("topic_id"),
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "text_score": float(text_scores[index]),
            "visual_score": float(visual_scores[index]),
            "temporal_score": float(temporal_scores[index]),
            "base_score": float(base_scores[index]),
            "selected": any(
                abs(segment["start"] - chosen["start"]) < 1e-6
                and abs(segment["end"] - chosen["end"]) < 1e-6
                for chosen in selected_segments
            )
        })

    _save_json(scores_path, score_records)

    topic_report = assess_topic_coverage(discovered_topics, selected_segments, minimum_topic_seconds)
    actual_duration = sum(s["end"]-s["start"] for s in selected_segments)
    target_duration = _probe_duration(video_file) * target_summary_fraction
    covered_weight = sum(t["importance_share"] * t["coverage_score"] for t in topic_report)
    overlap_pairs = 0
    redundant_pairs = 0
    semantic_similarities = []
    for i, first in enumerate(selected_segments):
        for second in selected_segments[i + 1:]:
            if min(first["end"], second["end"]) > max(first["start"], second["start"]):
                overlap_pairs += 1
            first_i = next(j for j, x in enumerate(candidate_segments) if x["start"] == first["start"] and x["end"] == first["end"])
            second_i = next(j for j, x in enumerate(candidate_segments) if x["start"] == second["start"] and x["end"] == second["end"])
            similarity = float(np.clip(embeddings[first_i] @ embeddings[second_i], -1, 1))
            semantic_similarities.append(similarity)
            redundant_pairs += int(similarity >= 0.85)
    _save_json(report_path, {"video_duration_seconds": _probe_duration(video_file),
        "target_summary_fraction": target_summary_fraction,
        "target_summary_duration_seconds": target_duration,
        "duration_budget_honored": actual_duration <= target_duration + 1.0,
        "actual_summary_duration_seconds": actual_duration,
        "achieved_compression_ratio": actual_duration / max(1.0, _probe_duration(video_file)),
        "importance_weighted_coverage": covered_weight,
        "topic_discovery_status": "unavailable_visual_only" if visual_only else "heuristic_discovered",
        "coverage_metric_definition": "Sum(topic importance share × capped selected evidence duration / estimated minimum useful duration). This is a duration proxy, not factual-completeness validation.",
        "topics": topic_report, "coverage_gaps": [t["topic_id"] for t in topic_report if t["coverage_status"] != "adequately_represented"],
        "modalities": {"timestamped_transcript": not visual_only, "semantic_embeddings": not visual_only, "visual_activity": visual_available,
            "visual_limitation": "Visual activity/scene change only; no object/action semantics"},
        "embedding_method": embedding_method,
        "text_summary": ("No transcript summary available; this is a visual-activity-only result." if visual_only else "\n".join(f"[{t['start']:.1f}s–{t['end']:.1f}s] {t['label']} — {t['coverage_status']}. Evidence: {' '.join(t['evidence'][:2])}" for t in topic_report)),
        "uncertainty": {"topic_clustering": "unavailable without transcript" if visual_only else "cosine-similarity heuristic; labels are source excerpts", "visual_error": None if visual_available else visual_error},
        "redundancy_diagnostics": {"selected_clip_count": len(selected_segments), "overlap_pairs": overlap_pairs,
            "high_similarity_pairs": redundant_pairs, "maximum_pairwise_cosine_similarity": max(semantic_similarities) if semantic_similarities else None,
            "selector_penalizes_semantic_similarity": True},
        "selected_clips": [{"start": s["start"], "end": s["end"], "text": s["text"], "topic_id": s.get("topic_id")} for s in selected_segments]})

    print("\nSelected highlights:")

    for index, segment in enumerate(selected_segments, start=1):
        print(
            f"{index}. {segment['start']:.2f}s - "
            f"{segment['end']:.2f}s | "
            f"Score: {segment['base_score']:.3f}"
        )

    # 10. Extract clips
    stage_start = time.perf_counter()
    print("\n10. Extracting clips...")

    # Each clip is independent once selection is complete. A small bounded
    # pool reduces repeated encoder startup / idle time without changing the
    # selected intervals or any encoding settings.
    clip_workers = min(4, max(1, int(os.getenv("MAX_PARALLEL_CLIP_ENCODERS", "2"))))
    clip_jobs = [
        (index, segment, highlights_dir / f"highlight_{index}.mp4")
        for index, segment in enumerate(selected_segments, start=1)
    ]

    def _render_clip(job):
        _, segment, output_path = job
        return extract_clip(
            str(video_file),
            start_time=segment["start"],
            end_time=segment["end"],
            output_path=str(output_path),
        )

    if clip_workers == 1 or len(clip_jobs) < 2:
        clip_paths = [_render_clip(job) for job in clip_jobs]
    else:
        with ThreadPoolExecutor(max_workers=clip_workers, thread_name_prefix="clip-render") as executor:
            # executor.map preserves the chronological output order.
            clip_paths = list(executor.map(_render_clip, clip_jobs))

    _record_stage(timings, telemetry, "clip_extraction_seconds", stage_start)
    timings["parallel_clip_encoder_count"] = min(clip_workers, len(clip_jobs))

    # 11. Merge clips
    stage_start = time.perf_counter()
    print("11. Merging clips...")

    if clip_paths:
        merge_clips(clip_paths, str(final_output))
    else:
        raise ValueError(
            "No highlight clips were selected."
        )

    _record_stage(timings, telemetry, "clip_merging_seconds", stage_start)

    timings["total_seconds"] = (
        time.perf_counter() - total_start
    )
    _write_json_atomic(output_dir / "telemetry.json", telemetry)
    logger.info(json.dumps({"event": "pipeline_complete", "total_seconds": timings["total_seconds"],
                            "timings": timings, "resources": _system_snapshot()}, separators=(",", ":")))

    _save_json(timings_path, timings)

    print(f"\nSummary video: {final_output}")
    print(f"Stage timings saved to: {timings_path}")
    print("\nProcessing times:")

    for name, seconds in timings.items():
        print(f"  {name}: {seconds:.2f} seconds")

    print("\n=== PIPELINE COMPLETE ===\n")

    return {
        "transcript": transcript,
        "candidate_segments": candidate_segments,
        "selected_segments": selected_segments,
        "highlight_clips": clip_paths,
        "final_video": str(final_output),
        "audio": str(audio_path),
        "transcript_file": str(transcript_path),
        "segments_file": str(segments_path),
        "scores_file": str(scores_path),
        "timings_file": str(timings_path),
        "coverage_report": str(report_path),
        "timings": timings
    }


def _probe_duration(video_path):
    """Read source duration without decoding the full video."""
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        check=True, capture_output=True, text=True)
    return max(0.0, float(result.stdout.strip()))


def _has_audio(video_path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video_path)],
        check=True, capture_output=True, text=True)
    return bool(result.stdout.strip())
