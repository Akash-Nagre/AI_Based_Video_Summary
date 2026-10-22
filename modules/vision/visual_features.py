import cv2
import os
import numpy as np


def extract_visual_features(video_path: str, sample_fps: float = 0.5,
                            profiling: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Decode the timeline once and return low-resolution visual-change samples."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    decode_threads = max(1, int(os.getenv("VIDEO_DECODE_THREADS", "4")))
    if hasattr(cv2, "CAP_PROP_N_THREADS"):
        try:
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_N_THREADS, decode_threads])
        except (TypeError, cv2.error):
            cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError("Could not determine video FPS")
    stride = max(int(round(fps / sample_fps)), 1)
    records = []
    previous = None
    frame_index = 0
    sampled_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride == 0:
            sampled_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))
            if previous is not None:
                change = float(np.mean(cv2.absdiff(gray, previous))) / 255.0
                records.append((frame_index / fps, change))
            previous = gray
        frame_index += 1
    cap.release()
    if profiling is not None:
        profiling.update({"video_frames_decoded": frame_index,
                          "video_frames_sampled": sampled_count,
                          "video_sample_fps": sample_fps,
                          "video_decode_threads": decode_threads,
                          "video_scene_change_samples": len(records),
                          "estimated_scene_change_count": int(sum(change >= 0.25 for _, change in records))})
    return (np.asarray([r[0] for r in records], dtype=float),
            np.asarray([r[1] for r in records], dtype=float))


def score_visual_features(times: np.ndarray, changes: np.ndarray, segments: list) -> np.ndarray:
    """Pool cached visual-change samples into candidate transcript intervals."""
    if not segments:
        return np.array([], dtype=float)
    times = np.asarray(times, dtype=float)
    changes = np.asarray(changes, dtype=float)
    if len(times) != len(changes):
        raise ValueError("Visual feature timestamps and values must have equal lengths")
    scores = []
    for segment in segments:
        mask = (times >= float(segment["start"])) & (times <= float(segment["end"]))
        # Activity is capped at the 90th percentile to reduce one-off cuts/noise.
        scores.append(float(np.percentile(changes[mask], 75)) if mask.any() else 0.0)
    scores = np.asarray(scores, dtype=float)
    low, high = scores.min(), scores.max()
    return (scores - low) / (high - low) if high > low else np.zeros_like(scores)


def calculate_visual_scores(video_path: str, segments: list, sample_fps: float = 0.5,
                            profiling: dict | None = None) -> np.ndarray:
    """Sample the video once and pool activity by transcript interval.

    This is a visual activity/scene-change signal, not object/action recognition.
    """
    if not segments:
        return np.array([], dtype=float)
    times, changes = extract_visual_features(video_path, sample_fps, profiling)
    return score_visual_features(times, changes, segments)
