import json
import os
import sys
import warnings
import os
from pathlib import Path
from functools import lru_cache


# ---------------------------------------------------------
# CUDA DLL configuration for Windows
# ---------------------------------------------------------

if os.name == "nt":

    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"

    cuda_directories = [
        site_packages / "cublas" / "bin",
        site_packages / "cudnn" / "bin",
        site_packages / "cuda_runtime" / "bin",
        site_packages / "cuda_nvrtc" / "bin",
    ]

    # Add existing CUDA DLL directories to PATH
    existing_paths = []

    for directory in cuda_directories:
        if directory.exists():
            existing_paths.append(str(directory))

    if existing_paths:
        os.environ["PATH"] = os.pathsep.join(existing_paths) + os.pathsep + os.environ.get("PATH", "")

    # Keep DLL directory handles alive
    _dll_handles = []

    for directory in cuda_directories:
        if directory.exists():
            _dll_handles.append(
                os.add_dll_directory(str(directory))
            )


from faster_whisper import WhisperModel


@lru_cache(maxsize=6)
def load_model(model_name: str = "small", cuda_compute_type: str | None = None,
               cpu_compute_type: str | None = None, cpu_threads: int | None = None):
    """
    Load and cache the Faster-Whisper model.
    """

    print(f"Loading Faster-Whisper model: {model_name}")
    cpu_threads = cpu_threads or max(1, int(os.getenv("WHISPER_CPU_THREADS", str(min(os.cpu_count() or 4, 8)))))
    cuda_compute_type = cuda_compute_type or os.getenv("WHISPER_CUDA_COMPUTE_TYPE", "float16")
    cpu_compute_type = cpu_compute_type or os.getenv("WHISPER_CPU_COMPUTE_TYPE", "int8")
    try:
        model = WhisperModel(model_name, device="cuda", compute_type=cuda_compute_type)
        print(f"Device: CUDA; compute: {cuda_compute_type}")
    except Exception as exc:
        warnings.warn(f"CUDA Whisper initialization failed; using CPU {cpu_compute_type}: {exc}")
        model = WhisperModel(model_name, device="cpu", compute_type=cpu_compute_type,
                             cpu_threads=cpu_threads, num_workers=1)
        print(f"Device: CPU; compute: {cpu_compute_type}; threads: {cpu_threads}")

    print("Faster-Whisper model loaded successfully.\n")

    return model


def transcribe_audio(
    audio_path: str,
    model_name: str = "small",
    output_path: str = "data/output/transcript.json"
):
    """
    Transcribe an audio file using Faster-Whisper.

    The transcript preserves:
    - detected language
    - full transcript
    - segment timestamps
    - word-level timestamps
    """

    model = load_model(model_name,
                       os.getenv("WHISPER_CUDA_COMPUTE_TYPE", "float16"),
                       os.getenv("WHISPER_CPU_COMPUTE_TYPE", "int8"),
                       max(1, int(os.getenv("WHISPER_CPU_THREADS", str(min(os.cpu_count() or 4, 8))))))

    print("Starting transcription...")
    print(f"Audio: {audio_path}")
    print(f"Model: {model_name}")
    print("Transcription device: selected automatically\n")

    beam_size = max(1, int(os.getenv("WHISPER_BEAM_SIZE", "3")))
    batch_size = max(1, int(os.getenv("WHISPER_BATCH_SIZE", "1")))
    if batch_size > 1:
        from faster_whisper import BatchedInferencePipeline
        inference = BatchedInferencePipeline(model)
        segments, info = inference.transcribe(
            audio_path, task="transcribe", beam_size=beam_size,
            condition_on_previous_text=False, word_timestamps=True,
            vad_filter=True, batch_size=batch_size)
    else:
        segments, info = model.transcribe(
            audio_path,
            task="transcribe",
            beam_size=beam_size,
            condition_on_previous_text=False,
            word_timestamps=True,
            vad_filter=True
        )

    transcript_segments = []
    full_text_parts = []

    for segment in segments:

        segment_text = segment.text.strip()

        if segment_text:
            full_text_parts.append(segment_text)

        words = []

        if segment.words:
            for word in segment.words:
                words.append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end
                })

        transcript_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment_text,
            "avg_logprob": (float(segment.avg_logprob) if getattr(segment, "avg_logprob", None) is not None else None),
            "words": words
        })

    transcript = {
        "language": info.language,
        "text": " ".join(full_text_parts).strip(),
        "segments": transcript_segments
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            transcript,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("Transcription complete.")
    print(f"Detected language: {info.language}")
    print(f"Transcript saved to: {output_file}\n")

    return transcript
