import subprocess
import os
from pathlib import Path


def extract_clip(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str
):
    """
    Extract a video clip with both video and audio.
    """

    output_file = Path(output_path)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    duration = end_time - start_time

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        str(start_time),
        "-i",
        video_path,
        "-t",
        str(duration),

        # Explicitly select video and audio streams
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",

        # Encode video and audio
        "-c:v",
        "libx264",
        "-preset",
        os.getenv("VIDEO_ENCODER_PRESET", "veryfast"),
        "-crf",
        os.getenv("VIDEO_ENCODER_CRF", "20"),
        "-c:a",
        "aac",

        # Make the MP4 broadly compatible
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",

        str(output_file)
    ]

    subprocess.run(
        command,
        check=True
    )

    return str(output_file)
