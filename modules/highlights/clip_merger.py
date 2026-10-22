import subprocess
from pathlib import Path


def merge_clips(
    clip_paths: list,
    output_path: str
):
    """
    Merge multiple MP4 highlight clips into one video.

    All clips are expected to have compatible
    video and audio formats.
    """

    if not clip_paths:
        raise ValueError("No clips provided for merging.")

    output_file = Path(output_path)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    list_file = output_file.parent / "concat_list.txt"

    with open(list_file, "w", encoding="utf-8") as file:
        for clip_path in clip_paths:
            absolute_path = Path(clip_path).resolve()

            file.write(
                f"file '{absolute_path.as_posix()}'\n"
            )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",

        "-f",
        "concat",

        "-safe",
        "0",

        "-i",
        str(list_file),

        "-c",
        "copy",

        "-movflags",
        "+faststart",

        str(output_file)
    ]

    subprocess.run(
        command,
        check=True
    )

    return str(output_file)
