
import re


SENTENCE_END = re.compile(r"""[.!?。！？]["')\]]*$""")


def segment_transcript(
    segments: list,
    max_duration: float = 20.0,
) -> list:
    """
    Group timestamped transcript chunks into candidate segments.

    Prefers sentence boundaries and keeps segments within a
    reasonable duration where possible. A single incoming
    Whisper segment may itself exceed max_duration.
    """

    if not segments:
        return []

    if max_duration <= 0:
        raise ValueError("max_duration must be positive.")

    grouped = []
    current = None

    for item in segments:
        text = str(item.get("text", "")).strip()

        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue

        if not text or end <= start:
            continue

        if current is None:
            current = {
                "start": start,
                "end": end,
                "text": text,
            }
            continue

        proposed_duration = end - current["start"]

        current_ends_sentence = bool(
            SENTENCE_END.search(current["text"])
        )

        close_chunk = (
            proposed_duration > max_duration
            or (
                current_ends_sentence
                and proposed_duration > 5.0
            )
        )

        if close_chunk:
            grouped.append(current)

            current = {
                "start": start,
                "end": end,
                "text": text,
            }
        else:
            current["end"] = end
            current["text"] += " " + text

    if current is not None:
        grouped.append(current)

    return grouped
