import re


def clean_text(text: str) -> str:
    """
    Clean a Whisper-generated transcript while preserving
    its original meaning.
    """

    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    words = text.split()
    cleaned_words = []

    for word in words:
        if not cleaned_words or word != cleaned_words[-1]:
            cleaned_words.append(word)

    return " ".join(cleaned_words)


def clean_transcript(segments: list) -> list:
    """
    Clean Whisper transcript segments while preserving
    timestamps and word-level timestamps.
    """

    cleaned_segments = []

    for segment in segments:

        cleaned_segment = {
            "start": segment["start"],
            "end": segment["end"],
            "text": clean_text(segment["text"]),
            "words": segment.get("words", []),
            "avg_logprob": segment.get("avg_logprob")
        }
        if segment.get("speaker") is not None:
            cleaned_segment["speaker"] = segment["speaker"]

        cleaned_segments.append(cleaned_segment)

    return cleaned_segments
