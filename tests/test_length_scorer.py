import json

from modules.nlp.importance_scorer import (
    calculate_length_scores,
)


TRANSCRIPT_PATH = "data/output/transcript.json"


print("Loading transcript...")

with open(
    TRANSCRIPT_PATH,
    "r",
    encoding="utf-8"
) as file:
    transcript = json.load(file)


segments = transcript["segments"]

scores = calculate_length_scores(
    segments
)


print("\nLength suitability scores:\n")

for index, (segment, score) in enumerate(
    zip(segments, scores),
    start=1
):

    duration = (
        segment["end"]
        - segment["start"]
    )

    print(
        f"{index}. "
        f"{duration:.2f}s → "
        f"{score:.3f}"
    )

print("\nTest completed successfully.")