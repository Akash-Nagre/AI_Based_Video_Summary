import json

from modules.nlp.segmenter import segment_transcript
from modules.nlp.temporal_scorer import calculate_temporal_scores


INPUT_PATH = "data/output/transcript.json"


with open(INPUT_PATH, "r", encoding="utf-8") as file:
    transcript = json.load(file)


candidate_segments = segment_transcript(
    transcript["segments"],
    max_duration=15.0
)


temporal_scores = calculate_temporal_scores(
    candidate_segments
)


print("\nTemporal Context Scores:\n")


for index, (segment, score) in enumerate(
    zip(candidate_segments, temporal_scores),
    start=1
):

    print(
        f"{index}. "
        f"[{segment['start']:.2f}s - "
        f"{segment['end']:.2f}s] "
        f"Temporal Score: {score:.3f}"
    )