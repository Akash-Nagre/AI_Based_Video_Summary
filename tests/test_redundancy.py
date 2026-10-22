import json

from modules.nlp.segmenter import segment_transcript
from modules.nlp.importance_scorer import (
    load_embedding_model,
    generate_embeddings
)
from modules.multimodal.redundancy import (
    calculate_redundancy_scores
)


INPUT_PATH = "data/output/transcript.json"


with open(INPUT_PATH, "r", encoding="utf-8") as file:
    transcript = json.load(file)


candidate_segments = segment_transcript(
    transcript["segments"],
    max_duration=15.0
)


print("Loading embedding model...")

model = load_embedding_model()

print("Generating embeddings...")

embeddings = generate_embeddings(
    candidate_segments,
    model
)


redundancy_scores = calculate_redundancy_scores(
    embeddings
)


print("\nRedundancy Scores:\n")


for index, (segment, score) in enumerate(
    zip(candidate_segments, redundancy_scores),
    start=1
):

    print(
        f"{index}. "
        f"[{segment['start']:.2f}s - "
        f"{segment['end']:.2f}s] "
        f"Redundancy: {score:.3f}"
    )