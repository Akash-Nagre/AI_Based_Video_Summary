import json

from modules.nlp.segmenter import segment_transcript
from modules.nlp.importance_scorer import (
    load_embedding_model,
    generate_embeddings,
    calculate_text_scores
)


INPUT_PATH = "data/output/transcript.json"


with open(INPUT_PATH, "r", encoding="utf-8") as file:
    transcript = json.load(file)


# Create candidate segments
candidate_segments = segment_transcript(
    transcript["segments"],
    max_duration=15.0
)


print("Candidate segments:", len(candidate_segments))
print("\nLoading embedding model...")

model = load_embedding_model()

print("Generating embeddings...")

embeddings = generate_embeddings(
    candidate_segments,
    model
)

print("Calculating text importance scores...")

scores = calculate_text_scores(embeddings)


print("\nCandidate Text Importance:\n")


for index, (segment, score) in enumerate(
    zip(candidate_segments, scores),
    start=1
):
    print(
        f"{index}. "
        f"[{segment['start']:.2f}s - {segment['end']:.2f}s] "
        f"Score: {score:.3f}"
    )

    print(f"   {segment['text']}")
    print()