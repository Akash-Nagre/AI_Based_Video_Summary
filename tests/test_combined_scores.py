import json

from modules.nlp.importance_scorer import (
    load_embedding_model,
    generate_embeddings,
    calculate_text_scores,
    calculate_length_scores,
    calculate_combined_text_scores,
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


print(f"Transcript segments: {len(segments)}")
print("\nLoading embedding model...")

model = load_embedding_model()

print("Generating embeddings...")

embeddings = generate_embeddings(
    segments,
    model
)

print("Calculating semantic scores...")

semantic_scores = calculate_text_scores(
    embeddings
)

print("Calculating length scores...")

length_scores = calculate_length_scores(
    segments
)

print("Calculating combined scores...")

combined_scores = calculate_combined_text_scores(
    semantic_scores,
    length_scores
)


print("\nCombined scores:\n")

for index, (
    segment,
    semantic,
    length,
    combined
) in enumerate(
    zip(
        segments,
        semantic_scores,
        length_scores,
        combined_scores
    ),
    start=1
):

    duration = (
        segment["end"]
        - segment["start"]
    )

    print(
        f"{index}. "
        f"{duration:.2f}s | "
        f"Semantic: {semantic:.3f} | "
        f"Length: {length:.3f} | "
        f"Combined: {combined:.3f}"
    )


print("\nTest completed successfully.")