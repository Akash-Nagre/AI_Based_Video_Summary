import json

from modules.nlp.importance_scorer import (
    load_embedding_model,
    generate_embeddings,
    calculate_text_scores,
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


print(
    f"Transcript segments: "
    f"{len(segments)}"
)

print("\nLoading embedding model...")

model = load_embedding_model()

print("Generating embeddings...")

embeddings = generate_embeddings(
    segments,
    model
)

print("Calculating TextRank semantic scores...")

scores = calculate_text_scores(
    embeddings
)


print("\nTextRank scores:\n")

for index, (
    segment,
    score
) in enumerate(
    zip(segments, scores),
    start=1
):

    duration = (
        segment["end"]
        - segment["start"]
    )

    print(
        f"{index}. "
        f"{duration:.2f}s | "
        f"Score: {score:.3f}"
    )

    print(
        f"   {segment['text']}"
    )


print("\nTest completed successfully.")