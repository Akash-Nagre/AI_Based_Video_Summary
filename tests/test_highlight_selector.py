import json
import numpy as np

from modules.nlp.importance_scorer import (
    load_embedding_model,
    generate_embeddings,
    calculate_text_scores,
)

from modules.multimodal.highlight_selector import (
    calculate_base_scores,
    select_highlights,
)

from modules.nlp.temporal_scorer import (
    calculate_temporal_scores,
)

from modules.vision.visual_features import (
    calculate_visual_scores,
)


TRANSCRIPT_PATH = "data/output/transcript.json"

VIDEO_PATH = (
    "data/input/"
    "vidssave.com GOOD NEWS FOR FIRST YEAR ENGINEERING "
    "STUDENTS_ALL UNIVERSITY_PRADEEP GIRI SIR 720P.mp4"
)


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

print("Calculating semantic scores...")

text_scores = calculate_text_scores(
    embeddings
)

print("Calculating temporal scores...")

temporal_scores = calculate_temporal_scores(
    segments
)

print("Calculating visual scores...")

visual_scores = calculate_visual_scores(
    VIDEO_PATH,
    segments
)

print("Calculating base scores...")

base_scores = calculate_base_scores(
    text_scores,
    visual_scores,
    temporal_scores
)


print("\nCandidate scores:\n")

for index, (
    segment,
    text,
    visual,
    temporal,
    base
) in enumerate(
    zip(
        segments,
        text_scores,
        visual_scores,
        temporal_scores,
        base_scores
    ),
    start=1
):

    print(
        f"{index}. "
        f"{segment['start']:.2f}s → "
        f"{segment['end']:.2f}s | "
        f"Text: {text:.3f} | "
        f"Visual: {visual:.3f} | "
        f"Temporal: {temporal:.3f} | "
        f"Base: {base:.3f}"
    )


print("\nSelecting highlights...")

selected_segments = select_highlights(
    segments,
    base_scores,
    embeddings,
    number_of_highlights=3,
    diversity_weight=0.35,
    minimum_gap=15.0
)


print("\nSelected highlights:\n")

for index, segment in enumerate(
    selected_segments,
    start=1
):

    print(
        f"{index}. "
        f"{segment['start']:.2f}s → "
        f"{segment['end']:.2f}s | "
        f"Score: {segment['base_score']:.3f}"
    )

    print(
        f"   {segment['text']}"
    )


print("\nTest completed successfully.")