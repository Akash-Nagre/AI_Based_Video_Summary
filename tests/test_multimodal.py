import json

from modules.nlp.segmenter import segment_transcript
from modules.nlp.importance_scorer import (
    load_embedding_model,
    generate_embeddings,
    calculate_text_scores
)
from modules.vision.visual_features import calculate_visual_scores
from modules.multimodal.scorer import calculate_multimodal_scores


VIDEO_PATH = (
    "data/input/"
    "vidssave.com GOOD NEWS FOR FIRST YEAR ENGINEERING "
    "STUDENTS_ALL UNIVERSITY_PRADEEP GIRI SIR 720P.mp4"
)

TRANSCRIPT_PATH = "data/output/transcript.json"


# Load transcript
with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as file:
    transcript = json.load(file)


# Create the same candidate segments
candidate_segments = segment_transcript(
    transcript["segments"],
    max_duration=15.0
)


# -------------------------
# TEXT ANALYSIS
# -------------------------

print("Loading embedding model...")

model = load_embedding_model()

print("Generating text embeddings...")

embeddings = generate_embeddings(
    candidate_segments,
    model
)

text_scores = calculate_text_scores(embeddings)


# -------------------------
# VISUAL ANALYSIS
# -------------------------

print("Calculating visual significance...")

visual_scores = calculate_visual_scores(
    VIDEO_PATH,
    candidate_segments
)


# -------------------------
# MULTIMODAL ANALYSIS
# -------------------------

final_scores = calculate_multimodal_scores(
    text_scores,
    visual_scores
)


# -------------------------
# RESULTS
# -------------------------

print("\nMultimodal Importance Scores:\n")


for index, (
    segment,
    text_score,
    visual_score,
    final_score
) in enumerate(
    zip(
        candidate_segments,
        text_scores,
        visual_scores,
        final_scores
    ),
    start=1
):

    print(
        f"{index}. "
        f"[{segment['start']:.2f}s - "
        f"{segment['end']:.2f}s]"
    )

    print(f"   Text Score:    {text_score:.3f}")
    print(f"   Visual Score:  {visual_score:.3f}")
    print(f"   Final Score:   {final_score:.3f}")
    print()