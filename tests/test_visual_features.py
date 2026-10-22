import json

from modules.nlp.segmenter import segment_transcript
from modules.vision.visual_features import calculate_visual_scores


VIDEO_PATH = (
    "data/input/"
    "vidssave.com GOOD NEWS FOR FIRST YEAR ENGINEERING "
    "STUDENTS_ALL UNIVERSITY_PRADEEP GIRI SIR 720P.mp4"
)

TRANSCRIPT_PATH = "data/output/transcript.json"


with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as file:
    transcript = json.load(file)


candidate_segments = segment_transcript(
    transcript["segments"],
    max_duration=15.0
)


print("Candidate segments:", len(candidate_segments))

print("\nCalculating visual significance...")

visual_scores = calculate_visual_scores(
    VIDEO_PATH,
    candidate_segments
)


print("\nVisual Importance Scores:\n")


for index, (segment, score) in enumerate(
    zip(candidate_segments, visual_scores),
    start=1
):

    print(
        f"{index}. "
        f"[{segment['start']:.2f}s - "
        f"{segment['end']:.2f}s] "
        f"Visual Score: {score:.3f}"
    )