import json

from modules.nlp.segmenter import segment_transcript


INPUT_PATH = "data/output/transcript.json"


with open(INPUT_PATH, "r", encoding="utf-8") as file:
    transcript = json.load(file)


grouped_segments = segment_transcript(
    transcript["segments"],
    max_duration=15.0
)


print("Original Whisper segments:", len(transcript["segments"]))
print("Candidate segments:", len(grouped_segments))

print("\nCandidate Segments:\n")

for index, segment in enumerate(grouped_segments, start=1):
    duration = segment["end"] - segment["start"]

    print(
        f"{index}. "
        f"[{segment['start']:.2f}s - {segment['end']:.2f}s] "
        f"({duration:.2f}s)"
    )

    print(f"   {segment['text']}")
    print()