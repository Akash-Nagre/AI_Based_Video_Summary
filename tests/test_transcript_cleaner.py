import json

from modules.nlp.transcript_cleaner import clean_transcript


INPUT_PATH = "data/output/transcript.json"


with open(INPUT_PATH, "r", encoding="utf-8") as file:
    transcript = json.load(file)


cleaned_segments = clean_transcript(transcript["segments"])


print("Original segments:", len(transcript["segments"]))
print("Cleaned segments:", len(cleaned_segments))
print("\nCleaned Transcript:\n")

for segment in cleaned_segments:
    print(
        f"[{segment['start']:.2f}s - {segment['end']:.2f}s] "
        f"{segment['text']}"
    )