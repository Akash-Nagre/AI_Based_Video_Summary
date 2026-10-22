import json


TRANSCRIPT_PATH = "data/output/transcript.json"


with open(
    TRANSCRIPT_PATH,
    "r",
    encoding="utf-8"
) as file:
    transcript = json.load(file)


segments = transcript["segments"]


print(f"Original Whisper segments: {len(segments)}")
print("\nRaw Whisper Segments:\n")


for index, segment in enumerate(
    segments,
    start=1
):

    duration = (
        segment["end"]
        - segment["start"]
    )

    print(
        f"{index}. "
        f"{segment['start']:.2f}s → "
        f"{segment['end']:.2f}s "
        f"({duration:.2f}s)"
    )

    print(
        f"   {segment['text']}"
    )


print("\nTest completed successfully.")