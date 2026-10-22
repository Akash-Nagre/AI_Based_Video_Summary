import whisper


AUDIO_PATH = "data/output/vidssave.com GOOD NEWS FOR FIRST YEAR ENGINEERING STUDENTS_ALL UNIVERSITY_PRADEEP GIRI SIR 720P_audio.wav"


print("Loading Whisper model...")

model = whisper.load_model("medium")

print("Transcribing...\n")

result = model.transcribe(
    AUDIO_PATH,
    task="transcribe",
    condition_on_previous_text=True,
    word_timestamps=False
)

print(f"Detected language: {result['language']}")
print(f"Segments: {len(result['segments'])}\n")

for index, segment in enumerate(
    result["segments"],
    start=1
):
    print(
        f"{index}. "
        f"{segment['start']:.2f}s → "
        f"{segment['end']:.2f}s "
        f"({segment['end'] - segment['start']:.2f}s)"
    )

    print(
        f"   {segment['text']}"
    )

print("\nTest completed successfully.")