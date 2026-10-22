from modules.highlights.clip_extractor import extract_clip


VIDEO_PATH = (
    "data/input/"
    "vidssave.com GOOD NEWS FOR FIRST YEAR ENGINEERING "
    "STUDENTS_ALL UNIVERSITY_PRADEEP GIRI SIR 720P.mp4"
)

HIGHLIGHTS = [
    {
        "start": 0.00,
        "end": 11.96
    },
    {
        "start": 39.16,
        "end": 50.10
    },
    {
        "start": 76.44,
        "end": 91.22
    }
]


OUTPUT_DIR = "data/output/highlights"


print("Extracting highlight clips...\n")


for index, highlight in enumerate(HIGHLIGHTS, start=1):

    output_path = (
        f"{OUTPUT_DIR}/highlight_{index}.mp4"
    )

    print(
        f"Highlight {index}: "
        f"{highlight['start']:.2f}s → "
        f"{highlight['end']:.2f}s"
    )

    extract_clip(
        VIDEO_PATH,
        start_time=highlight["start"],
        end_time=highlight["end"],
        output_path=output_path
    )

    print(f"Saved: {output_path}\n")


print("All highlight clips extracted successfully!")