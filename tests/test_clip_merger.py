from modules.highlights.clip_merger import merge_clips


CLIPS = [
    "data/output/highlights/highlight_1.mp4",
    "data/output/highlights/highlight_2.mp4",
    "data/output/highlights/highlight_3.mp4"
]

OUTPUT_PATH = "data/output/final_highlight.mp4"


print("Merging highlight clips...\n")

merge_clips(
    CLIPS,
    OUTPUT_PATH
)

print("\nFinal highlight video created!")
print(f"Output: {OUTPUT_PATH}")