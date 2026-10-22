from modules.pipeline import process_video


VIDEO_PATH = (
    "data/input/"
    "vidssave.com GOOD NEWS FOR FIRST YEAR ENGINEERING "
    "STUDENTS_ALL UNIVERSITY_PRADEEP GIRI SIR 720P.mp4"
)


result = process_video(
    VIDEO_PATH,
    number_of_highlights=3
)

print("Final video:")
print(result["final_video"])