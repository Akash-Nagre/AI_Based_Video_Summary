import shutil
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from modules.pipeline import process_video


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="AI Video Summarization System",
    description="AI-based multimodal video summarization API",
    version="1.0.0"
)


# ============================================================
# Directory Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

JOBS_DIR = DATA_DIR / "jobs"

JOBS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Job Storage
# ============================================================

jobs = {}
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "1")))
_processing_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS,
                                          thread_name_prefix="video-summary")


# ============================================================
# Helper Functions
# ============================================================

def get_job_directory(job_id: str) -> Path:
    """
    Return the directory belonging to a specific job.
    """

    return JOBS_DIR / job_id


def process_job(
    job_id: str,
    video_path: Path,
    job_dir: Path,
    target_summary_fraction: float = 0.25,
):
    """
    Run the AI video summarization pipeline
    in a background thread.
    """

    try:

        # The executor bounds CPU/GPU-heavy jobs and worker thread count.
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["stage"] = "AI video processing"
        jobs[job_id]["progress"] = 10

        result = process_video(
            video_path=str(video_path),
            output_dir=str(job_dir),
            target_summary_fraction=target_summary_fraction
        )

        # ----------------------------------------------------
        # Processing completed
        # ----------------------------------------------------

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["stage"] = "completed"
        jobs[job_id]["progress"] = 100

        jobs[job_id]["result"] = result

    except Exception as e:

        # ----------------------------------------------------
        # Handle processing error
        # ----------------------------------------------------

        jobs[job_id]["status"] = "failed"
        jobs[job_id]["stage"] = "error"
        jobs[job_id]["progress"] = 0
        jobs[job_id]["error"] = str(e)

        print(
            f"\nERROR processing job {job_id}:"
        )

        print(e)


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health_check():
    """
    Check whether the backend is running.
    """

    return {
        "status": "ok"
    }


# ============================================================
# Upload Video
# ============================================================

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...)
):
    """
    Upload a video and create a unique processing job.
    """

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    allowed_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm"
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Allowed formats: MP4, MOV, AVI, MKV, WEBM."
            )
        )

    # --------------------------------------------------------
    # Create unique job ID
    # --------------------------------------------------------

    job_id = uuid.uuid4().hex[:8]

    job_dir = get_job_directory(
        job_id
    )

    job_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save original video
    # --------------------------------------------------------

    video_path = job_dir / f"original{extension}"

    try:

        with open(
            video_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

    finally:

        await file.close()

    # --------------------------------------------------------
    # Store job information
    # --------------------------------------------------------

    jobs[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "status": "uploaded",
        "stage": "waiting",
        "progress": 0,
        "video_path": str(video_path),
        "job_directory": str(job_dir),
        "result": None,
        "error": None
    }

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "message": "Video uploaded successfully.",
        "job_id": job_id,
        "filename": file.filename,
        "status": "uploaded"
    }


# ============================================================
# Start Summarization
# ============================================================

@app.post("/summarize/{job_id}")
def summarize_video(
    job_id: str,
    target_summary_fraction: float = Query(0.25, ge=0.15, le=0.40)
):
    """
    Start AI processing for an uploaded video.
    """

    # --------------------------------------------------------
    # Check job
    # --------------------------------------------------------

    if job_id not in jobs:

        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    job = jobs[job_id]

    # --------------------------------------------------------
    # Prevent duplicate processing
    # --------------------------------------------------------

    if job["status"] == "processing":

        raise HTTPException(
            status_code=409,
            detail="This job is already being processed."
        )

    if job["status"] == "completed":

        return {
            "message": "This video has already been processed.",
            "job_id": job_id,
            "status": "completed"
        }

    # --------------------------------------------------------
    # Get paths
    # --------------------------------------------------------

    video_path = Path(
        job["video_path"]
    )

    job_dir = Path(
        job["job_directory"]
    )

    # --------------------------------------------------------
    # Check video exists
    # --------------------------------------------------------

    if not video_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Uploaded video file not found."
        )

    # --------------------------------------------------------
    # Start background processing
    # --------------------------------------------------------

    _processing_executor.submit(
        process_job,
            job_id,
            video_path,
            job_dir,
            target_summary_fraction
    )

    # --------------------------------------------------------
    # Update status
    # --------------------------------------------------------

    jobs[job_id]["status"] = "processing"
    jobs[job_id]["stage"] = "starting"
    jobs[job_id]["progress"] = 0

    return {
        "message": "Video summarization started.",
        "job_id": job_id,
        "status": "processing"
    }


# ============================================================
# Processing Status
# ============================================================

@app.get("/status/{job_id}")
def get_status(
    job_id: str
):
    """
    Get the current processing status of a job.
    """

    if job_id not in jobs:

        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    job = jobs[job_id]

    response = {
        "job_id": job_id,
        "filename": job["filename"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"]
    }

    if job["status"] == "failed":

        response["error"] = job.get(
            "error"
        )

    return response


# ============================================================
# Get Final Result
# ============================================================

@app.get("/result/{job_id}")
def get_result(
    job_id: str
):
    """
    Return the completed summarization result.
    """

    if job_id not in jobs:

        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    job = jobs[job_id]

    # --------------------------------------------------------
    # Still processing
    # --------------------------------------------------------

    if job["status"] == "processing":

        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Video is still being processed."
        }

    # --------------------------------------------------------
    # Failed
    # --------------------------------------------------------

    if job["status"] == "failed":

        raise HTTPException(
            status_code=500,
            detail=job.get(
                "error",
                "Video processing failed."
            )
        )

    # --------------------------------------------------------
    # Not processed yet
    # --------------------------------------------------------

    if job["status"] != "completed":

        return {
            "job_id": job_id,
            "status": job["status"],
            "message": "Video has not been processed yet."
        }

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "job_id": job_id,
        "status": "completed",
        "result": job["result"],
        "coverage_report": _read_coverage_report(job)
    }


def _read_coverage_report(job):
    import json
    path = Path(job["job_directory"]) / "coverage_report.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as source:
        return json.load(source)


@app.get("/coverage/{job_id}")
def get_coverage_report(job_id: str):
    """Return the completed job's structured topic coverage report."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    report = _read_coverage_report(jobs[job_id])
    if report is None:
        raise HTTPException(status_code=409, detail="Coverage report is not available yet.")
    return report


@app.get("/download/{job_id}")
def download_summary(job_id: str):
    """Download the rendered summary video for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    if jobs[job_id]["status"] != "completed":
        raise HTTPException(status_code=409, detail="Summary is not ready yet.")
    summary_path = Path(jobs[job_id]["job_directory"]) / "summary.mp4"
    if not summary_path.is_file():
        raise HTTPException(status_code=404, detail="Summary video not found.")
    return FileResponse(summary_path, media_type="video/mp4", filename="video-summary.mp4")


# Serve the browser interface from the same origin as the API.
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
