import shutil
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ai.video_processor import processor
from config import ALLOWED_VIDEO_EXTENSIONS, API_PREFIX, DEFAULT_CONFIDENCE, DEFAULT_FRAME_SKIP, DEFAULT_INPUT_SIZE, DEFAULT_MODEL, UPLOADS_DIR

router = APIRouter(prefix=API_PREFIX, tags=["detection"])

def video_metadata(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise HTTPException(422, "The uploaded file is not a readable video.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if not width or not height or not frames:
        raise HTTPException(422, "The uploaded video has no readable frames.")
    return {"resolution": f"{width}×{height}", "fps": round(fps, 2), "duration_seconds": round(frames / fps, 2) if fps else None, "frames": frames}

@router.post("/video/upload")
async def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(415, "Unsupported video format. Use .mp4, .avi, .mov, or .mkv.")
    upload_id = str(uuid.uuid4())
    destination = UPLOADS_DIR / f"{upload_id}{suffix}"
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        metadata = video_metadata(destination)
        return {"upload_id": upload_id, "filename": file.filename, "size_bytes": destination.stat().st_size, "status": "ready", **metadata}
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except Exception as error:
        destination.unlink(missing_ok=True)
        raise HTTPException(400, f"Upload failed: {error}") from error
    finally:
        await file.close()

@router.post("/detection/start")
async def start_detection(upload_id: str = Form(...), confidence: float = Form(DEFAULT_CONFIDENCE), input_size: int = Form(DEFAULT_INPUT_SIZE), frame_skip: int = Form(DEFAULT_FRAME_SKIP), model_name: str = Form(DEFAULT_MODEL)):
    candidates = list(UPLOADS_DIR.glob(f"{upload_id}.*"))
    if not candidates:
        raise HTTPException(404, "Uploaded video was not found. Upload it again.")
    if not 0.05 <= confidence <= 0.95:
        raise HTTPException(422, "Confidence must be between 0.05 and 0.95.")
    if input_size not in {320, 480, 640, 960}:
        raise HTTPException(422, "Input resolution must be one of 320, 480, 640, or 960.")
    if not 1 <= frame_skip <= 10:
        raise HTTPException(422, "Frame skip must be between 1 and 10.")
    job = processor.start(candidates[0], confidence, input_size, frame_skip, model_name)
    return job.public()

@router.get("/detection/status/{job_id}")
async def detection_status(job_id: str):
    job = processor.get(job_id)
    if not job:
        raise HTTPException(404, "Detection job not found.")
    return job.public()

@router.get("/detection/results/{job_id}")
async def detection_results(job_id: str):
    job = processor.get(job_id)
    if not job:
        raise HTTPException(404, "Detection job not found.")
    return {"job_id": job.id, "status": job.status, "model": job.model_name, "device": job.device, "frames": job.results, "processed_video_url": job.processed_video_url}
