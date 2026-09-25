import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2

from ai.detector import PersonDetector
from config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_FRAME_SKIP,
    DEFAULT_INPUT_SIZE,
    DEFAULT_MODEL,
    PROCESSED_DIR,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    source: Path
    confidence: float = DEFAULT_CONFIDENCE
    input_size: int = DEFAULT_INPUT_SIZE
    frame_skip: int = DEFAULT_FRAME_SKIP
    model_name: str = DEFAULT_MODEL
    status: str = "queued"
    error: str | None = None
    frame: int = 0
    total_frames: int = 0
    current_count: int = 0
    active_track_count: int = 0
    fps: float = 0.0
    inference_ms: float = 0.0
    processing_ms: float = 0.0
    device: str = "pending"
    gpu_name: str | None = None
    latest_frame_url: str | None = None
    processed_video_url: str | None = None
    results: list[dict] = field(default_factory=list)

    def public(self) -> dict:
        data = asdict(self)
        data["source"] = self.source.name
        data["progress"] = (
            round((self.frame / self.total_frames) * 100, 1)
            if self.total_frames
            else 0
        )
        # Frequent status polling doesn't need the full, ever-growing
        # per-frame history — that's served in full by
        # GET /detection/results/{job_id} (which reads job.results
        # directly, not public()).
        data.pop("results", None)
        return data


class VideoProcessor:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(
        self,
        source: Path,
        confidence: float,
        input_size: int,
        frame_skip: int,
        model_name: str,
    ) -> Job:
        job = Job(
            str(uuid.uuid4()),
            source,
            confidence,
            input_size,
            frame_skip,
            model_name,
        )

        with self._lock:
            self.jobs[job.id] = job

        threading.Thread(
            target=self._run,
            args=(job,),
            daemon=True,
            name=f"detection-{job.id[:8]}",
        ).start()

        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def _run(self, job: Job) -> None:
        capture = None
        writer = None

        try:
            job.status = "loading_model"

            detector = PersonDetector(job.model_name)

            job.device = detector.device
            job.gpu_name = detector.gpu_name

            capture = cv2.VideoCapture(str(job.source))

            if not capture.isOpened():
                raise ValueError(
                    "OpenCV could not open this video. "
                    "It may be corrupted or encoded with an unsupported codec."
                )

            job.total_frames = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            source_fps = (
                capture.get(cv2.CAP_PROP_FPS) or 25.0
            )

            width = int(
                capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            if width <= 0 or height <= 0:
                raise ValueError(
                    "The video has no readable frames."
                )

            video_path = (
                PROCESSED_DIR / f"{job.id}.mp4"
            )

            output_fps = source_fps / max(
                job.frame_skip,
                1,
            )

            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_fps,
                (width, height),
            )

            if not writer.isOpened():
                raise ValueError(
                    "Unable to create processed video output."
                )

            job.status = "processing"

            last_time = time.perf_counter()
            raw_frame = 0

            while True:
                ok, frame = capture.read()

                if not ok:
                    break

                raw_frame += 1

                if (raw_frame - 1) % job.frame_skip:
                    continue

                started = time.perf_counter()

                # Phase 3: YOLO person tracking
                detections, inference_ms = detector.track(
                    frame,
                    job.confidence,
                    job.input_size,
                )

                # Draw tracking results
                for detection in detections:
                    x1, y1, x2, y2 = detection.bbox

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (41, 187, 112),
                        2,
                    )

                    if detection.track_id is not None:
                        track_label = (
                            f"Person #{detection.track_id} "
                            f"{detection.confidence:.2f}"
                        )
                    else:
                        track_label = (
                            f"Person "
                            f"{detection.confidence:.2f}"
                        )

                    cv2.putText(
                        frame,
                        track_label,
                        (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (41, 187, 112),
                        2,
                    )

                writer.write(frame)

                job.frame = raw_frame
                job.current_count = len(detections)
                job.active_track_count = sum(
                    1
                    for detection in detections
                    if detection.track_id is not None
                )

                job.inference_ms = round(
                    inference_ms,
                    2,
                )

                job.processing_ms = round(
                    (time.perf_counter() - started) * 1000,
                    2,
                )

                elapsed = (
                    time.perf_counter() - last_time
                )

                job.fps = (
                    round(1 / elapsed, 2)
                    if elapsed
                    else 0
                )

                last_time = time.perf_counter()

                job.device = detector.device
                job.gpu_name = detector.gpu_name

                timestamp = round(
                    raw_frame / source_fps,
                    3,
                )

                result = {
                    "frame": raw_frame,
                    "timestamp": timestamp,
                    "person_count": len(detections),
                    "detections": [
                        asdict(item)
                        for item in detections
                    ],
                }

                job.results.append(result)

                latest_path = (
                    PROCESSED_DIR
                    / f"{job.id}_latest.jpg"
                )

                cv2.imwrite(
                    str(latest_path),
                    frame,
                )

                job.latest_frame_url = (
                    f"/media/processed/"
                    f"{latest_path.name}"
                )

            if not job.results:
                raise ValueError(
                    "No video frames were processed."
                )

            result_path = (
                RESULTS_DIR / f"{job.id}.json"
            )

            result_path.write_text(
                json.dumps(
                    {
                        "job_id": job.id,
                        "model": job.model_name,
                        "tracking": True,
                        "frames": job.results,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            job.processed_video_url = (
                f"/media/processed/"
                f"{video_path.name}"
            )

            job.status = "completed"

        except Exception as error:
            logger.exception(
                "Detection job %s failed",
                job.id,
            )

            job.status = "failed"
            job.error = str(error)

        finally:
            if capture is not None:
                capture.release()

            if writer is not None:
                writer.release()


processor = VideoProcessor()