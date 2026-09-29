from collections import deque
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
    ANOMALY_LEVEL_NORMAL,
    ANOMALY_ROLLING_WINDOW_FRAMES,
    DEFAULT_CAMERA_ZONE,
    DEFAULT_CONFIDENCE,
    DEFAULT_FRAME_SKIP,
    DEFAULT_INPUT_SIZE,
    DEFAULT_MODEL,
    FLOW_SPEED_UNIT,
    PROCESSED_DIR,
    RESULTS_DIR,
    ZoneDefinition,
    get_camera_zones,
)
from services.crowd_service import (
    calculate_crowd_anomalies,
    calculate_crowd_flow_and_congestion,
    calculate_frame_density,
    calculate_zone_metrics,
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
    unique_track_count: int = 0
    peak_crowd: int = 0
    density_index: float = 0.0
    visual_density_index: float = 0.0
    yolo_density: float = 0.0
    smoothed_density: float = 0.0
    occupancy_ratio: float = 0.0
    crowd_status: str = "LOW"
    flow_direction: str = "STATIONARY"
    flow_speed: float = 0.0
    flow_speed_unit: str = FLOW_SPEED_UNIT
    congestion_index: float = 0.0
    congestion_status: str = "LOW"
    active_flow_vectors: int = 0
    anomaly_score: float = 0.0
    anomaly_status: str = ANOMALY_LEVEL_NORMAL
    anomaly_reason: str = "Normal movement patterns"
    stagnation_anomaly: float = 0.0
    density_surge_anomaly: float = 0.0
    deceleration_anomaly: float = 0.0
    reversal_anomaly: float = 0.0
    turbulence_anomaly: float = 0.0
    reference_zone: str = DEFAULT_CAMERA_ZONE
    fps: float = 0.0
    inference_ms: float = 0.0
    processing_ms: float = 0.0
    device: str = "pending"
    gpu_name: str | None = None
    latest_frame_url: str | None = None
    processed_video_url: str | None = None
    results: list[dict] = field(default_factory=list)
    zones: list[dict] = field(default_factory=list)
    _observed_track_ids: set[int] = field(default_factory=set, repr=False)
    _track_history: dict[int, tuple[float, float, int, float]] = field(default_factory=dict, repr=False)
    _flow_vx: float = field(default=0.0, repr=False)
    _flow_vy: float = field(default=0.0, repr=False)
    _high_congestion_counter: int = field(default=0, repr=False)
    _anomaly_history: deque = field(default_factory=lambda: deque(maxlen=ANOMALY_ROLLING_WINDOW_FRAMES), repr=False)
    _high_anomaly_counter: int = field(default=0, repr=False)
    _zones: list[ZoneDefinition] = field(default_factory=list, repr=False)
    _zone_density_state: dict[str, float] = field(default_factory=dict, repr=False)
    _zone_track_history: dict[int, tuple[float, float, int, float]] = field(default_factory=dict, repr=False)
    _zone_velocity_state: dict[str, tuple[float, float]] = field(default_factory=dict, repr=False)
    _zone_congestion_state: dict[str, tuple[str, int]] = field(default_factory=dict, repr=False)
    _zone_anomaly_history: dict[str, deque] = field(default_factory=dict, repr=False)
    _zone_anomaly_counter: dict[str, int] = field(default_factory=dict, repr=False)

    def public(self) -> dict:
        return {
            "id": self.id,
            "job_id": self.id,
            "source": self.source.name,
            "confidence": self.confidence,
            "input_size": self.input_size,
            "frame_skip": self.frame_skip,
            "model_name": self.model_name,
            "status": self.status,
            "error": self.error,
            "frame": self.frame,
            "total_frames": self.total_frames,
            "progress": (
                round((self.frame / self.total_frames) * 100, 1)
                if self.total_frames
                else 0
            ),
            "current_count": self.current_count,
            "active_track_count": self.active_track_count,
            "unique_track_count": self.unique_track_count,
            "peak_crowd": self.peak_crowd,
            "density_index": self.density_index,
            "visual_density_index": self.visual_density_index,
            "yolo_density": self.yolo_density,
            "smoothed_density": self.smoothed_density,
            "occupancy_ratio": self.occupancy_ratio,
            "crowd_status": self.crowd_status,
            "flow_direction": self.flow_direction,
            "flow_speed": self.flow_speed,
            "flow_speed_unit": self.flow_speed_unit,
            "congestion_index": self.congestion_index,
            "congestion_status": self.congestion_status,
            "active_flow_vectors": self.active_flow_vectors,
            "anomaly_score": self.anomaly_score,
            "anomaly_status": self.anomaly_status,
            "anomaly_reason": self.anomaly_reason,
            "stagnation_anomaly": self.stagnation_anomaly,
            "density_surge_anomaly": self.density_surge_anomaly,
            "deceleration_anomaly": self.deceleration_anomaly,
            "reversal_anomaly": self.reversal_anomaly,
            "turbulence_anomaly": self.turbulence_anomaly,
            "reference_zone": self.reference_zone,
            "fps": self.fps,
            "inference_ms": self.inference_ms,
            "processing_ms": self.processing_ms,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "latest_frame_url": self.latest_frame_url,
            "processed_video_url": self.processed_video_url,
            "zones": list(self.zones),
        }





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
        default_zones = get_camera_zones("camera_07")
        job.zones = [
            {
                "zone_id": z.zone_id,
                "name": z.name,
                "rect": [round(float(c), 4) for c in z.rect],
                "capacity": z.reference_capacity,
                "current_count": 0,
                "active_track_count": 0,
                "density_index": 0.0,
                "smoothed_density": 0.0,
                "crowd_status": "LOW",
                "flow_direction": "STATIONARY",
                "flow_speed": 0.0,
                "flow_speed_unit": FLOW_SPEED_UNIT,
                "congestion_index": 0.0,
                "congestion_status": "LOW",
                "anomaly_score": 0.0,
                "anomaly_status": ANOMALY_LEVEL_NORMAL,
                "anomaly_reason": "Normal movement patterns",
                "risk": "NORMAL",
            }
            for z in default_zones
        ]

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
            job._zones = get_camera_zones("camera_07")
            job._zone_anomaly_history = {
                z.zone_id: deque(maxlen=ANOMALY_ROLLING_WINDOW_FRAMES)
                for z in job._zones
            }

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

                # Phase 4.2: Hybrid Image-Space Relative Crowd Density Calculation
                density_result = calculate_frame_density(
                    detections,
                    frame,
                    width,
                    height,
                    job.smoothed_density if raw_frame > job.frame_skip else None,
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

                # Maintain unique track IDs across the job
                for detection in detections:
                    if detection.track_id is not None:
                        job._observed_track_ids.add(detection.track_id)
                job.unique_track_count = len(job._observed_track_ids)

                # Track peak crowd observed across processed frames
                if job.current_count > job.peak_crowd:
                    job.peak_crowd = job.current_count

                job.density_index = density_result.density_index
                job.visual_density_index = density_result.visual_density_index
                job.yolo_density = density_result.yolo_density
                job.smoothed_density = density_result.smoothed_density_index
                job.occupancy_ratio = density_result.occupancy_ratio
                job.crowd_status = density_result.crowd_status

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

                # Phase 5: Crowd flow and persistent congestion calculation
                flow_result, job._track_history = calculate_crowd_flow_and_congestion(
                    detections=detections,
                    current_frame=raw_frame,
                    current_timestamp=timestamp,
                    track_history=job._track_history,
                    smoothed_density=job.smoothed_density,
                    previous_vx=job._flow_vx,
                    previous_vy=job._flow_vy,
                    previous_congestion_status=job.congestion_status,
                    high_congestion_counter=job._high_congestion_counter,
                )

                job._flow_vx = flow_result.smoothed_vx
                job._flow_vy = flow_result.smoothed_vy
                job._high_congestion_counter = flow_result.high_congestion_counter

                job.flow_direction = flow_result.flow_direction
                job.flow_speed = flow_result.flow_speed
                job.flow_speed_unit = flow_result.flow_speed_unit
                job.congestion_index = flow_result.congestion_index
                job.congestion_status = flow_result.congestion_status
                job.active_flow_vectors = flow_result.active_flow_vectors

                # Phase 6: Crowd behaviour anomaly intelligence calculation
                anomaly_result = calculate_crowd_anomalies(
                    smoothed_density=job.smoothed_density,
                    flow_speed=job.flow_speed,
                    smoothed_vx=job._flow_vx,
                    smoothed_vy=job._flow_vy,
                    stagnation_factor=flow_result.stagnation_factor,
                    individual_vectors=flow_result.individual_vectors,
                    history_buffer=job._anomaly_history,
                    previous_anomaly_status=job.anomaly_status,
                    high_anomaly_counter=job._high_anomaly_counter,
                )

                # Append current frame state to rolling history buffer (maxlen=50)
                job._anomaly_history.append({
                    "smoothed_density": job.smoothed_density,
                    "flow_speed": job.flow_speed,
                    "smoothed_vx": job._flow_vx,
                    "smoothed_vy": job._flow_vy,
                })
                job._high_anomaly_counter = anomaly_result.high_anomaly_counter

                job.anomaly_score = anomaly_result.anomaly_score
                job.anomaly_status = anomaly_result.anomaly_status
                job.anomaly_reason = anomaly_result.anomaly_reason
                job.stagnation_anomaly = anomaly_result.stagnation_anomaly
                job.density_surge_anomaly = anomaly_result.density_surge_anomaly
                job.deceleration_anomaly = anomaly_result.deceleration_anomaly
                job.reversal_anomaly = anomaly_result.reversal_anomaly
                job.turbulence_anomaly = anomaly_result.turbulence_anomaly

                # Phase 7 Step 2: Zone intelligence metrics calculation
                zone_results, job._zone_track_history = calculate_zone_metrics(
                    detections=detections,
                    frame_width=width,
                    frame_height=height,
                    current_frame=raw_frame,
                    current_timestamp=timestamp,
                    zones=job._zones,
                    zone_density_state=job._zone_density_state,
                    zone_track_history=job._zone_track_history,
                    zone_velocity_state=job._zone_velocity_state,
                    zone_congestion_state=job._zone_congestion_state,
                    zone_anomaly_history=job._zone_anomaly_history,
                    zone_anomaly_counter=job._zone_anomaly_counter,
                )
                job.zones = [zr.to_dict() for zr in zone_results]

                result = {
                    "frame": raw_frame,
                    "timestamp": timestamp,
                    "person_count": len(detections),
                    "current_count": len(detections),
                    "active_track_count": job.active_track_count,
                    "unique_track_count": job.unique_track_count,
                    "peak_crowd": job.peak_crowd,
                    "density_index": density_result.density_index,
                    "visual_density_index": density_result.visual_density_index,
                    "yolo_density": density_result.yolo_density,
                    "smoothed_density": density_result.smoothed_density_index,
                    "occupancy_ratio": density_result.occupancy_ratio,
                    "crowd_status": density_result.crowd_status,
                    "flow_direction": job.flow_direction,
                    "flow_speed": job.flow_speed,
                    "flow_speed_unit": job.flow_speed_unit,
                    "congestion_index": job.congestion_index,
                    "congestion_status": job.congestion_status,
                    "active_flow_vectors": job.active_flow_vectors,
                    "anomaly_score": job.anomaly_score,
                    "anomaly_status": job.anomaly_status,
                    "anomaly_reason": job.anomaly_reason,
                    "stagnation_anomaly": job.stagnation_anomaly,
                    "density_surge_anomaly": job.density_surge_anomaly,
                    "deceleration_anomaly": job.deceleration_anomaly,
                    "reversal_anomaly": job.reversal_anomaly,
                    "turbulence_anomaly": job.turbulence_anomaly,
                    "zones": job.zones,
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
                        "density_analysis": True,
                        "flow_analysis": True,
                        "anomaly_intelligence": True,
                        "zone_intelligence": True,
                        "unique_track_count": job.unique_track_count,
                        "peak_crowd": job.peak_crowd,
                        "reference_zone": job.reference_zone,
                        "zones": job.zones,
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