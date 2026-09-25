import logging
from dataclasses import dataclass
from typing import Any

import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: list[int]
    track_id: int | None = None


class PersonDetector:
    """YOLO person detector with tracking support."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = (
            torch.cuda.get_device_name(0)
            if self.device == "cuda"
            else None
        )
        self._model: YOLO | None = None

    def _load(self) -> YOLO:
        if self._model is None:
            logger.info(
                "Loading YOLO model=%s device=%s gpu=%s",
                self.model_name,
                self.device,
                self.gpu_name or "n/a",
            )
            self._model = YOLO(self.model_name)

        return self._model

    def detect(
        self,
        frame: Any,
        confidence: float,
        input_size: int,
    ) -> tuple[list[Detection], float]:

        model = self._load()

        try:
            result = model.predict(
                frame,
                classes=[0],
                conf=confidence,
                imgsz=input_size,
                device=self.device,
                verbose=False,
            )[0]

        except RuntimeError as error:
            if self.device != "cuda":
                raise

            logger.warning(
                "CUDA inference failed (%s); retrying on CPU",
                error,
            )

            self.device = "cpu"
            self.gpu_name = None

            result = model.predict(
                frame,
                classes=[0],
                conf=confidence,
                imgsz=input_size,
                device="cpu",
                verbose=False,
            )[0]

        detections = []

        if result.boxes is not None:
            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(
                    Detection(
                        class_name="person",
                        confidence=round(float(box.conf[0]), 4),
                        bbox=[
                            round(x1),
                            round(y1),
                            round(x2),
                            round(y2),
                        ],
                    )
                )

        return detections, float(
            result.speed.get("inference", 0.0)
        )

    def track(
        self,
        frame: Any,
        confidence: float,
        input_size: int,
    ) -> tuple[list[Detection], float]:

        model = self._load()

        try:
            result = model.track(
                frame,
                classes=[0],
                conf=confidence,
                imgsz=input_size,
                device=self.device,
                persist=True,
                verbose=False,
            )[0]

        except RuntimeError as error:
            if self.device != "cuda":
                raise

            logger.warning(
                "CUDA tracking failed (%s); retrying on CPU",
                error,
            )

            self.device = "cpu"
            self.gpu_name = None

            result = model.track(
                frame,
                classes=[0],
                conf=confidence,
                imgsz=input_size,
                device="cpu",
                persist=True,
                verbose=False,
            )[0]

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                track_id = None

                if box.id is not None:
                    track_id = int(box.id[0].item())

                detections.append(
                    Detection(
                        class_name="person",
                        confidence=round(float(box.conf[0]), 4),
                        bbox=[
                            round(x1),
                            round(y1),
                            round(x2),
                            round(y2),
                        ],
                        track_id=track_id,
                    )
                )

        return detections, float(
            result.speed.get("inference", 0.0)
        )