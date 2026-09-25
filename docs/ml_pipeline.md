# Phase 2: video person-detection pipeline

## Core concepts

Computer vision lets software obtain useful information from images and video. Object detection is a computer-vision task that identifies what an object is and where it appears in a frame. YOLO (You Only Look Once) is an object detector designed to make these predictions efficiently in a single pass through a neural network, which makes it a good fit for video.

A bounding box is the rectangle drawn around a detected object. A confidence score expresses how certain the model is about that detection. This project keeps only COCO's `person` class with a default confidence of 0.40, so unrelated objects are ignored.

OpenCV opens the uploaded video and reads one frame at a time; it never loads the full video into memory. YOLO detects people in that frame, and OpenCV draws the resulting boxes and labels. GPU acceleration can substantially reduce neural-network inference time. The application automatically uses CUDA when PyTorch reports an NVIDIA CUDA device; otherwise it runs safely on CPU.

## Pipeline

```text
Uploaded video → OpenCV frames → YOLO person detection → bounding boxes + confidence
→ current-frame count → annotated frame/video → dashboard
```

The **Current Detected People** value is only the number of person detections in the current frame. It is not a total or a unique-person count: someone appearing in 100 frames can be detected in all 100. Assigning persistent identities requires tracking, which is intentionally outside Phase 2.

## Runtime behaviour

The backend queues each upload as a background job, so FastAPI remains responsive. The frontend polls job status and shows the latest annotated frame, frame count, processing FPS, inference time, model, and active device. Results are saved as extensible JSON files for future tracking and zone-analysis phases.
