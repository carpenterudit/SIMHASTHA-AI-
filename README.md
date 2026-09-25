# SIMHASTHA AI — Command Center Foundation

A professional mock-data command-center dashboard for Simhastha Ujjain 2028. This foundation deliberately excludes AI/ML, CCTV feeds, drone control, and real integrations.

## Features

- Dark operational dashboard with responsive desktop, laptop, and tablet layouts
- Mock live CCTV panel, isolated Ujjain map, KPI cards, crowd chart, zone risks, incidents, emergency form, drones, and acknowledgable alerts
- Centralized frontend mock data and a small API service adapter
- FastAPI health endpoint: `GET /api/health`

## Technology

- Frontend: React, Vite, Lucide React, Recharts, CSS
- Backend: Python, FastAPI, Uvicorn

## Folder structure

```text
src/             React components, dashboard page, mock data, API adapter
backend/         FastAPI application, health route, future services/models folders
docs/            Architecture notes
```

## Run locally

Frontend (Node.js 20+):

```bash
npm install
npm run dev
```

Backend (Python 3.10+):

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://localhost:5173`. The health endpoint is `http://localhost:8000/api/health`.

## Current limitations

The live monitoring panel supports real uploaded-video person detection using YOLO11n. Other operational widgets remain mock UI data. The emergency form and fleet controls are non-dispatch UI actions only. Current detected people is a per-frame detection count, not unique-person tracking.

## Phase 2 setup

Install the additional computer-vision dependencies before running the backend:

```powershell
cd backend
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Open Live CCTV Monitoring on the dashboard, select an `.mp4`, `.avi`, `.mov`, or `.mkv` file, and select **Start AI Analysis**. On the first detection run, Ultralytics downloads the lightweight `yolo11n.pt` model automatically.

## Future modules

The frontend adapter and backend routing layout are prepared for `/api/detection`, `/api/tracking`, `/api/crowd`, `/api/congestion`, `/api/missing-person`, `/api/emergency`, `/api/medical`, `/api/traffic`, and `/api/drone`.
