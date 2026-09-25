from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config import DATA_ROOT
from routes.health import router as health_router
from routes.detection import router as detection_router

app = FastAPI(title="Simhastha AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(detection_router)
app.mount("/media", StaticFiles(directory=DATA_ROOT), name="media")
