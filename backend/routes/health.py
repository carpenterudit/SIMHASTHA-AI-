from fastapi import APIRouter
from config import API_PREFIX, SYSTEM_NAME

router = APIRouter(prefix=API_PREFIX, tags=["system"])

@router.get("/health")
async def health_check():
    return {"status": "online", "system": SYSTEM_NAME}
