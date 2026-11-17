import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from backend.api.routes.detection import router as detection_router, detection_service
from backend.api.routes.live import router as live_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="DeepShield API",
    description=(
        "Experimental AI-assisted video deepfake risk analysis. "
        "Scores are research outputs and are not proof of authenticity."
    ),
    version="1.0.0"
)

FRONTEND_FILE = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "DeepShield API",
        "model_status": detection_service.model_status,
        "model_name": "UADFV MobileNetV2",
        "model_load_error": detection_service.model_load_error
    }


@app.get("/", include_in_schema=False)
def home():
    if not FRONTEND_FILE.exists():
        return {"message": "DeepShield API is running. Open /docs for API documentation."}
    return FileResponse(FRONTEND_FILE)


app.include_router(detection_router)
app.include_router(live_router)
