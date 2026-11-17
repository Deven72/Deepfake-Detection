"""WebSocket endpoint for controlled live-camera frame scoring.

This sends individual JPEG frames to the backend. It is a prototype data
channel, not a complete peer-to-peer WebRTC calling/signalling system.
"""
import asyncio
import logging

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.routes.detection import detection_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Live Detection"])
MAX_FRAME_BYTES = 2 * 1024 * 1024


@router.websocket("/ws/live-detection")
async def live_detection(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_bytes()
            if not payload or len(payload) > MAX_FRAME_BYTES:
                await websocket.send_json({
                    "success": False,
                    "error": "Frame is empty or exceeds the 2 MB limit."
                })
                continue

            array = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({
                    "success": False,
                    "error": "Could not decode JPEG frame."
                })
                continue

            try:
                result = await asyncio.to_thread(
                    detection_service.analyze_frame, frame
                )
                await websocket.send_json({"success": True, "data": result})
            except Exception as exc:
                logger.exception("Live frame inference failed")
                await websocket.send_json({
                    "success": False,
                    "error": str(exc)
                })
    except WebSocketDisconnect:
        logger.info("Live detection client disconnected")
    except Exception:
        logger.exception("Live detection websocket failed")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
