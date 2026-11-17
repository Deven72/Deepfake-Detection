# Frontend Integration Contract

The included `frontend/index.html` is a working single-page demonstration for the current backend.

## REST video analysis

- Method: `POST`
- Path: `/api/v1/detection/video`
- Content type: `multipart/form-data`
- Form field: `file`
- Allowed extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
- Maximum size: 200 MB

Successful response shape:

```json
{
  "success": true,
  "data": {
    "video_info": {
      "width": 1280,
      "height": 720,
      "fps": 24.0,
      "total_frames": 192,
      "duration_seconds": 8.0
    },
    "frames_sampled": 8,
    "frames_with_face": 7,
    "faces_detected": 8,
    "faces_preprocessed": 8,
    "face_detector": "OpenCV Haar Cascade (diagnostic only)",
    "model_status": "experimental",
    "deepfake_probability": 0.51,
    "frames_analyzed_by_model": 8,
    "model_name": "UADFV MobileNetV2 (experimental)",
    "risk_score": 51.0,
    "risk_level": "MEDIUM",
    "risk_interpretation": "Heuristic band from an experimental model score; not a verified probability of manipulation."
  }
}
```

The numeric values above are illustrative response values, not a claim about the bundled sample video.

## WebSocket camera-frame demo

- Path: `/ws/live-detection`
- Send: binary JPEG bytes (maximum 2 MB)
- Receive: JSON message with `success` and `data`, or `success: false` and `error`.
- Suggested client rate: one frame every 1–2 seconds.
- The included UI uses 640-pixel-wide JPEG frames and does not send audio.

Example success message:

```json
{
  "success": true,
  "data": {
    "model_status": "experimental",
    "deepfake_probability": 0.51,
    "frames_analyzed_by_model": 1,
    "model_name": "UADFV MobileNetV2 (experimental)",
    "risk_score": 51.0,
    "risk_level": "MEDIUM",
    "faces_detected": 1,
    "message": "Single-frame experimental estimate; temporal evidence is not analyzed."
  }
}
```

## Frontend team notes

- Display the experimental warning near the score.
- Do not label the score as a verified authenticity probability.
- Keep model status visible.
- Handle HTTP 400 (invalid upload), 413 (file too large), 503 (model unavailable), and 500 (unexpected server error).
- Camera access requires user permission and a secure context (`localhost` or HTTPS).
- Use `wss://` when the page is served over HTTPS.
