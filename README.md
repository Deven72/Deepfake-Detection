# DeepShield
## AI-Assisted Video Deepfake Risk Analysis — Research Prototype

DeepShield is a college-project prototype for analyzing uploaded videos and demonstrating frame-by-frame camera scoring. It combines video metadata extraction, bounded frame sampling, face localization, an experimental MobileNetV2 classifier, and a heuristic risk-band display.

> **Important limitation:** DeepShield is not a production-grade or independently validated deepfake detector. Its model score is not calibrated confidence and is not proof that media is real or fake. Do not use it as the sole basis for identity, security, employment, legal, financial, or other high-impact decisions.

## Implemented scope

### Implemented in this package
- FastAPI backend and interactive HTML frontend.
- Video upload endpoint: `POST /api/v1/detection/video`.
- Video metadata extraction and memory-bounded frame sampling (up to 64 sampled frames).
- OpenCV Haar Cascade face localization and face-crop preprocessing diagnostics.
- UADFV MobileNetV2 model inference, sampling up to 8 frames per uploaded video.
- Heuristic LOW / MEDIUM / HIGH risk bands.
- WebSocket endpoint `/ws/live-detection` for browser-camera JPEG frame scoring.
- Health endpoint: `GET /health`.
- Swagger/OpenAPI documentation: `/docs`.

### Not implemented / future work
- Full WebRTC peer-to-peer video calling, call signalling, rooms, authentication, and multi-user session management.
- Audio deepfake detection (planned Phase 2).
- CNN-LSTM / CNN-GRU temporal model. The bundled model is frame-based MobileNetV2; it does not learn temporal relationships.
- Production-grade face tracking, robust face alignment, calibrated uncertainty, adversarial testing, or external benchmark certification.
- Persistent audit logs, user accounts, role-based access, and deployment hardening.

The live-camera feature is a controlled WebSocket frame-scoring demonstration, **not** a complete video-call product.

## Repository structure

```text
DeepShield/
├── backend/
│   ├── main.py
│   ├── api/routes/
│   │   ├── detection.py
│   │   └── live.py
│   └── services/
│       ├── detection/detection_service.py
│       ├── media/
│       │   ├── face_detector.py
│       │   ├── face_preprocessor.py
│       │   ├── frame_sampler.py
│       │   ├── video_analysis.py
│       │   └── video_processor.py
│       └── risk/risk_engine.py
├── detectors/
│   ├── base/detector.py
│   └── video/video_detector.py
├── frontend/index.html
├── models/
│   ├── uadfv_mobilenetv2_finetuned.keras
│   ├── uadfv_fold1_finetuned.keras
│   ├── uadfv_mobilenetv2_experimental.keras
│   ├── deepfake_cnn.keras
│   ├── face_detection/blaze_face_short_range.tflite
│   └── evaluation reports (*.json)
├── experiments/
├── datasets/test_video.mp4
├── requirements/
└── README.md
```

## Requirements

- Python 3.10–3.12 recommended (use a TensorFlow-supported Python/platform combination).
- 8 GB RAM recommended for CPU inference; more memory is helpful for long videos.
- Windows, Linux, or macOS for the API; TensorFlow wheel availability depends on OS and Python version.
- No GPU is required. CPU inference may be slower.
- Internet is needed only to install Python dependencies.

## Installation

### Windows PowerShell

```powershell
cd E:\1_Phase1\DeepShield

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements\backend.txt
```

If your project is in a different directory, replace the `cd` path. If TensorFlow cannot be installed for your Windows/Python combination, use a supported Python version or run the backend in WSL2/Linux.

### Linux / macOS

```bash
cd DeepShield
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/backend.txt
```

## Run

From the project root (the folder containing `backend`, `detectors`, `models`, and `frontend`):

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open:
- Frontend: http://127.0.0.1:8000/
- Health: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs

The model is loaded lazily on the first inference request. Before that, `/health` reports `model_status: not_loaded`. After inference initialization it reports `experimental` if the model loads, or `unavailable` if TensorFlow/the model cannot load. Video inference returns HTTP 503 when the model is unavailable; the health endpoint remains available.

## API

### `GET /health`

Example response:

```json
{
  "status": "healthy",
  "service": "DeepShield API",
  "model_status": "experimental",
  "model_name": "UADFV MobileNetV2",
  "model_load_error": null
}
```

### `POST /api/v1/detection/video`

Send `multipart/form-data` with the field name `file`.

Allowed extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`. Maximum upload size: 200 MB.

Example using curl:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/detection/video" \
  -F "file=@datasets/test_video.mp4"
```

The response contains video metadata, sample count, face-localization counts, model score, and heuristic risk band. The model uses up to 8 evenly spaced sampled frames. The media pipeline samples approximately 1 frame/second, capped at 64 frames.

### `WS /ws/live-detection`

Connect a WebSocket and send binary JPEG frames (maximum 2 MB per frame). The browser UI captures a camera frame approximately every 1.2 seconds. Each frame is scored independently; temporal consistency is not modeled.

The browser camera API generally requires `localhost` or HTTPS. For another device, serve the frontend over HTTPS and configure a secure WebSocket (`wss://`).

## Model and data notes

### Bundled inference model
The API loads `models/uadfv_mobilenetv2_finetuned.keras`. It is a frame-based MobileNetV2 sigmoid classifier:
- Label `0`: REAL
- Label `1`: FAKE
- Input: 160 × 160 RGB
- Preprocessing: MobileNetV2 `preprocess_input`, scaling pixels to [-1, 1].
- Video score: arithmetic mean of up to 8 frame-level fake scores.

Face detection is currently diagnostic; detected face crops are counted but are **not** fed into this bundled classifier. The classifier receives resized full frames, consistent with the training script.

### Existing reports supplied with the project
The included JSON reports describe different experiments/splits and must not be combined into one headline result:
- `uadfv_mobilenetv2_report.json`: frame-level validation accuracy about 75.6%, AUC about 0.825 on its recorded split.
- `uadfv_video_evaluation.json`: reports 90% video-level accuracy and AUC 0.94 on 20 videos for a particular evaluation split. This result is split-specific and should not be treated as external validation.
- `uadfv_5fold_cv_report.json`: mean video-level accuracy about 40.8%, mean AUC about 0.467 across five folds.
- `uadfv_fold1_finetuned_report.json`: 50% accuracy and AUC 0.60 for its fold-1 experiment.
- The separate FaceForensics++ c23 experiment discussed during development had 16 validation videos with video-level accuracy 50% and AUC 0.50 after fine-tuning. The FF++ training model is not included in this ZIP.

These results are inconsistent and the dataset is small. The strongest defensible statement is that the model is an experimental prototype requiring more independent, source-aware evaluation. Do not advertise the 90% result as guaranteed or real-world accuracy.

### Face detector
The backend uses OpenCV Haar Cascade for lightweight face localization. It is not a deepfake classifier. Sensitive parameters improve recall but may increase false positives. Small, side-facing, occluded, and low-resolution faces may be missed. A green bounding box is not proof that the detection is correct.

### Risk bands
The current risk bands are simple heuristic display thresholds:
- LOW: score < 0.30
- MEDIUM: 0.30 ≤ score < 0.70
- HIGH: score ≥ 0.70

They are not statistically calibrated probabilities or a clinical/security decision threshold.

## Tests

Install test tools:

```bash
pip install pytest
```

Run from project root:

```bash
python -m pytest -q
```

Quick checks:

```bash
python -m backend.tests.test_opencv
python -m backend.tests.test_frame_sampler
```

## Troubleshooting

- **`ModuleNotFoundError: backend`**: run Uvicorn from the project root, not from inside `backend`.
- **TensorFlow import/model load error**: check the Python/TensorFlow compatibility and run `python -c "import tensorflow as tf; print(tf.__version__)"`. `/health` includes the load error.
- **Model file missing**: verify `models/uadfv_mobilenetv2_finetuned.keras` exists.
- **Camera permission denied**: allow camera access and use `localhost` or HTTPS.
- **WebSocket fails behind a proxy**: ensure the proxy supports WebSocket upgrades and use `wss://` for HTTPS pages.
- **Video rejected**: use one of the allowed extensions and keep the file below 200 MB.
- **No face detected**: Haar Cascade is a simple detector; this does not automatically mean the frame contains no face.

## Security and privacy notes

- Uploaded videos are written to a temporary file and deleted after processing.
- The API does not store user videos or create user accounts.
- The sample test video is included for local smoke testing; obtain permission before processing other people's media.
- This is a local research prototype. Add authentication, HTTPS, rate limits, access controls, malware scanning, retention policies, and deployment monitoring before exposing it publicly.

## Academic project description

**DeepShield — AI-Based Deepfake Video Risk Detection System**

DeepShield is a research prototype that analyzes uploaded video content using frame sampling, face localization, and a MobileNetV2-based binary classifier. The backend is implemented with FastAPI and exposes REST and WebSocket interfaces. It aggregates frame-level model outputs into a video-level score and maps that score to heuristic risk bands. The current phase demonstrates uploaded-video analysis and controlled live-camera frame scoring. Future work includes full WebRTC call integration, temporal CNN-LSTM/GRU modeling, improved face alignment, audio deepfake detection, and broader source-independent evaluation.
