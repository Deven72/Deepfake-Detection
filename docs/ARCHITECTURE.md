# DeepShield Architecture

## High-level components

```text
Browser UI
  ├── Video upload ── HTTP multipart ──> FastAPI /api/v1/detection/video
  │                                      ├── Video metadata
  │                                      ├── Bounded frame sampler
  │                                      ├── OpenCV Haar face diagnostics
  │                                      └── UADFV MobileNetV2 (up to 8 frames)
  │                                             └── Mean frame score
  │                                                  └── Heuristic risk band
  │
  └── Camera preview ── JPEG binary ── WebSocket /ws/live-detection
                                         └── Single-frame model estimate
```

## Backend module responsibilities

| Module | Responsibility |
|---|---|
| `backend/api/routes/detection.py` | Validates upload extension and size, writes a temporary file, returns analysis JSON, removes temporary file |
| `backend/api/routes/live.py` | Receives JPEG frames over WebSocket and returns per-frame estimates |
| `backend/services/media/video_processor.py` | Reads width, height, FPS, frame count, and duration |
| `backend/services/media/frame_sampler.py` | Samples video frames with a maximum-frame cap |
| `backend/services/media/face_detector.py` | Localizes candidate faces using OpenCV Haar Cascade |
| `backend/services/media/face_preprocessor.py` | Crops and resizes face regions for diagnostics/future face-based models |
| `backend/services/media/video_analysis.py` | Combines metadata, frame sampling, and face diagnostics |
| `detectors/video/video_detector.py` | Loads the bundled UADFV MobileNetV2 model and runs frame inference |
| `backend/services/risk/risk_engine.py` | Maps a model score to simple display bands |
| `backend/services/detection/detection_service.py` | Orchestrates media analysis, model inference, and risk output |

## Current model flow

1. Decode and sample video frames at approximately one frame per second, capped at 64.
2. Independently run Haar Cascade face localization on sampled frames for diagnostic counts.
3. Select up to eight evenly spaced full frames for model inference.
4. Convert BGR to RGB and resize to 160×160.
5. Apply `tf.keras.applications.mobilenet_v2.preprocess_input` (pixel range [-1, 1]).
6. Predict sigmoid fake scores for each selected frame.
7. Average the frame scores into one video-level score.
8. Map the score to LOW, MEDIUM, or HIGH using fixed heuristic thresholds.

**Important:** Face crops are not currently passed to the bundled classifier. The classifier uses full frames, matching the bundled UADFV training script. There is no LSTM/GRU/Transformer temporal model in this build.

## Live-camera protocol

- Browser requests camera permission with `getUserMedia`.
- Browser captures a JPEG frame about every 1.2 seconds.
- JPEG bytes are sent to `/ws/live-detection`.
- Server decodes the image, runs a single-frame estimate, and returns JSON.
- This is a frame-scoring demo. It is not WebRTC peer-to-peer calling and has no call signalling or multi-party session layer.

## Future modular extensions

- Add a `BaseDetector` implementation for audio deepfake analysis.
- Add video temporal aggregation and a validated face alignment/cropping pipeline.
- Add WebRTC signalling, room management, identity/authentication, and session lifecycle.
- Add calibration and independent evaluation before using risk labels operationally.
