# DeepShield — Project Report Draft

## Title
**DeepShield: AI-Based Deepfake Video Risk Detection System**

## Abstract
DeepShield is an experimental prototype designed to analyze digital videos for potential visual manipulation. The system accepts an uploaded video, extracts representative frames, performs face localization for diagnostic purposes, and applies a MobileNetV2-based binary classifier to selected frames. Frame-level scores are averaged to produce a video-level score, which is displayed through heuristic LOW, MEDIUM, and HIGH risk bands. A FastAPI backend exposes REST and WebSocket interfaces, while a browser-based frontend supports uploaded-video analysis and a controlled live-camera frame-scoring demonstration. The prototype demonstrates an end-to-end AI integration workflow; however, its current model results are inconsistent across evaluation splits and are not sufficient for production use.

## 1. Introduction
Advances in generative models and face manipulation tools have made synthetic or altered media increasingly accessible. Deepfake videos may be used for misinformation, impersonation, and fraud. Automated analysis can help reviewers prioritize suspicious media, but model predictions must be interpreted carefully because compression, lighting, pose, and dataset differences can affect performance.

DeepShield explores a modular video-analysis pipeline that can later be extended with temporal and audio-based models.

## 2. Problem Statement
Manual inspection of video authenticity is time-consuming and subjective. The project aims to build a prototype that accepts video input, extracts visual evidence, runs an experimental machine-learning classifier, and returns a structured risk-oriented result through an API and user interface.

## 3. Objectives
- Build a backend service for video upload and analysis.
- Extract video metadata and sample frames within a memory limit.
- Localize candidate faces and prepare face crops for diagnostic/future use.
- Integrate a MobileNetV2-based REAL/FAKE frame classifier.
- Aggregate frame predictions into a video-level score.
- Present heuristic risk bands through a web interface.
- Demonstrate camera-frame communication using WebSocket.
- Keep the design modular for future temporal and audio detectors.

## 4. Technology Stack
- Python
- FastAPI and Uvicorn
- TensorFlow / Keras
- MobileNetV2
- OpenCV and NumPy
- HTML, CSS, JavaScript
- WebSocket
- Pytest

## 5. System Methodology
1. User uploads a supported video.
2. FastAPI validates the extension and enforces a 200 MB limit.
3. OpenCV reads video metadata.
4. The frame sampler extracts approximately one frame per second, capped at 64 frames.
5. Haar Cascade localizes candidate faces for diagnostic counts.
6. Up to eight evenly spaced full frames are resized to 160×160 and converted to MobileNetV2 preprocessing range [-1, 1].
7. The UADFV MobileNetV2 classifier produces sigmoid scores.
8. The scores are averaged to obtain one video-level model score.
9. The risk engine maps the score to heuristic bands: LOW (<0.30), MEDIUM (0.30–<0.70), and HIGH (≥0.70).
10. The frontend displays metadata, diagnostic counts, model score, and warning text.

Face localization is not used as the classifier input in the current build. The model uses resized full frames. The live-camera demo sends individual JPEG frames through WebSocket and scores each frame independently.

## 6. Evaluation and Results
The supplied project contains multiple UADFV reports with different evaluation procedures:
- One frame-level validation report records approximately 75.6% accuracy and 0.825 AUC.
- One 20-video evaluation report records 90% accuracy and 0.94 AUC on its particular split.
- Five-fold video-level cross-validation records mean accuracy of approximately 40.8% and mean AUC of approximately 0.467.
- A separate fold-1 fine-tuning report records 50% accuracy and 0.60 AUC.
- The separate FaceForensics++ c23 experiment discussed during development recorded 50% validation video accuracy and 0.50 AUC on 16 validation videos.

These values are not directly comparable and demonstrate that the model's performance is unstable across splits. The results do not establish reliable generalization to unseen real-world videos.

## 7. Limitations
- Limited training data and split sensitivity.
- No independent, large-scale external validation.
- Haar Cascade may miss faces or return false positives.
- Full-frame frame-based classification only; no temporal model.
- Risk bands are heuristic, not calibrated probabilities.
- Live camera demo is not full WebRTC calling.
- No audio deepfake detection in the current phase.
- Runtime depends on a compatible TensorFlow installation.

## 8. Future Scope
- Train and validate on larger, source-disjoint datasets.
- Add robust face alignment and higher-resolution face crops.
- Explore CNN-LSTM/GRU or video-transformer temporal modeling.
- Implement full WebRTC signalling and controlled video-call sessions.
- Add audio deepfake detection and multimodal fusion.
- Calibrate confidence and thresholds on held-out data.
- Add authentication, secure deployment, audit trails, and monitoring.

## 9. Conclusion
DeepShield demonstrates a modular software pipeline for video upload, metadata extraction, face localization, experimental MobileNetV2 inference, risk-score presentation, and WebSocket camera-frame analysis. The current model is a research prototype rather than a dependable authenticity verifier. Further dataset expansion, temporal modeling, independent evaluation, and system hardening are required before real-world use.
