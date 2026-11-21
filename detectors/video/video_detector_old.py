"""Experimental MobileNetV2 video-frame classifier.

The model was trained on the UADFV frame dataset. It is not a validated
production deepfake detector and should not be used as sole evidence.
"""
from pathlib import Path
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoDeepfakeDetector:
    IMAGE_SIZE = (160, 160)
    FRAMES_PER_VIDEO = 8

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]
        self.model_path = self.project_root / "models" / "uadfv_mobilenetv2_finetuned.keras"
        self.model = None
        self.tf = None
        self.load_error = None

        try:
            import tensorflow as tf
            self.tf = tf
            if not self.model_path.exists():
                raise FileNotFoundError(f"Trained model not found: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            logger.info("Loaded experimental model: %s", self.model_path.name)
        except Exception as exc:
            self.load_error = str(exc)
            logger.exception("Could not load deepfake model")

    @property
    def available(self) -> bool:
        return self.model is not None

    def preprocess_frames(self, frames):
        if not frames:
            raise ValueError("No video frames received for prediction.")

        indices = np.linspace(
            0, len(frames) - 1,
            min(len(frames), self.FRAMES_PER_VIDEO),
            dtype=int
        )
        images = []
        for index in indices:
            frame = frames[index]
            if frame is None or frame.size == 0:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, self.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            images.append(rgb.astype(np.float32))

        if not images:
            raise ValueError("No valid frames available for prediction")

        batch = np.asarray(images, dtype=np.float32)
        return self.tf.keras.applications.mobilenet_v2.preprocess_input(batch)

    def predict(self, input_data):
        if self.model is None:
            raise RuntimeError(
                "The experimental TensorFlow model is unavailable. "
                f"Load error: {self.load_error or 'unknown'}"
            )
        if not isinstance(input_data, (list, tuple)) or not input_data:
            raise ValueError("A non-empty list of video frames is required.")

        batch = self.preprocess_frames(input_data)
        predictions = self.model.predict(batch, verbose=0)
        frame_probabilities = np.asarray(predictions, dtype=np.float32).reshape(-1)

        if frame_probabilities.size == 0 or not np.isfinite(frame_probabilities).all():
            raise RuntimeError("Model returned invalid prediction values")

        fake_probability = float(np.clip(np.mean(frame_probabilities), 0.0, 1.0))
        return {
            "model_status": "experimental",
            "deepfake_probability": round(fake_probability, 4),
            "frames_analyzed_by_model": int(len(frame_probabilities)),
            "model_name": "UADFV MobileNetV2 (experimental)"
        }
