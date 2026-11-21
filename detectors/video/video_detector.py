
"""DeepShield FaceForensics++ MobileNetV2 video detector."""

from pathlib import Path
import logging

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

logger = logging.getLogger(__name__)


class VideoDeepfakeDetector:
    FRAME_SIZE = (160, 160)
    IMAGE_SIZE = (224, 224)
    FRAMES_PER_VIDEO = 8
    PADDING_RATIO = 1.70
    DETECTION_CONFIDENCE = 0.1

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]

        self.model_path = (
            self.project_root / "models" /
            "deepshield_mobilenetv2_finetuned.keras"
        )

        self.face_model_path = (
            self.project_root / "models" /
            "face_detection" / "blaze_face_short_range.tflite"
        )

        self.model = None
        self.face_detector = None
        self.tf = tf
        self.mp = mp
        self.load_error = None

        try:
            if not self.model_path.exists():
                raise FileNotFoundError(self.model_path)

            if not self.face_model_path.exists():
                raise FileNotFoundError(self.face_model_path)

            self.model = tf.keras.models.load_model(
                str(self.model_path),
                compile=False
            )

            options = mp.tasks.vision.FaceDetectorOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(self.face_model_path)
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                min_detection_confidence=self.DETECTION_CONFIDENCE
            )

            self.face_detector = (
                mp.tasks.vision.FaceDetector.create_from_options(options)
            )

            logger.info("DeepShield trained model loaded successfully.")

        except Exception as exc:
            self.load_error = str(exc)
            logger.exception("Could not initialize video detector")

    @property
    def available(self):
        return self.model is not None and self.face_detector is not None

    def extract_face_crop(self, frame):
        """Detect the highest-confidence face and create a padded crop."""

        if frame is None or frame.size == 0:
            return None

        # Match the 160x160 training-frame resolution.
        frame = cv2.resize(
            frame,
            self.FRAME_SIZE,
            interpolation=cv2.INTER_AREA
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.face_detector.detect(mp_image)

        if not result.detections:
            return None

        detection = max(
            result.detections,
            key=lambda d: max(d.categories[0].score, 0.0)
            if d.categories else 0.0
        )

        box = detection.bounding_box

        frame_h, frame_w = frame.shape[:2]

        x, y = box.origin_x, box.origin_y
        w, h = box.width, box.height

        if w <= 0 or h <= 0:
            return None

        # Square crop with 1.70 padding ratio.
        side = int(max(w, h) * self.PADDING_RATIO)

        center_x = x + w / 2
        center_y = y + h / 2

        x1 = max(0, int(center_x - side / 2))
        y1 = max(0, int(center_y - side / 2))
        x2 = min(frame_w, int(center_x + side / 2))
        y2 = min(frame_h, int(center_y + side / 2))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return crop

    def preprocess_frames(self, frames):
        if not frames:
            raise ValueError("No video frames received.")

        indices = np.linspace(
            0,
            len(frames) - 1,
            min(len(frames), self.FRAMES_PER_VIDEO),
            dtype=int
        )

        images = []
        for index in indices:
            crop = self.extract_face_crop(frames[index])

            if crop is None:
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            resized = cv2.resize(
                rgb,
                self.IMAGE_SIZE,
                interpolation=cv2.INTER_AREA
            )

            images.append(resized.astype(np.float32))

        if not images:
            raise ValueError(
                "No faces detected in sampled video frames."
            )

        batch = np.asarray(images, dtype=np.float32)

        return tf.keras.applications.mobilenet_v2.preprocess_input(
            batch
        )

    def predict(self, input_data):
        if not self.available:
            raise RuntimeError(
                f"Detector unavailable: {self.load_error or 'unknown'}"
            )

        if not isinstance(input_data, (list, tuple)) or not input_data:
            raise ValueError("A non-empty list of video frames is required.")

        batch = self.preprocess_frames(input_data)

        predictions = self.model.predict(batch, verbose=0)

        probabilities = np.asarray(
            predictions, dtype=np.float32
        ).reshape(-1)

        if not probabilities.size or not np.isfinite(probabilities).all():
            raise RuntimeError("Invalid model predictions.")

        fake_probability = float(
            np.clip(np.mean(probabilities), 0.0, 1.0)
        )

        return {
            "model_status": "trained",
            "deepfake_probability": round(fake_probability, 4),
            "frames_analyzed_by_model": int(len(probabilities)),
            "model_name": "DeepShield FaceForensics++ MobileNetV2"
        }
