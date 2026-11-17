"""Face crop and image preprocessing helpers."""
import cv2
import numpy as np


def crop_face(frame, face_box, margin: float = 0.20):
    """Crop a face with optional proportional context margin."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    if len(face_box) != 4:
        raise ValueError("face_box must contain x, y, width, height")

    x, y, width, height = map(int, face_box)
    if width <= 0 or height <= 0:
        return None

    frame_height, frame_width = frame.shape[:2]
    mx, my = int(width * margin), int(height * margin)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2 = min(frame_width, x + width + mx)
    y2 = min(frame_height, y + height + my)

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def preprocess_face(face, image_size=(224, 224)):
    """Convert a BGR crop to RGB float32 [0, 1]."""
    if face is None or getattr(face, "size", 0) == 0:
        return None
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, image_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0
