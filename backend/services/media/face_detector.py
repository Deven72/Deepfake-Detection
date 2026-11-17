"""OpenCV Haar Cascade face detector.

This lightweight detector is used for face localization only. It does NOT
classify a face as real or fake. Haar Cascade may miss small/side-facing faces
and can produce false positives; results should be treated as diagnostics.
"""
from pathlib import Path

import cv2


def _load_cascade() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Could not load OpenCV Haar cascade: {cascade_path}")
    return cascade


_FACE_CASCADE = _load_cascade()


def detect_faces(frame, scale_factor: float = 1.05, min_neighbors: int = 3,
                 min_size: tuple[int, int] = (20, 20)) -> list[tuple[int, int, int, int]]:
    """Detect faces in an OpenCV BGR frame; return (x, y, width, height)."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR frame.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    boxes = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    height, width = frame.shape[:2]
    results = []
    for x, y, w, h in boxes:
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(width, int(x + w)), min(height, int(y + h))
        if x2 > x1 and y2 > y1:
            results.append((x1, y1, x2 - x1, y2 - y1))
    return results
