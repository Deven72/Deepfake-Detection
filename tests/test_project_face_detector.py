import numpy as np

from backend.services.media.face_detector import detect_faces


def test_detector_returns_list_for_valid_frame():
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    result = detect_faces(frame)
    assert isinstance(result, list)


def test_detector_handles_none():
    assert detect_faces(None) == []
