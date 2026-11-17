import numpy as np

from backend.services.media.face_preprocessor import crop_face, preprocess_face


def test_crop_face_clamps_to_frame_and_returns_copy():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_face(frame, (5, 5, 30, 30), margin=0.2)
    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    crop[0, 0] = 255
    assert frame[0, 0].sum() == 0


def test_preprocess_face_shape_and_range():
    face = np.full((40, 50, 3), 128, dtype=np.uint8)
    output = preprocess_face(face, image_size=(160, 160))
    assert output.shape == (160, 160, 3)
    assert output.dtype == np.float32
    assert 0 <= output.min() <= output.max() <= 1


def test_none_face_returns_none():
    assert crop_face(None, (0, 0, 10, 10)) is None
    assert preprocess_face(None) is None
