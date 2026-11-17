"""Memory-bounded video frame sampling."""
import cv2
import numpy as np


def sample_frames(video_path: str, sample_fps: float = 1.0,
                  max_frames: int = 64) -> list:
    """Sample at approximately sample_fps, capped at max_frames.

    Uses frame seeking when video metadata is available, avoiding loading every
    decoded frame into memory. Returns OpenCV BGR frames in chronological order.
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero")
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total <= 0:
            raise ValueError("Video metadata is unavailable or invalid")

        interval = max(int(round(fps / sample_fps)), 1)
        indices = np.arange(0, total, interval, dtype=np.int64)
        if len(indices) > max_frames:
            positions = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
            indices = indices[positions]

        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                frames.append(frame)

        if not frames:
            raise ValueError("No frames could be decoded from the video")
        return frames
    finally:
        cap.release()
