import cv2


def get_video_info(video_path: str) -> dict:
    """Read basic metadata from a video file."""
    cap = cv2.VideoCapture(video_path)

    try:
        if not cap.isOpened():
            raise ValueError(
                "Could not open video. Check that the file is a valid, supported video."
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps > 0 else 0.0

        return {
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "total_frames": frame_count,
            "duration_seconds": round(duration, 2)
        }
    finally:
        cap.release()
