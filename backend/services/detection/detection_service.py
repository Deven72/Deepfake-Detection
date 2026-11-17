"""Orchestrates media diagnostics and lazy experimental model inference."""
from backend.services.media.video_analysis import analyze_video_media
from backend.services.media.face_detector import detect_faces
from backend.services.risk.risk_engine import calculate_risk
from detectors.video.video_detector import VideoDeepfakeDetector


class DetectionService:
    def __init__(self):
        self._video_detector = None
        self._model_load_attempted = False
        self._model_load_error = None

    def _get_detector(self):
        if not self._model_load_attempted:
            self._model_load_attempted = True
            try:
                self._video_detector = VideoDeepfakeDetector()
                self._model_load_error = self._video_detector.load_error
            except Exception as exc:
                self._model_load_error = str(exc)
                self._video_detector = None
        if self._video_detector is None or not self._video_detector.available:
            raise RuntimeError(
                "The experimental TensorFlow model is unavailable. "
                f"Load error: {self._model_load_error or 'unknown'}"
            )
        return self._video_detector

    @property
    def model_status(self):
        if not self._model_load_attempted:
            return "not_loaded"
        if self._video_detector is not None and self._video_detector.available:
            return "experimental"
        return "unavailable"

    @property
    def model_load_error(self):
        return self._model_load_error

    def analyze_video(self, video_path: str) -> dict:
        media_result = analyze_video_media(video_path, include_frames=True)
        frames = media_result.pop("_frames")
        prediction = self._get_detector().predict(frames)
        risk = calculate_risk(prediction["deepfake_probability"])
        return {
            **media_result,
            **prediction,
            **risk,
            "message": (
                "Analysis completed. Treat the score as experimental; "
                "do not use it as sole evidence of authenticity."
            )
        }

    def analyze_frame(self, frame) -> dict:
        prediction = self._get_detector().predict([frame])
        risk = calculate_risk(prediction["deepfake_probability"])
        faces = detect_faces(frame)
        return {
            **prediction,
            **risk,
            "faces_detected": len(faces),
            "message": "Single-frame experimental estimate; temporal evidence is not analyzed."
        }
