"""Video metadata, sampling, and face-localization diagnostics."""
from backend.services.media.video_processor import get_video_info
from backend.services.media.frame_sampler import sample_frames
from backend.services.media.face_detector import detect_faces
from backend.services.media.face_preprocessor import crop_face, preprocess_face


def analyze_video_media(video_path: str, sample_fps: float = 1.0,
                        max_frames: int = 64, include_frames: bool = False) -> dict:
    video_info = get_video_info(video_path)
    frames = sample_frames(video_path, sample_fps=sample_fps, max_frames=max_frames)

    faces_detected = 0
    faces_preprocessed = 0
    frames_with_face = 0

    for frame in frames:
        faces = detect_faces(frame)
        if faces:
            frames_with_face += 1
        faces_detected += len(faces)

        for face_box in faces:
            face = crop_face(frame, face_box)
            processed_face = preprocess_face(face)
            if processed_face is not None:
                faces_preprocessed += 1

    result = {
        "video_info": video_info,
        "frames_sampled": len(frames),
        "frames_with_face": frames_with_face,
        "faces_detected": faces_detected,
        "faces_preprocessed": faces_preprocessed,
        "face_detector": "OpenCV Haar Cascade (diagnostic only)"
    }
    if include_frames:
        result["_frames"] = frames
    return result
