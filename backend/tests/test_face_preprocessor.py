from backend.services.media.frame_sampler import sample_frames
from backend.services.media.face_detector import detect_faces
from backend.services.media.face_preprocessor import crop_face, preprocess_face

video_path = "datasets/test_video.mp4"

def main():
    frames = sample_frames(video_path, sample_fps=5)
    processed_count = 0
    for i, frame in enumerate(frames):
        faces = detect_faces(frame)
        if not faces:
            continue
        face = crop_face(frame, faces[0])
        processed_face = preprocess_face(face)
        if processed_face is not None:
            print(f"Frame {i}: Original={face.shape}, processed={processed_face.shape}")
            print(f"Pixel range: {processed_face.min():.2f} - {processed_face.max():.2f}")
            processed_count += 1
        if processed_count >= 5:
            break
    print("------------------")
    print("Faces preprocessed:", processed_count)

if __name__ == "__main__":
    main()
