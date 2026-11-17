from backend.services.media.frame_sampler import sample_frames
from backend.services.media.face_detector import detect_faces

video_path = "datasets/test_video.mp4"

def main():
    frames = sample_frames(video_path, sample_fps=5)
    total_faces = 0
    for i, frame in enumerate(frames):
        faces = detect_faces(frame)
        if len(faces) > 0:
            print(f"Frame {i}: {len(faces)} face(s) detected")
            total_faces += len(faces)
    print("------------------")
    print("Total detected faces:", total_faces)

if __name__ == "__main__":
    main()
