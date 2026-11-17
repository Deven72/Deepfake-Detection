from backend.services.media.frame_sampler import sample_frames

video_path = "datasets/test_video.mp4"

def main():
    frames = sample_frames(video_path, sample_fps=5)
    print("Frames sampled:", len(frames))
    if frames:
        print("First frame shape:", frames[0].shape)

if __name__ == "__main__":
    main()
