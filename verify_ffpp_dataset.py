
import csv
from pathlib import Path
import cv2

ROOT = Path("datasets/FaceForensics++")

FOLDERS = {
    "REAL": ROOT / "original_sequences/youtube/c23/videos",
    "FAKE": ROOT / "manipulated_sequences/Deepfakes/c23/videos"
}

results = []
total = 0
passed = 0
failed = 0

for label, folder in FOLDERS.items():
    videos = sorted(folder.glob("*.mp4"))

    print(f"\nChecking {label}: {len(videos)} videos")

    for video in videos:
        total += 1
        cap = cv2.VideoCapture(str(video))

        opened = cap.isOpened()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ok, frame = cap.read()
        decoded = bool(ok and frame is not None)

        cap.release()

        valid = (
            opened and decoded
            and fps > 0
            and frame_count > 0
            and width > 0 and height > 0
        )

        if valid:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        results.append({
            "label": label,
            "filename": video.name,
            "fps": fps,
            "frames": frame_count,
            "width": width,
            "height": height,
            "status": status
        })

        print(f"{status}: {video.name}")

output_csv = ROOT / "integrity_report.csv"

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n========== INTEGRITY SUMMARY ==========")
print("Total videos:", total)
print("Passed:", passed)
print("Failed:", failed)
print("Report:", output_csv)
