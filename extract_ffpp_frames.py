
import cv2
import numpy as np
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "datasets" / "FaceForensics++"

INPUT_CSV = DATASET_ROOT / "split_manifest.csv"

OUTPUT_ROOT = DATASET_ROOT / "extracted_frames"

FRAMES_PER_VIDEO = 8
IMAGE_SIZE = (160, 160)
JPEG_QUALITY = 95

FRAME_MANIFEST = OUTPUT_ROOT / "frame_manifest.csv"
FAILURE_REPORT = OUTPUT_ROOT / "extraction_failures.csv"


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def safe_name(value):
    """Convert a value into a safe folder name."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))


def extract_video_frames(row):

    relative_path = str(row["path"]).replace("\\", "/")
    video_path = DATASET_ROOT / relative_path

    label = str(row["label"]).upper().strip()
    split = str(row["split"]).lower().strip()

    pair_id = safe_name(row["pair_id"])
    video_id = safe_name(Path(relative_path).stem)

    video_output = (
        OUTPUT_ROOT / split / label /
        f"{pair_id}_{video_id}"
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open video")

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    original_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    original_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if frame_count < FRAMES_PER_VIDEO:
        cap.release()
        raise RuntimeError(
            f"Video has only {frame_count} frames"
        )

    # Select 8 evenly spaced frame positions
    frame_indices = np.linspace(
        0,
        frame_count - 1,
        FRAMES_PER_VIDEO,
        dtype=int
    )

    video_output.mkdir(
        parents=True,
        exist_ok=True
    )

    extracted_records = []

    for frame_number, frame_index in enumerate(
        frame_indices, start=1
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = cap.read()

        if not success or frame is None:
            cap.release()
            raise RuntimeError(
                f"Could not read frame {frame_index}"
            )

        # Resize frame
        resized = cv2.resize(
            frame,
            IMAGE_SIZE,
            interpolation=cv2.INTER_AREA
        )

        image_name = (
            f"frame_{frame_number:02d}_"
            f"idx_{frame_index:06d}.jpg"
        )

        image_path = video_output / image_name

        saved = cv2.imwrite(
            str(image_path),
            resized,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        if not saved:
            cap.release()
            raise RuntimeError(
                f"Could not save {image_name}"
            )

        extracted_records.append({
            "video_path": relative_path,
            "pair_id": str(row["pair_id"]),
            "split": split,
            "label": label,
            "frame_number": frame_number,
            "original_frame_index": int(frame_index),
            "frame_path": str(
                image_path.relative_to(DATASET_ROOT)
            ).replace("\\", "/"),
            "original_width": original_width,
            "original_height": original_height,
            "fps": fps,
            "total_video_frames": frame_count
        })

    cap.release()

    if len(extracted_records) != FRAMES_PER_VIDEO:
        raise RuntimeError(
            "Incorrect number of extracted frames"
        )

    return extracted_records


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():

    print("\n" + "=" * 55)
    print("DEEPSHIELD - FACEFORENSICS++ FRAME EXTRACTION")
    print("=" * 55)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Manifest not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = [
        "path", "label", "split", "pair_id"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing CSV columns: {missing}"
        )

    df = df.drop_duplicates(
        subset=["path"]
    ).reset_index(drop=True)

    valid_splits = [
        "train", "validation", "test"
    ]

    valid_labels = [
        "REAL", "FAKE"
    ]

    df["split"] = (
        df["split"].astype(str).str.lower().str.strip()
    )

    df["label"] = (
        df["label"].astype(str).str.upper().str.strip()
    )

    if not df["split"].isin(valid_splits).all():
        raise ValueError("Invalid split value found")

    if not df["label"].isin(valid_labels).all():
        raise ValueError("Invalid label value found")

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    all_records = []
    failures = []

    print(f"\nVideos to process: {len(df)}")
    print(f"Frames per video: {FRAMES_PER_VIDEO}")
    print(f"Image resolution: {IMAGE_SIZE}")
    print(
        f"Expected frames: "
        f"{len(df) * FRAMES_PER_VIDEO}"
    )

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Extracting frames"
    ):

        try:
            records = extract_video_frames(row)
            all_records.extend(records)

        except Exception as error:

            failures.append({
                "video_path": str(row["path"]),
                "label": str(row["label"]),
                "split": str(row["split"]),
                "error": str(error)
            })

    # Save frame manifest
    frame_df = pd.DataFrame(all_records)

    frame_df.to_csv(
        FRAME_MANIFEST,
        index=False
    )

    # Save failures
    failure_df = pd.DataFrame(
        failures,
        columns=[
            "video_path",
            "label",
            "split",
            "error"
        ]
    )

    failure_df.to_csv(
        FAILURE_REPORT,
        index=False
    )

    # ======================================
    # FINAL SUMMARY
    # ======================================

    print("\n" + "=" * 55)
    print("FRAME EXTRACTION SUMMARY")
    print("=" * 55)

    print(f"Total videos: {len(df)}")
    print(
        f"Successfully processed: "
        f"{len(df) - len(failures)}"
    )
    print(f"Failed videos: {len(failures)}")
    print(f"Total extracted frames: {len(all_records)}")

    print("\nFRAMES BY SPLIT AND CLASS:")

    if not frame_df.empty:
        print(
            frame_df.groupby(
                ["split", "label"]
            ).size().to_string()
        )

    print("\nReports saved:")
    print(f"Frame manifest: {FRAME_MANIFEST}")
    print(f"Failure report: {FAILURE_REPORT}")

    if failures:
        print(
            "\nWARNING: Some videos failed. "
            "Check extraction_failures.csv"
        )
    else:
        print(
            "\nSUCCESS: All videos processed!"
        )


if __name__ == "__main__":
    main()
