
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

ROOT = Path.home() / "Downloads" / "archive" / "UADFV"
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "uadfv_mobilenetv2_finetuned.keras"
)

IMAGE_SIZE = (160, 160)
FRAMES_PER_VIDEO = 8
SEED = 42
THRESHOLD = 0.5

random.seed(SEED)
np.random.seed(SEED)


def find_videos(class_name, label):
    base = ROOT / class_name / "frames"
    videos = []

    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue

        frames = sorted(folder.glob("*.png"))

        if frames:
            videos.append({
                "folder": folder,
                "label": label,
                "frames": frames,
            })

    return videos


def load_frames(video):
    frames = video["frames"]
    count = min(len(frames), FRAMES_PER_VIDEO)

    indices = np.linspace(
        0, len(frames) - 1, count, dtype=int
    )

    images = []

    for idx in indices:
        with Image.open(frames[idx]) as image:
            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE)
            images.append(
                np.asarray(image, dtype=np.float32)
            )

    return np.asarray(images, dtype=np.float32)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    model = tf.keras.models.load_model(MODEL_PATH)

    real = find_videos("real", 0)
    fake = find_videos("fake", 1)

    real_train, real_val = train_test_split(
        real, test_size=0.2, random_state=SEED
    )

    fake_train, fake_val = train_test_split(
        fake, test_size=0.2, random_state=SEED
    )

    # Exactly the same video-wise validation split as training.
    validation_videos = real_val + fake_val

    results = []

    for video in validation_videos:
        images = load_frames(video)

        images = (
            tf.keras.applications.mobilenet_v2
            .preprocess_input(images)
        )

        frame_scores = model.predict(
            images, verbose=0
        ).reshape(-1)

        video_score = float(np.mean(frame_scores))
        prediction = int(video_score >= THRESHOLD)

        results.append({
            "video": video["folder"].name,
            "actual": video["label"],
            "predicted": prediction,
            "fake_probability": video_score,
        })

        print(
            f"{video['folder'].name:20s} "
            f"Actual={video['label']} "
            f"Predicted={prediction} "
            f"FakeScore={video_score:.4f}"
        )

    y_true = [item["actual"] for item in results]
    y_pred = [item["predicted"] for item in results]
    y_score = [
        item["fake_probability"] for item in results
    ]

    print("\n========== VIDEO LEVEL RESULTS ==========")
    print("Videos evaluated:", len(results))
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print(
        "Precision:",
        round(precision_score(y_true, y_pred, zero_division=0), 4)
    )
    print(
        "Recall:",
        round(recall_score(y_true, y_pred, zero_division=0), 4)
    )
    print(
        "F1 Score:",
        round(f1_score(y_true, y_pred, zero_division=0), 4)
    )
    print(
        "ROC AUC:",
        round(roc_auc_score(y_true, y_score), 4)
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["REAL", "FAKE"],
            zero_division=0,
        )
    )

    output_path = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "uadfv_video_evaluation.json"
    )

    with open(output_path, "w") as file:
        json.dump({
            "threshold": THRESHOLD,
            "videos_evaluated": len(results),
            "metrics": {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(
                    precision_score(y_true, y_pred, zero_division=0)
                ),
                "recall": float(
                    recall_score(y_true, y_pred, zero_division=0)
                ),
                "f1": float(
                    f1_score(y_true, y_pred, zero_division=0)
                ),
                "roc_auc": float(roc_auc_score(y_true, y_score)),
            },
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            ).tolist(),
            "video_predictions": results,
        }, file, indent=2)

    print("\nReport saved:", output_path)


if __name__ == "__main__":
    main()
