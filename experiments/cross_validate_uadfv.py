
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path.home() / "Downloads" / "archive" / "UADFV"
OUTPUT = Path(__file__).resolve().parents[1] / "models"

IMAGE_SIZE = (160, 160)
FRAMES_PER_VIDEO = 8
BATCH_SIZE = 16
EPOCHS = 8
FOLDS = 5
SEED = 42


def find_videos():
    videos = []

    for class_name, label in [("real", 0), ("fake", 1)]:
        base = ROOT / class_name / "frames"

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


def load_video_frames(video):
    frames = video["frames"]
    count = min(len(frames), FRAMES_PER_VIDEO)

    indices = np.linspace(
        0, len(frames) - 1, count, dtype=int
    )

    images = []

    for idx in indices:
        try:
            with Image.open(frames[idx]) as image:
                image = image.convert("RGB")
                image = image.resize(IMAGE_SIZE)
                images.append(
                    np.asarray(image, dtype=np.float32)
                )
        except Exception as exc:
            print(f"Skipping {frames[idx]}: {exc}")

    return images


def make_arrays(videos):
    x, y, video_ids = [], [], []

    for video_index, video in enumerate(videos):
        images = load_video_frames(video)

        for image in images:
            x.append(image)
            y.append(video["label"])
            video_ids.append(video_index)

    if not x:
        raise RuntimeError("No images loaded.")

    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(video_ids),
    )


def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights="imagenet",
    )

    base.trainable = False

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    if not ROOT.exists():
        raise FileNotFoundError(ROOT)

    videos = find_videos()

    labels = np.array(
        [video["label"] for video in videos]
    )

    print("Total videos:", len(videos))
    print("REAL:", int(np.sum(labels == 0)))
    print("FAKE:", int(np.sum(labels == 1)))

    if len(videos) != 98:
        print("WARNING: Expected 98 videos. Check dataset.")

    splitter = StratifiedKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels),
        start=1,
    ):
        print(f"\n========== FOLD {fold}/{FOLDS} ==========")

        # Reset randomness and clear previous model.
        tf.keras.backend.clear_session()
        random.seed(SEED + fold)
        np.random.seed(SEED + fold)
        tf.random.set_seed(SEED + fold)

        train_videos = [videos[i] for i in train_idx]
        val_videos = [videos[i] for i in val_idx]

        train_x, train_y, _ = make_arrays(train_videos)
        val_x, val_y, val_video_ids = make_arrays(val_videos)

        train_x = (
            tf.keras.applications.mobilenet_v2.preprocess_input(
                train_x
            )
        )
        val_x = (
            tf.keras.applications.mobilenet_v2.preprocess_input(
                val_x
            )
        )

        print("Train videos:", len(train_videos))
        print("Validation videos:", len(val_videos))
        print("Train frames:", len(train_x))
        print("Validation frames:", len(val_x))

        model = build_model()

        model.fit(
            train_x,
            train_y,
            validation_data=(val_x, val_y),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True,
                )
            ],
            verbose=1,
        )

        # Aggregate frame predictions into one score per video.
        frame_probs = model.predict(
            val_x, batch_size=BATCH_SIZE, verbose=0
        ).ravel()

        video_probs = []
        video_labels = []

        for video_id in np.unique(val_video_ids):
            mask = val_video_ids == video_id

            video_probs.append(
                float(np.mean(frame_probs[mask]))
            )
            video_labels.append(
                int(val_y[mask][0])
            )

        video_probs = np.asarray(video_probs)
        video_labels = np.asarray(video_labels)
        predictions = (video_probs >= 0.5).astype(int)

        metrics = {
            "fold": fold,
            "videos": len(video_labels),
            "accuracy": float(
                accuracy_score(video_labels, predictions)
            ),
            "precision": float(
                precision_score(
                    video_labels, predictions, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    video_labels, predictions, zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    video_labels, predictions, zero_division=0
                )
            ),
            "auc": float(
                roc_auc_score(video_labels, video_probs)
            ),
            "confusion_matrix": confusion_matrix(
                video_labels, predictions
            ).tolist(),
        }

        fold_results.append(metrics)

        print("\nFold results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        del model, train_x, train_y, val_x, val_y

    metric_names = [
        "accuracy", "precision", "recall", "f1", "auc"
    ]

    summary = {}

    for metric in metric_names:
        values = [result[metric] for result in fold_results]

        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    report = {
        "dataset": "UADFV",
        "method": "5-Fold Stratified Video-Level Cross-Validation",
        "model": "MobileNetV2 ImageNet frozen feature extractor",
        "threshold": 0.5,
        "total_videos": len(videos),
        "folds": fold_results,
        "summary": summary,
        "warning": (
            "Small dataset. Cross-validation estimates performance "
            "on held-out folds, but does not establish external "
            "real-world generalization."
        ),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    report_path = OUTPUT / "uadfv_5fold_cv_report.json"

    with open(report_path, "w") as file:
        json.dump(report, file, indent=2)

    print("\n========== CROSS-VALIDATION SUMMARY ==========")

    for metric, values in summary.items():
        print(
            f"{metric}: "
            f"{values['mean']:.4f} "
            f"+/- {values['std']:.4f}"
        )

    print("\nReport saved:", report_path)


if __name__ == "__main__":
    main()
