
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

ROOT = Path.home() / "Downloads" / "archive" / "UADFV"
OUTPUT = Path(__file__).resolve().parents[1] / "models"

IMAGE_SIZE = (160, 160)
FRAMES_PER_VIDEO = 8
BATCH_SIZE = 8
EPOCHS_HEAD = 5
EPOCHS_FINE = 8
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
                    "frames": frames
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
        np.asarray(video_ids)
    )


def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights="imagenet"
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
    return model, base


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


def main():
    if not ROOT.exists():
        raise FileNotFoundError(ROOT)

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    videos = find_videos()
    labels = np.array([v["label"] for v in videos])

    print("Total videos:", len(videos))

    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    # Exactly the same Fold 1 split as cross_validate_uadfv.py
    train_idx, val_idx = next(
        splitter.split(np.zeros(len(labels)), labels)
    )

    train_videos = [videos[i] for i in train_idx]
    val_videos = [videos[i] for i in val_idx]

    print("Fold 1 train videos:", len(train_videos))
    print("Fold 1 validation videos:", len(val_videos))

    train_x, train_y, _ = make_arrays(train_videos)
    val_x, val_y, val_video_ids = make_arrays(val_videos)

    train_x = tf.keras.applications.mobilenet_v2.preprocess_input(
        train_x
    )
    val_x = tf.keras.applications.mobilenet_v2.preprocess_input(
        val_x
    )

    model, base = build_model()

    # Stage 1: Train classification head.
    compile_model(model, 1e-3)

    print("\n===== STAGE 1: FROZEN BACKBONE =====")

    model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=EPOCHS_HEAD,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    # Stage 2: Fine-tune last 30 backbone layers.
    print("\n===== STAGE 2: FINE-TUNING =====")

    base.trainable = True

    for layer in base.layers[:-30]:
        layer.trainable = False

    for layer in base.layers[-30:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, 1e-5)

    model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=EPOCHS_FINE,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    # Video-level evaluation.
    frame_probs = model.predict(
        val_x,
        batch_size=BATCH_SIZE,
        verbose=0
    ).ravel()

    video_probs = []
    video_labels = []

    for video_id in np.unique(val_video_ids):
        mask = val_video_ids == video_id
        video_probs.append(float(np.mean(frame_probs[mask])))
        video_labels.append(int(val_y[mask][0]))

    video_probs = np.asarray(video_probs)
    video_labels = np.asarray(video_labels)
    predictions = (video_probs >= 0.5).astype(int)

    results = {
        "fold": 1,
        "videos": len(video_labels),
        "accuracy": float(
            accuracy_score(video_labels, predictions)
        ),
        "precision": float(
            precision_score(video_labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(video_labels, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(video_labels, predictions, zero_division=0)
        ),
        "auc": float(
            roc_auc_score(video_labels, video_probs)
        ),
        "confusion_matrix": confusion_matrix(
            video_labels, predictions
        ).tolist()
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT / "uadfv_fold1_finetuned.keras"
    report_path = OUTPUT / "uadfv_fold1_finetuned_report.json"

    model.save(model_path)

    with open(report_path, "w") as file:
        json.dump(results, file, indent=2)

    print("\n===== FOLD 1 FINE-TUNED RESULTS =====")

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nModel saved:", model_path)
    print("Report saved:", report_path)


if __name__ == "__main__":
    main()
