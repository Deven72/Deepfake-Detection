
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

# Dataset and output paths
ROOT = Path.home() / "Downloads" / "archive" / "UADFV"
OUTPUT = Path(__file__).resolve().parents[1] / "models"

IMAGE_SIZE = (160, 160)
FRAMES_PER_VIDEO = 8
BATCH_SIZE = 16
EPOCHS = 8
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


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


def load_video_frames(video):
    frames = video["frames"]
    count = min(len(frames), FRAMES_PER_VIDEO)

    indices = np.linspace(
        0, len(frames) - 1, count, dtype=int
    )

    images = []
    labels = []

    for idx in indices:
        try:
            with Image.open(frames[idx]) as image:
                image = image.convert("RGB")
                image = image.resize(IMAGE_SIZE)
                images.append(np.asarray(image, dtype=np.float32))
                labels.append(video["label"])
        except Exception as exc:
            print(f"Skipping {frames[idx]}: {exc}")

    return images, labels


def make_arrays(videos):
    x, y = [], []

    for video in videos:
        images, labels = load_video_frames(video)
        x.extend(images)
        y.extend(labels)

    if not x:
        raise RuntimeError("No images loaded. Check dataset path.")

    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"Dataset not found: {ROOT}")

    real = find_videos("real", 0)
    fake = find_videos("fake", 1)

    print(f"Real videos: {len(real)}")
    print(f"Fake videos: {len(fake)}")

    if not real or not fake:
        raise RuntimeError("Real/fake video folders were not found.")

    # Split by video, not by frame.
    real_train, real_val = train_test_split(
        real, test_size=0.2, random_state=SEED
    )
    fake_train, fake_val = train_test_split(
        fake, test_size=0.2, random_state=SEED
    )

    train_videos = real_train + fake_train
    val_videos = real_val + fake_val

    random.shuffle(train_videos)
    random.shuffle(val_videos)

    train_x, train_y = make_arrays(train_videos)
    val_x, val_y = make_arrays(val_videos)

    print("Train videos:", len(train_videos))
    print("Validation videos:", len(val_videos))
    print("Train frames:", len(train_x))
    print("Validation frames:", len(val_x))
    print("Train label counts [real, fake]:",
          np.bincount(train_y.astype(int)))
    print("Validation label counts [real, fake]:",
          np.bincount(val_y.astype(int)))

    # MobileNetV2 expects pixels scaled to [-1, 1].
    train_x = tf.keras.applications.mobilenet_v2.preprocess_input(train_x)
    val_x = tf.keras.applications.mobilenet_v2.preprocess_input(val_x)

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
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT / "uadfv_mobilenetv2_experimental.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    results = model.evaluate(val_x, val_y, verbose=0, return_dict=True)

    report = {
        "dataset": "UADFV",
        "model": "MobileNetV2 transfer learning",
        "train_videos": len(train_videos),
        "validation_videos": len(val_videos),
        "train_frames": len(train_x),
        "validation_frames": len(val_x),
        "validation_metrics": {
            key: float(value) for key, value in results.items()
        },
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "warning": (
            "Small dataset; frame-level metrics are experimental. "
            "Frames from each video stay in one split, but video-level "
            "aggregation and testing on an independent dataset are still needed."
        ),
    }

    with open(OUTPUT / "uadfv_mobilenetv2_report.json", "w") as file:
        json.dump(report, file, indent=2)

    print("\nValidation results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    print(f"\nSaved model: {model_path}")
    print(f"Saved report: {OUTPUT / 'uadfv_mobilenetv2_report.json'}")


if __name__ == "__main__":
    main()
