import json
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path.home() / "Downloads" / "archive" / "UADFV"
OUTPUT = Path(__file__).resolve().parents[1] / "models" / "deepfake_cnn.keras"
IMAGE_SIZE = (96, 96)
FRAMES_PER_VIDEO = 8
EPOCHS = 6
BATCH_SIZE = 16


def get_video_folders():
    result = {"real": [], "fake": []}

    for label in ("real", "fake"):
        frame_root = ROOT / label / "frames"
        if not frame_root.exists():
            raise FileNotFoundError(f"Missing folder: {frame_root}")

        result[label] = sorted(
            p for p in frame_root.iterdir()
            if p.is_dir() and any(p.glob("*.png"))
        )

    for label in result:
        if len(result[label]) < 2:
            raise RuntimeError(
                f"Expected multiple {label} video-frame folders; "
                f"found {len(result[label])}"
            )

    return result


def split_videos(folders):
    train, val = [], []

    for label, paths in folders.items():
        paths = paths.copy()
        random.shuffle(paths)
        cut = max(1, int(len(paths) * 0.8))

        for path in paths[:cut]:
            train.append((path, 0 if label == "real" else 1))
        for path in paths[cut:]:
            val.append((path, 0 if label == "real" else 1))

    random.shuffle(train)
    random.shuffle(val)
    return train, val


def load_frames(video_items):
    images, labels = [], []
    used_videos = 0

    for folder, label in video_items:
        files = sorted(folder.glob("*.png"))
        if not files:
            continue

        # Sample up to 8 evenly spaced frames from this video.
        indices = np.linspace(
            0, len(files) - 1,
            min(FRAMES_PER_VIDEO, len(files)),
            dtype=int
        )

        loaded_this_video = 0
        for index in indices:
            image = cv2.imread(str(files[index]))
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)
            labels.append(label)
            loaded_this_video += 1

        if loaded_this_video:
            used_videos += 1

    if not images:
        raise RuntimeError("No readable PNG images found.")

    x = np.asarray(images, dtype=np.float32) / 255.0
    y = np.asarray(labels, dtype=np.float32)
    return x, y, used_videos


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(96, 96, 3)),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"Dataset not found: {ROOT}")

    folders = get_video_folders()
    train_items, val_items = split_videos(folders)

    print("Video folders:", {k: len(v) for k, v in folders.items()})
    print("Training videos:", len(train_items))
    print("Validation videos:", len(val_items))

    x_train, y_train, train_used = load_frames(train_items)
    x_val, y_val, val_used = load_frames(val_items)

    print("Training frames:", len(x_train), "| videos with readable frames:", train_used)
    print("Validation frames:", len(x_val), "| videos with readable frames:", val_used)
    print("Training real/fake:", int((y_train == 0).sum()), int((y_train == 1).sum()))
    print("Validation real/fake:", int((y_val == 0).sum()), int((y_val == 1).sum()))

    model = build_model()
    model.summary()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    history_path = OUTPUT.with_name("training_history.json")
    history_path.write_text(
        json.dumps(
            {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nTraining history saved:", history_path)
    print("Training accuracy by epoch:", history.history["accuracy"])
    print("Validation accuracy by epoch:", history.history["val_accuracy"])
    print("Training loss by epoch:", history.history["loss"])
    print("Validation loss by epoch:", history.history["val_loss"])
    loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
    model.save(OUTPUT)

    report = {
        "dataset": "UADFV",
        "model_type": "experimental_frame_cnn",
        "input_size": [96, 96, 3],
        "labels": {"0": "real", "1": "fake"},
        "training_videos": len(train_items),
        "validation_videos": len(val_items),
        "training_frames": len(x_train),
        "validation_frames": len(x_val),
        "validation_loss": float(loss),
        "validation_frame_accuracy": float(accuracy),
        "warning": (
            "Small dataset; experimental frame-level baseline only. "
            "Validation frames are grouped by video, but frame accuracy "
            "is not a reliable production-level video detector metric."
        )
    }

    report_path = OUTPUT.with_name("training_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nMODEL SAVED:", OUTPUT)
    print("REPORT SAVED:", report_path)
    print("Validation frame accuracy:", round(float(accuracy), 4))
    print("Note: this is not production-grade deepfake detection.")


if __name__ == "__main__":
    main()
