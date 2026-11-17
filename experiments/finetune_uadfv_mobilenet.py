
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path.home() / "Downloads" / "archive" / "UADFV"
PROJECT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT / "models" / "uadfv_mobilenetv2_experimental.keras"
OUTPUT_PATH = PROJECT / "models" / "uadfv_mobilenetv2_finetuned.keras"

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
    indices = np.linspace(
        0, len(frames) - 1,
        min(len(frames), FRAMES_PER_VIDEO),
        dtype=int
    )

    images = []

    for idx in indices:
        with Image.open(frames[idx]) as image:
            image = image.convert("RGB").resize(IMAGE_SIZE)
            images.append(np.asarray(image, dtype=np.float32))

    return images


def make_arrays(videos):
    x, y = [], []

    for video in videos:
        images = load_video_frames(video)
        x.extend(images)
        y.extend([video["label"]] * len(images))

    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    real = find_videos("real", 0)
    fake = find_videos("fake", 1)

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

    train_x = tf.keras.applications.mobilenet_v2.preprocess_input(train_x)
    val_x = tf.keras.applications.mobilenet_v2.preprocess_input(val_x)

    model = tf.keras.models.load_model(MODEL_PATH)

    base = model.get_layer("mobilenetv2_1.00_160")
    base.trainable = True

    # Freeze early layers and fine-tune only the last 30 layers.
    for layer in base.layers[:-30]:
        layer.trainable = False

    # Keep Batch Normalization stable for this small dataset.
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    print("Trainable base layers:",
          sum(layer.trainable for layer in base.layers))
    print("Training frames:", len(train_x))
    print("Validation frames:", len(val_x))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_PATH),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nEvaluating fine-tuned model:")
    best_model = tf.keras.models.load_model(OUTPUT_PATH)
    results = best_model.evaluate(
        val_x, val_y, verbose=0, return_dict=True
    )

    for name, value in results.items():
        print(f"{name}: {value:.4f}")

    print("\nFine-tuned model saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
