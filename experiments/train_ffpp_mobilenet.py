
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dropout,
    Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
STAGE1_EPOCHS = 12
STAGE2_EPOCHS = 8
SEED = 42

LABEL_MAP = {
    "REAL": 0,
    "FAKE": 1
}


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset-root",
    type=str,
    default="datasets/FaceForensics++"
)

parser.add_argument(
    "--output-dir",
    type=str,
    default="models/ffpp_experiment"
)

args = parser.parse_args()

ROOT = Path(args.dataset_root).resolve()
OUTPUT = Path(args.output_dir).resolve()

OUTPUT.mkdir(parents=True, exist_ok=True)

CSV_PATH = ROOT / "extracted_frames" / "frame_manifest.csv"

tf.keras.utils.set_random_seed(SEED)


# --------------------------------------------------
# GPU INFORMATION
# --------------------------------------------------

print("\nTENSORFLOW:", tf.__version__)
print("AVAILABLE GPUS:", tf.config.list_physical_devices("GPU"))

# --------------------------------------------------
# LOAD MANIFEST
# --------------------------------------------------

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Manifest not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

required_columns = [
    "video_path",
    "split",
    "label",
    "frame_path"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

df["label_id"] = df["label"].map(LABEL_MAP)

if df["label_id"].isna().any():
    raise ValueError("Unknown labels found in manifest")

df["full_path"] = df["frame_path"].apply(
    lambda p: str(ROOT / p)
)

# Verify image paths
missing = [
    p for p in df["full_path"]
    if not os.path.isfile(p)
]

if missing:
    raise FileNotFoundError(
        f"{len(missing)} image files are missing. "
        f"First missing file: {missing[0]}"
    )

# --------------------------------------------------
# SPLIT DATA
# --------------------------------------------------

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("\nDATASET SUMMARY")

for name, part in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:
    print(f"\n{name}")
    print("Frames:", len(part))
    print("Videos:", part["video_path"].nunique())
    print(part["label"].value_counts().to_dict())

# --------------------------------------------------
# VIDEO SPLIT LEAKAGE CHECK
# --------------------------------------------------

train_videos = set(train_df["video_path"])
val_videos = set(val_df["video_path"])
test_videos = set(test_df["video_path"])

assert train_videos.isdisjoint(val_videos)
assert train_videos.isdisjoint(test_videos)
assert val_videos.isdisjoint(test_videos)

print("\nVIDEO SPLIT CHECK PASSED")

# --------------------------------------------------
# TF.DATA PIPELINE
# --------------------------------------------------

def load_image(path, label):

    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_jpeg(
        image_bytes,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = preprocess_input(image)

    return image, tf.cast(label, tf.float32)


def make_dataset(dataframe, training=False):

    paths = dataframe["full_path"].values
    labels = dataframe["label_id"].values.astype("float32")

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# --------------------------------------------------
# BUILD MOBILENETV2
# --------------------------------------------------

def build_model():

    inputs = Input(
        shape=(160, 160, 3)
    )

    backbone = MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights="imagenet"
    )

    backbone.trainable = False

    x = backbone(inputs, training=False)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.3)(x)

    outputs = Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(
        inputs,
        outputs
    )

    return model


model = build_model()

model.summary()

# --------------------------------------------------
# STAGE 1: TRAIN CLASSIFICATION HEAD
# --------------------------------------------------

print("\nSTAGE 1: FEATURE EXTRACTION TRAINING")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

stage1_path = OUTPUT / "ffpp_stage1_best.keras"

callbacks1 = [
    ModelCheckpoint(
        str(stage1_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE1_EPOCHS,
    callbacks=callbacks1
)

# Load best Stage 1 model
model = tf.keras.models.load_model(
    str(stage1_path)
)

# --------------------------------------------------
# STAGE 2: FINE-TUNING
# --------------------------------------------------

print("\nSTAGE 2: MOBILENETV2 FINE-TUNING")

backbone = model.get_layer(
    "mobilenetv2_1.00_160"
)

backbone.trainable = True

# Freeze all except last 30 layers
for layer in backbone.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization frozen
for layer in backbone.layers:
    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False

print(
    "Trainable backbone layers:",
    sum(layer.trainable for layer in backbone.layers)
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

stage2_path = OUTPUT / "ffpp_finetuned_best.keras"

callbacks2 = [
    ModelCheckpoint(
        str(stage2_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE2_EPOCHS,
    callbacks=callbacks2
)

# Load best fine-tuned checkpoint
model = tf.keras.models.load_model(
    str(stage2_path)
)

final_model_path = OUTPUT / "ffpp_mobilenetv2_final.keras"

model.save(str(final_model_path))

# --------------------------------------------------
# SAVE TRAINING HISTORY
# --------------------------------------------------

history_data = {
    "stage1": history1.history,
    "stage2": history2.history
}

with open(
    OUTPUT / "training_history.json",
    "w"
) as f:
    json.dump(history_data, f, indent=4)

# --------------------------------------------------
# FINAL TEST EVALUATION
# Test set is used only now
# --------------------------------------------------

print("\nFINAL TEST EVALUATION")

test_probabilities = model.predict(
    test_ds,
    verbose=1
).reshape(-1)

test_true = test_df["label_id"].to_numpy()

test_predictions = (
    test_probabilities >= 0.5
).astype(int)

# Frame-level metrics
frame_results = {
    "accuracy": float(
        accuracy_score(test_true, test_predictions)
    ),

    "precision": float(
        precision_score(
            test_true,
            test_predictions,
            zero_division=0
        )
    ),

    "recall": float(
        recall_score(
            test_true,
            test_predictions,
            zero_division=0
        )
    ),

    "f1_score": float(
        f1_score(
            test_true,
            test_predictions,
            zero_division=0
        )
    ),

    "auc": float(
        roc_auc_score(
            test_true,
            test_probabilities
        )
    ),

    "confusion_matrix": confusion_matrix(
        test_true,
        test_predictions
    ).tolist(),

    "classification_report": classification_report(
        test_true,
        test_predictions,
        target_names=["REAL", "FAKE"],
        output_dict=True,
        zero_division=0
    )
}

# --------------------------------------------------
# VIDEO-LEVEL EVALUATION
# Mean probability of 8 frames per video
# --------------------------------------------------

video_df = test_df[
    ["video_path", "label_id"]
].copy()

video_df["probability"] = test_probabilities

video_results = video_df.groupby(
    "video_path"
).agg(
    actual_label=("label_id", "first"),
    predicted_probability=("probability", "mean"),
    frame_count=("probability", "count")
).reset_index()

video_true = video_results["actual_label"].to_numpy()

video_prob = video_results[
    "predicted_probability"
].to_numpy()

video_predictions = (
    video_prob >= 0.5
).astype(int)

video_metrics = {
    "videos_evaluated": int(len(video_results)),

    "accuracy": float(
        accuracy_score(video_true, video_predictions)
    ),

    "precision": float(
        precision_score(
            video_true,
            video_predictions,
            zero_division=0
        )
    ),

    "recall": float(
        recall_score(
            video_true,
            video_predictions,
            zero_division=0
        )
    ),

    "f1_score": float(
        f1_score(
            video_true,
            video_predictions,
            zero_division=0
        )
    ),

    "auc": float(
        roc_auc_score(video_true, video_prob)
    ),

    "confusion_matrix": confusion_matrix(
        video_true,
        video_predictions
    ).tolist(),

    "classification_report": classification_report(
        video_true,
        video_predictions,
        target_names=["REAL", "FAKE"],
        output_dict=True,
        zero_division=0
    )
}

# --------------------------------------------------
# SAVE FINAL REPORT
# --------------------------------------------------

report = {
    "dataset": "FaceForensics++",
    "architecture": "MobileNetV2",
    "image_size": IMAGE_SIZE,
    "frames_per_video": 8,
    "train_frames": int(len(train_df)),
    "validation_frames": int(len(val_df)),
    "test_frames": int(len(test_df)),
    "train_videos": int(len(train_videos)),
    "validation_videos": int(len(val_videos)),
    "test_videos": int(len(test_videos)),
    "frame_level": frame_results,
    "video_level": video_metrics
}

with open(
    OUTPUT / "ffpp_test_report.json",
    "w"
) as f:
    json.dump(report, f, indent=4)

video_results.to_csv(
    OUTPUT / "video_level_predictions.csv",
    index=False
)

# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\n" + "=" * 55)
print("FRAME-LEVEL TEST RESULTS")
print("=" * 55)

for key, value in frame_results.items():
    if key not in [
        "confusion_matrix",
        "classification_report"
    ]:
        print(f"{key}: {value}")

print("\nCONFUSION MATRIX:")
print(frame_results["confusion_matrix"])

print("\n" + "=" * 55)
print("VIDEO-LEVEL TEST RESULTS")
print("=" * 55)

for key, value in video_metrics.items():
    if key not in [
        "confusion_matrix",
        "classification_report"
    ]:
        print(f"{key}: {value}")

print("\nVIDEO CONFUSION MATRIX:")
print(video_metrics["confusion_matrix"])

print("\nFINAL MODEL:", final_model_path)
print("REPORT:", OUTPUT / "ffpp_test_report.json")

print("\nTRAINING AND EVALUATION COMPLETED!")
