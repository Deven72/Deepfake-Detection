import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2
import random

ROOT = Path.home() / "Downloads" / "archive" / "UADFV"
random.seed(42)

folders = {"real": [], "fake": []}
for label in folders:
    folders[label] = sorted(
        p for p in (ROOT / label / "frames").iterdir()
        if p.is_dir() and any(p.glob("*.png"))
    )

val_items = []
for label, paths in folders.items():
    paths = paths.copy()
    random.shuffle(paths)
    cut = max(1, int(len(paths) * 0.8))
    val_items.extend((p, 0 if label == "real" else 1) for p in paths[cut:])

random.shuffle(val_items)

images, labels = [], []
for folder, label in val_items:
    files = sorted(folder.glob("*.png"))
    indices = np.linspace(0, len(files)-1, min(8, len(files)), dtype=int)
    for idx in indices:
        im = cv2.imread(str(files[idx]))
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (96, 96))
        images.append(im.astype(np.float32) / 255.0)
        labels.append(label)

x = np.asarray(images)
y = np.asarray(labels)
model = tf.keras.models.load_model("models/deepfake_cnn.keras")
p = model.predict(x, verbose=0).ravel()

print("Validation frames:", len(y))
print("Real frames:", int((y == 0).sum()))
print("Fake frames:", int((y == 1).sum()))
print("Prediction min/max:", float(p.min()), float(p.max()))
print("Prediction mean:", float(p.mean()))
print("Predicted real:", int((p < 0.5).sum()))
print("Predicted fake:", int((p >= 0.5).sum()))
print("Mean score for real:", float(p[y == 0].mean()))
print("Mean score for fake:", float(p[y == 1].mean()))
