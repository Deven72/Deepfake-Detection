import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

ROOT = Path.home() / "Downloads" / "archive" / "UADFV"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

images, labels = [], []

for label, class_name in enumerate(["real", "fake"]):
    frame_root = ROOT / class_name / "frames"
    folders = sorted(p for p in frame_root.iterdir() if p.is_dir())
    random.shuffle(folders)

    # Use only 4 videos per class for this diagnostic.
    for folder in folders[:4]:
        files = sorted(folder.glob("*.png"))
        if not files:
            continue

        indices = np.linspace(0, len(files)-1, min(4, len(files)), dtype=int)
        for idx in indices:
            image = cv2.imread(str(files[idx]))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (96, 96))
            images.append(image.astype(np.float32) / 255.0)
            labels.append(label)

x = np.asarray(images)
y = np.asarray(labels, dtype=np.float32)

print("Images:", len(x))
print("Real:", int((y == 0).sum()), "| Fake:", int((y == 1).sum()))

model = tf.keras.models.load_model("models/deepfake_cnn.keras")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(x, y, epochs=12, batch_size=8, shuffle=True, verbose=1)

scores = model.predict(x, verbose=0).ravel()
print("Final training accuracy:", float(((scores >= 0.5) == y).mean()))
print("Real mean score:", float(scores[y == 0].mean()))
print("Fake mean score:", float(scores[y == 1].mean()))
