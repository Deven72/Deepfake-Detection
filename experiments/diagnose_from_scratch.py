exec(
    open("experiments/diagnose_learning.py", encoding="utf-8").read()
    .replace(
        'model = tf.keras.models.load_model("models/deepfake_cnn.keras")',
        '''model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(96, 96, 3)),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])'''
    )
)
print("Real scores:", np.round(scores[y == 0], 3))
print("Fake scores:", np.round(scores[y == 1], 3))