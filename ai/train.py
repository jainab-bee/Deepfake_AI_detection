"""Simple image prediction script using Keras."""

from pathlib import Path

import tensorflow as tf
from PIL import Image

from cnn import build_cnn
from config import MODEL_PATH
from preprocessing import val_preprocess


def predict(image_path: str):
    image = Image.open(image_path).convert("RGB")
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize(image, (224, 224)) / 255.0

    model = build_cnn(num_classes=2)
    model.load_weights(MODEL_PATH)

    result = model.predict(image, verbose=0)[0]
    pred = int(result.argmax())
    confidence = float(result[pred])

    if pred == 0:
        label = "AI Generated"
    else:
        label = "Real"

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}")


if __name__ == "__main__":
    sample = Path("../uploads/sample.jpg")
    if sample.exists():
        predict(str(sample))
    else:
        print("Put an image at uploads/sample.jpg first.")
