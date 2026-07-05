"""Prediction entry point for single image inference."""

from pathlib import Path

import torch
from PIL import Image

from preprocessing import build_transform
from cnn import SimpleCNN


def predict(image_path: str):
    image = Image.open(image_path).convert("RGB")
    transform = build_transform()
    tensor = transform(image).unsqueeze(0)

    model = SimpleCNN(num_classes=2)
    model.load_state_dict(torch.load("saved_models/simple_cnn.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()

    return "fake" if pred == 1 else "real"


if __name__ == "__main__":
    sample = Path("../uploads/sample.jpg")
    print(predict(str(sample)))
