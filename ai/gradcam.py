"""Grad-CAM visualization helpers for CNN explanations."""

import torch


def generate_gradcam(model, image_tensor):
    model.eval()
    image_tensor.requires_grad_(True)
    output = model(image_tensor)
    output[:, output.argmax(dim=1)].backward()
    return image_tensor.grad
