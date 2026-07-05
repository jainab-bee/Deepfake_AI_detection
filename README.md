# AI Image Authenticity Detector

This is a simple and beginner-friendly deep learning project that classifies images as either:

- 0 = AI Generated
- 1 = Real

## What this version includes

- A small CNN model
- Simple image preprocessing
- Training and prediction scripts
- A clean project folder structure

## Folder structure

- ai/: model, training, and prediction code
- app/: backend code (can be added later)
- frontend/: UI code (can be added later)
- uploads/: place your test images here
- outputs/: generated results
- reports/: reports and charts

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a dataset folder with two subfolders:
   - dataset/train/ai/
   - dataset/train/real/
3. Put sample images inside each folder.
4. Train the model:
   ```bash
   python ai/train.py
   ```
5. Run prediction:
   ```bash
   python ai/predict.py
   ```

## Notes

This version is intentionally simple so it is easy to learn and modify. It is a good starting point for a real-world deepfake detector project.
