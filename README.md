# AI Deepfake Detection

This project provides a deep learning pipeline for detecting deepfake images using convolutional neural networks and EfficientNet-based models.

## Project Structure

- `ai/`: training, preprocessing, model, and prediction scripts
- `app/`: backend application code
- `frontend/`: frontend assets and UI code
- `outputs/`: generated results and model outputs
- `uploads/`: uploaded images or videos for inference
- `reports/`: evaluation reports and visualizations

## Setup

1. Create a virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run training:
   ```bash
   python ai/train.py
   ```
4. Run inference:
   ```bash
   python ai/predict.py
   ```

## Notes

The current structure is meant as a starter scaffold for a deepfake detection system and can be expanded with a web app, dataset handling, and deployment support.
