from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = BASE_DIR / "ai" / "saved_models"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOADS_DIR = BASE_DIR / "uploads"

for folder in [DATASET_DIR, TRAIN_DIR, TEST_DIR, MODEL_DIR, OUTPUT_DIR, UPLOADS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 0.001
NUM_CLASSES = 2
RANDOM_SEED = 42
MODEL_PATH = MODEL_DIR / "simple_cnn.keras"