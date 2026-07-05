"""Custom dataset loader for image classification."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for image_path in class_dir.glob("*.*"):
                self.samples.append((image_path, class_dir.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_name = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = 1 if label_name.lower() == "fake" else 0
        return image, label
