"""Training entry point for the deepfake detection model."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from custom_dataset import CustomImageDataset
from preprocessing import build_transform
from cnn import SimpleCNN


def main():
    data_dir = Path("../datasets")
    train_dataset = CustomImageDataset(str(data_dir), transform=build_transform())
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = SimpleCNN(num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "saved_models/simple_cnn.pth")
    print("Training completed and model saved.")


if __name__ == "__main__":
    main()
