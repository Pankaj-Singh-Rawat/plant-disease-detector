from dataset import get_dataloaders
from model import build_model, get_device
import torch

def main():
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(batch_size=8, num_workers=0)
    print(f"Classes: {len(class_names)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model(num_classes=len(class_names)).to(device)

    # Pull one batch and run it through the model
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    print(f"Input batch shape: {images.shape}")
    print(f"Output shape: {outputs.shape}")  # should be [8, 38]

if __name__ == "__main__":
    main()