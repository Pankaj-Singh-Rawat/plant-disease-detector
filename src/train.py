from pathlib import Path
import torch
import torch.nn as nn
import wandb

from dataset import get_dataloaders
from model import build_model, get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# --- Config ---
CONFIG = {
    "batch_size": 32,
    "num_workers": 4,
    "epochs": 10,
    "lr": 1e-4,
    "weight_decay": 1e-4,
}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main():
    wandb.init(project="plant-disease-detector", config=CONFIG)
    config = wandb.config

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(
        batch_size=config.batch_size, num_workers=config.num_workers
    )
    print(f"Classes: {len(class_names)}")

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val_acc = 0.0

    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc,
                "epoch": epoch + 1,
            }, checkpoint_path)
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

    wandb.finish()
    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()