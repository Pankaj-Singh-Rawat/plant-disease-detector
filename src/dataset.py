from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Anchor to the project root regardless of where this script is run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "plantvillage" / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)"

IMG_SIZE = 224  # standard input size for MobileNetV2

# ImageNet normalization stats — required since we're using ImageNet-pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def get_dataloaders(batch_size: int = 32, num_workers: int = 4):
    train_dataset = datasets.ImageFolder(DATA_ROOT / "train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(DATA_ROOT / "valid", transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes
    return train_loader, val_loader, class_names