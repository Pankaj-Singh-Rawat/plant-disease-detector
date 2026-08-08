import os
from pathlib import Path

DATA_DIR = Path("data/plantvillage/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)")

def count_images_per_class(split_dir: Path):
    counts = {}
    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir():
            n = len(list(class_dir.glob("*.*")))
            counts[class_dir.name] = n
    return counts

if __name__ == "__main__":
    train_dir = DATA_DIR / "train"
    counts = count_images_per_class(train_dir)

    total = sum(counts.values())
    print(f"Total classes: {len(counts)}")
    print(f"Total images: {total}\n")

    for cls, n in sorted(counts.items(), key=lambda x: x[1]):
        print(f"{cls:50s} {n:6d}")

    imbalance_ratio = max(counts.values()) / min(counts.values())
    print(f"\nImbalance ratio (max/min class size): {imbalance_ratio:.2f}")