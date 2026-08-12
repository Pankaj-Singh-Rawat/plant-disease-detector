from pathlib import Path
import json
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pt"
LABELS_PATH = PROJECT_ROOT / "app" / "android" / "app" / "src" / "main" / "assets" / "labels.json"

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
class_names = checkpoint["class_names"]

LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(LABELS_PATH, "w") as f:
    json.dump(class_names, f, indent=2)

print(f"Saved {len(class_names)} class labels to {LABELS_PATH}")