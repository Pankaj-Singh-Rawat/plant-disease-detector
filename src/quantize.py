from pathlib import Path
import time
import copy
import torch
from torchvision.models.quantization import mobilenet_v2 as quantizable_mobilenet_v2

from dataset import get_dataloaders

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pt"
FLOAT_EXPORT_PATH = PROJECT_ROOT / "models" / "plant_disease_model.ptl"
QUANTIZED_EXPORT_PATH = PROJECT_ROOT / "models" / "plant_disease_model_quantized.ptl"

# ARM CPUs (phones, Apple Silicon) use the qnnpack quantized backend
torch.backends.quantized.engine = "qnnpack"


def load_quantizable_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]

    # quantize=False: load the architecture, we'll quantize manually below
    model = quantizable_mobilenet_v2(weights=None, quantize=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, val_acc {checkpoint['val_acc']:.4f}")
    return model, class_names


def calibrate(model, loader, num_batches: int = 100):
    """Run a handful of batches through the model in observe mode to collect
    activation statistics (min/max ranges) used to compute int8 scale factors."""
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(images)


def quantize_model(model, calib_loader):
    model.fuse_model()  # fuses Conv+BN+ReLU into single ops — required before quantization

    model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    torch.quantization.prepare(model, inplace=True)

    print("Calibrating...")
    calibrate(model, calib_loader)

    torch.quantization.convert(model, inplace=True)
    return model


@torch.no_grad()
def evaluate_accuracy(model, loader):
    correct = 0
    total = 0
    for images, labels in loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def benchmark_latency(model, num_runs: int = 50):
    example_input = torch.rand(1, 3, 224, 224)
    for _ in range(5):
        model(example_input)

    start = time.time()
    for _ in range(num_runs):
        model(example_input)
    elapsed = time.time() - start
    return (elapsed / num_runs) * 1000


def main():
    model, class_names = load_quantizable_model()
    _, val_loader, _ = get_dataloaders(batch_size=32, num_workers=0)

    # --- Float32 baseline (re-measured here, on CPU, for a fair comparison) ---
    print("\nEvaluating float32 model on full validation set...")
    float_model = copy.deepcopy(model)
    float_acc = evaluate_accuracy(float_model, val_loader)
    float_latency = benchmark_latency(float_model)
    float_size_mb = FLOAT_EXPORT_PATH.stat().st_size / (1024 * 1024)
    print(f"Float32 — val_acc: {float_acc:.4f}, latency: {float_latency:.2f} ms, size: {float_size_mb:.2f} MB")

    # --- Quantize ---
    print("\nQuantizing model (static int8)...")
    quantized_model = quantize_model(model, val_loader)

    example_input = torch.rand(1, 3, 224, 224)
    traced_quantized = torch.jit.trace(quantized_model, example_input)
    traced_quantized._save_for_lite_interpreter(str(QUANTIZED_EXPORT_PATH))

    quant_size_mb = QUANTIZED_EXPORT_PATH.stat().st_size / (1024 * 1024)

    print("\nEvaluating quantized model on full validation set...")
    quant_acc = evaluate_accuracy(quantized_model, val_loader)
    quant_latency = benchmark_latency(quantized_model)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("QUANTIZATION COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20}{'Float32':<20}{'Quantized (int8)':<20}")
    print(f"{'Val accuracy':<20}{float_acc:<20.4f}{quant_acc:<20.4f}")
    print(f"{'Latency (ms)':<20}{float_latency:<20.2f}{quant_latency:<20.2f}")
    print(f"{'Size (MB)':<20}{float_size_mb:<20.2f}{quant_size_mb:<20.2f}")
    print(f"\nSize reduction: {float_size_mb / quant_size_mb:.2f}x")
    print(f"Accuracy cost: {(float_acc - quant_acc) * 100:.2f} percentage points")
    print(f"\nSaved quantized model to: {QUANTIZED_EXPORT_PATH}")


if __name__ == "__main__":
    main()