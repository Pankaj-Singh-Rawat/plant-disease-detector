from pathlib import Path
import time
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from model import build_model, get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pt"
EXPORT_PATH = PROJECT_ROOT / "models" / "plant_disease_model.ptl"


def load_trained_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]

    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, val_acc {checkpoint['val_acc']:.4f}")
    return model, class_names


def export_torchscript_mobile(model):
    example_input = torch.rand(1, 3, 224, 224)

    # Trace the model — records the actual ops run for this input shape
    traced_model = torch.jit.trace(model, example_input)

    # Note: optimize_for_mobile() requires XNNPACK, which isn't enabled in the
    # default macOS ARM pip wheel. Skipping it — the traced model still exports
    # and runs correctly, just without the extra operator-fusion pass.
    traced_model._save_for_lite_interpreter(str(EXPORT_PATH))

    return traced_model


def benchmark_latency(model, num_runs: int = 50):
    example_input = torch.rand(1, 3, 224, 224)

    # Warm-up runs (first few calls are always slower — JIT warmup, cache effects)
    for _ in range(5):
        model(example_input)

    start = time.time()
    for _ in range(num_runs):
        model(example_input)
    elapsed = time.time() - start

    avg_latency_ms = (elapsed / num_runs) * 1000
    return avg_latency_ms


def main():
    model, class_names = load_trained_model()

    print("Exporting to TorchScript Mobile (.ptl)...")
    optimized_model = export_torchscript_mobile(model)

    size_mb = EXPORT_PATH.stat().st_size / (1024 * 1024)
    print(f"Exported model size: {size_mb:.2f} MB")

    print("Benchmarking CPU inference latency (single image)...")
    avg_latency_ms = benchmark_latency(optimized_model)
    print(f"Average inference latency: {avg_latency_ms:.2f} ms/image")

    print(f"\nSaved to: {EXPORT_PATH}")


if __name__ == "__main__":
    main()