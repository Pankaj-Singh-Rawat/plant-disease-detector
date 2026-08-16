"""
CI smoke test for the export pipeline. Uses an UNTRAINED model (random weights) —
this validates that export/quantization code runs correctly and produces a
loadable, correctly-shaped model. It does NOT validate real-world accuracy,
since that requires the trained checkpoint and full dataset, which aren't
tracked in this repo.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from model import build_model

NUM_CLASSES = 38
MAX_ALLOWED_SIZE_MB = 15  # generous ceiling — flags accidental bloat, not tuned tightly


def check_traceable_and_exportable():
    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()

    example_input = torch.rand(1, 3, 224, 224)

    # 1. Model runs and produces the right output shape
    output = model(example_input)
    assert output.shape == (1, NUM_CLASSES), f"Expected shape (1, {NUM_CLASSES}), got {output.shape}"
    print(f"✓ Forward pass output shape correct: {output.shape}")

    # 2. Model is traceable (this is what export.py relies on)
    traced = torch.jit.trace(model, example_input)
    print("✓ Model successfully traced with torch.jit.trace")

    # 3. Traced model can be saved and reloaded for mobile
    export_path = Path("ci_test_export.ptl")
    traced._save_for_lite_interpreter(str(export_path))

    size_mb = export_path.stat().st_size / (1024 * 1024)
    print(f"✓ Exported .ptl file size: {size_mb:.2f} MB")
    assert size_mb < MAX_ALLOWED_SIZE_MB, f"Exported model unexpectedly large: {size_mb:.2f}MB"

    # 4. Reloaded model produces correctly-shaped output too
    reloaded = torch.jit.load(str(export_path))
    reloaded_output = reloaded(example_input)
    assert reloaded_output.shape == (1, NUM_CLASSES), "Reloaded model output shape mismatch"
    print("✓ Reloaded exported model produces correct output shape")

    export_path.unlink()  # cleanup


if __name__ == "__main__":
    check_traceable_and_exportable()
    print("\n✅ Export pipeline validation passed.")