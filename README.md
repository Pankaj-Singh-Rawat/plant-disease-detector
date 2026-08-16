# 🌿 Plant Disease Detector

An on-device plant leaf disease classifier: a PyTorch model trained on the PlantVillage dataset, quantized and exported to TorchScript Mobile, and integrated into a native Android app via Flutter — with **no backend calls at inference time**. Every prediction runs entirely on-device.

[![Download APK](https://img.shields.io/badge/Download-APK-brightgreen?style=for-the-badge&logo=android)](https://github.com/Pankaj-Singh-Rawat/plant-disease-detector/releases/tag/v1.0.0)

---

## Overview

Most PyTorch tutorials stop at `torch.save()`. This project goes further: it trains a real image classifier, quantizes it for mobile, and ships it inside a working Android app that classifies plant leaf diseases from a photo — camera or gallery — with no internet connection required at prediction time.

**Why this domain:** PlantVillage is a clean, well-known, free dataset (~70K labeled leaf images across 38 disease/healthy classes) with genuine real-world framing — useful for farmers and gardeners, and a large enough class count to make the classification problem non-trivial.

## Pipeline

```
PyTorch Training (MobileNetV2, Apple M4 / MPS backend)
        │
        ▼
 best_model.pt  (float32 checkpoint, 99.82% val accuracy)
        │
        ▼
 TorchScript Mobile Export  (torch.jit.trace)
        │
        ▼
 Static INT8 Quantization  (qnnpack backend, calibrated on 100 batches)
        │
        ▼
 plant_disease_model_quantized.ptl  ──────►  GitHub Actions CI
        │                                     (validates export pipeline
        ▼                                      on every relevant code change)
 Flutter App
   ├─ Dart UI — camera / gallery picker
   └─ Kotlin native layer — PyTorch Android Lite runtime
        │
        ▼
   On-device prediction, zero network calls
```

## Results

### Training

MobileNetV2 (ImageNet-pretrained) fine-tuned on PlantVillage — 38 classes, ~70,295 training images, near-balanced dataset (1.23 max/min class ratio, confirmed before training, so no class reweighting was needed).

| Metric | Value |
|---|---|
| Best validation accuracy | **99.82%** |
| Epochs trained | 10 (best checkpoint at epoch 8) |
| Backbone | MobileNetV2 |
| Training hardware | Apple M4 (MPS backend) |
| Experiment tracking | Weights & Biases |

### Export & quantization — the core mobile-deployment tradeoff

| Metric | Float32 (TorchScript Mobile) | Quantized (INT8, static) |
|---|---|---|
| Validation accuracy | 98.01% | **95.48%** |
| Inference latency (CPU, single image) | 82.90 ms | **5.87 ms** |
| Model size | 9.20 MB | **2.71 MB** |

**Net result: ~14x faster inference, 3.39x smaller, at a cost of 2.54 percentage points of accuracy.**

This number wasn't accepted blindly — the first quantization attempt (20 calibration batches) cost 4.18pp of accuracy. Investigating showed the calibration set was too small to represent all 38 classes well; increasing to 100 batches recovered over half that gap, landing at the 2.54pp figure above. This kind of tradeoff (some accuracy for a large win in size and speed) is a standard, expected part of shipping ML models to constrained mobile hardware.

## Architecture decisions & why

- **MobileNetV2 over larger backbones** — chosen deliberately because it's designed for mobile inference from the ground up (depthwise separable convolutions, small parameter count), not bolted on as an afterthought.
- **TorchScript Mobile / Lite Interpreter over ONNX→TFLite** — kept things PyTorch-native end-to-end. (Note: PyTorch's own official direction has since moved to [ExecuTorch](https://docs.pytorch.org/executorch/stable/getting-started.html) as the successor to Lite Interpreter — this project predates that migration and would be a natural v2 direction.)
- **Native Kotlin platform channel over a community Flutter/PyTorch plugin** — the available third-party Flutter packages for PyTorch mobile inference (`pytorch_mobile` and similar) are thinly maintained, with no real-world example apps using them. Instead, Flutter handles the UI while a native Kotlin layer loads the model directly via `org.pytorch:pytorch_android_lite` — PyTorch's own official Android runtime — communicating with Dart over a `MethodChannel`. More setup, but far more robust.
- **Static quantization over dynamic** — MobileNetV2 is almost entirely `Conv2d` layers, which dynamic quantization doesn't cover well. Static quantization (with a calibration pass) properly quantizes both weights and activations for Conv layers, and was the right tool for this architecture.
- **`qnnpack` quantized backend** — the quantization engine built for ARM CPUs, matching both the training machine (Apple Silicon) and the deployment target (Android phones).

## Known environment constraints

Documented rather than hidden, since these are real limitations of the tooling at the time this was built:
- `torch.utils.mobile_optimizer.optimize_for_mobile()` requires XNNPACK, which isn't enabled in the default macOS ARM PyTorch pip wheel — the mobile export skips this optimization pass as a result (the traced model still exports and runs correctly, just without that extra operator-fusion step).
- Release builds required explicit R8/ProGuard keep rules for `org.pytorch.**` and `com.facebook.jni.**`, plus an explicit `jsr305` dependency — without these, Android's code shrinker strips classes the PyTorch native runtime needs at startup, causing a `ClassNotFoundException` crash on release (but not debug) builds.

## CI/CD

A GitHub Actions workflow ([`.github/workflows/validate-export.yml`](.github/workflows/validate-export.yml)) runs automatically whenever `model.py`, `export.py`, or `quantize.py` change. It builds a fresh (untrained) model, traces it, exports it to `.ptl`, reloads it, and checks the output shape and file size — catching pipeline-breaking regressions (e.g. an architecture change that breaks export) automatically, without needing the full dataset in CI.

This is intentionally scoped as a **pipeline-correctness check**, not a full accuracy re-validation — the latter would require the ~70K-image dataset and a full training run inside CI, which wasn't a fit for this project's scale. Dataset/model versioning via DVC was considered and deliberately left out for the same reason: this project isn't undergoing ongoing retraining, so there's nothing to version.

## Project structure

```
plant-disease-detector/
├── src/                    # PyTorch training, export, and quantization
│   ├── dataset.py          # DataLoader + augmentation pipeline
│   ├── model.py             # MobileNetV2 architecture setup
│   ├── train.py             # Training loop + W&B logging
│   ├── export.py            # TorchScript Mobile export
│   ├── quantize.py          # Static INT8 quantization + benchmarking
│   └── export_labels.py     # Extracts class names for the app
├── app/                     # Flutter application
│   ├── lib/main.dart        # UI — camera/gallery picker, results display
│   └── android/app/src/main/kotlin/.../PlantDiseaseClassifier.kt
│                             # Native inference layer (PyTorch Android Lite)
├── models/                  # Trained checkpoints & exported models (gitignored)
├── data/                    # PlantVillage dataset (gitignored)
└── .github/workflows/       # CI: export pipeline validation
```

## Scope

**In scope:** single-image classification (leaf photo in, disease label + confidence out), PyTorch training with proper experiment tracking, mobile export with measured quantization tradeoffs, a working Flutter/Android app with fully on-device inference, and automated export validation in CI.

**Deliberately out of scope:** multi-model ensembles or object detection, user accounts or cloud sync, iOS support, a custom-collected dataset, real-time video inference, and dataset/model versioning via DVC (no ongoing retraining planned for this project).

## Setup (for running the training pipeline yourself)

```bash
# Clone and set up the environment
git clone https://github.com/Pankaj-Singh-Rawat/plant-disease-detector.git
cd plant-disease-detector
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio wandb kaggle

# Download PlantVillage (requires a Kaggle API token in ~/.kaggle/kaggle.json)
cd data
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip new-plant-diseases-dataset.zip -d plantvillage
cd ..

# Train, export, quantize
python3 src/train.py
python3 src/export.py
python3 src/quantize.py
```

## Download

The Android app is available as a pre-built APK — no build steps required:

**[⬇ Download v1.0.0 APK](https://github.com/Pankaj-Singh-Rawat/plant-disease-detector/releases/tag/v1)**

---

Built by [Pankaj Singh Rawat](https://github.com/Pankaj-Singh-Rawat)
