<div align="center">

# Facial Skin Disease Classification & Synthetic Data Generation

**A reproducible deep-learning research pipeline for facial skin disease classification, synthetic image generation, multi-model benchmarking, weighted ensemble evaluation, and inference.**

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/github/license/shanmukhdatta/Facial_Skin_Diesease?style=for-the-badge" alt="MIT License">
</p>

</div>

> **Research / educational software.** This repository is an experimental computer-vision pipeline. It is **not a medical diagnostic device**, and predictions must not be treated as a substitute for evaluation by a qualified clinician.

---

## Overview

This project combines two complementary research workflows:

1. **Synthetic data generation** — controlled generation of facial skin-condition images using structured prompts and diffusion models, with a separate per-class StyleGAN2 (`stylegan2_pytorch`) path for Skin Cancer.
2. **Deep image classification** — preprocessing, quality control, leakage-aware splitting, training-set balancing, transfer learning, progressive fine-tuning, comprehensive evaluation, ensemble prediction, and standalone inference.

The codebase is organized as reusable Python modules under `src/`, configuration under `configs/`, CLI entry points under `scripts/`, prompt manifests under `prompts/`, and experimental notebooks under `Note books/`.

---

## Key Features

| Component | Capability |
|---|---|
| Data ingestion | Recursive discovery of JPG, JPEG, PNG, BMP, TIFF and WebP images |
| Data validation | Full image decoding and invalid/corrupt-image filtering |
| Dataset splitting | Stratified 70/15/15 train/validation/test split |
| Class balancing | Training-only bounded oversampling/capping |
| Input pipeline | TensorFlow `tf.data`, batching, shuffling and prefetching |
| Augmentation | Flip, rotation, zoom, contrast, brightness and translation |
| Backbones | EfficientNetV2-B0, ResNet-50, DenseNet-121, MobileNetV3-Large, ViT-B/16 |
| Training | Three-phase progressive unfreezing |
| Loss | Multi-class Focal Loss |
| Regularization | Dropout and L2 regularization |
| Evaluation | Accuracy, Macro-F1, Weighted-F1, AUC-ROC, Cohen's Kappa, MCC |
| Ensemble | Accuracy-weighted soft voting |
| Inference | Single-image prediction and probability visualization |
| Generation | Realistic Vision V5.1 (5 disease classes) + StyleGAN2 via `stylegan2_pytorch`, one independent model per class (Skin Cancer: BCC / SCC / melanoma) |
| Reproducibility | Fixed seeds, centralized configuration and verification suite |

---

## Pipeline Architecture

```text
Raw Dataset
    │
    ▼
Discover Images
    │
    ▼
Validate / Clean
    │
    ▼
Stratified 70 / 15 / 15 Split
    │
    ├──────────────► Validation / Test (untouched)
    │
    ▼
Balance Training Split
    │
    ▼
tf.data + Augmentation
    │
    ▼
┌─────────────────────────────────────────────┐
│          Three-Phase Transfer Learning       │
│                                             │
│  Phase 1 → Frozen backbone / head warm-up  │
│  Phase 2 → Partial backbone unfreezing     │
│  Phase 3 → Deep fine-tuning with AdamW     │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
             Model Evaluation
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
 Individual Metrics         Weighted Soft Voting
          │                         │
          └────────────┬────────────┘
                       ▼
              Inference / Analysis
```

---

## Repository Structure

```text
Facial_Skin_Diesease/
├── configs/
│   ├── config.py
│   └── generation_config.yaml
│
├── prompts/
│   ├── acne.txt
│   ├── acne_meta.jsonl
│   ├── fungal_infection.txt
│   ├── fungal_infection_meta.jsonl
│   ├── hyperpigmentation.txt
│   ├── hyperpigmentation_meta.jsonl
│   ├── normal_skin.txt
│   ├── normal_skin_meta.jsonl
│   ├── vitiligo.txt
│   ├── vitiligo_meta.jsonl
│   └── README.md
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── pipeline.py
│   │   └── utils.py
│   ├── generation/
│   │   ├── prompts.py
│   │   ├── diffusion_generator.py
│   │   ├── stylegan_generator.py
│   │   └── stylegan_trainer.py
│   ├── models/
│   │   ├── backbones.py
│   │   ├── builder.py
│   │   ├── custom_layers.py
│   │   └── losses.py
│   ├── training/
│   │   ├── callbacks.py
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── ensemble.py
│   │   └── visualization.py
│   └── inference/
│       └── predictor.py
│
├── scripts/
│   ├── run_generation.py
│   ├── run_training.py
│   ├── run_evaluation.py
│   ├── run_inference.py
│   └── verify_reproducibility.py
│
├── Note books/
│   ├── Modular_Pipeline.ipynb
│   └── Notebook.ipynb
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended for training and diffusion generation
- TensorFlow 2.15+
- PyTorch 2.0+
- Dependencies listed in `requirements.txt`

### Clone

```bash
git clone https://github.com/shanmukhdatta/Facial_Skin_Diesease.git
cd Facial_Skin_Diesease
```

### Virtual environment

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset Format

The classifier discovers class labels from subdirectories inside `data_dir`.

Recommended structure:

```text
data/
└── Real_data/
    ├── Acne/
    ├── Fungal_Infection/
    ├── Hyperpigmentation/
    ├── Normal_Skin/
    ├── Skin_Cancer/
    └── Vitiligo/
```

Supported image formats:

```text
.jpg
.jpeg
.png
.bmp
.tiff
.webp
```

The number of classification classes is determined from the dataset directories at runtime.

### Data processing

The pipeline:

1. Discovers valid image files.
2. Fully decodes images using Pillow.
3. Removes corrupt, unreadable or undersized images.
4. Performs a stratified split.
5. Balances **only the training split**.
6. Builds TensorFlow datasets.
7. Applies augmentation only during training.

---

## Configuration

Main classifier settings are centralized in `configs/config.py`.

| Parameter | Default |
|---|---:|
| Image size | `224 × 224` |
| Batch size | `32` |
| Default classes | `6` |
| Test-size parameter | `0.30` |
| Effective split | `70 / 15 / 15` |
| Seed | `42` |
| Phase 1 epochs | `15` |
| Phase 2 epochs | `25` |
| Phase 3 epochs | `30` |
| Phase 1 LR | `1e-3` |
| Phase 2 LR | `1e-4` |
| Phase 3 LR | `2e-5` |
| Dropout | `0.4` |
| L2 | `1e-4` |
| Focal gamma | `2.0` |
| Focal alpha | `0.25` |

Generation settings are stored in `configs/generation_config.yaml`.

---

# Training

The primary training entry point is:

```bash
python scripts/run_training.py
```

### Train one model

```bash
python scripts/run_training.py \
  --data_dir data/Real_data \
  --model ResNet50 \
  --batch_size 32
```

### Train all supported models

```bash
python scripts/run_training.py \
  --data_dir data/Real_data \
  --model all
```

### Supported backbones

```text
EfficientNetV2B0
ResNet50
DenseNet121
MobileNetV3Large
ViT_B16
```

---

## Training Strategy

### Phase 1 — Head Warm-up

The pretrained backbone is frozen while the classification head learns the task.

```text
Backbone: Frozen
Optimizer: Adam
Learning rate: 1e-3
Epochs: 15
```

### Phase 2 — Partial Fine-tuning

The final portion of the backbone is unfrozen.

```text
Backbone: Partially trainable
Optimizer: Adam
Learning rate: 1e-4
Epochs: 25
```

### Phase 3 — Deep Fine-tuning

Deeper backbone layers are unfrozen for representation adaptation.

```text
Backbone: Deep/full fine-tuning
Optimizer: AdamW
Learning rate: 2e-5
Weight decay: 1e-5
Epochs: 30
```

---

## Classification Head

For CNN-based architectures, the common head follows:

```text
Backbone
   │
Global Average Pooling
   │
Batch Normalization
   │
Dense(512, ReLU)
   │
Dropout
   │
Dense(256, ReLU)
   │
Dropout
   │
Dense(num_classes, Softmax)
```

ViT-B/16 uses a dedicated preprocessing layer and CLS-token extraction.

---

## Imbalance Handling

Two mechanisms are used:

- Training-set-only class balancing.
- Balanced class weights during optimization.

The project also uses multi-class **Focal Loss** to focus training on difficult examples.

---

# Synthetic Data Generation

The generation subsystem supports two distinct workflows.

## 1. Diffusion Generation

The configured diffusion pipeline uses:

- Realistic Vision V5.1
- Stable Diffusion VAE
- DPM-Solver++ / Karras scheduling
- Optional xFormers attention
- Attention slicing
- Configurable negative prompts
- Deterministic seeds
- Optional high-resolution refinement
- Automatic low-variance image rejection

Configured diffusion classes:

```text
acne
vitiligo
fungal_infection
normal_skin
hyperpigmentation
```

The prompt system uses disease-specific phenotypes and controlled variation across:

- Age
- Severity
- Fitzpatrick skin type
- Anatomical distribution
- Combined variations

The configuration targets 300 prompts × 5 images per prompt = 1,500 images per configured diffusion class.

### Verify prompts

```bash
python scripts/run_generation.py --method prompts_only
```

### Small test generation

```bash
python scripts/run_generation.py \
  --method diffusion \
  --classes acne vitiligo \
  --limit_prompts 5 \
  --images_per_prompt 2
```

### Full configured diffusion generation

```bash
python scripts/run_generation.py \
  --method diffusion \
  --classes acne vitiligo fungal_infection normal_skin hyperpigmentation
```

---

## 2. StyleGAN2 (Skin Cancer)

Skin Cancer generation trains **one independent, unconditional model per class**
(BCC / SCC / melanoma) using the `stylegan2_pytorch` PyPI package
(lucidrains/stylegan2-pytorch) — no external repository checkout needed, just
`pip install -r requirements.txt`. Each class's seed images live under
`configs/generation_config.yaml` → `stylegan2.paths.input_root`, one subfolder per class:

```text
data/skin_cancer_seed/
├── BCC/
├── SCC/
└── melanoma/
```

Training is time-boxed to fit a single GPU session (`stylegan2.training.total_time_budget_hours`):
it calibrates real steps/sec with a short timed run, scales the step target to the
remaining budget, then trains in small chunks, auto-resuming per class across sessions.

```bash
# Train (one class at a time, or --classes BCC SCC melanoma for all three)
python scripts/run_generation.py --method stylegan_train --classes SCC

# Generate from the trained checkpoints, with perceptual-hash de-duplication
python scripts/run_generation.py --method stylegan --classes SCC
```

The configured Skin Cancer categories are:

```text
BCC
SCC
melanoma
```

FID (real vs. generated, per class) can be computed with
`src/evaluation/metrics.py::compute_fid_for_classes` — see `stylegan2.evaluation.fid_dims`
in the config for the reduced-dimension Inception block used for this small-dataset setting.

An alternative NVIDIA `stylegan2-ada-pytorch`-based path (a single class-conditional or
combined model, loaded from a `.pkl` network snapshot) is kept in `stylegan_trainer.py` /
`stylegan_generator.py` / the `stylegan2_ada` config block for reference — it is **not**
the implementation used to produce this repo's published Skin Cancer images.

Large datasets, generated images, model weights and any external StyleGAN repository are intentionally excluded from Git.

---

# Evaluation

Run:

```bash
python scripts/run_evaluation.py \
  --data_dir data/Real_data \
  --save_dir outputs/saved_models \
  --plot_dir outputs/plots
```

## Metrics

The evaluation subsystem computes:

- Accuracy
- Macro-F1
- Weighted-F1
- Cohen's Kappa
- Matthews Correlation Coefficient
- Macro One-vs-Rest AUC-ROC
- Per-class precision
- Per-class recall
- Per-class F1

Visualizations include:

- Confusion matrices
- Normalized confusion matrices
- ROC curves
- Training curves
- Per-class metric charts
- Model comparison dashboard
- Radar chart

---

# Weighted Ensemble

The `WeightedEnsemble` module loads saved `*_final.keras` models and combines their probability outputs.

Current implementation:

```text
Individual model predictions
          │
          ▼
Validation-set accuracy per model
          │
          ▼
Normalize accuracies → ensemble weights
          │
          ▼
Weighted probability average (applied to test-set predictions)
          │
          ▼
Final class = argmax(probabilities), scored on the held-out test set
```

> Ensemble weights are computed from the validation set only. The test set is used exclusively for final scoring and never influences the weights, avoiding data leakage between weight selection and evaluation.

---

# Inference

Run inference on one image:

```bash
python scripts/run_inference.py \
  --image_path path/to/sample_image.jpg \
  --model_path outputs/saved_models/ResNet50_final.keras \
  --label_map outputs/saved_models/label_map.json \
  --save_plot outputs/plots/prediction.png
```

The inference engine:

1. Loads the trained Keras checkpoint.
2. Resizes the image to `224 × 224`.
3. Applies the model's internal preprocessing.
4. Produces the predicted class.
5. Returns the confidence and complete probability vector.
6. Can save a prediction visualization.

### Run all saved models

```bash
python scripts/run_inference.py \
  --image_path path/to/sample_image.jpg \
  --model_path all
```

---

# Notebooks

## Modular Pipeline

`Note books/Modular_Pipeline.ipynb`

Covers:

- Dataset discovery
- Cleaning
- Stratified splitting
- Balancing
- `tf.data` construction
- Augmentation
- Transfer learning
- Evaluation
- Ensemble loading
- Inference

Run:

```bash
jupyter lab "Note books/Modular_Pipeline.ipynb"
```

## Experimental Benchmark

`Note books/Notebook.ipynb`

Contains the broader multi-architecture experimental workflow.

Run:

```bash
jupyter lab "Note books/Notebook.ipynb"
```

For new experiments, the modular pipeline and CLI scripts are recommended.

---

# Reproducibility

Run the verification suite:

```bash
python scripts/verify_reproducibility.py
```

The verification workflow checks:

- Prompt manifests
- Synthetic generation
- GAN smoke-test behavior
- Data utilities
- Custom preprocessing layers
- Focal Loss
- Evaluation utilities
- Visualization
- Repository storage hygiene

Reproducibility features include:

- Fixed dataset split seed: `42`
- Deterministic per-class diffusion generation seeds (positional formula keyed on
  prompt number + variation index, not prompt content -- see `configs/generation_config.yaml`
  → `diffusion.seeding`)
- Centralized configuration
- Serialized `label_map.json`
- Large-artifact exclusion through `.gitignore`

---

# Output Artifacts

Typical outputs include:

```text
outputs/
├── saved_models/
│   ├── <Model>_final.keras
│   ├── <Model>_best.keras
│   ├── <Model>_training_log.csv
│   ├── label_map.json
│   ├── model_comparison.csv
│   └── model_comparison_with_ensemble.csv
│
├── plots/
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── <Model>_training_curves.png
│   ├── <Model>_confusion_matrix.png
│   ├── <Model>_roc_curves.png
│   ├── <Model>_per_class_metrics.png
│   ├── model_comparison_dashboard.png
│   └── model_radar_chart.png
│
└── synthetic_dataset/
    ├── <class>/
    └── metadata/
```

These runtime artifacts are ignored by Git.

---

# External Dependencies and Models

| Component | Requirement |
|---|---|
| TensorFlow/Keras | CNN training and inference |
| PyTorch | Generative pipeline |
| Diffusers | Stable Diffusion / Realistic Vision generation |
| Transformers | Transformer ecosystem support |
| scikit-learn | Splitting, metrics and class weighting |
| Pillow | Image loading and validation |
| stylegan2_pytorch | Skin Cancer generative workflow (lucidrains/stylegan2-pytorch, PyPI) |
| pytorch-fid / ImageHash | Skin Cancer generation evaluation (FID) and de-duplication (perceptual hash) |
| StyleGAN2-ADA | External repository, reference-only alternative StyleGAN workflow (not used to produce the published dataset) |
| Realistic Vision V5.1 | External pretrained diffusion model |
| Stable Diffusion VAE | External pretrained VAE |

---

# Research Limitations

### Dataset dependence

Performance depends on the quality, labeling, demographic diversity, acquisition conditions and clinical representativeness of the supplied dataset.

### Clinical deployment

This repository has not established clinical validity, regulatory compliance, treatment efficacy, or diagnostic safety.

---


# License

Released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---


