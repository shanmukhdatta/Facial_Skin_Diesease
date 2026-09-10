# 🏥 Facial Skin Disease Classification & Synthetic Data Generation Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: TensorFlow / PyTorch](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20PyTorch-orange.svg)](https://tensorflow.org)
[![Reproducibility: 100% Verified](https://img.shields.io/badge/Reproducibility-100%25%20Verified-success.svg)](scripts/verify_reproducibility.py)

An industry-grade, modular deep learning repository for multi-class facial skin disease classification and disease-adaptive synthetic data generation. Based on the research paper:

> **"Enhancing Facial Skin Disease Detection Through Synthetic Data Generation Using Diffusion Models, GANs, and Pre-Trained CNN Architectures"**

---

## 📁 Repository Layout

```
Facial_Skin_Disease_Synthetic_Generation/
│
├── configs/
│   ├── __init__.py
│   ├── config.py                     # Central classification configuration (dataclass, paths, hyperparameters)
│   └── generation_config.yaml        # Generative pipeline configuration (Realistic Vision V5.1 & StyleGAN2-ADA)
│
├── prompts/                          # 300 structured prompt manifests per disease class (Section 3.2.1)
│   ├── acne.txt                      # 300 clinical prompt texts (+ acne_prompts.txt)
│   ├── acne_meta.jsonl               # Structured metadata (phenotype, gender, age, severity, Fitzpatrick)
│   ├── vitiligo.txt (+ _meta.jsonl)
│   ├── fungal_infection.txt (+ _meta.jsonl)
│   ├── normal_skin.txt (+ _meta.jsonl)
│   ├── hyperpigmentation.txt (+ _meta.jsonl)
│   └── README.md                     # Prompt corpus design, axes & phenotypic descriptors
│
├── src/                              # Core modular Python package
│   ├── __init__.py
│   │
│   ├── generation/                   # Dual generative engine
│   │   ├── __init__.py
│   │   ├── prompts.py                # Hierarchical prompt builder (3 phenotypes x 5 variations x 2 genders)
│   │   ├── diffusion_generator.py    # Realistic Vision V5.1 Latent Diffusion engine (7,500 images)
│   │   ├── stylegan_generator.py     # StyleGAN2-ADA sampling engine for Skin Cancer (1,500 images)
│   │   └── stylegan_trainer.py       # StyleGAN2-ADA training orchestration & 1-step reproducibility engine
│   │
│   ├── data/                         # Data discovery, cleaning, splitting & tf.data
│   │   ├── __init__.py
│   │   ├── dataset.py                # discover_dataset, is_valid_image, clean_dataset, split_dataset, balance_dataset
│   │   ├── pipeline.py               # Robust image decoding, clinical augmentations, tf.data pipeline
│   │   └── utils.py                  # compute_class_weights, save_label_map, load_label_map
│   │
│   ├── models/                       # Architecture, backbones, custom layers & losses
│   │   ├── __init__.py
│   │   ├── custom_layers.py          # ResNet50Preprocess, DenseNet121Preprocess, ViTPreprocess, CUSTOM_OBJECTS
│   │   ├── losses.py                 # Numerically-stable FocalLoss
│   │   ├── backbones.py              # Backbone factory (EfficientNet, ResNet, DenseNet, MobileNet, ViT)
│   │   └── builder.py                # Classification head assembly, freeze/unfreeze helpers
│   │
│   ├── training/                     # Phased progressive unfreezing orchestration
│   │   ├── __init__.py
│   │   ├── callbacks.py              # ModelCheckpoint, EarlyStopping, ReduceLR, Cosine LR scheduler
│   │   └── trainer.py                # PhasedTrainer: 3-phase progressive unfreezing
│   │
│   ├── evaluation/                   # Metrics, visualizations & soft-voting ensemble
│   │   ├── __init__.py
│   │   ├── metrics.py                # evaluate_model (Accuracy, F1-macro, F1-weighted, Kappa, MCC, AUC-ROC)
│   │   ├── visualization.py          # Confusion matrices, ROC curves, training history, comparison dashboard
│   │   └── ensemble.py               # WeightedEnsemble: performance-weighted soft voting
│   │
│   └── inference/                    # Production inference engine
│       ├── __init__.py
│       └── predictor.py              # DiseasePredictor for single-image and batch predictions
│
├── scripts/                          # Clean command-line entry points
│   ├── run_generation.py             # Generate synthetic disease images & run GAN reproducibility test
│   ├── run_training.py               # Train CNN backbones with 3-phase progressive unfreezing
│   ├── run_evaluation.py             # Evaluate saved checkpoints and construct weighted ensemble
│   ├── run_inference.py              # Predict disease class on any input image
│   └── verify_reproducibility.py     # End-to-end reproducibility verification suite (7/7 tests)
│
├── Notebooks/
│   ├── Modular_Pipeline.ipynb        # Clean interactive modular notebook
│   └── Notebook.ipynb                # Preserved original research notebook
│
├── Papper/                           # Research paper manuscript & publication figures
│   ├── main.tex                      # Full LaTeX manuscript
│   └── figures/                      # 14 publication figures and diagrams
│
├── .gitignore                        # Standard rules protecting against committing model weights & large caches
├── requirements.txt                  # Categorized dependencies
└── README.md                         # Complete project documentation
```

---

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Facial_Skin_Disease_Synthetic_Generation.git
cd Facial_Skin_Disease_Synthetic_Generation

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Quick Reproducibility Verification (100% Score)

To immediately confirm that every component of the repository is intact, correctly configured, and working with zero runtime errors, execute the verification suite:

```bash
python scripts/verify_reproducibility.py
```

This tests 7 core subsystems:
1. **Prompts Schema**: Verifies all 1,500 prompts across all 5 classes match the paper specifications.
2. **Diffusion Generation**: Synthesizes a sample image from prompt metadata.
3. **Single-Step GAN Training**: Feeds the generated image into the GAN engine, executes 1-step backprop on both Discriminator and Generator layers, and verifies output generation.
4. **Data Engineering**: Tests image verification, balanced class weights, and label mapping.
5. **Model Custom Layers**: Validates preprocessing math and serialization.
6. **Metrics & Visualizations**: Verifies Accuracy, Macro F1, Cohen's Kappa, MCC, and confusion matrix plotting.
7. **Storage Footprint**: Confirms zero heavy `.keras` or `.pt` model weights exist in the repo (clean commit size $\approx$ 8.9 MB).

---

## 🎨 1. Synthetic Data Generation (Section 3.2)

The study introduces a **disease-adaptive dual-generative framework** to generate 9,000 synthetic images:
- **Realistic Vision V5.1 (Latent Diffusion)**: Synthesizes 5 classes (*Acne, Vitiligo, Fungal Infection, Normal Skin, Hyperpigmentation*).
  - 3 phenotypes/class $\times$ 5 variation groups (Age, Severity, Fitzpatrick I–VI, Anatomy, Combined) $\times$ 10 prompts $\times$ 2 genders $= 300$ prompts/class.
  - 300 prompts $\times$ 5 seed-controlled images $= 1,500$ images/class (Total: 7,500 images).
- **StyleGAN2-ADA**: Synthesizes *Skin Cancer* from a curated seed corpus of 300 real images (100 BCC, 100 SCC, 100 Melanoma) to produce 1,500 synthetic images.

### Usage Commands:

```bash
# 1. Verify prompt manifests only (no GPU compute required)
python scripts/run_generation.py --method prompts_only

# 2. Run a smoke-test generation (1 prompt per class) and verify single-step GAN training:
python scripts/run_generation.py --method diffusion --classes acne --limit_prompts 1 --images_per_prompt 1 --train_gan

# 3. Full diffusion generation for selected or all classes:
python scripts/run_generation.py --method diffusion --classes acne vitiligo fungal_infection normal_skin hyperpigmentation

# 4. Generate Skin Cancer images via trained StyleGAN2-ADA snapshot:
python scripts/run_generation.py --method stylegan --network_pkl /path/to/network-snapshot.pkl
```

---

## 🚀 2. Classifier Training (Section 3.3 & 3.4)

Train deep learning backbones (*ResNet-50, EfficientNetV2B0, DenseNet-121, MobileNetV3Large, ViT-B16*) using the **3-phase progressive unfreezing protocol**:
- **Phase 1 (Warmup)**: Backbone frozen, train classification head only (`lr = 1e-3`, Adam, 15 epochs).
- **Phase 2 (Partial Unfreeze)**: Top $N$ backbone layers unfrozen (`lr = 1e-4`, Adam, 25 epochs).
- **Phase 3 (Full Fine-Tuning)**: Deep backbone unfreeze (`lr = 2e-5`, AdamW with weight decay $1e-5$, 30 epochs).

```bash
# Train all models sequentially
python scripts/run_training.py --data_dir /path/to/Real_data --model all

# Train a specific backbone (e.g. ResNet50) with custom batch size
python scripts/run_training.py --data_dir /path/to/Real_data --model ResNet50 --batch_size 32
```

---

## 📊 3. Evaluation & Weighted Ensemble (Section 3.6)

Evaluates saved checkpoints on the held-out test split ($15\%$) across 6 metrics: Accuracy, Macro F1, Weighted F1, Cohen's Kappa, Matthews Correlation Coefficient (MCC), and One-vs-Rest AUC-ROC. Automatically constructs the **Weighted Soft-Voting Ensemble**:

```bash
python scripts/run_evaluation.py \
    --data_dir /path/to/Real_data \
    --save_dir outputs/saved_models \
    --plot_dir outputs/plots
```

---

## 🔍 4. Single-Image & Batch Inference (Section 3.7)

Run automated inference and plot the prediction confidence distribution for any dermatological image:

```bash
python scripts/run_inference.py \
    --image_path /path/to/test_image.jpg \
    --model_path outputs/saved_models/ResNet50_final.keras \
    --label_map outputs/saved_models/label_map.json \
    --save_plot outputs/plots/prediction_plot.png
```


---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
