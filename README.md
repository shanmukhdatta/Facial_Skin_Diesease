# Facial Skin Disease Classification and Synthetic Data Generation

Official code and experimental notebooks accompanying the research paper:
> **"Enhancing Facial Skin Disease Detection Through Synthetic Data Generation Using Diffusion Models, GANs, and Pre-Trained CNN Architectures"**

This repository contains the complete experimental pipeline, including synthetic image generation manifests, transfer learning workflows for deep neural backbones, soft-voting ensemble evaluation, and reproducible Jupyter notebooks.

---

## Repository Structure

```
Facial_Skin_Disease_Synthetic_Generation/
├── Note books/
│   ├── Modular_Pipeline.ipynb        # Interactive step-by-step pipeline using the modular src/ package
│   └── Notebook.ipynb                # Complete experimental benchmark notebook (all models & comparisons)
│
├── configs/
│   ├── config.py                     # Central configuration dataclass (paths, hyperparameters, splits)
│   └── generation_config.yaml        # Generative model configurations (Diffusion & StyleGAN2-ADA)
│
├── prompts/                          # Clinical prompt manifests and metadata for synthetic generation
│   ├── acne.txt                      # Prompt texts for Acne
│   ├── acne_meta.jsonl               # Structured metadata (phenotype, severity, Fitzpatrick scale)
│   ├── vitiligo.txt (+ _meta.jsonl)
│   ├── fungal_infection.txt (+ _meta.jsonl)
│   ├── normal_skin.txt (+ _meta.jsonl)
│   ├── hyperpigmentation.txt (+ _meta.jsonl)
│   └── README.md                     # Documentation of prompt design taxonomy
│
├── src/                              # Core Python source package
│   ├── data/
│   │   ├── dataset.py                # Dataset discovery, validation, cleaning, and stratified splitting
│   │   ├── pipeline.py               # tf.data input pipeline construction and clinical augmentations
│   │   └── utils.py                  # Class weighting, label mapping, and dataset serialization
│   ├── generation/
│   │   ├── prompts.py                # Prompt builder and variation generator
│   │   ├── diffusion_generator.py    # Latent Diffusion synthesis engine (Realistic Vision V5.1)
│   │   ├── stylegan_generator.py     # StyleGAN2-ADA inference and sampling engine
│   │   └── stylegan_trainer.py       # StyleGAN2-ADA training orchestration and step execution
│   ├── models/
│   │   ├── backbones.py              # Backbone loaders (EfficientNet, ResNet, DenseNet, MobileNet, ViT)
│   │   ├── builder.py                # Classification head assembly and layer freezing/unfreezing logic
│   │   ├── custom_layers.py          # Architecture-specific input preprocessing layers
│   │   └── losses.py                 # Multi-class Focal Loss implementation
│   ├── training/
│   │   ├── callbacks.py              # Checkpoint, early stopping, and learning rate scheduling callbacks
│   │   └── trainer.py                # PhasedTrainer: 3-phase progressive unfreezing protocol
│   ├── evaluation/
│   │   ├── metrics.py                # Evaluation metrics (Accuracy, Macro-F1, Kappa, MCC, AUC-ROC)
│   │   ├── ensemble.py               # Performance-weighted soft-voting ensemble
│   │   └── visualization.py          # Confusion matrices, ROC curves, and training history plots
│   └── inference/
│       └── predictor.py              # Standalone single-image and batch prediction engine
│
├── scripts/                          # Command-line entry points
│   ├── run_generation.py             # Script to generate synthetic images from prompt manifests
│   ├── run_training.py               # Script to train classification backbones
│   ├── run_evaluation.py             # Script to evaluate trained checkpoints and compute ensemble metrics
│   ├── run_inference.py              # Script to perform inference on input images
│   └── verify_reproducibility.py     # Script to verify environment, pipeline modules, and configs
│
├── Papper/
│   ├── main.tex                      # Research manuscript source in LaTeX
│   └── figures/                      # Manuscript figures, architecture diagrams, and charts
│
├── requirements.txt                  # Python package dependencies
├── LICENSE                           # MIT License
└── README.md                         # Project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU recommended for training and diffusion generation

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/Facial_Skin_Disease_Synthetic_Generation.git
   cd Facial_Skin_Disease_Synthetic_Generation
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Dataset Organization

Place your real and synthetic dataset in the `data/Real_data/` directory (or pass custom paths via `--data_dir`):

```
data/
└── Real_data/
    ├── Acne/
    │   ├── img_001.jpg
    │   └── ...
    ├── Fungal_Infection/
    ├── Hyperpigmentation/
    ├── Normal_Skin/
    ├── Skin_Cancer/
    └── Vitiligo/
```

Default paths and hyperparameters can be configured directly in `configs/config.py` or overridden using command-line arguments.

---

## Using the Notebooks

The repository provides two notebooks located in the `Note books/` directory:

### 1. `Note books/Modular_Pipeline.ipynb`
A clean, step-by-step interactive demonstration of the modular framework:
- Loads project configuration from `configs/config.py`.
- Discovers and audits dataset integrity using `src/data/dataset.py`.
- Constructs `tf.data` training, validation, and test streams with data augmentation.
- Trains a transfer learning backbone (e.g., EfficientNetV2-B0) via `PhasedTrainer`.
- Evaluates model performance, plots confusion matrices and ROC curves.
- Demonstrates multi-model soft-voting ensemble evaluation and single-image inference.

**To run:**
```bash
jupyter lab "Note books/Modular_Pipeline.ipynb"
```

### 2. `Note books/Notebook.ipynb`
The comprehensive experimental benchmark notebook:
- Contains end-to-end execution of all evaluated architectures: EfficientNetV2-B0, ResNet-50, DenseNet-121, MobileNetV3-Large, and Vision Transformer (ViT-B16).
- Executes comparative training benchmarks, class-imbalance mitigations, and cross-model performance analyses.

**To run:**
```bash
jupyter lab "Note books/Notebook.ipynb"
```

---

## Using the Command-Line Scripts

All pipeline stages can also be executed directly via command-line scripts in `scripts/`:

### 1. Verification
To verify that all modules, prompt files, custom layers, and dependencies are functional:
```bash
python scripts/verify_reproducibility.py
```

### 2. Synthetic Data Generation
Generate synthetic dermatological samples using prompt manifests:
```bash
# Verify prompt manifests without GPU generation
python scripts/run_generation.py --method prompts_only

# Generate sample images using Latent Diffusion
python scripts/run_generation.py --method diffusion --classes acne vitiligo --limit_prompts 5 --images_per_prompt 2

# Full generation across all classes
python scripts/run_generation.py --method diffusion --classes acne vitiligo fungal_infection normal_skin hyperpigmentation

# Sample from trained StyleGAN2-ADA snapshot for Skin Cancer
python scripts/run_generation.py --method stylegan --network_pkl /path/to/network-snapshot.pkl
```

### 3. Model Training
Train neural backbones using the 3-phase progressive unfreezing protocol:
```bash
# Train a specific backbone (options: EfficientNetV2B0, ResNet50, DenseNet121, MobileNetV3Large, ViT_B16)
python scripts/run_training.py --data_dir data/Real_data --model ResNet50 --batch_size 32

# Train all backbones sequentially
python scripts/run_training.py --data_dir data/Real_data --model all
```

Optional training arguments:
- `--epochs_p1`: Epochs for Phase 1 warmup (default: `15`).
- `--epochs_p2`: Epochs for Phase 2 partial unfreezing (default: `25`).
- `--epochs_p3`: Epochs for Phase 3 deep fine-tuning (default: `30`).
- `--save_dir`: Directory to save trained `.keras` checkpoints (default: `outputs/saved_models`).
- `--plot_dir`: Directory to save training plots (default: `outputs/plots`).

### 4. Evaluation and Ensembling
Evaluate saved model checkpoints on the test split and compute the soft-voting ensemble metrics:
```bash
python scripts/run_evaluation.py \
    --data_dir data/Real_data \
    --save_dir outputs/saved_models \
    --plot_dir outputs/plots
```

### 5. Standalone Image Inference
Run inference on a single test image using a saved model checkpoint:
```bash
python scripts/run_inference.py \
    --image_path path/to/sample_image.jpg \
    --model_path outputs/saved_models/ResNet50_final.keras \
    --label_map outputs/saved_models/label_map.json \
    --save_plot outputs/plots/prediction.png
```

---

## Paper Reference

If you use this codebase or notebooks in your research, please cite:

```bibtex
@article{facial_skin_disease_synthetic_generation,
  title   = {Enhancing Facial Skin Disease Detection Through Synthetic Data Generation Using Diffusion Models, GANs, and Pre-Trained CNN Architectures},
  author  = {Boda, Shanmukha Datta},
  year    = {2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
