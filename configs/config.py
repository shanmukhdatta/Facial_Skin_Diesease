import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ProjectConfig:
    """Central configuration for Face Skin Disease Classification pipeline."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "DATA_DIR",
                # Default Kaggle path or local fallback
                "/kaggle/input/datasets/shanmukhadattaboda/real-dataset-of-face-disease/Real_data"
                if Path("/kaggle/input").exists()
                else str(Path(__file__).resolve().parent.parent / "data" / "Real_data"),
            )
        )
    )
    # Synthetic dataset root: 5 prompt-generated classes (Acne, Fungal Infection,
    # Hyperpigmentation, Normal Skin, Vitiligo) + BCC/SCC/MEL Skin Cancer images.
    synthetic_data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "SYNTHETIC_DATA_DIR",
                "/kaggle/input/datasets/shanmukhadattaboda/facial-skin-diesease-dataset/Face_Dataset"
                if Path("/kaggle/input").exists()
                else str(Path(__file__).resolve().parent.parent / "data" / "Face_Dataset"),
            )
        )
    )
    # Real dataset root: same class-name set as synthetic_data_dir, real photos.
    real_data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "REAL_DATA_DIR",
                "/kaggle/input/datasets/shanmukhadattaboda/real-skin-diesease-images/Real_Dataset"
                if Path("/kaggle/input").exists()
                else str(Path(__file__).resolve().parent.parent / "data" / "Real_Dataset"),
            )
        )
    )
    # Two INDEPENDENT mixing knobs for the combined TRAIN split: per class,
    # synth_fraction of that class's own synthetic train pool and real_fraction
    # of that class's own real train pool are taken and unioned. Defaults
    # (1.0 / 1.0) match the notebook: the ENTIRE train partition of BOTH sources
    # is used. Val/Test always use 100% of both sources' val/test partitions.
    # Set real_fraction=None to fall back to the legacy single-source split_dataset().
    # Data source mode used by the split pipeline:
    #   "combined"  -> synthetic + real (default, as in the notebook)
    #   "synthetic" -> synthetic dataset only (prompt-group split)
    #   "real"      -> real dataset only (flip-aware group split)
    mode: str = field(
        default_factory=lambda: os.getenv("MODE", "combined").lower()
    )
    synth_fraction: float = 1.0
    real_fraction: float = 1.0


    # Real-data group split (notebook Section 6b)
    phash_size: int = 8                  # 8x8 -> 64-bit pHash
    dup_hamming_threshold: int = 5       # differing bits allowed between "same" images
    embed_tau_start: float = 0.90        # ResNet-50 cosine threshold (mean-centred)
    embed_tau_step: float = 0.02
    embed_tau_max: float = 0.98
    max_group_share: float = 0.02        # no group may hold > 2% of all real images
    group_split_tries: int = 200         # best-of-N seeded GroupShuffleSplit
    split_csv_override: Optional[str] = None  # e.g. path to a saved splits_v2.csv

    # Final leakage audit (notebook Section 6d)
    strict_md5_assert: bool = True
    auto_resolve_md5_dupes: bool = True
    run_cross_source_audit: bool = True

    # Section 7: True = use every combined train image as-is (no resampling);
    # False = bounded resampling via balance_dataset().
    use_all_train_images: bool = True
    save_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("SAVE_DIR", "/kaggle/working/saved_models" if Path("/kaggle").exists() else str(Path(__file__).resolve().parent.parent / "outputs" / "saved_models"))
        )
    )
    plot_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PLOT_DIR", "/kaggle/working/plots" if Path("/kaggle").exists() else str(Path(__file__).resolve().parent.parent / "outputs" / "plots"))
        )
    )

    # Image and Dataset
    img_size: int = 224
    batch_size: int = 32
    num_classes: int = 6
    test_size: float = 0.30  # 30% split (15% val, 15% test)
    seed: int = 42

    # Training Epochs per Phase
    epochs_phase1: int = 15
    epochs_phase2: int = 25
    epochs_phase3: int = 30

    # Learning Rates
    base_lr: float = 1e-3
    finetune_lr: float = 1e-4
    full_ft_lr: float = 2e-5

    # Regularization
    dropout_rate: float = 0.4
    l2_reg: float = 1e-4

    # Dataset Balancing
    max_images_per_class: int = 1500
    target_size_balance: int = 1200

    # Loss Configuration
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25

    def __post_init__(self):
        assert self.mode in ("combined", "synthetic", "real"), \
            f"[ERROR] mode must be combined | synthetic | real, got {self.mode!r}"
        self.save_dir = Path(self.save_dir)
        self.plot_dir = Path(self.plot_dir)
        self.data_dir = Path(self.data_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)


default_config = ProjectConfig()
