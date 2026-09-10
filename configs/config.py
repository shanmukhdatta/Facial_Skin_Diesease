import os
from dataclasses import dataclass, field
from pathlib import Path


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
        self.save_dir = Path(self.save_dir)
        self.plot_dir = Path(self.plot_dir)
        self.data_dir = Path(self.data_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)


default_config = ProjectConfig()
