import math
from pathlib import Path
from typing import List, Tuple, Union
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
)


def make_callbacks(
    model_name: str,
    save_dir: Union[str, Path],
    monitor: str = "val_accuracy",
    patience_es: int = 8,
) -> Tuple[List[keras.callbacks.Callback], str]:
    """Create standard training callbacks: Checkpoint, EarlyStopping, ReduceLR, CSVLogger."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(save_dir / f"{model_name}_best.keras")
    log_path = str(save_dir / f"{model_name}_training_log.csv")

    callbacks = [
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
            mode="max",
        ),
        EarlyStopping(
            monitor=monitor,
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(log_path),
    ]
    return callbacks, ckpt_path


def cosine_lr_schedule(base_lr: float, epochs: int, warmup: int = 3) -> keras.callbacks.LearningRateScheduler:
    """Cosine learning rate schedule with linear warmup."""

    def schedule(epoch):
        if epoch < warmup:
            return float(base_lr * (epoch + 1) / warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return float(base_lr * 0.5 * (1 + math.cos(math.pi * progress)))

    return keras.callbacks.LearningRateScheduler(schedule, verbose=0)
