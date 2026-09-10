import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, AdamW

from configs.config import ProjectConfig, default_config
from src.models.custom_layers import get_preprocess_layer
from src.models.losses import FocalLoss
from src.models.builder import (
    build_classification_head,
    build_vit_model,
    freeze_all,
    unfreeze_layers,
)
from src.training.callbacks import make_callbacks, cosine_lr_schedule


class PhasedTrainer:
    """
    Three-phase progressive unfreezing trainer for transfer learning:
      Phase 1 - Head warmup (backbone frozen)
      Phase 2 - Partial backbone unfreeze (last N layers)
      Phase 3 - Full/Deep backbone fine-tune with AdamW
    """

    def __init__(self, config: ProjectConfig = default_config):
        self.config = config

    def train_model(
        self,
        model_name: str,
        base_model: keras.Model,
        unfreeze_phase2: int,
        unfreeze_phase3: int,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        class_weight_dict: Optional[Dict[int, float]] = None,
        is_vit: bool = False,
    ) -> Tuple[keras.Model, List[Dict[str, Any]], str]:
        print(f"\n{'+'*70}")
        print(f"  [INFO] Training: {model_name}")
        print(f"{'+'*70}")

        if is_vit:
            model = build_vit_model(
                vit_base_model=base_model,
                num_classes=self.config.num_classes,
                dropout_rate=self.config.dropout_rate,
                l2_reg=self.config.l2_reg,
                img_size=self.config.img_size,
                name=model_name,
            )
        else:
            prep_layer = get_preprocess_layer(model_name)
            model = build_classification_head(
                base_model=base_model,
                num_classes=self.config.num_classes,
                dropout_rate=self.config.dropout_rate,
                l2_reg=self.config.l2_reg,
                img_size=self.config.img_size,
                name=model_name,
                preprocess_layer=prep_layer,
            )

        histories: List[Dict[str, Any]] = []
        loss_fn = FocalLoss(
            gamma=self.config.focal_loss_gamma,
            alpha=self.config.focal_loss_alpha,
        )
        metrics = [
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ]

        # -- PHASE 1 - Frozen Backbone Warm-up ----------------------------------
        print("\n[PHASE 1] Frozen backbone, warm up head")
        freeze_all(base_model)
        model.compile(optimizer=Adam(self.config.base_lr), loss=loss_fn, metrics=metrics)

        cbs1, _ = make_callbacks(f"{model_name}_p1", self.config.save_dir, patience_es=6)
        cbs1.append(cosine_lr_schedule(self.config.base_lr, self.config.epochs_phase1))

        h1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs_phase1,
            callbacks=cbs1,
            class_weight=class_weight_dict,
            verbose=1,
        )
        histories.append(h1.history)

        # -- PHASE 2 - Partial Unfreeze -----------------------------------------
        print(f"\n[PHASE 2] Partial unfreeze (last {unfreeze_phase2} layers)")
        unfreeze_layers(base_model, unfreeze_phase2)
        model.compile(optimizer=Adam(self.config.finetune_lr), loss=loss_fn, metrics=metrics)

        cbs2, _ = make_callbacks(f"{model_name}_p2", self.config.save_dir, patience_es=7)
        cbs2.append(cosine_lr_schedule(self.config.finetune_lr, self.config.epochs_phase2))

        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs_phase2,
            callbacks=cbs2,
            class_weight=class_weight_dict,
            verbose=1,
        )
        histories.append(h2.history)

        # -- PHASE 3 - Full Fine-tune -------------------------------------------
        print(f"\n[PHASE 3] Full fine-tune (last {unfreeze_phase3} layers)")
        unfreeze_layers(base_model, unfreeze_phase3)
        model.compile(
            optimizer=AdamW(learning_rate=self.config.full_ft_lr, weight_decay=1e-5),
            loss=loss_fn,
            metrics=metrics,
        )

        cbs3, _ = make_callbacks(f"{model_name}_p3", self.config.save_dir, patience_es=8)
        cbs3.append(cosine_lr_schedule(self.config.full_ft_lr, self.config.epochs_phase3))

        h3 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs_phase3,
            callbacks=cbs3,
            class_weight=class_weight_dict,
            verbose=1,
        )
        histories.append(h3.history)

        # -- Save Final Model ---------------------------------------------------
        final_path = str(self.config.save_dir / f"{model_name}_final.keras")
        model.save(final_path)
        print(f"\n[OK] Model saved -> {final_path}")

        return model, histories, final_path

    @staticmethod
    def cleanup_memory(model: Optional[keras.Model] = None) -> None:
        """Clear model from memory and reset Keras graph session."""
        if model is not None:
            del model
        gc.collect()
        K.clear_session()
