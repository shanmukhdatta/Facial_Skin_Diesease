from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras

from src.models.custom_layers import CUSTOM_OBJECTS
from src.models.losses import FocalLoss
from src.evaluation.visualization import plot_confusion_matrix, plot_roc_curves


class WeightedEnsemble:
    """Performance-weighted soft-voting ensemble across saved final model checkpoints."""

    def __init__(self, save_dir: Union[str, Path], plot_dir: Union[str, Path], class_names: List[str]):
        self.save_dir = Path(save_dir)
        self.plot_dir = Path(plot_dir)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.models: List[keras.Model] = []
        self.model_names: List[str] = []

    def load_models(self) -> int:
        """Find and load all *_final.keras models in save_dir."""
        model_files = sorted(self.save_dir.glob("*_final.keras"))
        self.models = []
        self.model_names = []

        for mf in model_files:
            try:
                m = keras.models.load_model(
                    str(mf),
                    custom_objects={**CUSTOM_OBJECTS, "FocalLoss": FocalLoss},
                )
                self.models.append(m)
                self.model_names.append(mf.stem.replace("_final", ""))
                print(f"  [OK] Loaded ensemble member: {mf.name}")
            except Exception as e:
                print(f"  [WARN] Could not load {mf.name}: {e}")

        return len(self.models)

    def evaluate_ensemble(
        self, test_ds: tf.data.Dataset
    ) -> Optional[Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
        """Run soft-voting ensemble prediction and evaluate metrics."""
        if len(self.models) < 2:
            print("[WARN] Need at least 2 loaded models to build an ensemble.")
            return None

        y_true = []
        probs_list = [[] for _ in self.models]

        for images, labels in test_ds:
            y_true.extend(labels.numpy())
            for idx, m in enumerate(self.models):
                probs_list[idx].extend(m.predict(images, verbose=0))

        y_true = np.array(y_true)
        probs_arrays = [np.array(p) for p in probs_list]

        # Performance-based weighting
        accuracies = [(np.argmax(p, axis=1) == y_true).mean() for p in probs_arrays]
        weights = np.array(accuracies)
        weights = weights / weights.sum()
        print(f"\nEnsemble weights: {dict(zip(self.model_names, weights.round(3)))}")

        ensemble_probs = sum(w * p for w, p in zip(weights, probs_arrays))
        y_pred = np.argmax(ensemble_probs, axis=1)

        ens_acc = (y_pred == y_true).mean()
        ens_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        ens_kappa = cohen_kappa_score(y_true, y_pred)

        y_bin = label_binarize(y_true, classes=np.arange(self.num_classes))
        try:
            ens_auc = roc_auc_score(y_bin, ensemble_probs, average="macro", multi_class="ovr")
        except Exception:
            ens_auc = float("nan")

        print(f"\n[INFO] ENSEMBLE Results:")
        print(f"  Accuracy      : {ens_acc:.4f}")
        print(f"  F1 (Macro)    : {ens_f1:.4f}")
        print(f"  AUC-ROC       : {ens_auc:.4f}")
        print(f"  Cohen's Kappa : {ens_kappa:.4f}")
        print("\n" + classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))

        plot_confusion_matrix(y_true, y_pred, self.class_names, "Ensemble", self.plot_dir)
        plot_roc_curves(y_true, ensemble_probs, self.class_names, "Ensemble", self.plot_dir)

        results = {
            "Model": "Ensemble (Weighted)",
            "Accuracy": round(float(ens_acc), 4),
            "F1 (Macro)": round(float(ens_f1), 4),
            "F1 (Weighted)": float("nan"),
            "AUC-ROC": round(float(ens_auc), 4),
            "Cohen's Kappa": round(float(ens_kappa), 4),
            "MCC": float("nan"),
        }
        return results, y_true, y_pred
