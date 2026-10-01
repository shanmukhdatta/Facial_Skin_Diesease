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
        self, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset
    ) -> Optional[Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
        """Run soft-voting ensemble prediction and evaluate metrics.

        Weights are computed from ``val_ds`` (data the ensemble weighting has
        not been scored against) and the final reported metrics are computed
        purely on ``test_ds``, so the same split is never used for both.
        """
        if len(self.models) < 2:
            print("[WARN] Need at least 2 loaded models to build an ensemble.")
            return None

        # --- Step 1: compute performance-based weights on the VALIDATION set ---
        # Ensemble weights are derived exclusively from validation Macro F1 scores (notebook Section 20).
        y_val = []
        val_probs_list = [[] for _ in self.models]

        for images, labels in val_ds:
            y_val.extend(labels.numpy())
            for idx, m in enumerate(self.models):
                val_probs_list[idx].extend(m.predict(images, verbose=0))

        y_val = np.array(y_val)
        val_probs_arrays = [np.array(p) for p in val_probs_list]

        validation_macro_f1 = np.array([
            f1_score(y_val, np.argmax(p, axis=1), average="macro", zero_division=0)
            for p in val_probs_arrays
        ])

        weight_sum = validation_macro_f1.sum()
        if weight_sum <= 0:
            # Edge case: fallback to equal weighting
            weights = np.full(len(self.models), 1.0 / len(self.models))
        else:
            weights = validation_macro_f1 / weight_sum

        print("\nValidation Macro F1:")
        for name, f1v in zip(self.model_names, validation_macro_f1):
            print(f"  {name:<20}: {f1v:.4f}")

        print(f"\nEnsemble weights: {dict(zip(self.model_names, weights.round(4)))}")

        # --- Step 2: score the weighted ensemble on the held-out TEST set ---
        y_true = []
        probs_list = [[] for _ in self.models]

        for images, labels in test_ds:
            y_true.extend(labels.numpy())
            for idx, m in enumerate(self.models):
                probs_list[idx].extend(m.predict(images, verbose=0))

        y_true = np.array(y_true)
        probs_arrays = [np.array(p) for p in probs_list]

        ensemble_probs = sum(w * p for w, p in zip(weights, probs_arrays))
        y_pred = np.argmax(ensemble_probs, axis=1)

        ens_acc = (y_pred == y_true).mean()
        ens_f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
        ens_f1_wt = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ens_kappa = cohen_kappa_score(y_true, y_pred)
        from sklearn.metrics import matthews_corrcoef
        ens_mcc = matthews_corrcoef(y_true, y_pred)

        y_bin = label_binarize(y_true, classes=np.arange(self.num_classes))
        try:
            ens_auc = roc_auc_score(y_bin, ensemble_probs, average="macro", multi_class="ovr")
        except Exception:
            ens_auc = float("nan")

        print(f"\n[INFO] ENSEMBLE Results:")
        print(f"  Accuracy      : {ens_acc:.4f}")
        print(f"  F1 (Macro)    : {ens_f1_mac:.4f}")
        print(f"  F1 (Weighted) : {ens_f1_wt:.4f}")
        print(f"  AUC-ROC       : {ens_auc:.4f}")
        print(f"  Cohen's Kappa : {ens_kappa:.4f}")
        print(f"  MCC           : {ens_mcc:.4f}")
        report_dict = classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0, output_dict=True)
        print("\n" + classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))

        plot_confusion_matrix(y_true, y_pred, self.class_names, "Ensemble", self.plot_dir)
        plot_roc_curves(y_true, ensemble_probs, self.class_names, "Ensemble", self.plot_dir)
        from src.evaluation.visualization import plot_per_class_metrics
        plot_per_class_metrics(report_dict, self.class_names, "Ensemble", self.plot_dir)

        results = {
            "Model": "Ensemble (Weighted)",
            "Accuracy": round(float(ens_acc), 4),
            "F1 (Macro)": round(float(ens_f1_mac), 4),
            "F1 (Weighted)": round(float(ens_f1_wt), 4),
            "AUC-ROC": round(float(ens_auc), 4),
            "Cohen's Kappa": round(float(ens_kappa), 4),
            "MCC": round(float(ens_mcc), 4),
        }
        return results, y_true, y_pred