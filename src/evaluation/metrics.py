from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
    model_name: str,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for a trained classifier."""
    y_true, y_pred_prob = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_prob.extend(preds)

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = (y_pred == y_true).mean()
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )

    y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    try:
        auc_roc = roc_auc_score(y_bin, y_pred_prob, average="macro", multi_class="ovr")
    except Exception:
        auc_roc = float("nan")

    return {
        "model": model_name,
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_mac), 4),
        "f1_weighted": round(float(f1_wt), 4),
        "kappa": round(float(kappa), 4),
        "mcc": round(float(mcc), 4),
        "auc_roc": round(float(auc_roc), 4),
        "report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_prob": y_pred_prob,
    }
