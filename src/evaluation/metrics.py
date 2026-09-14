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


def compute_fid_for_classes(
    real_dirs: Dict[str, str],
    generated_dirs: Dict[str, str],
    dims: int = 192,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Frechet Inception Distance between real and generated images, per class (Section
    3.2.2 evaluation of the Skin Cancer synthetic classes). Mirrors the FID cell in the
    GAN generation notebook exactly.

    Uses a lower-dimensional Inception feature block (`dims`) than the library default
    of 2048 -- pytorch-fid's own README recommends this for datasets much smaller than
    the ~2048 images/side its default features were tuned for; with only ~100 real
    images per class here, the default would be far noisier. Treat the returned scores
    as a *relative* signal for comparing your own classes/checkpoints against each
    other, not as an absolute number comparable to FID scores reported in papers
    trained on thousands of images per class.

    Args:
        real_dirs: {class_name: path_to_real_images_dir}
        generated_dirs: {class_name: path_to_generated_images_dir}
        dims: Inception feature block size (192, 768, 2048, ...).
        device: "cuda" or "cpu".

    Returns:
        {class_name: fid_value} for every class with >=2 images on both sides.
    """
    from pathlib import Path

    from pytorch_fid.fid_score import calculate_fid_given_paths

    fid_scores: Dict[str, float] = {}
    for class_name, real_dir in real_dirs.items():
        gen_dir = generated_dirs.get(class_name)
        if gen_dir is None:
            continue

        n_real = len(list(Path(real_dir).glob("*")))
        n_fake = len(list(Path(gen_dir).glob("*")))
        if n_real < 2 or n_fake < 2:
            print(f"[fid] skipping '{class_name}' -- need >=2 images on each side "
                  f"(real={n_real}, generated={n_fake}).")
            continue

        batch_size = min(50, n_real, n_fake)
        try:
            fid_value = calculate_fid_given_paths(
                [str(real_dir), str(gen_dir)],
                batch_size=batch_size,
                device=device,
                dims=dims,
            )
            fid_scores[class_name] = fid_value
            print(f"[fid] '{class_name}': FID = {fid_value:.2f} ({n_real} real vs {n_fake} generated, dims={dims})")
        except Exception as e:
            print(f"[fid] '{class_name}': computation failed ({e}) -- skipping.")

    return fid_scores
