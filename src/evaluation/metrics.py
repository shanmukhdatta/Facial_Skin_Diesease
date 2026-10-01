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


def evaluate_per_source(
    systems: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_df: Any,
    class_names: List[str],
    save_dir: Any = None,
) -> Tuple[Any, Any]:
    """
    Per-source evaluation (notebook Section 21c): separates test set metrics into
    real-only and synthetic-only subsets. The real-only metrics reflect performance
    on clinical real-world images.

    Args:
        systems: {model_name: (y_true, y_pred)}
        test_df: test partition dataframe (must contain 'source' column or match order)
        class_names: list of string class names
        save_dir: optional directory to save per_source_overall.csv & per_source_recall.csv

    Returns:
        (per_source_df, per_source_recall_df)
    """
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    if "source" in test_df.columns:
        is_syn = (test_df["source"] == "synthetic").values
    else:
        # Fallback if source column not present
        print("[WARN] 'source' column not in test_df; cannot separate real vs synthetic.")
        return pd.DataFrame(), pd.DataFrame()

    labels = list(range(len(class_names)))
    rows, rec = [], []
    for name, (yt, yp) in systems.items():
        yt = np.asarray(yt)
        yp = np.asarray(yp)
        for sname, msk in (("real", ~is_syn), ("synthetic", is_syn)):
            if msk.sum() == 0:
                continue
            acc = round(float(accuracy_score(yt[msk], yp[msk])), 4)
            f1m = round(float(f1_score(yt[msk], yp[msk], average="macro", labels=labels, zero_division=0)), 4)
            rows.append({
                "Model": name,
                "Test subset": sname,
                "n": int(msk.sum()),
                "Accuracy": acc,
                "F1 (macro)": f1m,
            })
            r = recall_score(yt[msk], yp[msk], average=None, labels=labels, zero_division=0)
            for ci, c in enumerate(class_names):
                rec.append({
                    "Model": name,
                    "subset": sname,
                    "class": c,
                    "recall": round(float(r[ci]), 3),
                })

    per_source = pd.DataFrame(rows)
    per_source_recall = pd.DataFrame(rec)

    print("\n" + "=" * 70)
    print("PER-SOURCE TEST METRICS (combined test = synthetic part + real part):")
    print("=" * 70)
    if not per_source.empty:
        print(per_source.to_string(index=False))
        for sname in ("real", "synthetic"):
            sub = per_source_recall[per_source_recall["subset"] == sname]
            if not sub.empty:
                print(f"\nPer-class recall -- {sname.upper()} test images only:")
                print(sub.pivot(index="class", columns="Model", values="recall").to_string())

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            per_source.to_csv(save_dir / "per_source_overall.csv", index=False)
            per_source_recall.to_csv(save_dir / "per_source_recall.csv", index=False)
            print(f"\n[OK] Per-source metrics saved to {save_dir}")

    print("\nNote: The REAL-only rows reflect true generalization on real-world clinical images.")
    return per_source, per_source_recall

