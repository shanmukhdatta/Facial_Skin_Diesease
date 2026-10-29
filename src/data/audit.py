"""Final combined leakage audit (notebook Section 6d / 6d-2).

* ``audit_and_resolve_md5``  – path-level + MD5 content-level audit of the final
  combined train/val/test sets, with automatic resolution of duplicate content.
* ``cross_source_audit``     – report-only real<->synthetic near-duplicate audit
  using EfficientNetV2-B0 features (image + mirror).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image as PILImage


def file_md5(path: str, chunk_size: int = 1 << 16) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _labeled_paths(train_df, val_df, test_df):
    return ([(p, "train") for p in np.unique(train_df["path"].values)] +
            [(p, "val") for p in val_df["path"].values] +
            [(p, "test") for p in test_df["path"].values])


def audit_and_resolve_md5(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_dir: Optional[Union[str, Path]] = None,
    strict: bool = True,
    auto_resolve: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """De-duplicate the FULL split pools before sampling training fractions.

    Every MD5 collision is considered, including within-partition collisions.
    Conflicting-label copies are all removed. With one label, preserve one
    test copy if present, else validation, else training. Report each removal,
    its source and its reason. No input image file is deleted on disk.
    """
    outdir = Path(save_dir) if save_dir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
    parts = [train_df.assign(_split="train"), val_df.assign(_split="val"),
             test_df.assign(_split="test")]
    full = pd.concat(parts, ignore_index=True)
    if not full["path"].is_unique:
        raise ValueError("The same path appears in multiple full-pool partitions")
    if "source" not in full:
        full["source"] = "unspecified"
    full["_md5"] = [file_md5(p) for p in full["path"]]
    # Preserve original order, choose test > validation > training as the survivor.
    priority = {"test": 0, "val": 1, "train": 2}
    drop_index = set()
    removed = []
    for digest, grp in full.groupby("_md5", sort=True):
        if len(grp) == 1:
            continue
        conflict = grp["class"].nunique() != 1
        keep_idx = None if conflict else min(
            grp.index, key=lambda i: (priority[full.at[i, "_split"]], str(full.at[i, "path"]))
        )
        for idx, row in grp.iterrows():
            if idx == keep_idx:
                continue
            drop_index.add(idx)
            removed.append({"md5": digest, "path": row["path"],
                            "source": row["source"], "class": row["class"],
                            "split": row["_split"],
                            "reason": "conflicting_label" if conflict else "exact_duplicate"})
    if drop_index and not auto_resolve:
        if strict:
            raise ValueError(f"{len(drop_index)} MD5 duplicates found; auto_resolve=False")
        print(f"[WARN] {len(drop_index)} duplicate rows retained because auto_resolve=False")
        drop_index.clear()
    kept = full.drop(index=list(drop_index))
    if outdir:
        fields = ["md5", "path", "source", "class", "split", "reason"]
        pd.DataFrame(removed, columns=fields).to_csv(outdir / "md5_removed_images.csv", index=False)
        stats = (pd.DataFrame(removed, columns=fields)
                 .groupby(["source", "reason"]).size().reset_index(name="removed")
                 if removed else pd.DataFrame(columns=["source", "reason", "removed"]))
        stats.to_csv(outdir / "md5_removals_by_source.csv", index=False)
        counts = full.groupby("source").size().rename("original").to_frame()
        after = kept.groupby("source").size().rename("retained")
        counts = counts.join(after, how="left").fillna(0).astype(int).reset_index()
        counts["removed"] = counts["original"] - counts["retained"]
        counts.to_csv(outdir / "md5_source_totals.csv", index=False)
    if strict and kept["_md5"].duplicated().any():
        raise AssertionError("Duplicate image bytes remained after MD5 resolution")
    def part(split):
        return (kept[kept["_split"] == split]
                .drop(columns=["_split", "_md5"]).reset_index(drop=True))
    print(f"[OK] Full-pool MD5 audit: {len(full)} images, {len(drop_index)} removals.")
    return part("train"), part("val"), part("test")


def cross_source_audit(
    synth_splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    real_splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    class_names: List[str],
    seed: int = 42,
    save_dir: Optional[Union[str, Path]] = None,
) -> Optional[pd.DataFrame]:
    """
    Report-only. EfficientNetV2-B0 features of every image and of its mirror; for
    each TEST image the similarity to its nearest TRAIN image, for
    real->real, real->synthetic, synthetic->real and synthetic->synthetic.
    Never raises: a failure is printed and training can proceed.
    """
    import gc
    import traceback
    try:
        from tensorflow.keras.applications import EfficientNetV2B0
        from tensorflow.keras import backend as K

        def _part(df, source, split):
            return pd.DataFrame({"path": df["path"].values, "label": df["label"].values,
                                 "source": source, "split": split})

        aud = pd.concat(
            [_part(d, "synthetic", n) for d, n in zip(synth_splits, ("train", "val", "test"))] +
            [_part(d, "real", n) for d, n in zip(real_splits, ("train", "val", "test"))],
            ignore_index=True)
        print(f"[INFO] Embedding {len(aud)} images (original + mirrored) for the cross-source audit...")

        net = EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=(224, 224, 3),
                               pooling="avg", include_preprocessing=True)
        E, Ef, paths = [], [], aud["path"].tolist()
        for s in range(0, len(paths), 64):
            arr = []
            for p in paths[s:s + 64]:
                with PILImage.open(p) as im:
                    arr.append(np.asarray(im.convert("RGB").resize((224, 224), PILImage.BILINEAR), dtype=np.float32))
            arr = np.stack(arr)
            E.append(net.predict(arr, verbose=0))
            Ef.append(net.predict(arr[:, :, ::-1, :].copy(), verbose=0))
        del net
        K.clear_session()
        gc.collect()
        E, Ef = np.concatenate(E).astype(np.float32), np.concatenate(Ef).astype(np.float32)
        mu = E.mean(axis=0, keepdims=True)
        E, Ef = E - mu, Ef - mu
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        Ef /= (np.linalg.norm(Ef, axis=1, keepdims=True) + 1e-8)

        def _nn_sim(q_src, q_split, r_src, r_split):
            qm = ((aud["source"] == q_src) & (aud["split"] == q_split)).values
            rm = ((aud["source"] == r_src) & (aud["split"] == r_split)).values
            return qm, np.maximum(E[qm] @ E[rm].T, E[qm] @ Ef[rm].T).max(axis=1)

        SETS = {"real TEST -> real TRAIN": ("real", "test", "real", "train"),
                "real TEST -> synthetic TRAIN": ("real", "test", "synthetic", "train"),
                "synthetic TEST -> real TRAIN": ("synthetic", "test", "real", "train"),
                "synthetic TEST -> synthetic TRAIN": ("synthetic", "test", "synthetic", "train")}
        rows, per_class = [], {}
        for name, args in SETS.items():
            qm, sim = _nn_sim(*args)
            rows.append({"query -> reference": name, "n queries": int(qm.sum()),
                         "median sim": round(float(np.median(sim)), 3),
                         "share >= 0.90": round(float((sim >= 0.90).mean()), 3),
                         "share >= 0.95": round(float((sim >= 0.95).mean()), 3)})
            per_class[name] = pd.Series(sim >= 0.95).groupby(
                np.array(class_names)[aud["label"].values[qm]]).mean().round(3)
        result = pd.DataFrame(rows)
        print("\nNearest-TRAIN similarity of every TEST image (cosine, mean-centred audit features):")
        print(result.to_string(index=False))
        print("\nShare of test images with a >= 0.95 neighbour in train, by class:")
        print(pd.DataFrame(per_class).to_string())

        rng = np.random.default_rng(seed)
        a, b = rng.integers(0, len(E), 20000), rng.integers(0, len(E), 20000)
        rnd = np.einsum("ij,ij->i", E[a], E[b])
        print(f"\nScale check - random pairs: p50={np.percentile(rnd, 50):.2f} "
              f"p95={np.percentile(rnd, 95):.2f} p99={np.percentile(rnd, 99):.2f}")
        if save_dir:
            save_dir = Path(save_dir)
            result.to_csv(save_dir / "cross_source_audit.csv", index=False)
            pd.DataFrame(per_class).to_csv(save_dir / "cross_source_audit_by_class.csv")

        xs = result.set_index("query -> reference")["share >= 0.95"]
        if max(xs["real TEST -> synthetic TRAIN"], xs["synthetic TEST -> real TRAIN"]) > 0.02:
            print("\n[WARN] More than 2% of test images have a near-identical image from the OTHER source in train.")
        else:
            print("\n[OK] Cross-source: <= 2% of test images have a >= 0.95 neighbour from the other source in train.")
        if xs["synthetic TEST -> synthetic TRAIN"] > 0.02:
            print("[WARN] Synthetic test images have near-identical synthetic train images "
                  "(check the Skin Cancer rows: that class is split at image level).")
        del E, Ef, aud
        gc.collect()
        return result
    except Exception:  # noqa: BLE001
        print("[WARN] Cross-source audit failed (training can still proceed):")
        traceback.print_exc()
        return None
