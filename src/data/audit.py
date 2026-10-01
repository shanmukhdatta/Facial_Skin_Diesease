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
    """
    Check 1: no file PATH in more than one of the combined partitions.
    Check 2: no identical file CONTENT (MD5) in more than one partition.

    Resolution (when ``auto_resolve``):
      (a) identical bytes, SAME class everywhere -> keep exactly one copy
          (preferring the one already in train) and drop the rest;
      (b) identical bytes, DIFFERENT class labels -> label cannot be trusted:
          every copy is excluded and logged to md5_conflicting_label_duplicates.csv.
    If ``auto_resolve`` is False and ``strict`` is True the function asserts.
    """
    save_dir = Path(save_dir) if save_dir else None

    set_tr, set_va, set_te = set(train_df["path"]), set(val_df["path"]), set(test_df["path"])
    assert not (set_tr & set_va), "[ERROR] Leakage: combined train ∩ val is non-empty!"
    assert not (set_tr & set_te), "[ERROR] Leakage: combined train ∩ test is non-empty!"
    assert not (set_va & set_te), "[ERROR] Leakage: combined val ∩ test is non-empty!"
    print("[OK] Path-level check passed: no file path crosses a combined partition boundary.")

    print("[INFO] Hashing all combined images (MD5) to catch identical-content duplicates...")
    all_paths_labeled = _labeled_paths(train_df, val_df, test_df)
    hash_to_splits: Dict[str, set] = {}
    hash_to_paths: Dict[str, List[str]] = {}
    for p, split in all_paths_labeled:
        h = file_md5(p)
        hash_to_splits.setdefault(h, set()).add(split)
        hash_to_paths.setdefault(h, []).append(p)

    leaking = {h: s for h, s in hash_to_splits.items() if len(s) > 1}
    if leaking:
        print(f"[WARN] {len(leaking)} identical-content image(s) found spanning multiple partitions!")
        for h, s in list(leaking.items())[:10]:
            print(f"   hash {h[:10]}... in {sorted(s)}: {hash_to_paths[h][:3]}")
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"md5": h, "partitions": sorted(s), "paths": hash_to_paths[h]}
                          for h, s in leaking.items()]
                         ).to_csv(save_dir / "md5_cross_partition_duplicates.csv", index=False)

    if leaking and auto_resolve:
        path_to_split = {p: s for p, s in all_paths_labeled}
        path_to_label = pd.concat([
            train_df[["path", "class"]], val_df[["path", "class"]], test_df[["path", "class"]],
        ]).drop_duplicates("path").set_index("path")["class"].to_dict()

        drop_paths, conflict_rows = set(), []
        for h in leaking:
            grp_paths = hash_to_paths[h]
            labels = {path_to_label.get(p) for p in grp_paths}
            if len(labels) > 1:
                drop_paths.update(grp_paths)
                conflict_rows.append({"md5": h, "labels": sorted(labels), "paths": grp_paths})
            else:
                keep = next((p for p in grp_paths if path_to_split.get(p) == "train"), grp_paths[0])
                drop_paths.update(p for p in grp_paths if p != keep)

        print(f"\n[INFO] Resolving: dropping {len(drop_paths)} duplicate-content image(s) "
              f"({len(conflict_rows)} conflicting-label group(s) excluded entirely, "
              f"{len(leaking) - len(conflict_rows)} same-label group(s) de-duplicated to one copy).")
        if conflict_rows and save_dir:
            pd.DataFrame(conflict_rows).to_csv(save_dir / "md5_conflicting_label_duplicates.csv", index=False)
            print(f"   [WARN] Conflicting-label duplicates logged -> "
                  f"{save_dir / 'md5_conflicting_label_duplicates.csv'} (review manually).")

        keep_rows = lambda d: d[~d["path"].isin(drop_paths)].reset_index(drop=True)
        train_df, val_df, test_df = keep_rows(train_df), keep_rows(val_df), keep_rows(test_df)
        print(f"[OK] After resolution - train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")

        # Re-verify: no duplicate content should remain across partitions.
        re_splits: Dict[str, set] = {}
        for p, split in _labeled_paths(train_df, val_df, test_df):
            re_splits.setdefault(file_md5(p), set()).add(split)
        still = {h: s for h, s in re_splits.items() if len(s) > 1}
        assert not still, "[ERROR] Duplicates still cross a partition boundary after resolution - investigate."
        print("[OK] Re-verified: no identical-content image crosses a partition boundary after resolution.")
    elif strict:
        assert not leaking, ("[ERROR] Content-hash leakage detected - identical image bytes appear in more "
                             "than one of train/val/test (see above).")
    elif leaking:
        print("[WARN] strict_md5_assert is False: continuing despite MD5 duplicates.")

    print(f"[OK] Content-hash check: {len(hash_to_splits)} unique images, "
          f"{len(leaking)} duplicate hash(es) originally crossing a partition boundary.")
    return train_df, val_df, test_df


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
