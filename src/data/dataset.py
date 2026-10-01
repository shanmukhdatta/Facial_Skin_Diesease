from __future__ import annotations
import glob
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
except ImportError:
    train_test_split = None
    GroupShuffleSplit = None

try:
    import imagehash
except ImportError:
    imagehash = None

from PIL import Image as PILImage, ImageOps

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def discover_dataset(data_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Walk the directory tree, collect (path, label) pairs, and return dataframe and class names."""
    data_dir = Path(data_dir)
    assert data_dir.exists(), f"[ERROR] DATA_DIR not found: {data_dir}"

    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    assert len(classes) > 0, f"[ERROR] No sub-folders found in DATA_DIR: {data_dir}"

    records, bad_files = [], []
    for label_idx, cls in enumerate(classes):
        cls_dir = data_dir / cls
        for f in cls_dir.rglob("*"):
            if f.suffix.lower() in VALID_EXTS:
                records.append({"path": str(f), "label": label_idx, "class": cls})
            elif f.is_file():
                bad_files.append(str(f))

    df = pd.DataFrame(records)
    print(f"\n[INFO] Dataset Summary")
    print(f"  Total valid images : {len(df)}")
    print(f"  Non-image files    : {len(bad_files)}")
    print(f"  Classes ({len(classes)})        : {classes}\n")
    if not df.empty:
        print(df.groupby("class")["path"].count().rename("count").to_string())

    return df, classes


def is_valid_image(path: str, min_size: int = 32) -> bool:
    """Fully decode image with PIL to catch any corrupt, empty, or unreadable files."""
    try:
        with PILImage.open(path) as img:
            img.load()  # forces full decode
            w, h = img.size
            if w < min_size or h < min_size:
                return False
            img.convert("RGB")
        return True
    except Exception:
        return False


def clean_dataset(df: pd.DataFrame, min_size: int = 32) -> pd.DataFrame:
    """Filter out corrupt or invalid images from the dataset dataframe."""
    print("[INFO] Scanning for corrupt / invalid images...")
    valid_mask = df["path"].apply(lambda p: is_valid_image(p, min_size=min_size))
    n_bad = (~valid_mask).sum()
    df_clean = df[valid_mask].reset_index(drop=True)
    print(f"  Removed corrupt/invalid images : {n_bad}")
    print(f"  Clean dataset size             : {len(df_clean)}")
    return df_clean


def split_dataset(
    df: pd.DataFrame, test_size: float = 0.30, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into Train (1 - test_size), Validation (test_size / 2), and Test (test_size / 2).
    Performed BEFORE any oversampling/balancing to prevent data leakage.
    """
    paths_all = df["path"].values
    labels_all = df["label"].values

    X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
        paths_all, labels_all, test_size=test_size, stratify=labels_all, random_state=seed
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    classes_map = dict(zip(df["label"], df["class"]))

    train_df = pd.DataFrame({
        "path": X_train_raw,
        "label": y_train_raw,
        "class": [classes_map[l] for l in y_train_raw]
    })
    val_df = pd.DataFrame({
        "path": X_val,
        "label": y_val,
        "class": [classes_map[l] for l in y_val]
    })
    test_df = pd.DataFrame({
        "path": X_test,
        "label": y_test,
        "class": [classes_map[l] for l in y_test]
    })

    print(f"  Train (raw, pre-balance) : {len(train_df)} images")
    print(f"  Validation               : {len(val_df)} images")
    print(f"  Test                     : {len(test_df)} images")
    return train_df, val_df, test_df


# Real dataset uses a plain per-class stratified split — this is exactly the
# same logic as split_dataset() above, kept as an explicit alias so calling
# code can name its intent (synthetic vs real) without duplicating the split.
# NOTE: superseded by split_real_grouped() below, which fixes the near-duplicate
# leakage this stratified split does not catch. Kept only for backward
# compatibility with older configs/scripts that still reference it by name.
split_real_stratified = split_dataset


class _UnionFind:
    """Minimal union-find (disjoint-set) with path compression + union by attach."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


# ─────────────────────────────────────────────────────────────────────────────
# Real dataset — flip-aware pHash + ResNet-50 embedding group split
# (mirrors notebook Sections 6.0 – 6.5)
# ─────────────────────────────────────────────────────────────────────────────
# Reference numbers reported by the notebook for the 10,227-image real dataset.
REFERENCE_REAL_SPLIT = {
    "train": 7144, "val": 1533, "test": 1550, "groups": 6298,
    "per_class_split": {  # (train, val, test)
        "Acne": (796, 183, 169), "Fungal Infection": (1199, 252, 251),
        "Hyperpigmentation": (499, 100, 101), "Normal Skin": (1035, 229, 233),
        "Skin Cancer": (2202, 475, 487), "Vitiligo": (1413, 294, 309),
    },
    "per_class_groups": {
        "Acne": 525, "Fungal Infection": 1439, "Hyperpigmentation": 346,
        "Normal Skin": 1391, "Skin Cancer": 1592, "Vitiligo": 1019,
    },
}


def _find_saved_split_csv(save_dir: Optional[Path], override: Optional[str]) -> Optional[str]:
    """Candidate order is identical to the notebook (override, /kaggle/input, SAVE_DIR)."""
    cands = ([str(override)] if override else []) \
        + sorted(glob.glob("/kaggle/input/**/splits_v2.csv", recursive=True)) \
        + ([str(Path(save_dir) / "splits_v2.csv")] if save_dir else [])
    return next((p for p in cands if p and os.path.exists(p)), None)


def _try_load_saved_split(df_clean_real: pd.DataFrame, csv_path: str):
    """Return a frame with columns path/split/group aligned to df_clean_real, or None."""
    try:
        sp = pd.read_csv(csv_path)
        assert {"path", "split", "group"} <= set(sp.columns), "CSV needs columns: path, split, group"
        if set(df_clean_real["path"]) <= set(sp["path"]):
            m = df_clean_real[["path"]].merge(sp[["path", "split", "group"]], on="path", how="left")
        else:  # dataset mounted elsewhere -> match on class/filename
            key = lambda p: f"{Path(p).parent.name}/{Path(p).name}"
            a = df_clean_real[["path"]].assign(_k=df_clean_real["path"].map(key))
            b = sp.assign(_k=sp["path"].map(key))
            assert a["_k"].is_unique and b["_k"].is_unique, "class/filename keys are not unique"
            m = a.merge(b[["_k", "split", "group"]], on="_k", how="left")
        assert len(m) == len(df_clean_real) and m["split"].notna().all(), \
            "some images of this dataset are missing from the CSV"
        return m
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Could not use {csv_path}: {e}\n       -> the split will be recomputed instead.")
        return None


def _hash_variants(path: str, phash_size: int) -> List[int]:
    """64-bit pHash of the image as original / H-flip / V-flip / 180° rotation."""
    with PILImage.open(path) as img:
        im = img.convert("RGB")
        variants = (im, ImageOps.mirror(im), ImageOps.flip(im), ImageOps.flip(ImageOps.mirror(im)))
        return [int(str(imagehash.phash(v, hash_size=phash_size)), 16) for v in variants]


def _embed_resnet50(paths: List[str], batch: int = 64):
    """ResNet-50 (ImageNet) avg-pool features of every image and of its mirror."""
    import gc
    from tensorflow import keras
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras import backend as K

    net = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg")
    grp, grp_f = [], []
    for s in range(0, len(paths), batch):
        arr = []
        for p in paths[s:s + batch]:
            with PILImage.open(p) as im:
                arr.append(np.asarray(im.convert("RGB").resize((224, 224), PILImage.BILINEAR), dtype=np.float32))
        arr = np.stack(arr)
        flip = arr[:, :, ::-1, :].copy()
        grp.append(net.predict(keras.applications.resnet50.preprocess_input(arr.copy()), verbose=0))
        grp_f.append(net.predict(keras.applications.resnet50.preprocess_input(flip.copy()), verbose=0))
    del net
    K.clear_session()
    gc.collect()
    return np.concatenate(grp).astype(np.float32), np.concatenate(grp_f).astype(np.float32)


def balanced_group_split(y, groups, test_size, tries: int = 200, seed: int = 42):
    """
    Each GROUP goes to exactly one partition. Because groups differ in size,
    the best of `tries` seeded GroupShuffleSplit runs is kept (smallest
    class-proportion deviation + size deviation) so small classes are not starved.
    """
    n_cls = int(y.max()) + 1
    overall = np.bincount(y, minlength=n_cls) / len(y)
    best = None
    for t in range(tries):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + t)
        a, b = next(gss.split(np.zeros(len(y)), y, groups=groups))
        pb = np.bincount(y[b], minlength=n_cls) / len(b)
        score = np.max(np.abs(pb / overall - 1)) + 2 * abs(len(b) / len(y) - test_size)
        if best is None or score < best[0]:
            best = (score, a, b)
    return best[1], best[2], best[0]


def split_real_grouped(
    df_clean_real: pd.DataFrame,
    test_size: float = 0.30,
    seed: int = 42,
    phash_size: int = 8,
    dup_hamming_threshold: int = 5,
    tau_start: float = 0.90,
    tau_step: float = 0.02,
    tau_max: float = 0.98,
    max_group_share: float = 0.02,
    n_tries: int = 200,
    chunk: int = 512,
    save_dir: Optional[Union[str, Path]] = None,
    split_csv_override: Optional[str] = None,
    emb_cache: Optional[Union[str, Path]] = "/tmp/emb_cache_v2.npz",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Corrected group-aware split for the REAL dataset (notebook Section 6b).

      1. Flip-aware pHash: every image is hashed as original / H-flip / V-flip /
         180°; two images are linked if the smallest Hamming distance over those
         orientations is <= dup_hamming_threshold. ALL pairs are compared
         exactly (no LSH banding), so nothing inside the threshold is missed.
      2. Embedding similarity: ResNet-50 features (mean-centred) of the image and
         its mirror; linked if cosine >= tau. tau starts at tau_start and is
         raised by tau_step (up to tau_max) only while one group would hold more
         than max_group_share of all images.
      3. Union-Find over both link sets -> dup_group_id.
      4. Whole groups are split 70/15/15 with balanced_group_split()
         (best of n_tries seeded GroupShuffleSplit runs for class balance).
      5. Leakage asserts: no path and no group crosses a partition.

    If a saved `splits_v2.csv` (columns: path, split, group) is found (override
    path, /kaggle/input/**, or <save_dir>/splits_v2.csv) it is used verbatim and
    the recomputation is skipped. The split actually used is written to
    <save_dir>/real_splits_used.csv; a freshly computed one also to splits_v2.csv.

    Limitations: no patient IDs exist, so groups are near-duplicate / same-session
    proxies; crops/zooms of one photo may only be partly caught by tau = 0.90.
    """
    if GroupShuffleSplit is None:
        raise ImportError("scikit-learn is required for split_real_grouped().")
    if imagehash is None:
        raise ImportError("ImageHash is required for split_real_grouped() (pip install ImageHash).")
    assert phash_size == 8, "[ERROR] the all-pairs Hamming kernel assumes 64-bit (8x8) hashes."

    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    df_clean_real = df_clean_real.reset_index(drop=True).copy()
    N = len(df_clean_real)
    X_all = df_clean_real["path"].values
    y_all = df_clean_real["label"].values
    cls_arr = df_clean_real["class"].values

    # ── 6.0 Saved reference split (fast path) ────────────────────────────────
    used_saved = False
    csv_path = _find_saved_split_csv(save_dir, split_csv_override)
    saved = _try_load_saved_split(df_clean_real, csv_path) if csv_path else None
    if saved is not None:
        used_saved = True
        print(f"[OK] Using the saved reference split: {csv_path}")
        groups_all = saved["group"].values
        sp = saved["split"].values
        train_idx = np.where(sp == "train")[0]
        val_idx = np.where(sp == "val")[0]
        test_idx = np.where(sp == "test")[0]
    else:
        if not csv_path:
            print("[INFO] No splits_v2.csv found -> the split will be recomputed (identical procedure).")

        # ── 6.1 Flip-aware perceptual hashes + exact all-pairs distances ─────
        print("[INFO] Hashing every image in 4 orientations (a few minutes)...")
        t0 = time.time()
        HV = np.array([_hash_variants(p, phash_size) for p in X_all], dtype=np.uint64)  # (N, 4)
        H0, H1, H2, H3 = HV[:, 0], HV[:, 1], HV[:, 2], HV[:, 3]
        print(f"  Hashed {N} images in {time.time() - t0:.0f}s.")

        POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

        def ham_block(a, b):  # (m,) x (n,) -> (m, n) uint8
            x = (a[:, None] ^ b[None, :]).view(np.uint8).reshape(len(a), len(b), 8)
            return POP[x].sum(axis=2, dtype=np.uint8)

        print("[INFO] Exact all-pairs Hamming distances (plain and flip-aware)...")
        t0 = time.time()
        D_plain = np.empty((N, N), dtype=np.uint8)  # original vs original (what the old split saw)
        D_flip = np.empty((N, N), dtype=np.uint8)   # min over the 4 orientations of the 2nd image
        for s in range(0, N, chunk):
            a = H0[s:s + chunk]
            d0 = ham_block(a, H0)
            d = d0.copy()
            for Hv in (H1, H2, H3):
                np.minimum(d, ham_block(a, Hv), out=d)
            D_plain[s:s + chunk] = d0
            D_flip[s:s + chunk] = d
        D_flip = np.minimum(D_flip, D_flip.T)  # symmetrise
        np.fill_diagonal(D_plain, 255)
        np.fill_diagonal(D_flip, 255)
        print(f"  Done in {time.time() - t0:.0f}s.")

        def _edges(mask):
            r, c = np.nonzero(mask)
            k = r < c
            return r[k], c[k]

        def _uf_groups(n, edge_lists):
            uf = _UnionFind(n)
            for ii, jj in edge_lists:
                for i, j in zip(ii.tolist(), jj.tolist()):
                    uf.union(i, j)
            return np.array([uf.find(i) for i in range(n)])

        e_plain = _edges(D_plain <= dup_hamming_threshold)
        e_flip = _edges(D_flip <= dup_hamming_threshold)
        g_plain = _uf_groups(N, [e_plain])
        g_flip = _uf_groups(N, [e_flip])
        del D_plain, D_flip

        def _groups_per_image(g):
            tmp = pd.DataFrame({"class": cls_arr, "g": g})
            return (tmp.groupby("class")["g"].nunique() / tmp.groupby("class").size()).round(3)

        print(f"\n  Linked pairs (Hamming <= {dup_hamming_threshold}): plain = {len(e_plain[0])} | "
              f"flip-aware = {len(e_flip[0])} (+{len(e_flip[0]) - len(e_plain[0])} only visible after flipping)")
        print(f"  Groups: plain pHash = {len(set(g_plain))} | flip-aware pHash = {len(set(g_flip))}")
        df_clean_real["phash"] = [f"{int(h):016x}" for h in H0]

        # ── 6.2 Embeddings (ResNet-50, image + mirror) ───────────────────────
        paths_list = df_clean_real["path"].tolist()
        E = Ef = None
        emb_cache = Path(emb_cache) if emb_cache else None
        if emb_cache is not None and emb_cache.exists():
            z = np.load(emb_cache, allow_pickle=True)
            if {"paths", "grp", "grp_f"} <= set(z.files):
                lk = {p: i for i, p in enumerate(z["paths"].tolist())}
                if all(p in lk for p in paths_list):
                    ix = np.array([lk[p] for p in paths_list])
                    E, Ef = z["grp"][ix], z["grp_f"][ix]
                    print("[INFO] Loaded cached embeddings.")
        if E is None:
            print("[INFO] Computing ResNet-50 embeddings (needs ImageNet weights)...")
            t0 = time.time()
            E, Ef = _embed_resnet50(paths_list)
            if emb_cache is not None:
                try:
                    np.savez(emb_cache, paths=np.array(paths_list, dtype=object), grp=E, grp_f=Ef)
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] Could not write embedding cache: {e}")
            print(f"  Done in {time.time() - t0:.0f}s.")

        mu = E.mean(axis=0, keepdims=True)  # unrelated pairs -> cosine ~ 0
        A, B = E - mu, Ef - mu
        A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)

        # ── 6.3 Final groups = flip-aware pHash links ∪ embedding links (>= tau)
        ei, ej, es = [], [], []
        for s in range(0, N, chunk):
            sim = np.maximum(A[s:s + chunk] @ A.T, A[s:s + chunk] @ B.T)  # flip-aware cosine
            r, c = np.nonzero(sim >= tau_start)
            k = c > (r + s)
            r, c = r[k], c[k]
            es.append(sim[r, c]); ei.append(r + s); ej.append(c)
        ei, ej, es = map(np.concatenate, (ei, ej, es))

        def _groups_at(tau):
            m = es >= tau
            return _uf_groups(N, [e_flip, (ei[m], ej[m])])

        tau = tau_start
        while True:
            g_final = _groups_at(tau)
            sizes = np.bincount(np.unique(g_final, return_inverse=True)[1])
            if sizes.max() / N <= max_group_share or tau >= tau_max:
                break
            tau = round(tau + tau_step, 2)
        if sizes.max() / N > max_group_share:
            print(f"[WARN] Largest group still {sizes.max()} images at tau={tau}; continuing anyway.")

        df_clean_real["dup_group_id"] = g_final
        df_clean_real["group_plain"] = g_plain
        df_clean_real["group_flip"] = g_flip
        print(f"\n  Chosen tau = {tau} | groups = {len(set(g_final))} | largest group = {sizes.max()} images "
              f"({sizes.max() / N:.2%}) | groups with >1 image = {int((sizes > 1).sum())}")
        gpi = pd.DataFrame({
            "plain pHash (old view)": _groups_per_image(g_plain),
            "flip-aware pHash": _groups_per_image(g_flip),
            "flip pHash + embeddings (FINAL)": _groups_per_image(g_final),
        })
        print("\n  Groups per image by class (lower = more duplicated; no patient IDs exist, these are proxies):")
        print(gpi.to_string())
        if save_dir:
            gpi.to_csv(save_dir / "group_diversity_v2.csv")
        span = df_clean_real.groupby("dup_group_id")["class"].nunique()
        print(f"\n  Groups containing images from >1 class (same picture under two labels): {int((span > 1).sum())}")

        # ── 6.4 Grouped, class-balanced 70/15/15 split ───────────────────────
        groups_all = g_final
        train_idx, temp_idx, _ = balanced_group_split(y_all, groups_all, test_size, tries=n_tries, seed=seed)
        va_rel, te_rel, _ = balanced_group_split(y_all[temp_idx], groups_all[temp_idx], 0.50,
                                                 tries=n_tries, seed=seed)
        val_idx, test_idx = temp_idx[va_rel], temp_idx[te_rel]

    # ── 6.5 Verification (both paths): leakage asserts ───────────────────────
    df_clean_real["dup_group_id"] = groups_all
    set_tr, set_va, set_te = set(X_all[train_idx]), set(X_all[val_idx]), set(X_all[test_idx])
    assert not (set_tr & set_va), "[ERROR] Leakage: real train ∩ val is non-empty!"
    assert not (set_tr & set_te), "[ERROR] Leakage: real train ∩ test is non-empty!"
    assert not (set_va & set_te), "[ERROR] Leakage: real val ∩ test is non-empty!"
    g_tr, g_va, g_te = set(groups_all[train_idx]), set(groups_all[val_idx]), set(groups_all[test_idx])
    assert not (g_tr & g_va), "[ERROR] A duplicate group is in both train and val!"
    assert not (g_tr & g_te), "[ERROR] A duplicate group is in both train and test!"
    assert not (g_va & g_te), "[ERROR] A duplicate group is in both val and test!"
    assert len(train_idx) + len(val_idx) + len(test_idx) == N

    split_name = np.empty(N, dtype=object)
    split_name[train_idx], split_name[val_idx], split_name[test_idx] = "train", "val", "test"

    print("=" * 64)
    print(f"REAL-DATA SPLIT USED: {'saved reference file' if used_saved else 'recomputed'}  (70 / 15 / 15, group-aware)")
    print("=" * 64)
    print(f"  Train: {len(train_idx)}  ({len(g_tr)} groups)")
    print(f"  Val  : {len(val_idx)}  ({len(g_va)} groups)")
    print(f"  Test : {len(test_idx)}  ({len(g_te)} groups)")
    print(f"  Total groups: {len(set(groups_all))}")
    tab = pd.crosstab(cls_arr, split_name, rownames=["class"], colnames=["split"])[["train", "val", "test"]]
    tab["groups"] = df_clean_real.groupby("class")["dup_group_id"].nunique()
    tab["groups_per_image"] = (tab["groups"] / tab[["train", "val", "test"]].sum(axis=1)).round(3)
    print("\n  Per-class counts and groups per class:")
    print(tab.to_string())

    # Compare with the notebook's reference split (only meaningful for that dataset).
    ref = REFERENCE_REAL_SPLIT
    if set(ref["per_class_split"]) == set(tab.index):
        ok = (len(train_idx), len(val_idx), len(test_idx)) == (ref["train"], ref["val"], ref["test"]) \
            and len(set(groups_all)) == ref["groups"]
        for c, trio in ref["per_class_split"].items():
            ok &= tuple(int(tab.loc[c, k]) for k in ("train", "val", "test")) == trio
            ok &= int(tab.loc[c, "groups"]) == ref["per_class_groups"][c]
        if ok:
            print("\n[OK] Matches the notebook reference split: 7,144 / 1,533 / 1,550 images, 6,298 groups.")
        else:
            print("\n[WARN] Does NOT match the notebook reference numbers (7,144 / 1,533 / 1,550 images, "
                  "6,298 groups). Attach splits_v2.csv, or check the dataset is the same 10,227 images.")

    if save_dir:
        out = pd.DataFrame({"path": X_all, "class": cls_arr, "group": groups_all, "split": split_name})
        if not used_saved:
            out.to_csv(save_dir / "splits_v2.csv", index=False)
            print(f"[OK] Split assignment saved -> {save_dir / 'splits_v2.csv'}")
        out.to_csv(save_dir / "real_splits_used.csv", index=False)
        print(f"[OK] Split actually used saved -> {save_dir / 'real_splits_used.csv'}")

    def _mk(idx):
        return pd.DataFrame({"path": X_all[idx], "label": y_all[idx], "class": cls_arr[idx]}).reset_index(drop=True)

    return _mk(train_idx), _mk(val_idx), _mk(test_idx)


_PROMPT_RE = re.compile(r"_p_(\d+)_v_(\d+)$", re.IGNORECASE)
_CANCER_RE = re.compile(r"_(BCC|SCC|MEL)_(\d+)$", re.IGNORECASE)


def _parse_stem(path_str: str) -> Tuple[str, Optional[str], Optional[int]]:
    """Classify a filename as a synthetic prompt/variant image or a
    Skin-Cancer BCC/SCC/MEL image, ignoring the extension."""
    stem = Path(path_str).stem
    m = _PROMPT_RE.search(stem)
    if m:
        return "synthetic", m.group(1), int(m.group(2))
    m = _CANCER_RE.search(stem)
    if m:
        return "cancer", m.group(1).upper(), int(m.group(2))
    return "unknown", None, None


def split_synthetic_grouped(
    df_clean: pd.DataFrame, test_size: float = 0.30, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the synthetic dataset with source-aware logic:
      - The 5 prompt-generated classes (Acne, Fungal Infection, Hyperpigmentation,
        Normal Skin, Vitiligo): grouped by (class + prompt_id) so all 5 variants
        of a prompt land in exactly ONE partition (GROUP split, no stratify —
        prevents prompt-level leakage).
      - Skin Cancer (BCC/SCC/MEL): split independently per subtype at the image
        level (PLAIN random split, no stratify), then recombined under the
        single "Skin Cancer" class label.
    Ratio is test_size/2 : test_size/2 for val/test either way (default 70/15/15).
    Images whose filename matches neither pattern are excluded and reported.
    """
    if train_test_split is None:
        raise ImportError("scikit-learn is required for split_synthetic_grouped().")

    df_clean = df_clean.copy()
    kinds, keys, variants = [], [], []
    for p in df_clean["path"]:
        kind, key, var = _parse_stem(p)
        kinds.append(kind)
        keys.append(key)
        variants.append(var)

    df_clean["file_kind"] = kinds
    df_clean["prompt_id"] = [k if kd == "synthetic" else None for kd, k in zip(kinds, keys)]
    df_clean["variant"] = [v if kd == "synthetic" else None for kd, v in zip(kinds, variants)]
    df_clean["subtype"] = [k if kd == "cancer" else None for kd, k in zip(kinds, keys)]
    df_clean["group_key"] = [
        f"{cls}_{pid}" if kd == "synthetic" else None
        for kd, cls, pid in zip(kinds, df_clean["class"], df_clean["prompt_id"])
    ]

    n_unknown = (df_clean["file_kind"] == "unknown").sum()
    if n_unknown:
        print(f"[WARN] {n_unknown} image(s) matched neither the prompt/variant nor "
              f"BCC/SCC/MEL filename pattern and will be EXCLUDED from the split.")

    # ── Prompt classes: GROUP split (70/15/15 by default) ──
    df_synth = df_clean[df_clean["file_kind"] == "synthetic"]
    complete_group_keys, incomplete_report = [], []
    for gk in sorted(df_synth["group_key"].unique()):
        grp = df_synth[df_synth["group_key"] == gk]
        variants_present = sorted(int(v) for v in grp["variant"])
        if len(grp) == 5 and len(set(variants_present)) == 5:
            complete_group_keys.append(gk)
        else:
            incomplete_report.append({
                "group_key": gk, "n_images": len(grp),
                "variants_present": variants_present,
                "duplicates": len(grp) - len(set(variants_present)),
            })
    complete_group_keys = sorted(complete_group_keys)  # deterministic order before shuffling

    print(f"  Synthetic prompt-group validation: complete (5/5) = {len(complete_group_keys)} | "
          f"incomplete/duplicate (excluded) = {len(incomplete_report)}")
    for r in incomplete_report[:15]:
        print(f"    {r['group_key']}: {r['n_images']} images, variants={r['variants_present']}, "
              f"duplicates={r['duplicates']}")

    assert len(complete_group_keys) > 0, (
        "[ERROR] No complete (5-variant) prompt groups found in synthetic data."
    )

    train_groups, temp_groups = train_test_split(
        complete_group_keys, test_size=test_size, random_state=seed, shuffle=True
    )
    val_groups, test_groups = train_test_split(
        temp_groups, test_size=0.50, random_state=seed, shuffle=True
    )
    train_groups_set, val_groups_set, test_groups_set = set(train_groups), set(val_groups), set(test_groups)

    synth_train_idx = df_synth.index[df_synth["group_key"].isin(train_groups_set)].tolist()
    synth_val_idx = df_synth.index[df_synth["group_key"].isin(val_groups_set)].tolist()
    synth_test_idx = df_synth.index[df_synth["group_key"].isin(test_groups_set)].tolist()

    # ── Skin Cancer: PLAIN per-subtype split (no stratify) ──
    df_cancer = df_clean[df_clean["file_kind"] == "cancer"]
    cancer_train_idx, cancer_val_idx, cancer_test_idx = [], [], []
    cancer_report = {}
    for subtype in sorted(df_cancer["subtype"].unique()):
        sub = df_cancer[df_cancer["subtype"] == subtype]
        idx_all = sub.index.to_numpy()
        if len(idx_all) < 3:
            cancer_train_idx.extend(idx_all)
            cancer_report[subtype] = (len(idx_all), 0, 0)
            continue
        idx_train, idx_temp = train_test_split(idx_all, test_size=test_size, random_state=seed)
        idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=seed)
        cancer_train_idx.extend(idx_train)
        cancer_val_idx.extend(idx_val)
        cancer_test_idx.extend(idx_test)
        cancer_report[subtype] = (len(idx_train), len(idx_val), len(idx_test))

    train_idx = list(synth_train_idx) + list(cancer_train_idx)
    val_idx = list(synth_val_idx) + list(cancer_val_idx)
    test_idx = list(synth_test_idx) + list(cancer_test_idx)

    # Leakage checks — path level and prompt-group level.
    set_train, set_val, set_test = set(df_clean.loc[train_idx, "path"]), \
        set(df_clean.loc[val_idx, "path"]), set(df_clean.loc[test_idx, "path"])
    assert not (set_train & set_val), "[ERROR] Leakage: synthetic train ∩ val is non-empty!"
    assert not (set_train & set_test), "[ERROR] Leakage: synthetic train ∩ test is non-empty!"
    assert not (set_val & set_test), "[ERROR] Leakage: synthetic val ∩ test is non-empty!"
    assert not (train_groups_set & val_groups_set), "[ERROR] A prompt_id is in both train and val!"
    assert not (train_groups_set & test_groups_set), "[ERROR] A prompt_id is in both train and test!"
    assert not (val_groups_set & test_groups_set), "[ERROR] A prompt_id is in both val and test!"

    # Sanity check: every complete prompt group landed ENTIRELY in exactly one
    # partition (all 5 of its images), never split across partitions.
    train_set_i, val_set_i, test_set_i = set(train_idx), set(val_idx), set(test_idx)
    for gk in complete_group_keys:
        members = df_synth.index[df_synth["group_key"] == gk]
        n_tr = sum(i in train_set_i for i in members)
        n_va = sum(i in val_set_i for i in members)
        n_te = sum(i in test_set_i for i in members)
        buckets_used = sum(n > 0 for n in (n_tr, n_va, n_te))
        assert buckets_used == 1 and (n_tr + n_va + n_te) == 5, \
            f"[ERROR] Prompt group {gk} was split across partitions!"
    # Skin Cancer per-subtype allocation validation.
    for subtype, (ntr, nva, nte) in cancer_report.items():
        n_total = int((df_cancer["subtype"] == subtype).sum())
        assert n_total < 3 or (ntr + nva + nte) == n_total, \
            f"[ERROR] Skin Cancer subtype {subtype}: allocation does not add up!"
    for subtype in ["BCC", "SCC", "MEL"]:
        if subtype in cancer_report:
            ntr, nva, nte = cancer_report[subtype]
            print(f"  Skin Cancer {subtype} -> Train: {ntr} | Val: {nva} | Test: {nte}")

    train_df = df_clean.loc[train_idx, ["path", "label", "class"]].reset_index(drop=True)
    val_df = df_clean.loc[val_idx, ["path", "label", "class"]].reset_index(drop=True)
    test_df = df_clean.loc[test_idx, ["path", "label", "class"]].reset_index(drop=True)

    print(f"  Synthetic Train (raw, pre-balance) : {len(train_df)} images")
    print(f"  Synthetic Validation               : {len(val_df)} images")
    print(f"  Synthetic Test                     : {len(test_df)} images")
    return train_df, val_df, test_df


def mix_train_pools(
    train_df_synth: pd.DataFrame,
    train_df_real: pd.DataFrame,
    synth_fraction: float = 1.0,
    real_fraction: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build the final combined TRAIN pool with two INDEPENDENT fraction knobs
    (defaults 1.0 / 1.0 = the notebook's 'use 100% of both sources' behaviour):
    per class, take synth_fraction of that class's own SYNTHETIC train pool
    and real_fraction of that class's own REAL train pool, then union the two
    subsets. Each fraction is applied to its own source's pool size only —
    there is no shared ratio constraint between them (e.g. synth_fraction=0.5,
    real_fraction=0.5 simply keeps half of each pool, regardless of how large
    the synthetic pool is relative to the real one).

    Only the TRAIN split is mixed — val/test must be built separately as the
    full union of both sources' val/test partitions (never fractioned), so
    evaluation always reflects the true combined distribution. Sampling is
    done per class, deterministically (seed), so each source's own class
    balance is preserved within its fractioned subset.
    """
    assert 0.0 < synth_fraction <= 1.0, "[ERROR] synth_fraction must be in (0, 1]."
    assert 0.0 < real_fraction <= 1.0, "[ERROR] real_fraction must be in (0, 1]."

    class_names = sorted(set(train_df_synth["class"]) | set(train_df_real["class"]))
    parts, mix_report = [], []
    for cls in class_names:
        synth_pool = train_df_synth[train_df_synth["class"] == cls]
        real_pool = train_df_real[train_df_real["class"] == cls]
        n_synth_avail, n_real_avail = len(synth_pool), len(real_pool)

        # round() + max(1, ...) ensures a class with a nonzero pool never gets
        # zeroed out entirely by a small fraction.
        n_synth_take = min(n_synth_avail, max(1, int(round(n_synth_avail * synth_fraction)))) if n_synth_avail > 0 else 0
        n_real_take = min(n_real_avail, max(1, int(round(n_real_avail * real_fraction)))) if n_real_avail > 0 else 0

        synth_sel = synth_pool.sample(n=n_synth_take, random_state=seed) if n_synth_take > 0 else synth_pool.iloc[0:0]
        real_sel = real_pool.sample(n=n_real_take, random_state=seed) if n_real_take > 0 else real_pool.iloc[0:0]
        parts.extend([synth_sel, real_sel])

        mix_report.append({
            "class": cls,
            "synthetic_available": n_synth_avail, "real_available": n_real_avail,
            "synthetic_used": n_synth_take, "real_used": n_real_take,
            "achieved_real_%": round(100 * n_real_take / max(1, n_synth_take + n_real_take), 1),
        })

    combined = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  TRAIN MIX (synth_fraction={synth_fraction:.0%}, real_fraction={real_fraction:.0%})")
    print(pd.DataFrame(mix_report).to_string(index=False))
    print(f"  Combined TRAIN size : {len(combined)}")
    return combined


def balance_dataset(df: pd.DataFrame, target: int, max_cap: int, seed: int = 42) -> pd.DataFrame:
    """
    Oversample/cap class distribution for training split only.
    Leaves validation and test sets untouched in real-world distribution.
    """
    parts = []
    for cls_name, grp in df.groupby("class"):
        n = len(grp)
        if n < target:
            extra = grp.sample(target - n, replace=True, random_state=seed)
            parts.append(pd.concat([grp, extra]))
        elif n > max_cap:
            parts.append(grp.sample(max_cap, random_state=seed))
        else:
            parts.append(grp)
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
