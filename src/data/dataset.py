from __future__ import annotations
import re
from pathlib import Path
from typing import List, Optional, Tuple
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

from PIL import Image as PILImage

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


def split_real_grouped(
    df_clean_real: pd.DataFrame,
    test_size: float = 0.30,
    seed: int = 42,
    phash_size: int = 8,
    dup_hamming_threshold: int = 5,
    n_bands: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the real dataset with a perceptual-hash near-duplicate GROUP split,
    replacing the old plain per-class stratified split (split_real_stratified).

    A plain stratified split does not protect against near-identical images
    (exact duplicates, resaves, minor recompressions) ending up on both sides
    of the train/test boundary, which leaks information and inflates held-out
    metrics. This mirrors the same "group, don't split" principle used for the
    synthetic prompt groups in split_synthetic_grouped(), but the group key is
    a perceptual-hash near-duplicate cluster instead of a filename-derived
    prompt ID:
      1. Compute a perceptual hash (imagehash.phash) for every real image.
      2. Cluster near-duplicates (Hamming distance <= dup_hamming_threshold)
         with a Union-Find over an LSH-banded candidate index (avoids the
         full O(n^2) pairwise comparison). n_bands must be strictly greater
         than dup_hamming_threshold so no true near-duplicate pair is missed
         (pigeonhole principle).
      3. Every image in a duplicate cluster becomes one indivisible group.
      4. GroupShuffleSplit assigns each entire cluster to exactly ONE
         partition (70/15/15 by default) — no duplicate cluster can appear
         in more than one partition.

    Trade-off: GroupShuffleSplit splits on groups and therefore cannot
    stratify on the label, so per-class proportions across partitions are
    approximate rather than exact.
    """
    if GroupShuffleSplit is None:
        raise ImportError("scikit-learn is required for split_real_grouped().")
    if imagehash is None:
        raise ImportError("ImageHash is required for split_real_grouped() "
                           "(pip install ImageHash).")

    n_bits_total = phash_size * phash_size
    band_bits = n_bits_total // n_bands
    assert n_bands > dup_hamming_threshold, (
        "[ERROR] n_bands must exceed dup_hamming_threshold or the LSH index "
        "can miss near-duplicates."
    )

    def _compute_phash(path: str):
        with PILImage.open(path) as img:
            return imagehash.phash(img.convert("RGB"), hash_size=phash_size)

    print("[INFO] Computing perceptual hashes for REAL dataset images...")
    real_paths_arr = df_clean_real["path"].values
    real_phashes = [_compute_phash(p) for p in real_paths_arr]
    hash_ints = [int(str(h), 16) for h in real_phashes]  # plain Python ints — a
    # 64-bit phash can exceed np.int64 range at phash_size=8.

    # ── LSH banding: bucket images that share at least one band ──
    band_buckets = [dict() for _ in range(n_bands)]
    for idx, hv in enumerate(hash_ints):
        hv_int = int(hv) & ((1 << n_bits_total) - 1)
        for b in range(n_bands):
            band_val = (hv_int >> (band_bits * b)) & ((1 << band_bits) - 1)
            band_buckets[b].setdefault(band_val, []).append(idx)

    uf = _UnionFind(len(hash_ints))

    def _hamming(a: int, b: int) -> int:
        return bin(int(a) ^ int(b)).count("1")

    for b in range(n_bands):
        for idxs in band_buckets[b].values():
            if len(idxs) < 2:
                continue
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a_idx, b_idx = idxs[i], idxs[j]
                    if uf.find(a_idx) == uf.find(b_idx):
                        continue
                    if _hamming(hash_ints[a_idx], hash_ints[b_idx]) <= dup_hamming_threshold:
                        uf.union(a_idx, b_idx)

    dup_group_ids = [uf.find(i) for i in range(len(hash_ints))]
    df_clean_real = df_clean_real.copy()
    df_clean_real["dup_group_id"] = dup_group_ids

    n_unique_groups = len(set(dup_group_ids))
    cluster_sizes = pd.Series(dup_group_ids).value_counts()
    n_dup_clusters = int((cluster_sizes > 1).sum())
    n_images_in_clusters = int(cluster_sizes[cluster_sizes > 1].sum())
    print(f"  Real images                   : {len(df_clean_real)}")
    print(f"  Unique near-duplicate groups  : {n_unique_groups}")
    print(f"  Duplicate clusters (size > 1) : {n_dup_clusters}")
    print(f"  Images inside those clusters  : {n_images_in_clusters}")
    print(f"  Largest cluster                : {int(cluster_sizes.max())} images")

    # ── Diagnostic: clusters whose images do NOT all share one class label ──
    # A near-duplicate cluster spanning two classes means the same (or a
    # visually identical) image is filed under two different diagnoses —
    # label noise worth reporting in the paper. The split is unaffected: the
    # whole cluster still travels into exactly one partition.
    labels_per_group = df_clean_real.groupby("dup_group_id")["class"].nunique()
    cross_class_groups = labels_per_group[labels_per_group > 1]
    if len(cross_class_groups) > 0:
        print(f"[WARN] {len(cross_class_groups)} near-duplicate cluster(s) span more than one class label:")
        for gid in cross_class_groups.index[:10]:
            clss = sorted(df_clean_real.loc[df_clean_real["dup_group_id"] == gid, "class"].unique())
            print(f"    cluster {gid}: {clss}")
    else:
        print("  Every near-duplicate cluster is label-consistent (no cross-class duplicates).")

    X_all_real = df_clean_real["path"].values
    y_all_real = df_clean_real["label"].values
    groups_all_real = df_clean_real["dup_group_id"].values

    gss_temp = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx_real, temp_idx_real = next(gss_temp.split(X_all_real, y_all_real, groups=groups_all_real))

    X_temp_real = X_all_real[temp_idx_real]
    y_temp_real = y_all_real[temp_idx_real]
    groups_temp_real = groups_all_real[temp_idx_real]

    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx_real, test_idx_real = next(gss_val_test.split(X_temp_real, y_temp_real, groups=groups_temp_real))

    X_train_real, y_train_real = X_all_real[train_idx_real], y_all_real[train_idx_real]
    X_val_real, y_val_real = X_temp_real[val_idx_real], y_temp_real[val_idx_real]
    X_test_real, y_test_real = X_temp_real[test_idx_real], y_temp_real[test_idx_real]

    # ── Leakage validation: no image path and no duplicate-cluster ID crosses
    # a partition boundary. ──
    set_train, set_val, set_test = set(X_train_real), set(X_val_real), set(X_test_real)
    assert not (set_train & set_val), "[ERROR] Leakage: real train ∩ val is non-empty!"
    assert not (set_train & set_test), "[ERROR] Leakage: real train ∩ test is non-empty!"
    assert not (set_val & set_test), "[ERROR] Leakage: real val ∩ test is non-empty!"

    groups_train_set = set(groups_all_real[train_idx_real])
    groups_val_set = set(groups_temp_real[val_idx_real])
    groups_test_set = set(groups_temp_real[test_idx_real])
    assert not (groups_train_set & groups_val_set), "[ERROR] A duplicate cluster is in both train and val!"
    assert not (groups_train_set & groups_test_set), "[ERROR] A duplicate cluster is in both train and test!"
    assert not (groups_val_set & groups_test_set), "[ERROR] A duplicate cluster is in both val and test!"

    # ── Leakage validation #3: no near-duplicate PAIR straddles a partition ──
    # Independent re-derivation of validation #2 above: confirms that no two
    # images within Hamming distance of each other end up in different
    # partitions, by re-deriving each image's partition directly from its path.
    partition_of = {}
    for p in X_train_real:
        partition_of[p] = "train"
    for p in X_val_real:
        partition_of[p] = "val"
    for p in X_test_real:
        partition_of[p] = "test"
    cross_partition_pairs = 0
    for gid, grp in df_clean_real.groupby("dup_group_id"):
        if len(grp) < 2:
            continue
        if len({partition_of[p] for p in grp["path"]}) > 1:
            cross_partition_pairs += 1
    assert cross_partition_pairs == 0, (
        f"[ERROR] {cross_partition_pairs} near-duplicate cluster(s) were split across partitions!"
    )

    classes_map = dict(zip(df_clean_real["label"], df_clean_real["class"]))
    train_df = pd.DataFrame({
        "path": X_train_real, "label": y_train_real,
        "class": [classes_map[l] for l in y_train_real],
    })
    val_df = pd.DataFrame({
        "path": X_val_real, "label": y_val_real,
        "class": [classes_map[l] for l in y_val_real],
    })
    test_df = pd.DataFrame({
        "path": X_test_real, "label": y_test_real,
        "class": [classes_map[l] for l in y_test_real],
    })

    print(f"  Real Train (raw, pre-balance) : {len(train_df)} images "
          f"({len(groups_train_set)} groups)")
    print(f"  Real Validation               : {len(val_df)} images "
          f"({len(groups_val_set)} groups)")
    print(f"  Real Test                     : {len(test_df)} images "
          f"({len(groups_test_set)} groups)")

    # Per-class breakdown across the three partitions (diagnostic only).
    real_split_df = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ])
    print("  Per-class counts:")
    print(real_split_df.groupby(["class", "split"]).size().unstack(fill_value=0).to_string())

    return train_df, val_df, test_df


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
    complete_group_keys = []
    for gk in sorted(df_synth["group_key"].unique()):
        grp = df_synth[df_synth["group_key"] == gk]
        variants_present = sorted(int(v) for v in grp["variant"])
        if len(grp) == 5 and len(set(variants_present)) == 5:
            complete_group_keys.append(gk)
    complete_group_keys = sorted(complete_group_keys)

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
    for subtype in sorted(df_cancer["subtype"].unique()):
        sub = df_cancer[df_cancer["subtype"] == subtype]
        idx_all = sub.index.to_numpy()
        if len(idx_all) < 3:
            cancer_train_idx.extend(idx_all)
            continue
        idx_train, idx_temp = train_test_split(idx_all, test_size=test_size, random_state=seed)
        idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=seed)
        cancer_train_idx.extend(idx_train)
        cancer_val_idx.extend(idx_val)
        cancer_test_idx.extend(idx_test)

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
    synth_fraction: float = 0.50,
    real_fraction: float = 0.50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build the final combined TRAIN pool with two INDEPENDENT fraction knobs:
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
