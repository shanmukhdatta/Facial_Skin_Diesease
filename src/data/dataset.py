from __future__ import annotations
import re
from pathlib import Path
from typing import List, Optional, Tuple
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

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
split_real_stratified = split_dataset


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
    real_fraction: float = 0.50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build the final combined TRAIN pool by mixing, PER CLASS, real_fraction of
    real images with (1 - real_fraction) of synthetic images. Only the TRAIN
    split is mixed — val/test must be built separately as the full union of
    both sources' val/test partitions (never fractioned), so evaluation always
    reflects the true combined distribution. No oversampling/duplication is
    done here — the largest achievable total respecting both the requested
    ratio and each source's real availability is used per class.
    """
    synthetic_fraction = 1.0 - real_fraction
    assert 0.0 <= real_fraction <= 1.0, "[ERROR] real_fraction must be between 0 and 1."

    class_names = sorted(set(train_df_synth["class"]) | set(train_df_real["class"]))
    parts = []
    for cls in class_names:
        synth_pool = train_df_synth[train_df_synth["class"] == cls]
        real_pool = train_df_real[train_df_real["class"] == cls]
        n_synth_avail, n_real_avail = len(synth_pool), len(real_pool)

        if real_fraction == 0:
            n_synth_take, n_real_take = n_synth_avail, 0
        elif synthetic_fraction == 0:
            n_synth_take, n_real_take = 0, n_real_avail
        else:
            max_total_by_synth = n_synth_avail / synthetic_fraction
            max_total_by_real = n_real_avail / real_fraction
            total = min(max_total_by_synth, max_total_by_real)
            n_synth_take = min(n_synth_avail, int(round(total * synthetic_fraction)))
            n_real_take = min(n_real_avail, int(round(total * real_fraction)))

        synth_sel = synth_pool.sample(n=n_synth_take, random_state=seed) if n_synth_take > 0 else synth_pool.iloc[0:0]
        real_sel = real_pool.sample(n=n_real_take, random_state=seed) if n_real_take > 0 else real_pool.iloc[0:0]
        parts.extend([synth_sel, real_sel])

    combined = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  Combined TRAIN size : {len(combined)} "
          f"(target ratio {synthetic_fraction:.0%} synthetic / {real_fraction:.0%} real per class)")
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
