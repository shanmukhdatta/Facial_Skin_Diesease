from pathlib import Path
from typing import List, Tuple
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
