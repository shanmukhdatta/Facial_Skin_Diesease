"""CPU-level research protocol checks. No patient images or GPU required."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from src.data.dataset import normalize_class_folder, split_synthetic_grouped
from src.data.audit import audit_and_resolve_md5

def _img(p, color=(11, 22, 33)):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(p)
    return str(p)

def test_folder_normalization():
    assert normalize_class_folder("Fungal Infection") == "Fungal_Infection"
    assert normalize_class_folder("Hyper_Pigmentation") == "Hyperpigmentation"
    assert normalize_class_folder("Skin cancer") == "Skin_Cancer"
    assert normalize_class_folder("Normal Skin") == "Normal_Skin"

def test_prompt_variant_isolation_and_freeze(tmp_path):
    records = []
    for prompt in range(1, 21):
        for variant in range(1, 6):
            p = tmp_path / "Acne" / f"cls01_p_{prompt:03d}_v_{variant}.jpg"
            records.append({"path": _img(p, (prompt, variant, 90)), "label": 0, "class": "Acne"})
    frame = pd.DataFrame(records)
    a = split_synthetic_grouped(frame, seed=42, save_dir=tmp_path)
    manifest = tmp_path / "synthetic_frozen_v1.csv"
    assert manifest.exists()
    b = split_synthetic_grouped(frame, seed=123, split_csv_override=str(manifest))
    for x, y in zip(a, b):
        assert set(x.path) == set(y.path)
    partitions = {}
    for i, part in enumerate(a):
        for p in part.path:
            key = Path(p).stem.split("_v_")[0]
            partitions.setdefault(key, set()).add(i)
    assert all(len(s) == 1 for s in partitions.values())

def test_md5_full_pool_counts_and_stable_test(tmp_path):
    x = _img(tmp_path / "real" / "a.png", (3, 4, 5))
    y = _img(tmp_path / "synthetic" / "b.png", (3, 4, 5))
    z = _img(tmp_path / "real" / "c.png", (9, 4, 5))
    def f(path, source):
        return {"path": path, "source": source, "class": "Acne", "label": 0}
    tr = pd.DataFrame([f(x, "real")])
    va = pd.DataFrame([f(z, "real")])
    te = pd.DataFrame([f(y, "synthetic")])
    new_tr, new_val, new_test = audit_and_resolve_md5(tr, va, te, save_dir=tmp_path)
    assert not len(new_tr) and len(new_val) == len(new_test) == 1
    totals = pd.read_csv(tmp_path / "md5_source_totals.csv").set_index("source")
    assert totals.loc["real", "removed"] == 1

def test_reuse_real_e2_portable_manifest(tmp_path):
    from hashlib import sha256
    from src.data.dataset import _try_load_saved_split
    a = _img(tmp_path / "Normal_Skin" / "one.jpg", (1, 2, 3))
    b = _img(tmp_path / "Skin_cancer" / "two.jpg", (4, 5, 6))
    real = pd.DataFrame([{"path": a}, {"path": b}])
    manifest = pd.DataFrame([
        {"relative_path": "Normal_Skin/one.jpg", "sha256": sha256(Path(a).read_bytes()).hexdigest(),
         "split": "test", "group": "g0", "class": "Normal_Skin"},
        {"relative_path": "Skin_cancer/two.jpg", "sha256": sha256(Path(b).read_bytes()).hexdigest(),
         "split": "train", "group": "g1", "class": "Skin_Cancer"},
    ])
    fn = tmp_path / "splits_v3.csv"
    manifest.to_csv(fn, index=False)
    reused = _try_load_saved_split(real, str(fn))
    assert list(reused["split"]) == ["test", "train"]
    _img(Path(a), (100, 20, 30))
    try:
        _try_load_saved_split(real, str(fn))
    except ValueError:
        pass
    else:
        raise AssertionError("Altered image contents must invalidate the split")
