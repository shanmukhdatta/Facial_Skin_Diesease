"""Single entry point that builds train/val/test exactly as the notebook does.

Shared by run_training.py, run_evaluation.py and run_inference.py so all three
see the *same* held-out sets (notebook Sections 4 -> 6d, 7).

config.mode selects the data source:
  "combined"  synthetic + real (notebook behaviour, default)
  "synthetic" synthetic dataset only
  "real"      real dataset only
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.data.dataset import (
    discover_dataset, clean_dataset, split_dataset,
    split_real_grouped, split_synthetic_grouped, mix_train_pools,
)
from src.data.audit import audit_and_resolve_md5, cross_source_audit

_EMPTY_COLS = ["path", "label", "class"]


def use_combined_pipeline(config) -> bool:
    """False -> legacy single-source stratified split on config.data_dir."""
    return config.real_fraction is not None


def required_dirs(config) -> List[Path]:
    """Dataset roots that must exist for the selected mode."""
    if not use_combined_pipeline(config):
        return [Path(config.data_dir)]
    if config.mode == "synthetic":
        return [Path(config.synthetic_data_dir)]
    if config.mode == "real":
        return [Path(config.real_data_dir)]
    return [Path(config.synthetic_data_dir), Path(config.real_data_dir)]


def _real_split(config, df_clean_real):
    return split_real_grouped(
        df_clean_real, test_size=config.test_size, seed=config.seed,
        phash_size=config.phash_size, dup_hamming_threshold=config.dup_hamming_threshold,
        tau_start=config.embed_tau_start, tau_step=config.embed_tau_step, tau_max=config.embed_tau_max,
        max_group_share=config.max_group_share, n_tries=config.group_split_tries,
        save_dir=config.save_dir, split_csv_override=config.split_csv_override,
    )


def build_splits(config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Return (train_df_raw, val_df, test_df, class_names)."""
    if not use_combined_pipeline(config):
        # Legacy single-source pipeline (unchanged).
        df_full, class_names = discover_dataset(config.data_dir)
        df_clean = clean_dataset(df_full)
        train_df_raw, val_df, test_df = split_dataset(df_clean, test_size=config.test_size, seed=config.seed)
        return train_df_raw, val_df, test_df, class_names

    mode = config.mode
    print(f"[INFO] Data mode: {mode}")
    synth_splits = real_splits = None

    if mode in ("combined", "synthetic"):
        df_full_synth, class_names = discover_dataset(config.synthetic_data_dir)
    if mode in ("combined", "real"):
        df_full_real, class_names_real = discover_dataset(config.real_data_dir)
        if mode == "real":
            class_names = class_names_real
    if mode == "combined":
        assert class_names == class_names_real, (
            f"[ERROR] Class name mismatch between datasets!\n"
            f"  Synthetic : {class_names}\n  Real      : {class_names_real}"
        )
    config.num_classes = len(class_names)

    # Clean corrupt files, independently per source; then each source's own group split.
    if mode in ("combined", "synthetic"):
        df_clean_synth = clean_dataset(df_full_synth)
        s_tr, s_va, s_te = split_synthetic_grouped(df_clean_synth, test_size=config.test_size, seed=config.seed)
        s_tr = s_tr.assign(source="synthetic")
        s_va = s_va.assign(source="synthetic")
        s_te = s_te.assign(source="synthetic")
        synth_splits = (s_tr, s_va, s_te)

    if mode in ("combined", "real"):
        df_clean_real = clean_dataset(df_full_real)
        r_tr, r_va, r_te = _real_split(config, df_clean_real)
        r_tr = r_tr.assign(source="real")
        r_va = r_va.assign(source="real")
        r_te = r_te.assign(source="real")
        real_splits = (r_tr, r_va, r_te)

    empty = pd.DataFrame(columns=_EMPTY_COLS + ["source"])
    s_tr, s_va, s_te = synth_splits if synth_splits else (empty, empty, empty)
    r_tr, r_va, r_te = real_splits if real_splits else (empty, empty, empty)

    # Train = fractions of whichever sources are active; val/test = full union of active sources.
    train_df_raw = mix_train_pools(
        s_tr, r_tr,
        synth_fraction=config.synth_fraction if synth_splits else 1.0,
        real_fraction=config.real_fraction if real_splits else 1.0,
        seed=config.seed,
    )
    val_df = pd.concat([s_va, r_va], ignore_index=True)
    test_df = pd.concat([s_te, r_te], ignore_index=True)
    print(f"  Combined TRAIN {len(train_df_raw)} | VAL {len(val_df)} | TEST {len(test_df)}")

    # Final leakage audit (path + MD5, auto-resolve) - runs in every mode.
    train_df_raw, val_df, test_df = audit_and_resolve_md5(
        train_df_raw, val_df, test_df, save_dir=config.save_dir,
        strict=config.strict_md5_assert, auto_resolve=config.auto_resolve_md5_dupes,
    )

    # Cross-source near-duplicate audit only makes sense when both sources are present.
    if mode == "combined" and config.run_cross_source_audit:
        cross_source_audit(synth_splits, real_splits, class_names, seed=config.seed, save_dir=config.save_dir)
    else:
        print("[INFO] Cross-source audit skipped.")

    return train_df_raw, val_df, test_df, class_names
