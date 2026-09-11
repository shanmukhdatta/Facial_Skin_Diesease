import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import ProjectConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Saved Models and Run Weighted Ensemble.")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset directory.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory with saved .keras models.")
    parser.add_argument("--plot_dir", type=str, default=None, help="Directory to save evaluation plots.")
    parser.add_argument("--label_map", type=str, default=None, help="Path to label_map.json.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import pandas as pd
        from src.data.dataset import discover_dataset, clean_dataset, split_dataset
        from src.data.pipeline import make_dataset
        from src.data.utils import load_label_map
        from src.evaluation.ensemble import WeightedEnsemble
        from src.evaluation.metrics import evaluate_model
        from src.evaluation.visualization import (
            plot_comparison_dashboard,
            plot_radar_chart,
        )
    except ImportError as e:
        print(f"[ERROR] Required ML dependency missing: {e}")
        print("        Please ensure your evaluation environment has requirements installed:")
        print("        pip install -r requirements.txt")
        sys.exit(1)

    config = ProjectConfig()

    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.save_dir:
        config.save_dir = Path(args.save_dir)
    if args.plot_dir:
        config.plot_dir = Path(args.plot_dir)

    label_map_path = Path(args.label_map) if args.label_map else config.save_dir / "label_map.json"
    assert label_map_path.exists(), f"Label map not found: {label_map_path}"
    label_map = load_label_map(label_map_path)
    class_names = [label_map[i] for i in sorted(label_map.keys())]

    print("======================================================================")
    print("[INFO] Evaluating Checkpoints & Ensemble Voting")
    print(f"Save Directory: {config.save_dir}")
    print(f"Classes: {class_names}")
    print("======================================================================")

    # Re-create val/test splits (val is needed for ensemble weighting, test for final scoring)
    df_full, _ = discover_dataset(config.data_dir)
    df_clean = clean_dataset(df_full)
    _, val_df, test_df = split_dataset(df_clean, test_size=config.test_size, seed=config.seed)

    val_ds = make_dataset(
        paths=val_df["path"].values,
        labels=val_df["label"].values,
        img_size=config.img_size,
        batch_size=config.batch_size,
        augment=False,
        shuffle=False,
    )

    test_ds = make_dataset(
        paths=test_df["path"].values,
        labels=test_df["label"].values,
        img_size=config.img_size,
        batch_size=config.batch_size,
        augment=False,
        shuffle=False,
    )

    ensemble = WeightedEnsemble(
        save_dir=config.save_dir,
        plot_dir=config.plot_dir,
        class_names=class_names,
    )
    n_loaded = ensemble.load_models()
    print(f"[INFO] Successfully loaded {n_loaded} models for ensemble.")

    if n_loaded < 2:
        print("[WARN] At least 2 models are needed for ensemble evaluation.")
        return

    ens_results = ensemble.evaluate_ensemble(val_ds, test_ds)
    if ens_results:
        metrics_dict, _, _ = ens_results
        comp_csv = config.save_dir / "model_comparison.csv"
        if comp_csv.exists():
            df_compare = pd.read_csv(comp_csv)
            ens_row = pd.DataFrame([metrics_dict])
            df_final = pd.concat([df_compare, ens_row], ignore_index=True).sort_values("Accuracy", ascending=False)
        else:
            df_final = pd.DataFrame([metrics_dict])

        out_csv = config.save_dir / "model_comparison_with_ensemble.csv"
        df_final.to_csv(out_csv, index=False)
        print("\n[INFO] Final Comparison Table:")
        print(df_final.to_string(index=False))

        plot_comparison_dashboard(df_final, config.plot_dir)
        plot_radar_chart(df_final, config.plot_dir)
        print(f"\n[OK] Saved final comparison table to {out_csv}")


if __name__ == "__main__":
    main()