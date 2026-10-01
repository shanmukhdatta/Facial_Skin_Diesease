import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import ProjectConfig

MODEL_SPECS = {
    "EfficientNetV2B0": {"unfreeze_p2": 100, "unfreeze_p3": 200, "is_vit": False},
    "ResNet50": {"unfreeze_p2": 50, "unfreeze_p3": 100, "is_vit": False},
    "DenseNet121": {"unfreeze_p2": 60, "unfreeze_p3": 120, "is_vit": False},
    "MobileNetV3Large": {"unfreeze_p2": 25, "unfreeze_p3": 60, "is_vit": False},
    "ViT_B16": {"unfreeze_p2": 20, "unfreeze_p3": -1, "is_vit": True},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Face Skin Disease Classification Models.")
    parser.add_argument("--data_dir", type=str, default=None,
                         help="Path to a single dataset dir (legacy single-source mode).")
    parser.add_argument("--synthetic_data_dir", type=str, default=None,
                         help="Path to synthetic dataset root (combined mode).")
    parser.add_argument("--real_data_dir", type=str, default=None,
                         help="Path to real dataset root (combined mode).")
    parser.add_argument("--synth_fraction", type=float, default=None,
                         help="Fraction of each class's own SYNTHETIC train pool to keep "
                              "(0-1, default 1.0 = use all, as in the notebook).")
    parser.add_argument("--real_fraction", type=float, default=None,
                         help="Fraction of each class's own REAL train pool to keep "
                              "(0-1, default 1.0 = use all, as in the notebook).")
    parser.add_argument("--mode", type=str, default=None, choices=["combined", "synthetic", "real"],
                         help="Data source: combined (synthetic+real, default), synthetic only, or real only.")
    parser.add_argument("--single_source", action="store_true",
                         help="Skip the combined synthetic+real pipeline and use the legacy "
                              "single-source stratified split on --data_dir.")
    parser.add_argument("--split_csv", type=str, default=None,
                         help="Path to a saved splits_v2.csv (path, split, group) for the real data.")
    parser.add_argument("--skip_cross_source_audit", action="store_true",
                         help="Skip the report-only real<->synthetic near-duplicate audit (6d-2).")
    parser.add_argument("--rebalance_train", action="store_true",
                         help="Use bounded resampling (balance_dataset) instead of all train images.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "EfficientNetV2B0", "ResNet50", "DenseNet121", "MobileNetV3Large", "ViT_B16"],
        help="Which model to train or 'all'.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs_p1", type=int, default=15, help="Epochs for Phase 1 (warmup).")
    parser.add_argument("--epochs_p2", type=int, default=25, help="Epochs for Phase 2 (partial unfreeze).")
    parser.add_argument("--epochs_p3", type=int, default=30, help="Epochs for Phase 3 (full fine-tune).")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save models.")
    parser.add_argument("--plot_dir", type=str, default=None, help="Directory to save plots.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import pandas as pd
        from src.data.dataset import balance_dataset
        from src.data.splits import build_splits
        from src.data.pipeline import make_dataset
        from src.data.utils import compute_class_weights, save_label_map
        from src.models.backbones import get_backbone
        from src.training.trainer import PhasedTrainer
        from src.evaluation.metrics import evaluate_model
        from src.evaluation.visualization import (
            plot_class_distribution,
            plot_sample_images,
            plot_training_history,
            plot_confusion_matrix,
            plot_roc_curves,
            plot_per_class_metrics,
            plot_comparison_dashboard,
            plot_radar_chart,
        )
    except ImportError as e:
        print(f"[ERROR] Required ML dependency missing: {e}")
        print("        Please ensure your training environment has requirements installed:")
        print("        pip install -r requirements.txt")
        sys.exit(1)

    config = ProjectConfig()

    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.synthetic_data_dir:
        config.synthetic_data_dir = Path(args.synthetic_data_dir)
    if args.real_data_dir:
        config.real_data_dir = Path(args.real_data_dir)
    if args.synth_fraction is not None:
        config.synth_fraction = args.synth_fraction
    if args.real_fraction is not None:
        config.real_fraction = args.real_fraction
    if args.mode:
        config.mode = args.mode
    if args.single_source:
        config.real_fraction = None
    if args.split_csv:
        config.split_csv_override = args.split_csv
    if args.skip_cross_source_audit:
        config.run_cross_source_audit = False
    if args.rebalance_train:
        config.use_all_train_images = False
    if args.save_dir:
        config.save_dir = Path(args.save_dir)
    if args.plot_dir:
        config.plot_dir = Path(args.plot_dir)

    config.batch_size = args.batch_size
    config.epochs_phase1 = args.epochs_p1
    config.epochs_phase2 = args.epochs_p2
    config.epochs_phase3 = args.epochs_p3

    print("======================================================================")
    print("[INFO] Starting Face Skin Disease Classification Training Pipeline")
    print(f"Data mode: {'single-source (legacy)' if config.real_fraction is None else config.mode}")
    print(f"Data Dir: {config.data_dir}")
    print(f"Save Dir: {config.save_dir}")
    print(f"Plot Dir: {config.plot_dir}")
    print("======================================================================")

    # 1-6d. Discover -> clean -> synthetic prompt-group split + real flip-aware group split
    #        -> combine -> MD5 / cross-source leakage audit (identical to the notebook).
    #        Legacy single-source mode is handled inside build_splits().
    train_df_raw, val_df, test_df, class_names = build_splits(config)
    config.num_classes = len(class_names)

    # 7. Training set: use every combined train image (default) or bounded resampling
    if config.use_all_train_images:
        # 100% of both sources' train partitions: nothing dropped, nothing duplicated.
        # Imbalance is handled by the class-weighted loss.
        df_balanced = train_df_raw.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    else:
        train_target = int(round(train_df_raw["class"].value_counts().mean()))
        df_balanced = balance_dataset(
            train_df_raw, target=train_target, max_cap=train_target, seed=config.seed
        )
    plot_class_distribution(train_df_raw, df_balanced, config.plot_dir)

    # 5. Label Map & Class Weights
    save_label_map(class_names, config.save_dir / "label_map.json")
    class_weights = compute_class_weights(df_balanced["label"].values, config.num_classes)
    print(f"Calculated class weights: {class_weights}")

    # 6. Build tf.data Pipelines
    train_ds = make_dataset(
        paths=df_balanced["path"].values,
        labels=df_balanced["label"].values,
        img_size=config.img_size,
        batch_size=config.batch_size,
        augment=True,
        shuffle=True,
        seed=config.seed,
    )
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

    # Plot sample augmented images (Cell 16)
    try:
        plot_sample_images(train_ds, class_names, config.plot_dir)
        print("[OK] Sample images plotted to sample_images.png")
    except Exception as e:
        print(f"[WARN] Could not plot sample images: {e}")

    trainer = PhasedTrainer(config)
    models_to_train = list(MODEL_SPECS.keys()) if args.model == "all" else [args.model]

    all_metrics = []
    for model_name in models_to_train:
        spec = MODEL_SPECS[model_name]
        try:
            base_model = get_backbone(model_name, img_size=config.img_size)
            trained_model, histories, _ = trainer.train_model(
                model_name=model_name,
                base_model=base_model,
                unfreeze_phase2=spec["unfreeze_p2"],
                unfreeze_phase3=spec["unfreeze_p3"],
                train_ds=train_ds,
                val_ds=val_ds,
                class_weight_dict=class_weights,
                is_vit=spec["is_vit"],
            )

            # Evaluate on held-out test split
            metrics = evaluate_model(trained_model, test_ds, class_names, model_name)
            all_metrics.append(metrics)

            # Save Visualizations
            plot_training_history(histories, model_name, config.plot_dir)
            plot_confusion_matrix(metrics["y_true"], metrics["y_pred"], class_names, model_name, config.plot_dir)
            plot_roc_curves(metrics["y_true"], metrics["y_pred_prob"], class_names, model_name, config.plot_dir)
            plot_per_class_metrics(metrics["report"], class_names, model_name, config.plot_dir)

            trainer.cleanup_memory(trained_model)
            del base_model
        except Exception as e:
            print(f"[ERROR] Error training {model_name}: {e}")

    # 7. Comparison Summary & Final Recommendations (Cell 38, 40, 44)
    if all_metrics:
        comparison_rows = [
            {
                "Model": m["model"],
                "Accuracy": m["accuracy"],
                "F1 (Macro)": m["f1_macro"],
                "F1 (Weighted)": m["f1_weighted"],
                "AUC-ROC": m["auc_roc"],
                "Cohen's Kappa": m["kappa"],
                "MCC": m["mcc"],
            }
            for m in all_metrics
        ]
        df_compare = pd.DataFrame(comparison_rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
        csv_path = config.save_dir / "model_comparison.csv"
        df_compare.to_csv(csv_path, index=False)

        print("\n" + "=" * 75)
        print("  FULL MODEL COMPARISON TABLE")
        print("=" * 75)
        print(df_compare.to_string(index=False))
        print("=" * 75)

        plot_comparison_dashboard(df_compare, config.plot_dir)
        plot_radar_chart(df_compare, config.plot_dir)

        # Section 21c: Per-source evaluation (real-only vs synthetic-only test metrics)
        if config.mode == "combined" and "source" in test_df.columns:
            from src.evaluation.metrics import evaluate_per_source
            systems = {m["model"]: (m["y_true"], m["y_pred"]) for m in all_metrics}
            evaluate_per_source(systems, test_df, class_names, save_dir=config.save_dir)

        best = max(all_metrics, key=lambda x: x["accuracy"])
        print("\n" + "=" * 75)
        print("  FINAL RECOMMENDATION")
        print("=" * 75)

        print(f"Top Model        : {best['model']}")
        print(f"   Accuracy      : {best['accuracy']:.4f}")
        print(f"   F1 (Macro)    : {best['f1_macro']:.4f}")
        print(f"   AUC-ROC       : {best['auc_roc']:.4f}")
        print(f"   Cohen kappa   : {best['kappa']:.4f}")
        print(f"   MCC           : {best['mcc']:.4f}")
        print(f"""
Deployment Guidance:
  * Best Accuracy  -> Ensemble (Weighted Soft Voting)
  * Best Speed     -> MobileNetV3Large
  * Best Balance   -> {best['model']}

Saved in {config.save_dir}:
  * <Model>_final.keras        -- full trained model
  * <Model>_best.keras         -- best checkpoint per phase
  * <Model>_training_log.csv   -- epoch logs
  * model_comparison.csv
  * label_map.json
""")

    print("\n[OK] Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
