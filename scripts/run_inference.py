import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import ProjectConfig
from src.inference.predictor import DiseasePredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Run Facial Skin Disease Prediction on an Image.")
    parser.add_argument("--image_path", type=str, default=None, help="Path to input image file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="all",
        help="Path to .keras model checkpoint or 'all' to evaluate on all saved final checkpoints.",
    )
    parser.add_argument("--label_map", type=str, default=None, help="Path to label_map.json file.")
    parser.add_argument("--save_plot", type=str, default=None, help="Filepath to save visualization plot.")
    parser.add_argument("--demo", action="store_true", help="Run automated inference demo on held-out test data.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ProjectConfig()

    # Determine label map
    label_map_path = Path(args.label_map) if args.label_map else config.save_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(
            f"Label map not found at {label_map_path}. Train models first or specify --label_map."
        )

    # Automated demo mode (Cell 46)
    image_path = args.image_path
    true_label = None
    if args.demo or image_path is None:
        if config.data_dir.exists():
            print("[INFO] Selecting held-out test sample for inference demo...")
            from src.data.dataset import discover_dataset, clean_dataset, split_dataset
            df_full, class_names = discover_dataset(config.data_dir)
            df_clean = clean_dataset(df_full)
            _, _, test_df = split_dataset(df_clean, test_size=config.test_size, seed=config.seed)
            sample_row = test_df.iloc[0]
            image_path = sample_row["path"]
            true_label = sample_row["class"]
            print(f"Demo image: {image_path}")
            print(f"Ground Truth Label: {true_label}")
        else:
            if not image_path:
                raise ValueError("Must provide --image_path or valid dataset path for --demo.")

    # Find models to execute
    if args.model_path == "all":
        model_files = sorted(config.save_dir.glob("*_final.keras"))
        if not model_files:
            raise FileNotFoundError(f"No *_final.keras checkpoints found in {config.save_dir}")
    else:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model_files = [model_path]

    print("\n" + "=" * 65)
    print("INFERENCE RESULTS SUMMARY")
    print("=" * 65)
    print(f"Image: {image_path}")
    if true_label:
        print(f"True Label: {true_label}")
    print("-" * 65)
    print(f"{'Model':<22} | {'Predicted Class':<22} | {'Confidence':<10} | Match")
    print("-" * 65)

    for mf in model_files:
        model_name = mf.stem.replace("_final", "")
        try:
            predictor = DiseasePredictor(
                model_path=mf,
                label_map=label_map_path,
                img_size=config.img_size,
            )
            pred_class, confidence, probs = predictor.predict_image(image_path)
            match_str = ""
            if true_label:
                match_str = "[MATCH]" if pred_class == true_label else "[MISMATCH]"

            print(f"{model_name:<22} | {pred_class:<22} | {confidence:>9.2%} | {match_str}")

            plot_path = args.save_plot
            if plot_path and len(model_files) > 1:
                p = Path(plot_path)
                plot_path = p.parent / f"{model_name}_{p.name}"

            if plot_path:
                predictor.plot_prediction(
                    image_path=image_path,
                    pred_class=pred_class,
                    confidence=confidence,
                    probs=probs,
                    model_name=model_name,
                    save_path=plot_path,
                )
        except Exception as e:
            print(f"{model_name:<22} | ERROR: {e}")

    print("=" * 65)


if __name__ == "__main__":
    main()
