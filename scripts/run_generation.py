import argparse
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.prompts import load_or_build_prompts
from src.generation.diffusion_generator import DiffusionGenerator
from src.generation.stylegan_generator import StyleGANCancerGenerator
from src.generation.stylegan_trainer import StyleGANTrainer, run_single_step_gan


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic Facial Skin Disease Generation Pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/generation_config.yaml",
        help="Path to generation YAML configuration.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="diffusion",
        choices=["diffusion", "stylegan", "prompts_only", "all"],
        help="Generation method to execute.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Specific classes to generate (default: all diffusion classes in config).",
    )
    parser.add_argument(
        "--limit_prompts",
        type=int,
        default=None,
        help="Optional debug cap on number of prompts per class.",
    )
    parser.add_argument(
        "--images_per_prompt",
        type=int,
        default=None,
        help="Override number of images to generate per prompt (default from config).",
    )
    parser.add_argument(
        "--train_gan",
        action="store_true",
        help="Execute single-step GAN training pass on a generated image to verify reproducibility.",
    )
    parser.add_argument(
        "--network_pkl",
        type=str,
        default=None,
        help="Path to StyleGAN2-ADA network snapshot .pkl (for Skin Cancer).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (e.g. cuda or cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    assert config_path.exists(), f"Configuration file not found: {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    diff_cfg = cfg.get("diffusion", {})
    stylegan_cfg = cfg.get("stylegan2_ada", {})

    if args.device:
        diff_cfg["model"]["device"] = args.device

    prompts_dir = Path(diff_cfg["paths"]["prompts_dir"])
    output_dir = Path(diff_cfg["paths"]["output_dir"])
    metadata_dir = Path(diff_cfg["paths"]["metadata_dir"])
    rejected_log = Path(diff_cfg["paths"]["rejected_log"])

    classes_to_run = args.classes or diff_cfg.get("classes", [])

    print("======================================================================")
    print("[INFO] Synthetic Skin Disease Generation Pipeline")
    print(f"Config: {config_path}")
    print(f"Method: {args.method}")
    print(f"Classes: {classes_to_run}")
    print("======================================================================")

    # 1. Build or Verify Prompts
    prompts_by_class = {}
    for cls in classes_to_run:
        prompts = load_or_build_prompts(cls, diff_cfg, prompts_dir)
        prompts_by_class[cls] = prompts
        print(f"[OK] Loaded/Generated {len(prompts)} prompts for class: {cls}")

    if args.method == "prompts_only":
        print("\n[OK] Prompt manifests successfully verified. Exiting (prompts_only).")
        return

    # 2. Diffusion Generation
    generated_sample_paths = []
    if args.method in ["diffusion", "all"]:
        diff_gen = DiffusionGenerator(diff_cfg)
        diff_gen.load_pipeline()

        for cls in classes_to_run:
            prompts = prompts_by_class[cls]
            n_done = diff_gen.run_class(
                class_name=cls,
                prompts=prompts,
                out_dir=output_dir,
                metadata_dir=metadata_dir,
                rejected_log_path=rejected_log,
                limit_prompts=args.limit_prompts,
                images_per_prompt=args.images_per_prompt,
            )
            print(f"[DONE] Completed {n_done} synthetic images for {cls}")
            cls_dir = output_dir / cls
            if cls_dir.exists():
                generated_sample_paths.extend(list(cls_dir.glob("*.png")))

    # 3. StyleGAN2-ADA Generation (Skin Cancer)
    if args.method in ["stylegan", "all"]:
        network_pkl = args.network_pkl or stylegan_cfg["paths"].get("network_pkl")
        if not network_pkl:
            print("\n[WARN] StyleGAN network snapshot .pkl path not provided. Skipping StyleGAN generation.")
            print("       Pass --network_pkl <path_to_snapshot.pkl> to run Skin Cancer generation.")
        else:
            stylegan_gen = StyleGANCancerGenerator(
                repo_dir=Path(stylegan_cfg["paths"]["stylegan2_ada_repo"]),
                network_pkl=network_pkl,
                device=diff_cfg["model"].get("device", "cuda"),
            )
            n_cancer = stylegan_gen.generate(
                num_images=stylegan_cfg["generation"].get("num_images", 1500),
                out_dir=Path(stylegan_cfg["paths"]["generated_dir"]),
                truncation_psi=stylegan_cfg["generation"].get("truncation_psi", 0.7),
            )
            print(f"[DONE] Completed {n_cancer} synthetic Skin Cancer images.")

    # 4. Optional GAN Training Verification (Single-step Reproducibility Smoke Test)
    if args.train_gan:
        print("\n======================================================================")
        print("[INFO] Single-Step GAN Reproducibility Training Verification")
        print("======================================================================")
        sample_img = None
        if generated_sample_paths:
            sample_img = generated_sample_paths[0]
        else:
            # Check if any image exists in output_dir
            existing = list(output_dir.glob("**/*.png"))
            if existing:
                sample_img = existing[0]

        if sample_img is None:
            print("[WARN] No generated image found to train GAN. Generating 1 test sample first...")
            diff_gen = DiffusionGenerator(diff_cfg)
            first_prompt = prompts_by_class[classes_to_run[0]][0]["prompt"]
            test_img = diff_gen.generate_synthetic_test_image(first_prompt, seed=42)
            test_path = output_dir / "test_sample_for_gan.png"
            test_path.parent.mkdir(parents=True, exist_ok=True)
            test_img.save(test_path)
            sample_img = test_path

        print(f"[INFO] Training GAN (1 step / 1 layer) using generated image: {sample_img}")
        gan_metrics = run_single_step_gan(input_image=sample_img)
        print(f"[OK] GAN Training Step Completed:")
        print(f"     - Discriminator loss (before update): {gan_metrics['d_loss_before']}")
        print(f"     - Generator loss (before update):     {gan_metrics['g_loss_before']}")
        print(f"     - Generator loss (after update):      {gan_metrics['g_loss_after']}")
        print(f"     - D(x_real) prediction:               {gan_metrics['d_output_real']}")
        print(f"     - D(G(z)) fake prediction:            {gan_metrics['d_output_fake']}")
        print(f"     - Verified Output Image Generated at: {gan_metrics['output_image_path']}")
        print(f"     - Layers updated:                     {', '.join(gan_metrics['trained_layers'])}")

    print("\n[OK] Generation workflow finished successfully.")


if __name__ == "__main__":
    main()
