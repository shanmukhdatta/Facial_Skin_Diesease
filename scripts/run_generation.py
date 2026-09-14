import argparse
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.prompts import load_or_build_prompts
from src.generation.diffusion_generator import DiffusionGenerator
from src.generation.stylegan_generator import StyleGAN2PyTorchGenerator
from src.generation.stylegan_trainer import StyleGAN2PyTorchTrainer, run_single_step_gan


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
        choices=["diffusion", "stylegan_train", "stylegan", "prompts_only", "all"],
        help="Generation method to execute. 'stylegan_train' trains the per-class "
             "stylegan2_pytorch models (Skin Cancer); 'stylegan' samples from already-"
             "trained checkpoints.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Specific classes to generate (default: all classes for the chosen --method, "
             "from config -- diffusion.classes for 'diffusion', stylegan2.dataset.classes "
             "for 'stylegan'/'stylegan_train').",
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
        "--load_from",
        type=int,
        default=None,
        help="Optional checkpoint number to generate from instead of the latest "
             "(passed through to stylegan2_pytorch's ModelLoader).",
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
    stylegan_cfg = cfg.get("stylegan2", {})

    if args.device:
        diff_cfg["model"]["device"] = args.device

    prompts_dir = Path(diff_cfg["paths"]["prompts_dir"])
    output_dir = Path(diff_cfg["paths"]["output_dir"])
    metadata_dir = Path(diff_cfg["paths"]["metadata_dir"])
    rejected_log = Path(diff_cfg["paths"]["rejected_log"])

    is_stylegan_method = args.method in ["stylegan", "stylegan_train"]
    if is_stylegan_method:
        classes_to_run = args.classes or stylegan_cfg.get("dataset", {}).get("classes", [])
    else:
        classes_to_run = args.classes or diff_cfg.get("classes", [])

    print("======================================================================")
    print("[INFO] Synthetic Skin Disease Generation Pipeline")
    print(f"Config: {config_path}")
    print(f"Method: {args.method}")
    print(f"Classes: {classes_to_run}")
    print("======================================================================")

    # 1. Build or Verify Prompts (diffusion classes only -- stylegan methods have no prompts)
    prompts_by_class = {}
    if not is_stylegan_method:
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

    # 3. StyleGAN2 Training -- Skin Cancer (BCC / SCC / melanoma), one model per class
    if args.method in ["stylegan_train", "all"]:
        trainer = StyleGAN2PyTorchTrainer(stylegan_cfg)
        input_root = Path(stylegan_cfg["paths"]["input_root"])
        train_cfg = stylegan_cfg.get("training", {})
        total_budget_h = train_cfg.get("total_time_budget_hours", 5.5)
        gen_reserve_h = train_cfg.get("generation_reserve_hours", 0.75)
        session_budget_s = (total_budget_h - gen_reserve_h) * 3600
        per_class_budget_s = session_budget_s / max(len(classes_to_run), 1)
        for cls in classes_to_run:
            trainer.run_class_within_budget(cls, input_root, per_class_budget_s)

    # 4. StyleGAN2 Generation -- samples from already-trained per-class checkpoints
    if args.method in ["stylegan", "all"]:
        gen_cfg = {**stylegan_cfg.get("generation", {}), **stylegan_cfg.get("quality_control", {})}
        stylegan_gen = StyleGAN2PyTorchGenerator(
            cfg=gen_cfg,
            models_dir=Path(stylegan_cfg["paths"]["models_dir"]),
        )
        out_root = Path(stylegan_cfg["paths"]["generated_dir"])
        target_per_class = gen_cfg.get("images_per_class_target", 500)
        for cls in classes_to_run:
            n_cls = stylegan_gen.generate_class(
                class_name=cls,
                target_count=target_per_class,
                out_root=out_root,
                load_from=args.load_from,
            )
            print(f"[DONE] Completed {n_cls} synthetic {cls} images.")

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
