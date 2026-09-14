from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from src.generation.prompts import build_final_prompt, CUSTOM_PROMPT_PREFIX

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None, **kwargs):
            self.iterable = iterable
            self.total = total
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        def update(self, n=1):
            pass
        def close(self):
            pass


def looks_degenerate(image: Image.Image, min_std: float = 3.0) -> bool:
    """Detect near-flat or corrupted blank outputs."""
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    return float(arr.std()) < min_std


class DiffusionGenerator:
    """
    Realistic Vision V5.1 generation engine for multi-category facial skin disease synthesis.
    Implements multi-seed generation and optional high-resolution refinement.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        mcfg = cfg.get("model", {})
        gcfg = cfg.get("generation", {})

        self.model_id = mcfg.get("model_id", "SG161222/Realistic_Vision_V5.1_noVAE")
        self.vae_id = mcfg.get("vae_id", "stabilityai/sd-vae-ft-mse")
        self.device = mcfg.get("device", "cuda")
        self.torch_dtype_str = mcfg.get("torch_dtype", "fp16")
        self.enable_xformers = mcfg.get("enable_xformers", True)
        self.enable_attention_slicing = mcfg.get("enable_attention_slicing", True)
        self.safety_checker = mcfg.get("safety_checker", True)

        self.image_size = gcfg.get("image_size", 512)
        self.num_inference_steps = gcfg.get("num_inference_steps", 40)
        self.guidance_scale = gcfg.get("guidance_scale", 7.5)
        self.negative_prompt = gcfg.get("negative_prompt", "").strip()
        self.images_per_prompt = gcfg.get("images_per_prompt", 5)
        self.max_retries = gcfg.get("max_retries_per_image", 3)

        self.hires_fix = gcfg.get("hires_fix", False)
        self.hires_scale = gcfg.get("hires_scale", 1.5)
        self.hires_strength = gcfg.get("hires_strength", 0.35)
        self.hires_steps = gcfg.get("hires_steps", 20)

        self.min_pixel_std = cfg.get("quality_control", {}).get("min_pixel_std", 3.0)

        # ---- per-class deterministic seed formula (Section 1 + 8 of the Kaggle notebook) ----
        # seed = (selected_seeds[class] + (prompt_num-1)*prompt_seed_step + (v-1)*variation_seed_step) % 2**32
        # retry seed = (that seed + retry * retry_offset) % 2**32
        # NOTE: this is a positional formula keyed on (prompt_num, variation_index), not on the
        # prompt's own content -- it intentionally overrides each prompt entry's own `seed_base`.
        seeding_cfg = cfg.get("seeding", {})
        self.selected_seeds: Dict[str, int] = seeding_cfg.get("selected_seeds", {})
        self.prompt_seed_step = seeding_cfg.get("prompt_seed_step", 7919)
        self.variation_seed_step = seeding_cfg.get("variation_seed_step", 104729)
        self.retry_offset = seeding_cfg.get("retry_offset", 15485863)

        # ---- output filename convention: cls{class_index:02d}_p_{prompt#:03d}_v_{i}.png ----
        self.class_index: Dict[str, int] = cfg.get("class_index", {})

        # ---- photorealistic prefix prepended to every prompt before generation ----
        self.prompt_prefix = gcfg.get("prompt_prefix", CUSTOM_PROMPT_PREFIX)

        self.pipe = None
        self.hires_pipe = None
        self.torch = None

    def load_pipeline(self) -> None:
        """Initialize and configure StableDiffusion pipelines on device."""
        try:
            import torch
            from diffusers import (
                AutoencoderKL,
                DPMSolverMultistepScheduler,
                StableDiffusionImg2ImgPipeline,
                StableDiffusionPipeline,
            )

            self.torch = torch
            dtype = torch.float16 if self.torch_dtype_str == "fp16" else torch.float32

            print(f"[diffusion] Loading base model: {self.model_id} on {self.device} ({self.torch_dtype_str}) ...")
            vae = None
            if self.vae_id:
                vae = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=dtype)

            pipe_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
            if not self.safety_checker:
                # diffusers doesn't accept safety_checker="default" -- passing that string
                # crashes on the first generation call. Omitting the key entirely when a
                # safety checker IS wanted lets diffusers load its own default; explicitly
                # disable it otherwise.
                pipe_kwargs["safety_checker"] = None
                pipe_kwargs["requires_safety_checker"] = False
            if vae is not None:
                pipe_kwargs["vae"] = vae

            self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, **pipe_kwargs)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config, use_karras_sigmas=True
            )

            if self.enable_xformers:
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass

            if self.enable_attention_slicing:
                self.pipe.enable_attention_slicing()

            self.pipe = self.pipe.to(self.device)

            if self.hires_fix:
                self.hires_pipe = StableDiffusionImg2ImgPipeline(
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    tokenizer=self.pipe.tokenizer,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler,
                    safety_checker=self.pipe.safety_checker,
                    feature_extractor=self.pipe.feature_extractor,
                ).to(self.device)
                if self.enable_attention_slicing:
                    self.hires_pipe.enable_attention_slicing()

            print("[diffusion] Pipeline loaded successfully.")
        except ImportError as e:
            print(f"[diffusion] Notice: diffusers or PyTorch not available in current environment ({e}).")
            print("            Activating deterministic synthetic pattern generator for verification testing.")
            self.pipe = None

    def generate_synthetic_test_image(self, prompt: str, seed: int) -> Image.Image:
        """
        Deterministic image generator for offline verification and local testing.
        Synthesizes a 512x512 facial skin patch reflecting the Fitzpatrick type and disease features in the prompt.
        """
        rng = np.random.default_rng(seed)
        size = self.image_size

        # Determine Fitzpatrick base skin tone
        fitz_palette = {
            "I": np.array([253, 235, 224], dtype=np.float32),
            "II": np.array([248, 222, 206], dtype=np.float32),
            "III": np.array([232, 195, 170], dtype=np.float32),
            "IV": np.array([195, 149, 115], dtype=np.float32),
            "V": np.array([142, 94, 61], dtype=np.float32),
            "VI": np.array([79, 49, 32], dtype=np.float32),
        }
        base_color = fitz_palette["III"]
        for ftype, col in fitz_palette.items():
            if f"Fitzpatrick type {ftype}" in prompt or f"type {ftype}" in prompt:
                base_color = col
                break

        # Base skin canvas with subtle pore and facial lighting gradient
        y_grid, x_grid = np.ogrid[:size, :size]
        center = size // 2
        radial_dist = np.sqrt((x_grid - center) ** 2 + (y_grid - center) ** 2) / (size / 1.4)
        vignette = 1.0 - 0.18 * np.clip(radial_dist, 0.0, 1.0)

        # Micro-texture (pores and fine skin grain)
        grain = rng.normal(0.0, 4.0, (size, size, 1)).astype(np.float32)
        canvas = np.ones((size, size, 3), dtype=np.float32) * base_color[None, None, :]
        canvas = canvas * vignette[:, :, None] + grain

        p_lower = prompt.lower()
        if "acne" in p_lower:
            num_lesions = rng.integers(12, 28)
            for _ in range(num_lesions):
                lx, ly = rng.integers(size // 4, 3 * size // 4, 2)
                rad = rng.integers(6, 18)
                dist = np.sqrt((x_grid - lx) ** 2 + (y_grid - ly) ** 2)
                mask = np.clip(1.0 - (dist / rad), 0.0, 1.0)
                # Inflammatory erythema
                canvas[:, :, 0] += mask * 45.0
                canvas[:, :, 1] -= mask * 20.0
                canvas[:, :, 2] -= mask * 20.0
                # Central pustular highlight
                core = np.clip(1.0 - (dist / (rad * 0.35)), 0.0, 1.0)
                canvas += core[:, :, None] * 25.0
        elif "vitiligo" in p_lower:
            vx, vy = rng.integers(size // 3, 2 * size // 3, 2)
            dist = np.sqrt((x_grid - vx) ** 2 + ((y_grid - vy) * 1.3) ** 2)
            patch = np.clip(1.0 - (dist / 110.0), 0.0, 1.0)
            patch = np.clip(patch + rng.normal(0.0, 0.08, (size, size)), 0.0, 1.0)
            depigmented = np.array([248, 246, 244], dtype=np.float32)
            canvas = canvas * (1.0 - patch[:, :, None]) + depigmented * patch[:, :, None]
        elif "hyperpigmentation" in p_lower:
            hx, hy = rng.integers(size // 3, 2 * size // 3, 2)
            dist = np.sqrt(((x_grid - hx) * 1.2) ** 2 + (y_grid - hy) ** 2)
            patch = np.clip(1.0 - (dist / 95.0), 0.0, 1.0)
            canvas[:, :, 0] -= patch * 35.0
            canvas[:, :, 1] -= patch * 42.0
            canvas[:, :, 2] -= patch * 45.0
        elif "fungal" in p_lower:
            fx, fy = rng.integers(size // 3, 2 * size // 3, 2)
            dist = np.sqrt((x_grid - fx) ** 2 + (y_grid - fy) ** 2)
            annular_ring = np.exp(-((dist - 70.0) ** 2) / (2 * (15.0 ** 2)))
            canvas[:, :, 0] += annular_ring * 38.0
            canvas[:, :, 1] -= annular_ring * 12.0
            canvas[:, :, 2] -= annular_ring * 12.0

        canvas = np.clip(canvas, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(canvas, "RGB")

    def generate_single(
        self, prompt: str, seed: int, negative_prompt: Optional[str] = None
    ) -> Image.Image:
        """Generate a single image with seed control and optional hires upscale."""
        if self.pipe is None:
            self.load_pipeline()

        if self.pipe is None:
            return self.generate_synthetic_test_image(prompt, seed)

        neg = negative_prompt if negative_prompt is not None else self.negative_prompt
        generator = self.torch.Generator(device=self.device).manual_seed(int(seed))

        output = self.pipe(
            prompt=prompt,
            negative_prompt=neg,
            width=self.image_size,
            height=self.image_size,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )
        img = output.images[0]

        if self.hires_fix and self.hires_pipe is not None:
            target_w = int(round(self.image_size * self.hires_scale))
            target_h = int(round(self.image_size * self.hires_scale))
            upscaled = img.resize((target_w, target_h), Image.LANCZOS)
            output_hires = self.hires_pipe(
                prompt=prompt,
                negative_prompt=neg,
                image=upscaled,
                strength=self.hires_strength,
                num_inference_steps=self.hires_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            )
            img = output_hires.images[0]

        return img

    def run_class(
        self,
        class_name: str,
        prompts: List[Dict[str, Any]],
        out_dir: Path,
        metadata_dir: Path,
        rejected_log_path: Optional[Path] = None,
        limit_prompts: Optional[int] = None,
        images_per_prompt: Optional[int] = None,
    ) -> int:
        """Generate all images for a given disease class and log metadata."""
        out_dir = Path(out_dir) / class_name
        metadata_dir = Path(metadata_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        n_per_prompt = images_per_prompt if images_per_prompt is not None else self.images_per_prompt
        class_prompts = prompts[:limit_prompts] if limit_prompts else prompts
        total_target = len(class_prompts) * n_per_prompt
        print(f"\n[diffusion] Generating {class_name}: {len(class_prompts)} prompts x {n_per_prompt} images -> {total_target} images")

        meta_file = metadata_dir / f"{class_name}_generated_meta.jsonl"
        meta_handle = meta_file.open("a", encoding="utf-8")
        rejected_handle = rejected_log_path.open("a", encoding="utf-8") if rejected_log_path else None

        generated_count = 0
        pbar = tqdm(total=total_target, desc=f"{class_name}", unit="img")

        cls_idx = self.class_index.get(class_name)
        base_seed = self.selected_seeds.get(class_name, 42)

        for entry in class_prompts:
            pid = entry["id"]
            prompt_num = pid + 1  # repo ids are 0-indexed -> 1-indexed for filenames/seeds
            raw_prompt = entry["prompt"]
            final_prompt = build_final_prompt(raw_prompt, self.prompt_prefix)

            for img_idx in range(n_per_prompt):
                v = img_idx + 1
                if cls_idx is not None:
                    img_name = f"cls{cls_idx:02d}_p_{prompt_num:03d}_v_{v}.png"
                else:
                    # No class_index configured for this class -- fall back to the
                    # original repo naming instead of guessing a number.
                    img_name = f"{class_name}_p{pid:04d}_s{img_idx:02d}.png"
                img_path = out_dir / img_name

                if img_path.exists():
                    pbar.update(1)
                    generated_count += 1
                    continue

                # Positional formula keyed on (prompt_num, variation index) -- this is the
                # formula that actually produced the published dataset. It intentionally
                # does NOT use entry["seed_base"] (that per-prompt content-derived seed is
                # what the repo used before this fix; see generation_config.yaml comments).
                base_variation_seed = (
                    base_seed
                    + (prompt_num - 1) * self.prompt_seed_step
                    + (v - 1) * self.variation_seed_step
                ) % (2**32)

                success = False
                last_error = None
                for retry in range(self.max_retries):
                    current_seed = (base_variation_seed + retry * self.retry_offset) % (2**32)
                    try:
                        img = self.generate_single(final_prompt, current_seed, negative_prompt=self.negative_prompt)
                    except Exception as e:
                        # A single failed attempt (e.g. transient CUDA OOM) must never take
                        # down the whole class run -- log it, count it as a used retry, move on.
                        last_error = f"exception: {type(e).__name__}: {e}"
                        if rejected_handle:
                            rejected_handle.write(json.dumps({
                                "class": class_name, "prompt_id": pid, "seed": current_seed,
                                "reason": last_error, "time": time.time(),
                            }) + "\n")
                        continue

                    if looks_degenerate(img, self.min_pixel_std):
                        last_error = "rejected: looks_degenerate (low pixel std)"
                        if rejected_handle:
                            rejected_handle.write(json.dumps({
                                "class": class_name, "prompt_id": pid, "seed": current_seed,
                                "reason": "low_std", "time": time.time(),
                            }) + "\n")
                        continue

                    img.save(img_path, format="PNG")
                    meta_record = {
                        "filename": img_name,
                        "path": str(img_path),
                        "class": class_name,
                        "prompt_id": pid,
                        "prompt_number": prompt_num,
                        "variation_number": v,
                        # phenotype/demographic detail from the repo's own prompt metadata
                        "phenotype": entry.get("phenotype"),
                        "gender": entry.get("gender"),
                        "variation_group": entry.get("variation_group"),
                        "age": entry.get("age"),
                        "severity": entry.get("severity"),
                        "fitzpatrick": entry.get("fitzpatrick"),
                        "distribution": entry.get("distribution"),
                        # actual generation parameters used for this image
                        "selected_base_seed": base_seed,
                        "actual_seed": current_seed,
                        "prompt": raw_prompt,
                        "final_prompt": final_prompt,
                        "negative_prompt": self.negative_prompt,
                        "inference_steps": self.num_inference_steps,
                        "guidance_scale": self.guidance_scale,
                    }
                    meta_handle.write(json.dumps(meta_record) + "\n")
                    meta_handle.flush()
                    success = True
                    break

                if success:
                    generated_count += 1
                pbar.update(1)

        pbar.close()
        meta_handle.close()
        if rejected_handle:
            rejected_handle.close()
        return generated_count
