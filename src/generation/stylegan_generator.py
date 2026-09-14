from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

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
    """Check for flat/blank outputs."""
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    return float(arr.std()) < min_std


class StyleGAN2PyTorchGenerator:
    """
    Sampling engine for the `stylegan2_pytorch` (lucidrains) per-class Skin Cancer models
    -- the implementation actually used to generate the published dataset (Section 3.2.2).

    Generates each class (BCC / SCC / melanoma) from its own independently-trained model
    into `generated_dir/<class>/`, with perceptual-hash de-duplication (reject a sample if
    it's within `max_phash_distance` of one already accepted -- resample instead) and an
    optional wall-clock time cap so a run always terminates even if de-duplication turns
    out to need many resamples.

    Expects `cfg` to be the `stylegan2.generation` block (merged with `quality_control`) of
    `configs/generation_config.yaml`.
    """

    def __init__(self, cfg: Dict[str, Any], models_dir: Union[str, Path]):
        self.cfg = cfg
        self.models_dir = Path(models_dir)

    def generate_class(
        self,
        class_name: str,
        target_count: int,
        out_root: Union[str, Path],
        load_from: Optional[int] = None,
        time_budget_seconds: Optional[float] = None,
    ) -> int:
        import time as _time

        import imagehash
        import torch
        from stylegan2_pytorch import ModelLoader

        cfg = self.cfg
        out_dir = Path(out_root) / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / f"{class_name}_meta.jsonl"

        loader = ModelLoader(base_dir=str(self.models_dir.parent), name=class_name, load_from=load_from)

        min_pixel_std = cfg.get("min_pixel_std", 5.0)
        truncation_psi = cfg.get("truncation_psi", 0.75)
        batch_size = cfg.get("batch_size", 16)
        max_phash_distance = cfg.get("max_phash_distance", 4)
        max_resample_attempts = cfg.get("max_resample_attempts_multiplier", 6) * target_count
        seed_base = cfg.get("seed_base", 1000)

        seen_hashes: List[Any] = []
        generated = 0
        attempts = 0
        start_time = _time.time()
        torch.manual_seed(seed_base)
        meta_f = meta_path.open("w", encoding="utf-8")

        print(f"[stylegan2_pytorch] generating {target_count} images for '{class_name}'"
              + (f" (time cap: {time_budget_seconds/60:.1f} min)" if time_budget_seconds else ""))

        while generated < target_count and attempts < max_resample_attempts:
            if time_budget_seconds is not None and (_time.time() - start_time) > time_budget_seconds:
                print(f"[stylegan2_pytorch] '{class_name}': time cap reached at "
                      f"{generated}/{target_count} -- saving what we have.")
                break

            batch_n = min(batch_size, target_count - generated)
            noise = torch.randn(batch_n, 512).cuda()
            styles = loader.noise_to_styles(noise, trunc_psi=truncation_psi)
            images = loader.styles_to_images(styles)  # (batch_n, 3, H, W), float in [0, 1]

            for i in range(batch_n):
                attempts += 1
                arr = (images[i].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
                pil_img = Image.fromarray(arr)

                if looks_degenerate(pil_img, min_pixel_std):
                    continue

                phash = imagehash.phash(pil_img)
                if any(phash - h <= max_phash_distance for h in seen_hashes):
                    continue  # near-duplicate of an already-accepted image -> resample

                filename = f"{class_name}_{generated:04d}.png"
                pil_img.save(out_dir / filename)
                seen_hashes.append(phash)
                meta_f.write(json.dumps({"file": filename, "phash": str(phash)}) + "\n")
                generated += 1
                if generated >= target_count:
                    break

        meta_f.close()
        elapsed = _time.time() - start_time
        print(f"[stylegan2_pytorch] '{class_name}': generated {generated}/{target_count} "
              f"after {attempts} attempts in {elapsed/60:.1f} min.")
        return generated


class StyleGANCancerGenerator:
    """
    StyleGAN2-ADA sampling engine for the Skin Cancer synthetic class (Section 3.2.2).

    NOT the implementation used to produce this repo's published Skin Cancer images --
    kept as an alternative path for a single class-conditional (or unconditional combined)
    model loaded from an NVIDIA-ADA `.pkl` snapshot. See `StyleGAN2PyTorchGenerator` above
    for the actual per-class method.
    """

    def __init__(self, repo_dir: Path, network_pkl: str, device: str = "cuda"):
        self.repo_dir = Path(repo_dir)
        self.network_pkl = str(network_pkl)
        self.device = device
        self.generator = None
        self.torch = None

    def load_model(self) -> None:
        """Dynamically import official stylegan2-ada-pytorch and load network snapshot."""
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        try:
            import dnnlib  # noqa: F401
            import legacy
            import torch
        except ImportError as e:
            raise ImportError(
                f"Could not import stylegan2-ada-pytorch from {self.repo_dir}. "
                "Clone it from https://github.com/NVlabs/stylegan2-ada-pytorch"
            ) from e

        self.torch = torch
        print(f"[stylegan2] Loading network snapshot from: {self.network_pkl} ...")
        with dnnlib.util.open_url(self.network_pkl) as f:
            data = legacy.load_network_pkl(f)
        self.generator = data["G_ema"].to(self.device).eval()
        print(f"[stylegan2] Model loaded (z_dim={self.generator.z_dim}, c_dim={self.generator.c_dim}).")

    def tensor_to_pil(self, img_tensor) -> Image.Image:
        """Convert StyleGAN normalized output tensor to PIL RGB Image."""
        img = (img_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to("cpu", dtype=self.torch.uint8)
        return Image.fromarray(img[0].numpy(), "RGB")

    def generate(
        self,
        num_images: int = 1500,
        out_dir: Path = Path("outputs/synthetic_dataset/skin_cancer"),
        truncation_psi: float = 0.7,
        noise_mode: str = "const",
        batch_size: int = 8,
        seed_base: int = 1000,
        min_pixel_std: float = 3.0,
    ) -> int:
        """Sample latent space to generate synthetic skin cancer images."""
        if self.generator is None:
            self.load_model()

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_file = out_dir / "skin_cancer_meta.jsonl"
        meta_handle = meta_file.open("a", encoding="utf-8")

        print(f"[stylegan2] Generating {num_images} Skin Cancer images with truncation_psi={truncation_psi} ...")
        generated_count = 0
        pbar = tqdm(total=num_images, desc="Skin Cancer (StyleGAN2)", unit="img")

        current_seed = seed_base
        while generated_count < num_images:
            batch_n = min(batch_size, num_images - generated_count)
            z = self.torch.from_numpy(
                np.random.RandomState(current_seed).randn(batch_n, self.generator.z_dim)
            ).to(self.device, dtype=self.torch.float32)

            c = None
            if self.generator.c_dim > 0:
                c = self.torch.zeros([batch_n, self.generator.c_dim], device=self.device)

            img_tensors = self.generator(z, c, truncation_psi=truncation_psi, noise_mode=noise_mode)

            for i in range(batch_n):
                pil_img = self.tensor_to_pil(img_tensors[i: i + 1])
                if looks_degenerate(pil_img, min_pixel_std):
                    current_seed += 1
                    continue

                filename = f"skin_cancer_{generated_count:05d}.png"
                img_path = out_dir / filename
                pil_img.save(img_path, format="PNG")

                meta_record = {
                    "filename": filename,
                    "path": str(img_path),
                    "class": "skin_cancer",
                    "seed": current_seed,
                    "truncation_psi": truncation_psi,
                }
                meta_handle.write(json.dumps(meta_record) + "\n")
                meta_handle.flush()

                generated_count += 1
                current_seed += 1
                pbar.update(1)

        pbar.close()
        meta_handle.close()
        return generated_count
