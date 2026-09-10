from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
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


class StyleGANCancerGenerator:
    """
    StyleGAN2-ADA sampling engine for the Skin Cancer synthetic class (Section 3.2.2).
    Uses a curated seed corpus (BCC, SCC, melanoma) to produce 1,500 synthetic images.
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
