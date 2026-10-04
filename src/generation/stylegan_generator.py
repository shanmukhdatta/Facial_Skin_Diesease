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


# Canonical alias for the active StyleGAN2 generator
StyleGAN2Generator = StyleGAN2PyTorchGenerator

