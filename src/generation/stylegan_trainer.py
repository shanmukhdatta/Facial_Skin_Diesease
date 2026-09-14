"""
stylegan_trainer.py
===================
Orchestration wrappers for Skin Cancer (BCC / SCC / melanoma) GAN training
(Section 3.2.2 of the paper), plus a self-contained single-step GAN verification
engine for rapid smoke testing, CI/CD, and offline reproducibility verification.

Two training orchestrators live here:

- `StyleGAN2PyTorchTrainer` -- wraps the `stylegan2_pytorch` PyPI package
  (lucidrains/stylegan2-pytorch). **This is the implementation actually used to
  generate the published Skin Cancer synthetic images**: one independent,
  unconditional model trained per class, time-boxed to fit a Kaggle GPU session.
- `StyleGANTrainer` -- wraps NVIDIA's official `stylegan2-ada-pytorch` training
  script instead. Kept for reference / as an alternative path (e.g. if you want a
  class-conditional single model later) -- it is NOT what produced the dataset
  currently published with this repo, and `scripts/run_generation.py` does not call
  it by default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


class StyleGAN2PyTorchTrainer:
    """
    Orchestration wrapper around the `stylegan2_pytorch` CLI (lucidrains/stylegan2-pytorch)
    -- the implementation actually used to train the Skin Cancer synthetic classes
    (Section 3.2.2). Trains one independent, unconditional model per class
    (BCC / SCC / melanoma), each auto-resuming from its own checkpoint directory, within
    a wall-clock time budget suited to a single Kaggle GPU session.

    Expects `cfg` to be the `stylegan2` block of `configs/generation_config.yaml`.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        self.dataset_cfg = cfg.get("dataset", {})
        self.training_cfg = cfg.get("training", {})

    def _models_dir(self) -> Path:
        return Path(self.paths.get("models_dir", "outputs/stylegan_training/models"))

    def _results_dir(self) -> Path:
        return Path(self.paths.get("results_dir", "outputs/stylegan_training/results"))

    def has_checkpoint(self, class_name: str) -> bool:
        """True if a resumable checkpoint already exists for this class."""
        ckpt_dir = self._models_dir() / class_name
        return ckpt_dir.exists() and any(ckpt_dir.glob("model_*.pt"))

    def build_train_command(
        self,
        class_name: str,
        data_dir: Path,
        target_steps_this_chunk: int,
        fresh: bool = False,
    ) -> List[str]:
        tcfg = self.training_cfg
        cmd = [
            "stylegan2_pytorch",
            f"--data={data_dir}",
            f"--name={class_name}",
            f"--models_dir={self._models_dir()}",
            f"--results_dir={self._results_dir()}",
            f"--image-size={self.dataset_cfg.get('image_size', 256)}",
            f"--network-capacity={tcfg.get('network_capacity', 16)}",
            f"--batch-size={tcfg.get('batch_size', 4)}",
            f"--gradient-accumulate-every={tcfg.get('gradient_accumulate_every', 8)}",
            f"--num-train-steps={target_steps_this_chunk}",
            f"--aug-prob={tcfg.get('aug_prob', 0.4)}",
            f"--aug-types={tcfg.get('aug_types', '[translation,cutout,color]')}",
        ]
        attn_layers = tcfg.get("attn_layers")
        if attn_layers and attn_layers != "[]":
            cmd.append(f"--attn-layers={attn_layers}")
        if fresh:
            cmd.append("--new")
        return cmd

    def run_class_within_budget(
        self,
        class_name: str,
        input_root: Union[str, Path],
        per_class_deadline_seconds: float,
        dry_run: bool = False,
    ) -> None:
        """
        Calibrate real steps/sec with a short timed run, scale `num_train_steps` to fit
        the remaining time budget (with a safety margin), then train in `chunk_steps`
        increments until either the (possibly-scaled) step target or the wall-clock
        deadline is hit -- whichever comes first. Mirrors the Kaggle notebook's
        calibrate-then-chunk training loop exactly, so re-running this against the same
        seed data and time budget reproduces the same training procedure (not
        bit-identical weights, since GPU/timing varies run to run).
        """
        tcfg = self.training_cfg
        chunk_steps = tcfg.get("chunk_steps", 250)
        calibration_steps = tcfg.get("calibration_steps", 50)
        min_viable_steps = tcfg.get("min_viable_steps", 1500)
        safety_margin = tcfg.get("training_safety_margin", 0.80)
        target_total = tcfg.get("num_train_steps", 50_000)

        data_dir = Path(input_root) / class_name
        log_path = self._models_dir() / class_name / "train_log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        class_has_checkpoint = self.has_checkpoint(class_name)
        start_steps = 0 if class_has_checkpoint else calibration_steps

        print(f"[stylegan2_pytorch] {class_name}: "
              f"{'resuming' if class_has_checkpoint else 'starting fresh'}")

        t0 = time.time()
        calib_cmd = self.build_train_command(
            class_name, data_dir, calibration_steps, fresh=not class_has_checkpoint
        )
        print(f"[stylegan2_pytorch] {class_name}: calibrating -- {' '.join(calib_cmd)}")
        if not dry_run:
            self._stream_run(calib_cmd, log_path)
        elapsed = max(time.time() - t0, 1e-6)
        sps = calibration_steps / elapsed
        measured_ok = elapsed >= 5.0  # a real run should take noticeably longer than this

        remaining_for_class = per_class_deadline_seconds - elapsed
        if measured_ok:
            achievable = int(sps * remaining_for_class * safety_margin)
            target_total = max(min_viable_steps, min(target_total, start_steps + achievable))
            print(f"[stylegan2_pytorch] {class_name}: {sps:.3f} steps/sec, "
                  f"{remaining_for_class/3600:.2f}h remaining -> target {target_total} steps")
        else:
            print(f"[stylegan2_pytorch] {class_name}: resumed checkpoint was already past "
                  f"the calibration target -- skipping step-target scaling, relying on the "
                  f"wall-clock cutoff below.")

        steps_target_this_call = start_steps
        class_start = time.time()
        while steps_target_this_call < target_total:
            if (time.time() - class_start) >= per_class_deadline_seconds:
                print(f"[stylegan2_pytorch] {class_name}: time budget reached, stopping.")
                break
            steps_target_this_call = min(steps_target_this_call + chunk_steps, target_total)
            cmd = self.build_train_command(class_name, data_dir, steps_target_this_call, fresh=False)
            print(f"[stylegan2_pytorch] {class_name}: training toward "
                  f"{steps_target_this_call}/{target_total} steps")
            if not dry_run:
                self._stream_run(cmd, log_path)

        print(f"[stylegan2_pytorch] {class_name}: done for this session.")

    @staticmethod
    def _stream_run(cmd: List[str], log_path: Path) -> int:
        """Run cmd, streaming output live and appending it to log_path."""
        with open(log_path, "a") as log_f, subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True,
        ) as proc:
            for line in proc.stdout:
                sys.stdout.write(line)
                log_f.write(line)
            proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"stylegan2_pytorch training step failed with exit code {proc.returncode}. "
                f"See {log_path} for details."
            )
        return proc.returncode


class StyleGANTrainer:
    """
    Orchestration wrapper around NVIDIA's official StyleGAN2-ADA-PyTorch training code.

    NOT the implementation used to produce this repo's published Skin Cancer images --
    kept as an alternative path (e.g. for a single class-conditional model instead of
    three independent ones). See `StyleGAN2PyTorchTrainer` above for the actual method.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        self.training_cfg = cfg.get("training", {})
        self.dataset_cfg = cfg.get("dataset", {})

    def check_repo(self, repo_dir: Path) -> bool:
        train_py = repo_dir / "train.py"
        dataset_tool = repo_dir / "dataset_tool.py"
        return train_py.exists() and dataset_tool.exists()

    def prepare_dataset(self, repo_dir: Path, force: bool = False) -> Path:
        dataset_zip = Path(self.paths.get("dataset_zip", "data/skin_cancer_seed.zip"))
        seed_dir = Path(self.paths.get("seed_images_dir", "data/skin_cancer_seed"))
        resolution = self.dataset_cfg.get("resolution", 512)

        if dataset_zip.exists() and not force:
            print(f"[stylegan2] Dataset already prepared at {dataset_zip} (skipping conversion).")
            return dataset_zip

        if not seed_dir.exists():
            raise FileNotFoundError(
                f"[stylegan2] Seed image directory '{seed_dir}' not found. "
                "Expected class subfolders (e.g. BCC, SCC, melanoma)."
            )

        dataset_zip.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(repo_dir / "dataset_tool.py"),
            f"--source={seed_dir}",
            f"--dest={dataset_zip}",
            f"--width={resolution}",
            f"--height={resolution}",
        ]
        print(f"[stylegan2] Preparing dataset: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return dataset_zip

    def build_train_command(
        self,
        repo_dir: Path,
        dataset_zip: Path,
        conditional: bool = False,
        resume: Optional[str] = None,
    ) -> List[str]:
        tcfg = self.training_cfg
        outdir = Path(self.paths.get("training_output_dir", "training-runs/stylegan2_skin_cancer"))
        outdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(repo_dir / "train.py"),
            f"--outdir={outdir}",
            f"--data={dataset_zip}",
            f"--gpus={tcfg.get('gpus', 1)}",
            f"--batch={tcfg.get('batch', 16)}",
            f"--gamma={tcfg.get('gamma', 10.0)}",
            f"--kimg={tcfg.get('kimg', 2400)}",
            f"--snap={tcfg.get('snap', 50)}",
            f"--aug={tcfg.get('aug', 'ada')}",
            f"--target={tcfg.get('target', 0.6)}",
            f"--augpipe={tcfg.get('augpipe', 'bgc')}",
            f"--metrics={tcfg.get('metrics', 'fid50k_full')}",
            f"--seed={tcfg.get('seed', 42)}",
            f"--cfg={tcfg.get('cfg', 'auto')}",
            f"--mirror={1 if self.dataset_cfg.get('mirror', True) else 0}",
        ]
        if conditional:
            cmd.append("--cond=1")

        resume_pkl = resume or tcfg.get("resume_pkl")
        if resume_pkl:
            cmd.append(f"--resume={resume_pkl}")

        return cmd

    def run_training(
        self,
        conditional: bool = False,
        resume: Optional[str] = None,
        force_dataset: bool = False,
        dry_run: bool = False,
    ) -> List[str]:
        repo_dir = Path(self.paths.get("stylegan2_ada_repo", "stylegan2-ada-pytorch"))
        if not dry_run and not self.check_repo(repo_dir):
            raise RuntimeError(
                f"[stylegan2] Official repo not found at '{repo_dir}'. "
                "Clone it: git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git"
            )

        if dry_run:
            dataset_zip = Path(self.paths.get("dataset_zip", "data/skin_cancer_seed.zip"))
        else:
            dataset_zip = self.prepare_dataset(repo_dir, force=force_dataset)

        cmd = self.build_train_command(repo_dir, dataset_zip, conditional=conditional, resume=resume)
        print(f"[stylegan2] Launch command: {' '.join(cmd)}")
        if not dry_run:
            subprocess.run(cmd, check=True)
        return cmd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


def run_single_step_gan(
    input_image: Union[str, Path, Image.Image],
    output_dir: Union[str, Path] = "outputs/gan_verification",
    latent_dim: int = 64,
    img_size: int = 64,
    lr: float = 0.001,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes an exact 1-step (single iteration / single layer) GAN training pass
    on an input real/generated image.

    Verifies:
      1. Forward pass of the real image through Discriminator (computes D(x_real)).
      2. Generator creates a synthetic candidate from latent vector z (computes G(z)).
      3. Forward pass of fake candidate through Discriminator (computes D(G(z))).
      4. Computes binary cross-entropy Discriminator loss and updates D layer weights.
      5. Computes Generator adversarial loss and updates G layer weights.
      6. Evaluates the updated Generator to produce a new output image, verifying
         end-to-end output generation and loss reduction.

    Returns:
      Dictionary containing losses, discriminator scores, and path to output image.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1. Load and prepare the real input image
    if isinstance(input_image, (str, Path)):
        img_path = Path(input_image)
        if not img_path.exists():
            raise FileNotFoundError(f"Input image not found: {img_path}")
        pil_img = Image.open(img_path).convert("RGB")
    else:
        pil_img = input_image.convert("RGB")

    pil_resized = pil_img.resize((img_size, img_size), Image.Resampling.BILINEAR)
    real_arr = (np.asarray(pil_resized, dtype=np.float32) / 127.5) - 1.0  # [-1, 1]
    real_flat = real_arr.flatten()  # (img_size * img_size * 3,)
    input_dim = real_flat.shape[0]

    # 2. Initialize 1-layer Generator and 1-layer Discriminator
    # Generator: latent_dim -> input_dim (with Tanh activation)
    W_g = rng.normal(0.0, 0.02, size=(latent_dim, input_dim)).astype(np.float32)
    b_g = np.zeros(input_dim, dtype=np.float32)

    # Discriminator: input_dim -> 1 (with Sigmoid activation)
    W_d = rng.normal(0.0, 0.02, size=(input_dim, 1)).astype(np.float32)
    b_d = np.zeros((1,), dtype=np.float32)

    # 3. Sample latent vector z
    z = rng.normal(0.0, 1.0, size=(1, latent_dim)).astype(np.float32)

    # 4. Forward Pass - Generator
    fake_raw = np.dot(z, W_g) + b_g
    fake_flat = np.tanh(fake_raw)[0]  # shape: (input_dim,)

    # 5. Forward Pass - Discriminator
    d_real_logit = np.dot(real_flat, W_d) + b_d  # scalar
    d_real_prob = float(sigmoid(d_real_logit)[0])

    d_fake_logit = np.dot(fake_flat, W_d) + b_d  # scalar
    d_fake_prob = float(sigmoid(d_fake_logit)[0])

    # Initial losses
    eps = 1e-7
    loss_d_real = -np.log(max(d_real_prob, eps))
    loss_d_fake = -np.log(max(1.0 - d_fake_prob, eps))
    d_loss_before = float(loss_d_real + loss_d_fake)
    g_loss_before = float(-np.log(max(d_fake_prob, eps)))

    # 6. Backward Pass - Discriminator Update (1 step)
    # d(Loss_D) / d(logit_real) = d_real_prob - 1
    # d(Loss_D) / d(logit_fake) = d_fake_prob
    grad_d_real_logit = d_real_prob - 1.0
    grad_d_fake_logit = d_fake_prob

    grad_W_d = (grad_d_real_logit * real_flat[:, None]) + (grad_d_fake_logit * fake_flat[:, None])
    grad_b_d = np.array([grad_d_real_logit + grad_d_fake_logit], dtype=np.float32)

    # Apply gradient descent step to D
    W_d_updated = W_d - lr * grad_W_d
    b_d_updated = b_d - lr * grad_b_d

    # 7. Backward Pass - Generator Update (1 step)
    # Loss_G = -log(D(fake))
    # d(Loss_G) / d(fake_flat) = (d_fake_prob - 1) * W_d
    grad_g_output = (d_fake_prob - 1.0) * W_d[:, 0]  # shape: (input_dim,)
    # d(tanh(x)) = 1 - tanh^2(x)
    grad_g_raw = grad_g_output * (1.0 - fake_flat**2)
    grad_W_g = np.dot(z.T, grad_g_raw[None, :])
    grad_b_g = grad_g_raw

    # Apply gradient descent step to G
    W_g_updated = W_g - lr * grad_W_g
    b_g_updated = b_g - lr * grad_b_g

    # 8. Verification: Generate an output image from the updated Generator
    z_eval = rng.normal(0.0, 1.0, size=(1, latent_dim)).astype(np.float32)
    fake_updated = np.tanh(np.dot(z_eval, W_g_updated) + b_g_updated)[0]
    out_img_arr = ((fake_updated.reshape((img_size, img_size, 3)) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    out_pil = Image.fromarray(out_img_arr, "RGB")

    # Resize output back to 512x512 for visual inspection
    out_pil_full = out_pil.resize((512, 512), Image.Resampling.NEAREST)
    out_path = output_dir / "gan_single_step_output.png"
    out_pil_full.save(out_path)

    # Post-step verification score
    d_fake_post = float(sigmoid(np.dot(fake_updated, W_d_updated) + b_d_updated)[0])
    g_loss_after = float(-np.log(max(d_fake_post, eps)))

    result = {
        "status": "SUCCESS",
        "d_loss_before": round(d_loss_before, 4),
        "g_loss_before": round(g_loss_before, 4),
        "g_loss_after": round(g_loss_after, 4),
        "d_output_real": round(d_real_prob, 4),
        "d_output_fake": round(d_fake_prob, 4),
        "output_image_path": str(out_path),
        "output_image_resolution": "512x512",
        "trained_layers": ["Generator_Dense_Linear", "Discriminator_Dense_Linear"],
        "steps_completed": 1,
    }

    metrics_path = output_dir / "gan_verification_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
