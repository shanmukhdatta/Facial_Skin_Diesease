"""Synthetic image generation package (Diffusion Models & StyleGAN2)."""
from src.generation.prompts import PromptBuilder, load_or_build_prompts, build_final_prompt
from src.generation.diffusion_generator import DiffusionGenerator
from src.generation.stylegan_generator import StyleGAN2PyTorchGenerator, StyleGANCancerGenerator
from src.generation.stylegan_trainer import StyleGAN2PyTorchTrainer, StyleGANTrainer, run_single_step_gan

__all__ = [
    "PromptBuilder",
    "load_or_build_prompts",
    "build_final_prompt",
    "DiffusionGenerator",
    "StyleGAN2PyTorchGenerator",   # actual Skin Cancer generation method used
    "StyleGANCancerGenerator",     # reference-only NVIDIA-ADA alternative
    "StyleGAN2PyTorchTrainer",     # actual Skin Cancer training method used
    "StyleGANTrainer",             # reference-only NVIDIA-ADA alternative
    "run_single_step_gan",
]
