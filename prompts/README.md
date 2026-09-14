# Structured Diffusion Prompts for Facial Skin Disease Synthesis

This directory contains the structured diffusion prompt corpus for the 5 prompt-generated
disease classes, plus the metadata each prompt was built from. Prompts are consumed by
[`scripts/run_generation.py`](../scripts/run_generation.py) via
[`src/generation/prompts.py`](../src/generation/prompts.py), and the actual sampling
parameters (scheduler, steps, guidance scale, seeding, negative prompt) live in
[`configs/generation_config.yaml`](../configs/generation_config.yaml) under the
`diffusion:` key, not in this directory.

The Skin Cancer class (BCC / SCC / melanoma) is **not** generated from these text prompts —
it is produced by a separate per-class StyleGAN2 pipeline; see
[`src/generation/stylegan_generator.py`](../src/generation/stylegan_generator.py) /
[`src/generation/stylegan_trainer.py`](../src/generation/stylegan_trainer.py) and the
`stylegan2:` block of `configs/generation_config.yaml`. That pipeline uses the
`stylegan2_pytorch` package (lucidrains/stylegan2-pytorch) — an alternative path built on
NVIDIA's official `stylegan2-ada-pytorch` is kept in the same files under the `stylegan2_ada`
config key purely for reference, and is **not** what this repository's generation scripts
run by default.

---

## File Inventory

Each class has a plain-text prompt file (one prompt per line) and a matching `_meta.jsonl`
file (one JSON record per line, in the same order, with the structured fields — phenotype,
gender, variation group, age, severity, Fitzpatrick type, anatomical distribution, and
per-image seed — that the prompt was built from). `scripts/run_generation.py` loads these
if present, and otherwise regenerates them from `PromptBuilder` and writes them back out.

| File | Target Class | Phenotypes |
| :--- | :--- | :--- |
| [`acne.txt`](./acne.txt) / [`acne_prompts.txt`](./acne_prompts.txt), [`acne_meta.jsonl`](./acne_meta.jsonl) | Acne | comedonal, papulopustular, nodular |
| [`vitiligo.txt`](./vitiligo.txt) / [`vitiligo_prompts.txt`](./vitiligo_prompts.txt), [`vitiligo_meta.jsonl`](./vitiligo_meta.jsonl) | Vitiligo | segmental, non-segmental, acrofacial |
| [`fungal_infection.txt`](./fungal_infection.txt) / [`fungal_infection_prompts.txt`](./fungal_infection_prompts.txt), [`fungal_infection_meta.jsonl`](./fungal_infection_meta.jsonl) | Fungal Infection | tinea faciei, tinea versicolor, cutaneous candidiasis |
| [`normal_skin.txt`](./normal_skin.txt) / [`normal_skin_prompts.txt`](./normal_skin_prompts.txt), [`normal_skin_meta.jsonl`](./normal_skin_meta.jsonl) | Normal Skin | oily, dry, normal |
| [`hyperpigmentation.txt`](./hyperpigmentation.txt) / [`hyperpigmentation_prompts.txt`](./hyperpigmentation_prompts.txt), [`hyperpigmentation_meta.jsonl`](./hyperpigmentation_meta.jsonl) | Hyperpigmentation | melasma, dyschromia, post-inflammatory hyperpigmentation |

Each class has exactly 300 prompts: 3 phenotypes × 2 genders × 5 variation groups × 10 prompts per group. `scripts/run_generation.py` generates `images_per_prompt` images per prompt (5 by default, set in `configs/generation_config.yaml` → `diffusion.generation.images_per_prompt`), for up to 1,500 images per class.

---

## Hierarchical Prompt Structure

Rather than unconstrained free-text prompts, each prompt is built deterministically by
`PromptBuilder` (`src/generation/prompts.py`) from a fixed hierarchy:

```
Disease Class (e.g., acne)
│
├── Phenotype (3 per class, 100 prompts each)
│   ├── comedonal
│   ├── papulopustular
│   └── nodular
│
├── Gender (2 per phenotype, 50 prompts each)
│   ├── female
│   └── male
│
└── Variation Group (5 per gender, 10 prompts each)
    ├── age                     — age drawn from one of 3 configured age ranges
    ├── severity                — mild / moderate / severe
    ├── skin_tone               — Fitzpatrick type I–VI
    ├── anatomical_distribution — one of the class's configured facial locations
    └── combined                — age, severity, Fitzpatrick type and distribution
                                    all randomized together
```

Outside its own variation axis, each group uses the configured default age range,
severity, Fitzpatrick type, and the first configured distribution for that class — see
`configs/generation_config.yaml` → `diffusion.variations` for the exact defaults and
value lists. Every prompt's per-image seed is derived deterministically from
`(class, phenotype, gender, variation_group, index)` via `stable_seed()`, so rebuilding
a prompt file from scratch reproduces byte-identical prompts and metadata.

---

## Phenotype Descriptors

Each phenotype's descriptive text (used verbatim inside the generated prompts) is defined
in `PHENOTYPES` in `src/generation/prompts.py`:

### Acne
- **Comedonal:** blackheads, whiteheads, clogged pores, small non-inflamed bumps, minimal redness.
- **Papulopustular:** red papules, inflamed bumps, pustules with white pus-filled centers, surrounding redness.
- **Nodular:** large deep nodules, firm swollen lumps, intense inflammation, pronounced redness, deep lesions.

### Vitiligo
- **Segmental:** unilateral, sharply localized depigmented patch with an irregular border.
- **Non-segmental:** multiple symmetric depigmented patches with well-demarcated borders.
- **Acrofacial:** depigmented patches around the eyes, mouth, and nose with irregular margins.

### Fungal Infection
- **Tinea faciei:** annular, scaly, erythematous plaque with a raised, well-defined border and mild central clearing.
- **Tinea versicolor:** multiple small hypo/hyperpigmented scaly patches with fine surface scaling.
- **Cutaneous candidiasis:** moist erythematous patches with satellite pustules in a skin-fold area.

### Normal Skin
- **Oily:** shiny skin with visible sebum, slightly enlarged pores, smooth even complexion.
- **Dry:** matte texture with fine flaking, subtle roughness, even healthy complexion.
- **Normal:** balanced, healthy, even-toned skin, minimal pore visibility, no visible lesions.

### Hyperpigmentation
- **Melasma:** symmetric brown-to-gray patches on the cheeks, forehead, and upper lip.
- **Dyschromia:** diffuse, mottled areas of uneven skin tone.
- **Post-inflammatory hyperpigmentation:** localized dark brown-black macules at sites of previous inflammation.

See `DISTRIBUTIONS` in the same file for each class's set of anatomical-location phrases used by the `anatomical_distribution` and `combined` variation groups.

---

## Final Prompt Assembly

The text stored in this directory is the *repo-generated* prompt. Before a prompt is
actually sent to the diffusion model, `build_final_prompt()`
(`src/generation/prompts.py`) prepends a fixed photorealism prefix and strips the
repo's own generic trailing clause (starting at `"Natural facial anatomy"`) so it isn't
duplicated — see that function's docstring for the exact matching rule. The prefix
currently configured in `configs/generation_config.yaml` is:

```text
Photorealistic clinical dermatology photograph, real human face, natural realistic
skin texture, visible pores, true-to-life skin color, unretouched skin, single face,
neutral expression, sharp photographic detail.
```

---

## Generation Command

```bash
# Verify/rebuild prompt manifests only, without touching a GPU or loading the model
python scripts/run_generation.py --method prompts_only

# Generate the full configured set for one or more classes
python scripts/run_generation.py --method diffusion --classes acne vitiligo

# Small test batch
python scripts/run_generation.py --method diffusion --classes acne --limit_prompts 5 --images_per_prompt 2
```

Sampling parameters (scheduler, inference steps, guidance scale, seeding formula,
negative prompt) are not CLI flags — they're set in `configs/generation_config.yaml`
under `diffusion.model` / `diffusion.generation` / `diffusion.seeding`. As configured
there:

- **Base model:** `SG161222/Realistic_Vision_V5.1_noVAE`, with the `stabilityai/sd-vae-ft-mse` VAE
- **Scheduler:** DPM-Solver++ with Karras sigmas
- **Steps:** 40
- **Guidance scale:** 7.5
- **Resolution:** 512 × 512 (resized to 224 × 224 for classifier training)
- **Negative prompt:** see `diffusion.generation.negative_prompt` in `configs/generation_config.yaml`

Generated images below a minimum pixel-standard-deviation threshold (`diffusion.quality_control.min_pixel_std`) are treated as degenerate/blank and automatically retried with a different seed, up to `diffusion.generation.max_retries_per_image` attempts; rejections are logged to `diffusion.paths.rejected_log`.
