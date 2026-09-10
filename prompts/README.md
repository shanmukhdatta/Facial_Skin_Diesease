# Structured Diffusion Prompts for Facial Skin Disease Synthesis

This directory contains the complete corpus of **1,500 structured diffusion prompts** used to generate 7,500 photorealistic facial skin disease images across five disease classes using **Realistic Vision V5.1** (Stable Diffusion 1.5 fine-tune). Combined with 1,500 StyleGAN2-ADA generated Skin Cancer images, this forms the **9,000-image balanced synthetic dataset** described in the paper:

> **"Enhancing Facial Skin Disease Detection Through Synthetic Data Generation Using Diffusion Models, GANs, and Pre-Trained CNN Architectures"**

---

## 🗂️ File Inventory

| File | Target Class | Subtypes / Phenotypes | Genders | Prompts | Images (5 seeds/prompt) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`acne_prompts.txt`](./acne_prompts.txt) | **Acne** | Comedonal, Papulopustular, Nodulocystic | Female (150), Male (150) | 300 | 1,500 |
| [`vitiligo_prompts.txt`](./vitiligo_prompts.txt) | **Vitiligo** | Segmental, Non-segmental / Generalized, Acrofacial | Female (150), Male (150) | 300 | 1,500 |
| [`fungal_infection_prompts.txt`](./fungal_infection_prompts.txt) | **Fungal Infection** | Tinea faciei, Tinea versicolor, Cutaneous candidiasis | Female (150), Male (150) | 300 | 1,500 |
| [`normal_skin_prompts.txt`](./normal_skin_prompts.txt) | **Normal Skin** | Oily, Dry, Normal surface texture | Female (150), Male (150) | 300 | 1,500 |
| [`hyperpigmentation_prompts.txt`](./hyperpigmentation_prompts.txt) | **Hyperpigmentation** | Melasma, Post-inflammatory (PIH), Dyschromia | Female (150), Male (150) | 300 | 1,500 |
| **Total (5 Classes)** | | **15 Subtypes** | **Balanced (1:1)** | **1,500** | **7,500** |

*(Note: Skin Cancer is generated via StyleGAN2-ADA using 300 real seed lesion images [100 BCC, 100 SCC, 100 Melanoma] yielding 1,500 images; see [`generation/stylegan2_ada/`](../generation/stylegan2_ada/README.md).)*

---

## 🌳 Hierarchical Condition Tree

Rather than using generic, unconstrained text prompts (e.g., *"a person with acne"*), prompts are systematically synthesized via a 5-level hierarchical generation condition tree:

```
Disease Class (e.g., Acne)
│
├── Subtype / Phenotype (3 per class, 100 prompts each)
│   ├── Comedonal acne
│   ├── Papulopustular acne
│   └── Nodular / nodulocystic acne
│
├── Demographic Gender (Balanced 1:1, 50 prompts per gender per subtype)
│   ├── Female (Prompts 001–050)
│   └── Male   (Prompts 051–100)
│
└── Variation Axes (10 prompts per group per gender)
    ├── [01–10 AGE VARIATION]                     (Ages 15 to 48 years)
    ├── [11–20 SEVERITY VARIATION]                (Mild, Moderate, Severe)
    ├── [21–30 SKIN-TONE VARIATION]               (Fitzpatrick Types I to VI)
    ├── [31–40 ANATOMICAL-DISTRIBUTION VARIATION] (Cheeks, forehead, chin, perioral, etc.)
    └── [41–50 COMBINED ORTHOGONAL VARIATION]     (Multi-factorial combinations)
```

---

## 🔬 Subtypes and Phenotypic Descriptors

Each prompt explicitly specifies pathognomonic clinical features, lesion morphology, and negative diagnostic exclusions:

### 1. Acne (`acne_prompts.txt`)
- **Comedonal Acne:** Open and closed comedones (blackheads and whiteheads), visibly clogged follicular orifices, non-inflamed papules, absence of dominant nodules or cysts.
- **Papulopustular Acne:** Inflammatory erythematous papules, superficial pustules containing purulent exudate, surrounding perilesional erythema.
- **Nodular / Nodulocystic Acne:** Deep-seated inflammatory nodules, cystic lesions, pronounced edema, induration, and tissue involvement.

### 2. Vitiligo (`vitiligo_prompts.txt`)
- **Segmental Vitiligo:** Unilateral depigmented macules/patches strictly respecting the midline, dermatomal distribution, trichrome borders.
- **Non-segmental (Generalized) Vitiligo:** Bilateral, symmetric chalk-white macules, progressive borders, follicular repigmentation islands.
- **Acrofacial Vitiligo:** Distal extremity and periorificial facial predilection (perioral, periocular, nasal tip).

### 3. Fungal Infection (`fungal_infection_prompts.txt`)
- **Tinea Faciei:** Annular erythematous plaques with central clearing, active scaly advancing borders, vesicular margins.
- **Tinea Versicolor (Pityriasis Versicolor):** Confluent hypo- or hyperpigmented macules with delicate branny scale (collarette).
- **Cutaneous Candidiasis:** Beefy red erythematous patches, macerated skin folds, characteristic satellite pustules and papules.

### 4. Normal Skin (`normal_skin_prompts.txt`)
- **Oily Skin Phenotype:** Visible follicular sebum, cutaneous shine, mildly dilated pores, greasy reflection, absence of pathological eruptions.
- **Dry Skin Phenotype:** Fine epidermal xerosis, subtle scaling, matte surface, absence of erythema or excoriation.
- **Normal Surface Phenotype:** Balanced epidermal hydration, smooth stratum corneum, uniform texture, physiological pores.

### 5. Hyperpigmentation (`hyperpigmentation_prompts.txt`)
- **Melasma:** Reticulated, symmetrical light-to-dark brown macules/patches in centrofacial, malar, or mandibular patterns.
- **Post-Inflammatory Hyperpigmentation (PIH):** Irregular hyperpigmented macules localized to sites of resolved inflammatory lesions.
- **Dyschromia:** Mottled, patchy variations in cutaneous pigmentary uniformity with sun-exposed facial distribution.

---

## 🎨 Controlled Variation Axes

Across every subtype and gender block, 5 orthogonal axes ensure extensive intra-class diversity:

| Variation Group | Prompt Range | Parameterized Variables | Clinical Rationale |
| :--- | :--- | :--- | :--- |
| **Age Variation** | `01–10` | Exact ages: 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 (female) / 25–48 (male) | Alters epidermal thickness, elasticity, sebum production, and wrinkle baselines. |
| **Severity Variation** | `11–20` | `Mild` (11–13), `Moderate` (14–16, 20), `Severe` (17–19) | Prevents classifiers from overfitting solely to extreme lesion presentations. |
| **Skin-Tone Variation** | `21–30` | **Fitzpatrick Types I to VI** (Fair, Pale, Medium, Olive, Brown, Dark brown/black) | Eliminates demographic bias across diverse patient phototypes. |
| **Anatomical Distribution** | `31–40` | Cheeks, forehead, chin, temples, nasal region, jawline, perioral, multifocal | Trains spatial invariance across different facial anatomical landmarks. |
| **Combined Variation** | `41–50` | Orthogonal cross-product of age, Fitzpatrick type, severity, and distribution | Stresses generalizability across intersectional demographic/pathological factors. |

---

## ⚙️ Generative Pipeline & Execution

Prompts are parsed and executed using [`generation/diffusion/generate_diffusion.py`](../generation/diffusion/generate_diffusion.py):

```bash
# Example: Generate 1,500 images for Acne (300 prompts × 5 seeds)
python generation/diffusion/generate_diffusion.py \
    --config generation/diffusion/config.yaml \
    --classes acne \
    --output-dir data/synthetic/acne \
    --images-per-prompt 5
```

Sampling parameters (`guidance_scale`, `num_inference_steps`, `seed_base`,
scheduler, etc.) are not CLI flags — set them in
[`generation/diffusion/config.yaml`](../generation/diffusion/config.yaml)
instead. Use `--dry-run` to inspect/rebuild the prompt files without
touching the GPU or loading the diffusion model.

### Recommended Diffusion Hyperparameters
- **Base Model:** Realistic Vision V5.1 (SD 1.5 fine-tune)
- **Sampler:** DPM++ 2M Karras
- **Steps:** 30
- **Guidance Scale (CFG):** 7.0
- **Resolution:** 512 × 512 (resized to 224 × 224 for CNN training)
- **Negative Prompt:**
  ```text
  deformed, bad anatomy, bad eyes, disfigured, poorly drawn face, mutation, mutated,
  extra limb, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs,
  malformed hands, blur, out of focus, long neck, surreal, cartoon, 3d render, anime,
  plastic skin, mannequin, airbrushed, smooth filter, watermark, signature
  ```

---

## 🩺 Quality Validation (Human-in-the-Loop)

As detailed in Section 3.2.3 of the paper, all generated images are screened before training inclusion:
1. **Clinical Plausibility:** Verifying that lesion morphology matches the target disease and subtype description.
2. **Anatomical Coherence:** Eliminating structural facial artifacts, double eyes, distorted noses, or uncanny skin blurring.
3. **Artifact-Free Boundaries:** Checking that the transition between affected lesion and unaffected skin is natural and gradual.
4. **Degenerate Sample Rejection:** Automatic filtering based on edge density and variance thresholds in `generate_diffusion.py`.
