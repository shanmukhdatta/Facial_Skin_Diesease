from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# --------------------------------------------------------------------------- #
# Disease-specific phenotype / subtype descriptor vocabulary (Section 3.2.1)
# --------------------------------------------------------------------------- #
PHENOTYPES: Dict[str, List[Tuple[str, str]]] = {
    "acne": [
        ("comedonal", "blackheads, whiteheads, clogged pores, small non-inflamed bumps, and minimal redness"),
        ("papulopustular", "red papules, inflamed bumps, pustules, white pus-filled centers, and surrounding redness"),
        ("nodular", "large deep nodules, firm swollen lumps, intense inflammation, pronounced redness, and deep lesions"),
    ],
    "vitiligo": [
        ("segmental", "a unilateral, sharply localized depigmented patch on one side of the face with an irregular border and clear contrast with the surrounding skin"),
        ("non-segmental", "multiple symmetric depigmented patches distributed on both sides of the face with well-demarcated borders"),
        ("acrofacial", "depigmented patches concentrated around the eyes, mouth, and nose with irregular margins"),
    ],
    "fungal_infection": [
        ("tinea faciei", "an annular, scaly, erythematous plaque with a raised, well-defined border and mild central clearing"),
        ("tinea versicolor", "multiple small hypopigmented or hyperpigmented scaly patches with fine surface scaling"),
        ("cutaneous candidiasis", "moist erythematous patches with satellite pustules in a skin-fold area of the face"),
    ],
    "normal_skin": [
        ("oily", "shiny, oily skin with visible sebum, slightly enlarged pores, and a smooth even complexion without lesions"),
        ("dry", "matte, dry skin texture with fine flaking, subtle roughness, and an even healthy complexion without lesions"),
        ("normal", "balanced, healthy, even-toned skin with a smooth surface, minimal pore visibility, and no visible lesions"),
    ],
    "hyperpigmentation": [
        ("melasma", "symmetric brown-to-gray patches on the cheeks, forehead, and upper lip with irregular but well-defined borders"),
        ("dyschromia", "diffuse, mottled areas of uneven skin tone with patchy pigmentation across the face"),
        ("post-inflammatory hyperpigmentation", "localized dark brown-black macules at sites of previous inflammation or lesions"),
    ],
}

DISTRIBUTIONS: Dict[str, List[str]] = {
    "acne": [
        "on the cheeks", "on the forehead", "on the chin and jawline",
        "across the T-zone", "on the cheeks and nose", "diffusely across the face"
    ],
    "vitiligo": [
        "on one side of the face", "symmetrically on both cheeks",
        "around the eyes and mouth", "on the forehead", "along the jawline", "diffusely across the face"
    ],
    "fungal_infection": [
        "on the cheek", "on the jawline", "around the mouth and chin",
        "on the forehead", "on the temple", "across the lower face"
    ],
    "normal_skin": [
        "across the entire face", "on the T-zone", "on the cheeks",
        "across the forehead and cheeks", "evenly across the face", "on the cheeks and chin"
    ],
    "hyperpigmentation": [
        "on the cheeks and forehead", "on the upper lip and cheeks", "around the eyes",
        "on the forehead", "on the jawline", "diffusely across the face"
    ],
}

VARIATION_GROUPS = ["age", "severity", "skin_tone", "anatomical_distribution", "combined"]


def stable_seed(*parts: Any) -> int:
    """Deterministic 32-bit integer seed derived from hashable string components."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) % (2**32)


def rng_for(*parts: Any) -> np.random.Generator:
    """Return a deterministic NumPy random generator seeded by key parts."""
    return np.random.default_rng(stable_seed(*parts))


class PromptBuilder:
    """
    Hierarchical prompt-conditioning builder for diffusion-based skin disease synthesis.
    Implements: 3 phenotypes x 5 variation groups x 10 prompts x 2 genders = 300 prompts per class.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        variations = cfg.get("variations", {})
        self.age_ranges: Dict[str, List[int]] = variations.get("age_ranges", {
            "15-24": [16, 24],
            "25-39": [26, 38],
            "40-59": [41, 58],
        })
        self.default_age_range: str = variations.get("default_age_range", "25-39")
        self.severities: List[str] = variations.get("severity_levels", ["mild", "moderate", "severe"])
        self.default_severity: str = variations.get("default_severity", "moderate")
        self.fitz_types: List[str] = variations.get("fitzpatrick_types", ["I", "II", "III", "IV", "V", "VI"])
        self.default_fitz: str = variations.get("default_fitzpatrick", "III")

        prompt_struct = cfg.get("prompt_structure", {})
        self.genders: List[str] = prompt_struct.get("genders", ["female", "male"])
        self.n_per_group: int = prompt_struct.get("prompts_per_group", 10)

    def _age_value(self, age_range_label: str, rng: np.random.Generator) -> int:
        lo, hi = self.age_ranges[age_range_label]
        return int(rng.integers(lo, hi + 1))

    def _default_age(self) -> int:
        lo, hi = self.age_ranges[self.default_age_range]
        return (lo + hi) // 2

    @staticmethod
    def _article(word: str) -> str:
        return "an" if word[:1].lower() in "aeiou" else "a"

    _DISEASE_NOUN = {"acne": "acne", "vitiligo": "vitiligo"}

    def _build_text(
        self,
        class_name: str,
        phenotype_label: str,
        phenotype_desc: str,
        gender: str,
        age: int,
        severity: str,
        fitz: str,
        distribution: str,
    ) -> str:
        noun = self._DISEASE_NOUN.get(class_name)
        phenotype_display = f"{phenotype_label} {noun}" if noun else phenotype_label

        if class_name == "normal_skin":
            article = self._article(phenotype_label)
            return (
                f"Photorealistic close-up facial photograph of a {age}-year-old {gender} "
                f"with Fitzpatrick type {fitz} skin, showing {article} {phenotype_label} skin "
                f"phenotype {distribution}, with {phenotype_desc}. Natural facial anatomy, "
                f"realistic skin texture, healthy even complexion, and clinically plausible "
                f"skin surface characteristics, shot in soft even studio lighting."
            )
        return (
            f"Photorealistic close-up facial photograph of a {age}-year-old {gender} "
            f"with Fitzpatrick type {fitz} skin, showing {severity} {phenotype_display} "
            f"{distribution}, with {phenotype_desc}. Natural facial anatomy, realistic "
            f"skin texture, visible pores, and clinically plausible disease morphology, "
            f"shot in soft even studio lighting."
        )

    def build_for_class(self, class_name: str) -> List[Dict[str, Any]]:
        """Construct exactly 300 structured prompts with metadata for a given disease class."""
        if class_name not in PHENOTYPES:
            raise ValueError(f"Unknown class: {class_name}. Supported: {list(PHENOTYPES.keys())}")

        phenotypes = PHENOTYPES[class_name]
        distributions = DISTRIBUTIONS[class_name]
        entries: List[Dict[str, Any]] = []
        pid = 0

        for phenotype_label, phenotype_desc in phenotypes:
            for gender in self.genders:
                for group in VARIATION_GROUPS:
                    for idx in range(self.n_per_group):
                        rng = rng_for(class_name, phenotype_label, gender, group, idx)

                        age_range = self.default_age_range
                        severity = self.default_severity
                        fitz = self.default_fitz
                        distribution = distributions[0]
                        age = self._default_age()

                        if group == "age":
                            age_range = list(self.age_ranges.keys())[idx % 3]
                            age = self._age_value(age_range, rng)
                        elif group == "severity":
                            severity = self.severities[idx % len(self.severities)]
                        elif group == "skin_tone":
                            fitz = self.fitz_types[idx % len(self.fitz_types)]
                        elif group == "anatomical_distribution":
                            distribution = distributions[idx % len(distributions)]
                        elif group == "combined":
                            age_range = rng.choice(list(self.age_ranges.keys()))
                            age = self._age_value(str(age_range), rng)
                            severity = rng.choice(self.severities)
                            fitz = rng.choice(self.fitz_types)
                            distribution = rng.choice(distributions)

                        text = self._build_text(
                            class_name=class_name,
                            phenotype_label=phenotype_label,
                            phenotype_desc=phenotype_desc,
                            gender=gender,
                            age=age,
                            severity=severity,
                            fitz=fitz,
                            distribution=distribution,
                        )

                        entries.append({
                            "id": pid,
                            "class": class_name,
                            "phenotype": phenotype_label,
                            "gender": gender,
                            "variation_group": group,
                            "age": age,
                            "severity": severity,
                            "fitzpatrick": fitz,
                            "distribution": distribution,
                            "prompt": text,
                            "seed_base": stable_seed(class_name, phenotype_label, gender, group, idx, "img"),
                        })
                        pid += 1
        return entries


# --------------------------------------------------------------------------- #
# Final-prompt post-processing (Section 5b of the Kaggle generation notebook)
# --------------------------------------------------------------------------- #
# Every prompt actually sent to the diffusion model has this short photorealistic
# prefix prepended, and the repo's own generic trailing clause (which starts with
# GENERIC_TRAILING_ANCHOR) stripped so it isn't duplicated. The strip only fires when
# that anchor text runs, uninterrupted, all the way to the prompt's final period --
# if the wording upstream ever changes, nothing is removed and the prefix is simply
# prepended to the untouched original text, rather than risking a cut of real
# disease/subtype/severity detail.
CUSTOM_PROMPT_PREFIX = (
    "Photorealistic clinical dermatology photograph, "
    "real human face, natural realistic skin texture, visible pores, "
    "true-to-life skin color, unretouched skin, single face, "
    "neutral expression, sharp photographic detail. "
)

GENERIC_TRAILING_ANCHOR = "Natural facial anatomy"


def build_final_prompt(raw_prompt: str, prefix: str = CUSTOM_PROMPT_PREFIX) -> str:
    """
    Prepend the photorealistic prefix to a repo-generated prompt.

    Strips the repo's own known generic trailing clause (starting at
    GENERIC_TRAILING_ANCHOR) ONLY when it matches the exact known template ending
    (i.e. the anchor text runs, uninterrupted, to the final period). This never
    touches anything before the anchor -- all disease/subtype/severity/age/
    Fitzpatrick/location/morphology content from the repo prompt is preserved
    unchanged. If the anchor isn't found in the expected trailing position, no text
    is removed -- some duplicated generic wording is accepted rather than risking a
    cut of real content.
    """
    trimmed = raw_prompt
    anchor_idx = raw_prompt.find(GENERIC_TRAILING_ANCHOR)
    if anchor_idx != -1:
        tail = raw_prompt[anchor_idx:]
        if tail.rstrip().endswith("."):
            trimmed = raw_prompt[:anchor_idx].rstrip()
    return f"{prefix}{trimmed}"


def load_or_build_prompts(class_name: str, cfg: Dict[str, Any], prompts_dir: Path) -> List[Dict[str, Any]]:
    """Load prompts from `<prompts_dir>/<class>.txt` if present, else build and save."""
    prompts_dir = Path(prompts_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    txt_path = prompts_dir / f"{class_name}.txt"
    if not txt_path.exists() and (prompts_dir / f"{class_name}_prompts.txt").exists():
        txt_path = prompts_dir / f"{class_name}_prompts.txt"
    meta_path = prompts_dir / f"{class_name}_meta.jsonl"

    if txt_path.exists():
        lines = [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if meta_path.exists():
            meta = [json.loads(ln) for ln in meta_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if len(meta) == len(lines):
                return meta
        return [
            {
                "id": i,
                "class": class_name,
                "phenotype": "custom",
                "gender": "unspecified",
                "variation_group": "custom",
                "prompt": text,
                "seed_base": stable_seed(class_name, "custom", i),
            }
            for i, text in enumerate(lines)
        ]

    builder = PromptBuilder(cfg)
    entries = builder.build_for_class(class_name)
    txt_path.write_text("\n".join(e["prompt"] for e in entries), encoding="utf-8")
    with meta_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return entries
