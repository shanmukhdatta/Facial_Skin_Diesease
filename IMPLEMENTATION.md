# DermaLens Facial Skin Research — Implementation & Execution Tracker

> **Repository:** `shanmukhdatta/Facial_Skin_Diesease`  
> **Tracking style:** Work packages and evidence gates, **not calendar days**.  
> **Status updated:** 2026-10-30.  
> **Scope:** Six-class facial skin image classification, controlled synthetic generation and research reporting. This is research software, **not a diagnostic medical device**.

## 0. Operating rules

- [ ] Treat this document as the execution checklist; tick items **only after evidence exists**. A notebook being written or passing a static review is not equivalent to a successful Kaggle run.
- [ ] Retain the existing notebooks and code; patch only demonstrated defects. Do not replace working pipelines or change research hypotheses merely to make notebooks run.
- [ ] Keep separate records for **implemented**, **statically checked**, **pilot executed**, **full executed**, **verified**.
- [ ] Never claim a perfect clinical image, a leakage-free split, or a published-grade result without review evidence.
- [ ] Record Git commit SHA, Kaggle notebook version, dataset version, configuration, random seed, model/checkpoint revision, output location, validation metrics, and failures for each run.
- [ ] Preserve immutable validation/test sets and avoid test-driven model selection.
- [ ] Check source licenses, consent/privacy constraints and permitted synthetic reuse before release; de-identify research outputs as appropriate.
- [ ] Do not commit datasets, generated image archives, large checkpoints, tokens or API credentials to Git. Use Kaggle dataset versions/artifact storage and Git-tracked manifests.

## 1. Current evidence baseline (do not mistake this for GPU verification)

- [x] Six target classes defined: `Acne`, `Fungal_Infection`, `Vitiligo`, `Hyperpigmentation`, `Skin_Cancer`, `Normal`.
- [x] Real-image duplicate grouping and grouped train/validation/test splitting **implemented** in an existing Kaggle notebook.
- [x] Synthetic prompt-group separation accepted as the leakage-control strategy.
- [x] Notebook generation/training infrastructure and static review reported in previous project work.
- [ ] Reconcile existing repository notebooks against the **latest local/Kaggle 30-notebook audit**. The repository README may describe an earlier model/generation stack; it is not evidence that current Kaggle notebooks are synchronized.
- [ ] Confirm which notebooks are genuinely available in this Git branch and which exist only in Kaggle/local storage.
- [ ] Capture a reproducibility inventory: notebook filename, class, model, git SHA, input paths, expected outputs, dependencies, and whether runtime was actually tested.

**Important discrepancy to resolve:** The repository README currently describes a Realistic Vision/StyleGAN2 generation route and an older classifier lineup, while the latest discussed research plan uses class-specific generation experiments and four classifiers (EfficientNet-B0, ConvNeXt V2 Tiny, DINOv2 ViT-B/14, PanDerm Base). The **actual notebook code/checkpoint provenance**, not the README, must determine what can run. Do not silently substitute these models.

## 2. WP-A — Real dataset gate (dependency for trustworthy training)

- [ ] Mount the real dataset at `/kaggle/input/datasets/shanmukhadattaboda/facial-skin-conditions-real-image-collection/Face_Skin_datset` or resolve the exact current Kaggle mount path.
- [ ] Produce counts, source attribution and file manifests for all six classes.
- [ ] **Manually review labels**, especially files labeled Skin_Cancer that may actually be SJS/TEN and Normal files that may depict inflammatory conditions or contain source watermarks.
- [ ] Exclude or relabel only with documented justification; record removals and remaining class counts. Never treat those mislabeled images as ground truth.
- [ ] Check corrupt/black/near-empty images, face/head relevance and duplicates.
- [ ] Run real-image SHA/pixel hash, flip-aware pHash and embedding-based grouping with review of ambiguous near-duplicate clusters.
- [ ] Export frozen grouped `train/val/test` assignments (target near 70/15/15 as group sizes permit), class distribution and leakage report.
- [ ] Preserve original image aspect ratio during preprocessing and use model-specific normalizations.
- [ ] **Gate A PASS:** review log signed off, no known cross-split exact/near-duplicate leakage, immutable manifest, dataset version recorded.

## 3. WP-B — Synthetic prompts and generation contracts

- [ ] Verify source-controlled prompt manifests for each class; confirm whether the intended 300 prompts per class and 210/45/45 prompt-group allocation are actually present, not merely planned.
- [ ] Ensure each image references `class`, `subtype`, `prompt_id`, `prompt_family_id`, `seed`, `checkpoint_revision`, `adapter_revision`, `guidance`, `steps` and `split`.
- [ ] Group semantically related/reworded prompts into one split before sampling. **No prompt family may cross train/val/test.**
- [ ] Confirm prompt length/truncation behaviour for each actual model tokenizer; test any chunking method rather than assuming it increases the underlying model's context window.
- [ ] Verify exact genuine checkpoints, compatible architecture and license for each class, especially skin cancer subtypes and the Normal control.
- [ ] Freeze one generation configuration per experimental arm and record deviations.
- [ ] **Gate B PASS:** versioned prompt manifest, split audit, checkpoint provenance and pilot-ready configurations.

## 4. WP-C — Pilot generation (before full-scale jobs)

Run the six classes independently: **Acne, Fungal Infection, Vitiligo, Hyperpigmentation, Skin Cancer (BCC/SCC/melanoma), Normal.**

For **each class**:

- [ ] Start a fresh Kaggle environment; install pinned libraries; run import and version checks.
- [ ] Fetch the correct GitHub prompt manifest and verify its checksum/record count.
- [ ] Download or attach the real checkpoint/LoRA; verify weight format, tokenizer, scheduler and device precision.
- [ ] Generate a small pilot batch (e.g. 15–20 images across several prompts, **not** a definitive quality verdict).
- [ ] Record out-of-memory, auth/network, incompatible checkpoint and corrupt-output failures.
- [ ] Review morphology, location, photographic plausibility, skin-tone representation, prompt adherence, artifacts, text/watermarks and disease ambiguity. Clinical-label claims require knowledgeable review.
- [ ] Save pilot images, metadata and contact sheets in a versioned Kaggle output.
- [ ] **Class pilot PASS:** repeatable run, complete metadata, acceptable reviewed pilot with documented limitations.

**Skin cancer guardrail:** Distinguish melanoma, BCC and SCC in source metadata even if the classifier's final class is `Skin_Cancer`; do not assert a generated image has clinically proven pathology.

## 5. WP-D — Scale generation + synthetic curation

- [ ] Scale **only** classes passing their pilot. Run in resumable shards with stable `prompt_id + seed` IDs.
- [ ] Save incremental results and checkpointed progress, so a quota interruption never requires a complete restart.
- [ ] Enforce output format, RGB validity and image-shape consistency.
- [ ] Filter obvious artifacts/corruption and exact/repeated duplicates; inspect samples manually across class, subtype, prompt family and appearance strata.
- [ ] Audit near-duplicates against real training/validation/test data; document potential training contamination.
- [ ] Keep rejected files/IDs in rejection manifests (not as silently discarded evidence).
- [ ] Export `synthetic_manifest.csv`, prompt-family split manifest, approved-image counts, quality-review table and archive checksums.
- [ ] **Gate D PASS:** curated versioned datasets and reviewed class/subtype counts; generation alone does not pass.

## 6. WP-E — Four classifier notebooks

**Use four separate classifier notebooks:**
1. **EfficientNet-B0:** pretrained head training → partial last-block unfreeze → optional full fine-tuning.
2. **ConvNeXt V2 Tiny:** model-appropriate pretrained classifier head → partial final-stage unfreeze → optional full fine-tuning.
3. **DINOv2 ViT-B/14:** pretrained self-supervised frozen-feature/linear head baseline → last-block fine-tuning if justified → optional full tuning.
4. **PanDerm Base:** genuine dermatology-pretrained weights and official compatible classification procedure; stop if weight provenance or architecture cannot be verified.

For each classifier:

- [ ] Confirm imports, pretrained weights, class order, RGB transforms, aspect ratio, normalization and corrupt-image handling.
- [ ] Lock same real grouped assignments and same synthetic prompt-family split across all models.
- [ ] Run **Arm A: Real-only** with frozen real train/val/test.
- [ ] Run **Arm B: Real + Synthetic TRAIN only** with exactly the same untouched real validation/test, if the research question is synthetic augmentation effectiveness.
- [ ] If synthetic-held-out evaluation is required, report it separately. Do **not** mix synthetic into the primary real clinical test metric or change test composition between A and B.
- [ ] Use augmentations on training only; sensible optimizer groups/LRs for frozen vs unfrozen parameters; checkpoint best validation macro-F1.
- [ ] Validate class/sample counts and no overlap between partitions prior to each training run.
- [ ] Save metrics, class predictions, curves, configs, environment snapshot and best checkpoint; reload checkpoint and repeat test inference.
- [ ] **Gate E PASS:** both arms successfully trained for each valid model and run outputs independently reloaded. Any unrun model is marked pending rather than assigned invented metrics.

## 7. WP-F — Evaluation, comparisons and paper

- [ ] Primary comparison: macro-F1 on the **same frozen real test set** across all classifiers and A/B arms.
- [ ] Also report per-class sensitivity/recall, specificity, precision, confusion matrices, balanced accuracy and appropriate uncertainty intervals.
- [ ] Use repeated seeds when feasible; distinguish variation due to training randomness from variation in dataset partitions.
- [ ] Audit demographic/skin-tone representation and subgroup error patterns where labels are valid and counts adequate.
- [ ] Evaluate artifacts, synthetic-to-real gap, calibration, abstention/uncertainty and failure cases without making diagnostic claims.
- [ ] Run statistically justified paired comparisons on aligned test predictions; report confidence intervals and multiple-testing handling as appropriate.
- [ ] Check the research report's methodology against actual code, model weights, prompt manifests, dataset splits and results.
- [ ] Update LaTeX tables/figures **only with measured results**; state synthetic generation limitations, ethics, image rights, clinical review limitations and reproducibility details.
- [ ] Compile LaTeX cleanly; verify citations, figure references and page limits.
- [ ] **Gate F PASS:** complete experiment provenance and reproducible quantitative claims.

## 8. Parallel Kaggle execution plan — 4–5 sessions, if permitted

**Do not assume five simultaneous GPU sessions are allowed.** Kaggle availability, session limits and GPU time quotas are account- and platform-dependent and must be checked in the running UI. Each session owns its own output; never write multiple jobs to the same mutable dataset or directory.

| Slot | Independent task while gates permit | Required input | Output |
|---|---|---|---|
| S1 | Acne synthetic pilot/scale | Gate B | Acne shards + metadata |
| S2 | Vitiligo synthetic pilot/scale | Gate B | Vitiligo shards + metadata |
| S3 | Hyperpigmentation pilot/scale | Gate B | Hyperpigmentation shards + metadata |
| S4 | Fungal Infection pilot/scale | Gate B | Fungal shards + metadata |
| S5 | Normal and Skin Cancer pilots/scale **sequentially in this slot**, or separate sessions as capacity frees | Gate B | Normal/cancer shards + metadata |

If only 2–3 sessions are permitted, assign slots dynamically; finish one class and reuse the slot. Image-quality review and dataset labeling can run on CPU/laptop while GPUs generate images. Classifier training can start on the frozen **Real-only** dataset after Gate A, independently of unfinished synthetic generation. **Real+Synthetic** runs wait for Gate D. Do not schedule GPU-heavy generation and training together in one shared Kaggle session if they compete for VRAM.

### Simple dependency graph

```text
WP-A real labels + grouped split -----+----> Real-only classifier A runs ---+
                                      |                                     |
WP-B prompts / checkpoint checks ----> WP-C 6 independent pilots            |
                                      |                                     |
                                      +--> WP-D curate all six classes -----+--> Real+Synthetic classifier B runs
                                                                            |
                                                                            v
                                                              WP-F paired evaluation -> LaTeX
```

### Execution matrix

| Work item | S1 | S2 | S3 | S4 | S5 | Gate |
|---|---|---|---|---|---|---|
| First GPU wave | Acne | Vitiligo | Hyperpigmentation | Fungal | Normal | B |
| Next GPU wave | Retry/scale | Retry/scale | Retry/scale | Retry/scale | Skin Cancer subtypes | C |
| Training wave after A/D | EfficientNet-B0 | ConvNeXt V2 Tiny | DINOv2 | PanDerm | Reserve for validation/failed rerun | A/D |
| Follow-up training | Each model's other experimental arm as outputs become available | | | | | E |

These are **logical slots**, not a promise of GPU concurrency or performance. GPU-intensive runs require persistent Kaggle output snapshots between sessions.

## 9. Per-run evidence template (copy for each notebook)

```yaml
work_package: WP-C
class_or_model: Acne
experiment_arm: pilot
notebook_name:
kaggle_notebook_url:
kaggle_version:
git_commit:
dataset_name_and_version:
prompt_manifest_sha256:
checkpoint_name_and_revision:
adapter_revision:
seed:
gpu_type:
python_torch_diffusers_versions:
started_utc:
status: NOT_RUN  # NOT_RUN | RUNNING | BLOCKED | FAILED | PILOT_PASS | VERIFIED
images_attempted:
images_saved:
images_reviewed:
outputs_uri:
output_checksums:
key_metrics:
failure_log_uri:
reviewer_notes:
```

## 10. Work-first completion board

| Work package | Code/static status | Runtime verification | Sign-off |
|---|---|---|---|
| A. Real data, labels and split | Implemented in prior notebook; audit pending | ☐ | ☐ |
| B. Prompts and checkpoint provenance | Previously planned/generated; repo reconciliation pending | ☐ | ☐ |
| C. Six generation pilots | Not verified | ☐ | ☐ |
| D. Full generation and curation | Not verified | ☐ | ☐ |
| E. Four classifiers × two arms | Notebook implementations reported; runtime unverified | ☐ | ☐ |
| F. Statistical analysis + LaTeX | Planned/prepared; measured outputs pending | ☐ | ☐ |

## 11. What to execute first

1. **First non-GPU task:** run WP-A label audit, freeze real split and verify WP-B prompt/checkpoint manifests. Record actual notebook inventory.
2. **First GPU task:** execute the Acne **pilot only**, inspect metadata and output quality; do not launch mass generation yet.
3. Once checkpoint and dependencies are verified, run **up to four other class pilots** independently in available Kaggle sessions.
4. Start Real-only classifier experiments after Gate A independently of synthetic-image generation.
5. Only after each class's quality gates pass, scale synthetic images and start the Real+Synthetic comparisons.
6. Complete statistical analysis, update LaTeX and mark gates only when their evidence is available.

### Git workflow for this tracker

- Tracker-only change: keep notebooks, data, experiments and manuscript unchanged.
- Keep this file under source control; update checkboxes and link exact evidence after every successful stage.
- Prefer a dedicated documentation branch and reviewable PR rather than direct edits to protected `main`.
