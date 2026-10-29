# Reproducibility protocol (October 2026)

This repository's modular CLI code is not the executed paper evidence. The paper's
E2 and E3 Kaggle notebooks and their frozen manifests must be preserved verbatim
from their verified run. Do not publish any result until a matching split/fingerprint
and generated CSV/log/figure have been checked.

* Configure dataset roots through DATA_DIR, REAL_DATA_DIR and SYNTHETIC_DATA_DIR
  (or CLI arguments). No dataset mount path is baked into notebooks by this update.
* The first run creates a new real split and synthetic_frozen_v1.csv. Subsequent
  E3 mixtures pass --split_csv and --synthetic_split_csv explicitly.
* Real saved splits require file SHA-256 and group membership verification.
* The full-pool MD5 audit precedes the 100:100, 70:30 and 30:70 independent
  train-pool fractions. Each removed image is logged by source and reason.
* Per-source tests require actual real-only and synthetic-only held-out predictions.
* No accuracy, duplicate count, GPU training outcome, or clinical ground truth is
  claimed by the CPU unit checks.
* Preserve original dataset-source provenance separately if available. Real-vs-
  synthetic source labels do not identify the public dataset each real photo came from.
* Git history is used for rollbacks; do not commit raw clinical images or weights.
