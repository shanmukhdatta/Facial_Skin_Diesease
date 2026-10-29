# DermaLens EfficientNet-B0 notebook

Uses the frozen six-class real and synthetic splits. Preprocessing: EXIF, RGB, letterbox 224, normalization. Stage one head-only; stage two last EfficientNet feature blocks; optional full fine-tuning. Save best-validation checkpoints and export predictions.

Note: local runtime tests were unavailable; run on Kaggle GPU.
