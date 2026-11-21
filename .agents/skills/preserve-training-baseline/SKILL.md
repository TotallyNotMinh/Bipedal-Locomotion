---
name: preserve-training-baseline
description: Preserve this repository's terrain configuration and training environment when preparing, running, resuming, or troubleshooting training and when handling run outputs.
---

# Preserve the Training Baseline

Keep the existing terrain configuration and environment setup fixed before, during, and after every training run. Apply the same rule when a run fails, is interrupted, or is retried.

- Use the established terrain configuration, normally `configs/terrain.yaml`. Do not edit, regenerate, or replace it, and do not use flags or alternate configs to override the baseline terrain settings.
- Reuse the configured training environment. Do not rerun `setup_kaggle.sh`, install or update packages, switch runtimes or accelerators, or change persistent environment variables as part of training or cleanup.
- Keep checkpoints, logs, and generated heightfields in `results/` or the requested run output directory. Normal curriculum progression within a run is allowed.
- If training cannot proceed with the existing setup, report the blocker and ask before changing the baseline.
