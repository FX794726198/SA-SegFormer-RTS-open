# Baselines

These wrappers train comparison models with the same released manifest and the required fixed split:

- training: 837 chips
- validation: 179 chips
- testing: 179 chips

All wrappers call `scripts/train_baseline.py`. Use `--rgb-only` if a baseline should consume only optical RGB channels; otherwise the default input is RGB plus the 10 factor rasters.

Example:

```bash
python baselines/train_unet.py --epochs 200 --batch-size 8
python baselines/train_convnext.py --rgb-only --epochs 200
```
