# Baselines

These wrappers train the twelve paper baseline models with the same released manifest and the required fixed split:

- training: 837 chips
- validation: 179 chips
- testing: 179 chips

All wrappers call `scripts/train_baseline.py`. Baselines consume optical RGB channels by default, matching the manuscript benchmark protocol. Use `--include-factors` only for ablations that intentionally use RGB plus the 10 factor rasters.

Included wrappers:

```text
train_cnn.py, train_unet.py, train_unetpp.py, train_deeplabv3plus.py,
train_resnet.py, train_convnext.py, train_swin.py, train_segformer.py,
train_sa_segformer.py, train_ca_segformer.py, train_sa_convnext.py,
train_ca_convnext.py
```

Example:

```bash
python baselines/train_unet.py --epochs 200 --batch-size 8
python baselines/train_convnext.py --epochs 200
```
