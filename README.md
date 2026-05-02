# SA-SegFormer RTS

This repository contains the code, configs, scripts, and dataset manifests for retrogressive thaw slump (RTS) binary segmentation with SA-SegFormer, plus baseline scripts used for comparison.

The full raster dataset is distributed separately on Zenodo to avoid repository-size and Git LFS quota limits. Download the dataset archive from the DOI listed below, then unzip it at the repository root so that `data/2023/`, `data/2024/`, and `data/folds/` are restored locally.

## Repository Layout

```text
src/sa_segformer_rts/      Reusable model, dataset, metrics, training, and export code
scripts/                  Training, inference, evaluation, and dataset utilities
baselines/                Original comparison-model scripts
slurm/                    Portable SLURM job templates
configs/                  Default experiment configuration
data/                     Versioned manifests plus expected local dataset layout
```

The SA-SegFormer input is 13 channels: RGB optical imagery plus 10 factor rasters (`DEM`, `EVI`, `FTI`, `LST`, `NBR`, `NDMI`, `NDVI`, `TCB`, `TCG`, `TCW`).

## Setup

```bash
pip install -e .
```

For a plain environment without editable install:

```bash
pip install -r requirements.txt
```

## Data

GitHub contains the code and lightweight CSV manifests. The complete raster chips are hosted on Zenodo.

Dataset download: [https://doi.org/10.5281/zenodo.19972661](https://doi.org/10.5281/zenodo.19972661)

Recommended download layout:

```bash
unzip SA-SegFormer-RTS-dataset-zenodo.zip -d .
python scripts/summarize_dataset.py
```

Expected local paths after unzip:

```text
data/2023/factors/
data/2024/factors/
data/2024/optical/
data/folds/
```

The large local dataset directories are intentionally ignored by Git. The dataset is released under CC BY 4.0; see `DATA_LICENSE`.

## Build and Check Manifests

The manifests map each tile to its optical image, mask, and factor rasters.

```bash
python scripts/data/build_manifests.py
python scripts/summarize_dataset.py
```

Expected manifest columns:

```text
year, region, stem, optical_path, mask_path, DEM, EVI, FTI, LST, NBR, NDMI, NDVI, TCB, TCG, TCW
```

## Required Fixed Split

All main training scripts use the required 1195-chip split:

```text
training   837
validation 179
testing    179
```

Create the split manifest after building the base manifests:

```bash
python scripts/data/create_fixed_split.py \
  --manifest data/manifests/manifest_2023.csv \
  --output data/manifests/manifest_2023_split_837_179_179.csv \
  --require-exact-total
```

## Train SA-SegFormer

```bash
python scripts/train_sa_segformer.py \
  --manifest data/manifests/manifest_2023_split_837_179_179.csv \
  --output-dir runs/sa_segformer \
  --run-name 2023_main \
  --epochs 400 \
  --batch-size 8
```

Quick dataloader check:

```bash
python scripts/train_sa_segformer.py --dry-run --max-train-samples 8 --max-val-samples 4
```

## Train Baselines

Each baseline wrapper uses the same `837/179/179` split manifest by default.

```bash
python baselines/train_unet.py --epochs 200 --batch-size 8
python baselines/train_deeplabv3plus.py --epochs 200 --batch-size 8
python baselines/train_convnext.py --epochs 200 --batch-size 8
```

## Inference

```bash
python scripts/infer_sa_segformer.py \
  --manifest data/manifests/manifest_2023_split_837_179_179.csv \
  --checkpoint runs/sa_segformer/2023_main/checkpoints/best.pth \
  --output-dir runs/inference/2023_main
```

Inference exports binary masks, probability maps, overlays, panels, and `prediction_manifest.csv`.

## Region-Holdout Evaluation

```bash
python scripts/eval_region_holdout.py \
  --manifest data/manifests/manifest_2023.csv \
  --test-regions "北1区域数据集" \
  --run-name holdout_north1 \
  --epochs 200
```

Dry run:

```bash
python scripts/eval_region_holdout.py \
  --test-regions "北1区域数据集" \
  --dry-run
```

## Cross-Year Temporal Transfer

The temporal transfer experiment intentionally keeps its original 8:2 shared-key split and does not use the fixed 837/179/179 split.

```bash
python scripts/eval_temporal_transfer.py \
  --source-manifest data/manifests/manifest_2024.csv \
  --target-manifest data/manifests/manifest_2023.csv \
  --run-name 2024_to_2023 \
  --regions "中1区域数据集,北2区域数据集,北3区域数据集" \
  --epochs 200
```

## License

Code is released under the MIT License. The external dataset is released under CC BY 4.0; see `DATA_LICENSE`.
