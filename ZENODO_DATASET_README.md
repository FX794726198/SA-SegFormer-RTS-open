# SA-SegFormer RTS Dataset

This archive contains the raster chips and manifests used by the SA-SegFormer RTS segmentation experiments.

## Contents

```text
DATA_LICENSE
ZENODO_DATASET_README.md
data/
  2023/
    factors/
  2024/
    factors/
    optical/
  folds/
  manifests/
```

## Dataset Summary

- 2023 manifest: 1195 complete chips.
- Required main split: 837 training, 179 validation, 179 testing.
- 2024 manifest: 717 complete chips.
- Input channels: RGB optical imagery plus `DEM`, `EVI`, `FTI`, `LST`, `NBR`, `NDMI`, `NDVI`, `TCB`, `TCG`, and `TCW`.
- Region names and tile filenames are preserved from the original dataset.

## How To Use With The Code Repository

Download the code repository from GitHub, then unzip this archive at the repository root:

```bash
unzip SA-SegFormer-RTS-dataset-zenodo.zip -d SA-SegFormer-RTS-open
cd SA-SegFormer-RTS-open
python scripts/summarize_dataset.py
```

The scripts read `data/manifests/*.csv`, so the local folder names should remain unchanged after extraction.

## License

The dataset is released under CC BY 4.0. See `DATA_LICENSE`.

The accompanying code repository is released under the MIT License.

## Citation

Please cite the associated paper and the Zenodo dataset DOI once available.
