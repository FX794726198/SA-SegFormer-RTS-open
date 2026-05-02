# Data Layout

This GitHub repository tracks only lightweight dataset manifests and metadata. The full raster dataset is distributed separately to avoid GitHub repository-size and Git LFS quota limits.

Before public release, add the final Zenodo DOI here:

```text
TODO: Zenodo DOI / archive URL
```

The Zenodo archive should be uploaded as:

```text
SA-SegFormer-RTS-dataset-zenodo.zip
```

After downloading the dataset, unzip the archive at the repository root:

```bash
unzip SA-SegFormer-RTS-dataset-zenodo.zip -d .
python scripts/summarize_dataset.py
```

Expected local layout:

```text
data/
  2023/
    factors/
  2024/
    factors/
    optical/
  folds/
  manifests/
```

The large local directories `data/2023/`, `data/2024/`, and `data/folds/` are ignored by Git on purpose. The scripts use `data/manifests/*.csv` to locate local files, so the original Chinese region names and tile names should remain unchanged.

Dataset files are released under CC BY 4.0. Code is released under MIT.
