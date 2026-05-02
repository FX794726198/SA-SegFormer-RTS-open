"""Dataset scanning, manifests, and raster loading."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image, UnidentifiedImageError

try:
    from torch.utils.data import Dataset
    import torch
except ImportError:
    torch = None

    class Dataset:
        pass

from .constants import (
    DEFAULT_FEATURE_NAMES,
    DEFAULT_IMAGE_SIZE,
    IMAGE_DIR_CANDIDATES,
    IMAGE_EXTENSIONS,
    LABEL_DIR_CANDIDATES,
    OPTICAL_DIR_CANDIDATES,
    RASTER_EXTENSIONS,
)


def first_existing_dir(base: Path, candidates: Iterable[str]) -> Path | None:
    for candidate in candidates:
        path = base / candidate
        if path.is_dir():
            return path
    return None


def candidate_files(directory: Path, stem: str, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> list[Path]:
    return [directory / f"{stem}{ext}" for ext in extensions]


def first_existing_file(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def load_raster_gray_float01(path: str | Path) -> np.ndarray:
    path = Path(path)
    arr = None
    try:
        arr = tiff.imread(str(path))
    except Exception:
        try:
            arr = np.array(Image.open(path))
        except UnidentifiedImageError:
            import cv2

            arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Cannot read raster: {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin > 1e-6:
        return (arr - vmin) / (vmax - vmin)
    return np.zeros_like(arr, dtype=np.float32)


def load_rgb01_safe(path: str | Path, main_image_bands: list[int] | None = None) -> np.ndarray:
    path = Path(path)
    try:
        image = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        return image.astype(np.float32) / 255.0
    except Exception:
        arr = np.nan_to_num(tiff.imread(str(path)), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if main_image_bands is not None:
            channel_count = arr.shape[2]
            bands = [band if band < channel_count else channel_count - 1 for band in main_image_bands]
            arr = arr[..., bands]
        elif arr.shape[2] < 3:
            last = arr[..., -1]
            arr = np.stack([arr[..., idx] if idx < arr.shape[2] else last for idx in range(3)], axis=-1)
        else:
            arr = arr[..., :3]
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin > 1e-6:
        return (arr - vmin) / (vmax - vmin)
    return np.clip(arr, 0, 1)


def load_binary_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        mask = np.array(Image.open(path))
    except Exception:
        mask = tiff.imread(str(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (np.nan_to_num(mask) > 0).astype(np.uint8)


def mask_has_positive(path: str | Path) -> bool:
    path = Path(path)
    return path.is_file() and bool(load_binary_mask(path).any())


def region_prefix(region: str) -> str:
    return region.replace("区域数据集", "区域")


def find_optical_root(factor_root: Path, optical_root: str | Path | None = None) -> Path | None:
    if optical_root:
        path = Path(optical_root)
        return path if path.exists() else None
    return first_existing_dir(factor_root, OPTICAL_DIR_CANDIDATES)


def find_feature_path(factor_root: Path, feature_name: str, region: str, stem: str) -> Path | None:
    feature_region = factor_root / feature_name / region
    image_dir = first_existing_dir(feature_region, IMAGE_DIR_CANDIDATES)
    if image_dir is None:
        return None
    return first_existing_file(candidate_files(image_dir, stem, RASTER_EXTENSIONS))


def find_optical_for_factor_sample(optical_root: Path, region: str, stem: str) -> Path | None:
    prefix = region_prefix(region)
    search_dirs = [
        optical_root / "img",
        optical_root / "images",
        optical_root,
        optical_root / "split_dataset" / "Train_img",
        optical_root / "split_dataset" / "Test_img",
    ]
    stems = [stem, f"{prefix}_{stem}"]
    candidates = []
    for directory in search_dirs:
        if directory.is_dir():
            for candidate_stem in stems:
                candidates.extend(candidate_files(directory, candidate_stem))
    return first_existing_file(candidates)


def scan_region_optical_samples(
    factor_root: Path,
    optical_root: Path,
    feature_names: list[str],
    year: str,
) -> list[dict[str, str]]:
    rows = []
    region_dirs = [item for item in optical_root.iterdir() if item.is_dir() and item.name.endswith("区域数据集")]
    for region_dir in sorted(region_dirs):
        image_dir = first_existing_dir(region_dir, IMAGE_DIR_CANDIDATES)
        label_dir = first_existing_dir(region_dir, LABEL_DIR_CANDIDATES)
        if image_dir is None or label_dir is None:
            continue
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
            stem = image_path.stem
            mask_path = first_existing_file(candidate_files(label_dir, stem, RASTER_EXTENSIONS))
            if mask_path is None:
                continue
            row = {
                "year": str(year),
                "region": region_dir.name,
                "stem": stem,
                "optical_path": str(image_path),
                "mask_path": str(mask_path),
            }
            for feature_name in feature_names:
                feature_path = find_feature_path(factor_root, feature_name, region_dir.name, stem)
                row[feature_name] = str(feature_path) if feature_path else ""
            rows.append(row)
    return rows


def scan_factor_indexed_samples(
    factor_root: Path,
    optical_root: Path | None,
    feature_names: list[str],
    year: str,
    index_feature: str = "DEM",
) -> list[dict[str, str]]:
    feature_root = factor_root / index_feature
    if not feature_root.is_dir():
        raise FileNotFoundError(f"Index feature root not found: {feature_root}")

    rows = []
    for region_dir in sorted(item for item in feature_root.iterdir() if item.is_dir() and item.name.endswith("区域数据集")):
        image_dir = first_existing_dir(region_dir, IMAGE_DIR_CANDIDATES)
        label_dir = first_existing_dir(region_dir, LABEL_DIR_CANDIDATES)
        if image_dir is None or label_dir is None:
            continue
        for feature_image in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in RASTER_EXTENSIONS):
            stem = feature_image.stem
            mask_path = first_existing_file(candidate_files(label_dir, stem, RASTER_EXTENSIONS))
            if mask_path is None:
                continue
            optical_path = find_optical_for_factor_sample(optical_root, region_dir.name, stem) if optical_root else None
            if optical_path is None:
                continue
            row = {
                "year": str(year),
                "region": region_dir.name,
                "stem": stem,
                "optical_path": str(optical_path),
                "mask_path": str(mask_path),
            }
            for feature_name in feature_names:
                feature_path = find_feature_path(factor_root, feature_name, region_dir.name, stem)
                row[feature_name] = str(feature_path) if feature_path else ""
            rows.append(row)
    return rows


def scan_dataset(
    factor_root: str | Path,
    year: str,
    optical_root: str | Path | None = None,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    factor_root = Path(factor_root).expanduser().resolve()
    optical_path = find_optical_root(factor_root, optical_root)
    feature_names = list(feature_names or DEFAULT_FEATURE_NAMES)

    rows: list[dict[str, str]] = []
    if optical_path is not None:
        region_dirs = [item for item in optical_path.iterdir() if item.is_dir() and item.name.endswith("区域数据集")]
        if region_dirs:
            rows = scan_region_optical_samples(factor_root, optical_path, feature_names, year)
    if not rows:
        rows = scan_factor_indexed_samples(factor_root, optical_path, feature_names, year)
    if not rows:
        raise RuntimeError(f"No samples found under factor_root={factor_root} optical_root={optical_path}")

    columns = ["year", "region", "stem", "optical_path", "mask_path", *feature_names]
    return pd.DataFrame(rows, columns=columns)


def relativize_manifest_paths(df: pd.DataFrame, relative_to: str | Path | None) -> pd.DataFrame:
    if relative_to is None:
        return df.copy()
    base = Path(relative_to).expanduser().resolve()
    out = df.copy()
    path_columns = [column for column in out.columns if column.endswith("_path") or column in DEFAULT_FEATURE_NAMES]
    for column in path_columns:
        out[column] = out[column].map(lambda value: _relative_path_string(value, base))
    return out


def _relative_path_string(value, base: Path) -> str:
    if not isinstance(value, str) or not value:
        return ""
    path = Path(value)
    try:
        return str(path.resolve().relative_to(base))
    except ValueError:
        return str(path)


def build_manifest(
    out_path: str | Path,
    factor_root: str | Path,
    year: str,
    optical_root: str | Path | None = None,
    feature_names: list[str] | None = None,
    relative_to: str | Path | None = None,
) -> pd.DataFrame:
    df = scan_dataset(factor_root=factor_root, optical_root=optical_root, year=year, feature_names=feature_names)
    manifest = relativize_manifest_paths(df, relative_to)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    return manifest


def load_manifest(manifest_path: str | Path, repo_root: str | Path | None = None) -> list[dict[str, str]]:
    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path).fillna("")
    root = Path(repo_root).expanduser().resolve() if repo_root else manifest_path.resolve().parents[2]
    path_columns = [column for column in df.columns if column.endswith("_path") or column in DEFAULT_FEATURE_NAMES]
    rows = []
    for row in df.to_dict("records"):
        for column in path_columns:
            value = row.get(column, "")
            if isinstance(value, str) and value:
                path = Path(value)
                row[column] = str(path if path.is_absolute() else root / path)
        rows.append(row)
    return rows


class LandslideDataset(Dataset):
    """RTS segmentation dataset using optical RGB plus optional factor rasters."""

    def __init__(
        self,
        samples: list[dict[str, str]],
        feature_names: list[str] | None = None,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        is_train: bool = True,
        main_image_bands: list[int] | None = None,
    ):
        self.samples = samples
        if torch is None:
            raise ImportError("torch is required for LandslideDataset. Install requirements.txt.")
        self.feature_names = list(feature_names or DEFAULT_FEATURE_NAMES)
        self.image_size = tuple(image_size)
        self.is_train = is_train
        self.main_image_bands = main_image_bands
        try:
            import albumentations as A
        except ImportError as exc:
            raise ImportError("albumentations is required for LandslideDataset. Install requirements.txt.") from exc
        self.train_geo_tf = A.Compose(
            [
                A.RandomResizedCrop(height=self.image_size[0], width=self.image_size[1], scale=(0.5, 1.0), ratio=(0.9, 1.1), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )
        self.val_tf = A.Compose([A.Resize(height=self.image_size[0], width=self.image_size[1])])
        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        rgb = load_rgb01_safe(sample["optical_path"], main_image_bands=self.main_image_bands)
        height, width = rgb.shape[:2]

        feature_arrays = []
        for feature_name in self.feature_names:
            feature_path = sample.get(feature_name, "")
            if feature_path and Path(feature_path).is_file():
                arr = load_raster_gray_float01(feature_path)[..., None]
                if arr.shape[:2] != (height, width):
                    import cv2

                    arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)[..., None]
            else:
                arr = np.zeros((height, width, 1), dtype=np.float32)
            feature_arrays.append(arr)

        feature_stack = np.concatenate(feature_arrays, axis=-1) if feature_arrays else np.zeros((height, width, 0), dtype=np.float32)
        image = np.concatenate([rgb, feature_stack], axis=-1)
        mask = load_binary_mask(sample["mask_path"])

        if self.is_train:
            augmented = self.train_geo_tf(image=image, mask=mask)
        else:
            augmented = self.val_tf(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        mean = np.concatenate([self.rgb_mean, np.zeros(len(self.feature_names), dtype=np.float32)])
        std = np.concatenate([self.rgb_std, np.ones(len(self.feature_names), dtype=np.float32)])
        image = (image - mean) / (std + 1e-7)

        image_chw = np.transpose(image.astype(np.float32), (2, 0, 1))
        mask_1hw = mask.astype(np.float32)[None, ...]
        from .utils import numpy_to_tensor_safe

        return numpy_to_tensor_safe(image_chw, dtype=torch.float32), numpy_to_tensor_safe(mask_1hw, dtype=torch.float32)
