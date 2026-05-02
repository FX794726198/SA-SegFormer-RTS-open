"""Training and evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .constants import DEFAULT_FEATURE_NAMES, DEFAULT_IMAGE_SIZE, DEFAULT_SEED, DEFAULT_SPLIT_COUNTS, DEFAULT_THRESHOLD
from .data import LandslideDataset, load_binary_mask, mask_has_positive
from .metrics import PIXEL_METRIC_KEYS, aggregate_object_metrics, compute_batch_metrics, object_level_metrics
from .model import FusionSASegFormer
from .utils import tensor_to_numpy_safe


def build_model(
    feature_names: list[str] | None = None,
    backbone: str = "mit_b0",
    decoder_dim: int = 256,
    pretrained: bool = True,
    use_decoder_sa: bool = True,
    sa_heads: int = 4,
    sa_downsample: int = 8,
) -> FusionSASegFormer:
    feature_names = list(DEFAULT_FEATURE_NAMES if feature_names is None else feature_names)
    return FusionSASegFormer(
        in_channels=3 + len(feature_names),
        num_classes=1,
        decoder_dim=decoder_dim,
        backbone=backbone,
        pretrained=pretrained,
        use_decoder_sa=use_decoder_sa,
        sa_heads=sa_heads,
        sa_downsample=sa_downsample,
    )


def build_criterion(pos_weight: float = 5.0, bce_weight: float = 0.5, dice_weight: float = 1.0, focal_weight: float = 1.0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    dice = smp.losses.DiceLoss(mode="binary")
    focal = smp.losses.FocalLoss(mode="binary", alpha=0.8)

    def criterion(preds, targets):
        return bce_weight * bce(preds, targets) + dice_weight * dice(preds, targets) + focal_weight * focal(preds, targets)

    return criterion


def build_loader(
    samples: list[dict[str, str]],
    feature_names: list[str] | None = None,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 8,
    num_workers: int = 0,
    is_train: bool = False,
    weighted_sampling: bool = True,
):
    dataset = LandslideDataset(samples, feature_names=feature_names, image_size=image_size, is_train=is_train)
    if is_train and weighted_sampling:
        weights = [2.0 if mask_has_positive(sample["mask_path"]) else 1.0 for sample in samples]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)
    return dataset, loader


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item()) * images.size(0)
    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def validate(model, loader, criterion, device, threshold: float = DEFAULT_THRESHOLD):
    model.eval()
    total_loss = 0.0
    metric_sums = {key: 0.0 for key in PIXEL_METRIC_KEYS}
    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        total_loss += float(criterion(logits, masks).item()) * images.size(0)
        batch_metrics = compute_batch_metrics(logits, masks, threshold=threshold)
        for key, value in batch_metrics.items():
            metric_sums[key] += float(value) * images.size(0)
    count = max(1, len(loader.dataset))
    return total_loss / count, {key: metric_sums[key] / count for key in metric_sums}


@torch.no_grad()
def evaluate_object_level(model, loader, device, threshold: float = DEFAULT_THRESHOLD):
    model.eval()
    rows = []
    sample_offset = 0
    for images, _masks in tqdm(loader, desc="Object metrics", leave=False):
        images = images.to(device)
        probs = torch.sigmoid(model(images))
        probs_np = tensor_to_numpy_safe(probs.squeeze(1), dtype=np.float32)
        batch_size = probs_np.shape[0]
        for idx in range(batch_size):
            sample = loader.dataset.samples[sample_offset + idx]
            pred_mask = (probs_np[idx] > threshold).astype(np.uint8)
            gt_mask = load_binary_mask(sample["mask_path"])
            if gt_mask.shape != pred_mask.shape:
                pred_mask = cv2_resize_nearest(pred_mask, gt_mask.shape)
            metrics = object_level_metrics(pred_mask, gt_mask)
            rows.append({"region": sample.get("region", ""), "stem": sample.get("stem", ""), **metrics})
        sample_offset += batch_size
    return aggregate_object_metrics(rows), rows


def cv2_resize_nearest(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(mask.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)


def split_random(samples: list[dict[str, str]], val_ratio: float = 0.2, seed: int = DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(samples))
    rng.shuffle(idxs)
    n_train = int((1.0 - val_ratio) * len(samples))
    train_samples = [samples[int(idx)] for idx in idxs[:n_train]]
    val_samples = [samples[int(idx)] for idx in idxs[n_train:]]
    return train_samples, val_samples


def split_fixed_counts(
    samples: list[dict[str, str]],
    split_counts: dict[str, int] | None = None,
    seed: int = DEFAULT_SEED,
    require_exact_total: bool = False,
):
    split_counts = dict(split_counts or DEFAULT_SPLIT_COUNTS)
    total_required = sum(split_counts.values())
    if len(samples) < total_required:
        raise ValueError(f"Need at least {total_required} samples for fixed split, got {len(samples)}.")
    if require_exact_total and len(samples) != total_required:
        raise ValueError(f"Expected exactly {total_required} samples, got {len(samples)}.")

    rng = np.random.default_rng(seed)
    idxs = np.arange(len(samples))
    rng.shuffle(idxs)
    selected = [samples[int(idx)] for idx in idxs[:total_required]]

    out = {}
    offset = 0
    for split_name in ("train", "val", "test"):
        count = int(split_counts[split_name])
        out[split_name] = selected[offset : offset + count]
        for sample in out[split_name]:
            sample["split"] = split_name
        offset += count
    return out["train"], out["val"], out["test"]


def split_from_manifest_column(samples: list[dict[str, str]], split_column: str = "split"):
    grouped = {"train": [], "val": [], "test": []}
    for sample in samples:
        split_name = str(sample.get(split_column, "")).strip().lower()
        if split_name in {"training", "train"}:
            grouped["train"].append(sample)
        elif split_name in {"validation", "valid", "val"}:
            grouped["val"].append(sample)
        elif split_name in {"testing", "test"}:
            grouped["test"].append(sample)
    if not grouped["train"] or not grouped["val"] or not grouped["test"]:
        raise ValueError(f"Manifest split column '{split_column}' must contain train, val, and test rows.")
    return grouped["train"], grouped["val"], grouped["test"]


def save_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_samples_csv(path: str | Path, samples: list[dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(samples).to_csv(path, index=False)


def load_samples_csv(path: str | Path) -> list[dict[str, str]]:
    return pd.read_csv(path).fillna("").to_dict("records")
