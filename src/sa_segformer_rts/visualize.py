"""Prediction export and interpretability utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .data import load_binary_mask, load_rgb01_safe
from .utils import tensor_to_numpy_safe


def grad_x_input_per_channel(
    model,
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
    device="cuda",
    fallback: str = "pred",
    min_fg_pixels: int = 50,
    prob_thresh: float = 0.5,
):
    model.eval()
    image = image.clone().detach().to(device)
    image.requires_grad_(True)
    with torch.enable_grad():
        logits = model(image)
        probs = torch.sigmoid(logits)
        use_mask = None
        if mask is not None and mask.sum().item() >= min_fg_pixels:
            use_mask = mask.to(device)
        if use_mask is None and fallback == "pred":
            pred_bin = (probs > prob_thresh).float()
            if pred_bin.sum().item() >= min_fg_pixels:
                use_mask = pred_bin
        target = probs.mean() if use_mask is None else (probs * use_mask).sum() / (use_mask.sum() + 1e-6)
        model.zero_grad(set_to_none=True)
        target.backward()
    saliency = (image.grad.detach() * image.detach()).squeeze(0).abs()
    saliency_norm = saliency / (saliency.amax(dim=(1, 2), keepdim=True) + 1e-8)
    saliency_np = tensor_to_numpy_safe(saliency_norm, dtype=np.float32)
    channel_scores = tensor_to_numpy_safe(saliency.mean(dim=(1, 2)), dtype=np.float32)
    return saliency_np, channel_scores


def normalize_rgb_for_display(rgb: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


def add_label(image_bgr: np.ndarray, text: str) -> np.ndarray:
    image = image_bgr.copy()
    cv2.rectangle(image, (0, 0), (max(120, 12 * len(text)), 28), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def overlay_mask(rgb_uint8: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha: float = 0.45) -> np.ndarray:
    out = rgb_uint8.copy()
    color_layer = np.zeros_like(out)
    color_layer[:] = color
    mask_bool = mask.astype(bool)
    out[mask_bool] = cv2.addWeighted(out, 1.0 - alpha, color_layer, alpha, 0)[mask_bool]
    return out


@torch.no_grad()
def export_predictions(
    model,
    loader,
    device,
    out_dir: str | Path,
    threshold: float = 0.4,
    max_preview_side: int = 512,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    pred_dir = out_dir / "pred_masks"
    prob_dir = out_dir / "prob_maps"
    overlay_dir = out_dir / "overlays"
    panel_dir = out_dir / "panels"
    for directory in (pred_dir, prob_dir, overlay_dir, panel_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model.eval()
    rows = []
    sample_offset = 0
    for images, _masks in loader:
        images = images.to(device)
        probs = torch.sigmoid(model(images)).squeeze(1)
        probs_np = tensor_to_numpy_safe(probs, dtype=np.float32)
        for idx in range(probs_np.shape[0]):
            sample = loader.dataset.samples[sample_offset + idx]
            prob = probs_np[idx]
            pred = (prob > threshold).astype(np.uint8)
            rgb = normalize_rgb_for_display(load_rgb01_safe(sample["optical_path"]))
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            height, width = rgb_uint8.shape[:2]
            pred_full = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
            prob_full = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)
            gt = load_binary_mask(sample["mask_path"])
            if gt.shape != (height, width):
                gt = cv2.resize(gt, (width, height), interpolation=cv2.INTER_NEAREST)

            region = str(sample.get("region", "region")).replace("/", "_")
            stem = str(sample.get("stem", Path(sample["optical_path"]).stem))
            root = f"{region}__{stem}"

            pred_path = pred_dir / f"{root}_pred.png"
            prob_path = prob_dir / f"{root}_prob.png"
            overlay_path = overlay_dir / f"{root}_overlay.png"
            panel_path = panel_dir / f"{root}_panel.png"

            cv2.imwrite(str(pred_path), pred_full * 255)
            cv2.imwrite(str(prob_path), np.clip(prob_full * 255, 0, 255).astype(np.uint8))

            rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
            pred_overlay = overlay_mask(rgb_bgr, pred_full, color=(0, 0, 255))
            gt_overlay = overlay_mask(rgb_bgr, gt, color=(0, 180, 0))
            compare = overlay_mask(overlay_mask(rgb_bgr, gt, color=(0, 180, 0), alpha=0.35), pred_full, color=(0, 0, 255), alpha=0.35)
            cv2.imwrite(str(overlay_path), pred_overlay)

            preview_items = [
                add_label(rgb_bgr, "RGB"),
                add_label(gt_overlay, "GT overlay"),
                add_label(pred_overlay, "Prediction"),
                add_label(cv2.applyColorMap(np.clip(prob_full * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET), "Probability"),
                add_label(compare, "GT green / Pred red"),
            ]
            preview_items = [_resize_preview(item, max_preview_side) for item in preview_items]
            panel = np.concatenate(preview_items, axis=1)
            cv2.imwrite(str(panel_path), panel)

            rows.append(
                {
                    "region": sample.get("region", ""),
                    "stem": sample.get("stem", stem),
                    "optical_path": sample["optical_path"],
                    "mask_path": sample["mask_path"],
                    "pred_mask_path": str(pred_path),
                    "prob_map_path": str(prob_path),
                    "overlay_path": str(overlay_path),
                    "panel_path": str(panel_path),
                }
            )
        sample_offset += probs_np.shape[0]

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "prediction_manifest.csv", index=False)
    return manifest


def _resize_preview(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_side / max(height, width), 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return image
