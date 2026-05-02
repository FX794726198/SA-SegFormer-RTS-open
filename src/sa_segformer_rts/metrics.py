"""Pixel- and object-level metrics for binary segmentation."""

from __future__ import annotations

import cv2
import numpy as np
import torch


PIXEL_METRIC_KEYS = [
    "precision_pos",
    "recall_pos",
    "f1_pos",
    "iou_pos",
    "acc_pos",
    "precision_neg",
    "recall_neg",
    "f1_neg",
    "iou_neg",
    "acc_neg",
    "precision",
    "recall",
    "f1_score",
    "iou",
    "pixel_acc",
]


def compute_batch_metrics(preds: torch.Tensor, masks: torch.Tensor, threshold: float = 0.4, eps: float = 1e-6) -> dict[str, float]:
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()

    tp = (preds_bin * masks).sum((1, 2, 3))
    fp = (preds_bin * (1 - masks)).sum((1, 2, 3))
    fn = ((1 - preds_bin) * masks).sum((1, 2, 3))
    tn = ((1 - preds_bin) * (1 - masks)).sum((1, 2, 3))

    precision_pos = (tp / (tp + fp + eps)).mean().item()
    recall_pos = (tp / (tp + fn + eps)).mean().item()
    f1_pos = (2 * tp / (2 * tp + fp + fn + eps)).mean().item()
    iou_pos = (tp / (tp + fp + fn + eps)).mean().item()
    precision_neg = (tn / (tn + fn + eps)).mean().item()
    recall_neg = (tn / (tn + fp + eps)).mean().item()
    f1_neg = (2 * tn / (2 * tn + fn + fp + eps)).mean().item()
    iou_neg = (tn / (tn + fn + fp + eps)).mean().item()

    return {
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "iou_pos": iou_pos,
        "acc_pos": recall_pos,
        "precision_neg": precision_neg,
        "recall_neg": recall_neg,
        "f1_neg": f1_neg,
        "iou_neg": iou_neg,
        "acc_neg": recall_neg,
        "precision": 0.5 * (precision_pos + precision_neg),
        "recall": 0.5 * (recall_pos + recall_neg),
        "f1_score": 0.5 * (f1_pos + f1_neg),
        "iou": 0.5 * (iou_pos + iou_neg),
        "pixel_acc": ((tp + tn) / (tp + tn + fp + fn + eps)).mean().item(),
    }


def object_level_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, min_overlap: int = 1) -> dict[str, int | float]:
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)
    pred_count, pred_labels = cv2.connectedComponents(pred_bin, connectivity=8)
    gt_count, gt_labels = cv2.connectedComponents(gt_bin, connectivity=8)

    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    for pred_id in range(1, pred_count):
        gt_ids = np.unique(gt_labels[pred_labels == pred_id])
        gt_ids = [int(gt_id) for gt_id in gt_ids if int(gt_id) != 0]
        has_match = False
        for gt_id in gt_ids:
            overlap = int(np.sum((pred_labels == pred_id) & (gt_labels == gt_id)))
            if overlap >= min_overlap:
                has_match = True
                matched_gt.add(gt_id)
        if has_match:
            tp += 1
        else:
            fp += 1

    fn = max(0, (gt_count - 1) - len(matched_gt))
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1_score": f1_score}


def aggregate_object_metrics(rows: list[dict[str, int | float]]) -> dict[str, int | float]:
    tp = int(sum(int(row["tp"]) for row in rows))
    fp = int(sum(int(row["fp"]) for row in rows))
    fn = int(sum(int(row["fn"]) for row in rows))
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1_score": f1_score}
