#!/usr/bin/env python
"""Region-holdout training and evaluation for FusionSA-SegFormer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sa_segformer_rts.constants import DEFAULT_FEATURE_NAMES, DEFAULT_SEED, DEFAULT_THRESHOLD  # noqa: E402
from sa_segformer_rts.data import load_manifest  # noqa: E402
from sa_segformer_rts.train import build_criterion, build_loader, build_model, evaluate_object_level, train_one_epoch, validate  # noqa: E402
from sa_segformer_rts.utils import set_all_seeds  # noqa: E402
from sa_segformer_rts.visualize import export_predictions  # noqa: E402


def parse_regions(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="FusionSA-SegFormer region-holdout evaluation")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--manifest", default="data/manifests/manifest_2023.csv")
    parser.add_argument("--output-dir", default="runs/region_holdout")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--test-regions", required=True)
    parser.add_argument("--val-regions", default="")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-test-predictions", action="store_true")
    return parser.parse_args()


def split_samples(samples, test_regions, val_regions, val_ratio, seed):
    test_regions = set(test_regions)
    val_regions = set(val_regions)
    test = [sample for sample in samples if sample["region"] in test_regions]
    remaining = [sample for sample in samples if sample["region"] not in test_regions]
    if val_regions:
        val = [sample for sample in remaining if sample["region"] in val_regions]
        train = [sample for sample in remaining if sample["region"] not in val_regions]
    else:
        rng = np.random.default_rng(seed)
        idxs = np.arange(len(remaining))
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(remaining) * val_ratio)))
        val_idx = set(int(idx) for idx in idxs[:n_val])
        val = [sample for idx, sample in enumerate(remaining) if idx in val_idx]
        train = [sample for idx, sample in enumerate(remaining) if idx not in val_idx]
    return train, val, test


def evaluate_split(name, model, criterion, loader, device, threshold, metrics_dir):
    loss, pixel = validate(model, loader, criterion, device, threshold=threshold)
    obj, per_sample = evaluate_object_level(model, loader, device, threshold=threshold)
    (metrics_dir / f"{name}_pixel.json").write_text(json.dumps({"loss": loss, **pixel}, indent=2, ensure_ascii=False), encoding="utf-8")
    (metrics_dir / f"{name}_object.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(per_sample).to_csv(metrics_dir / f"{name}_object_per_sample.csv", index=False)
    return {"loss": loss, "pixel": pixel, "object": obj}


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    set_all_seeds(args.seed)
    test_regions = parse_regions(args.test_regions)
    val_regions = parse_regions(args.val_regions)
    samples = load_manifest(repo_root / args.manifest, repo_root=repo_root)

    train_samples, val_samples, test_samples = split_samples(samples, test_regions, val_regions, args.val_ratio, args.seed)
    run_name = args.run_name or f"test-{'_'.join(test_regions)}"
    save_dir = repo_root / args.output_dir / run_name
    split_dir = save_dir / "splits"
    metrics_dir = save_dir / "metrics"
    ckpt_dir = save_dir / "checkpoints"
    for directory in (split_dir, metrics_dir, ckpt_dir):
        directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_samples).to_csv(split_dir / "train.csv", index=False)
    pd.DataFrame(val_samples).to_csv(split_dir / "val.csv", index=False)
    pd.DataFrame(test_samples).to_csv(split_dir / "test.csv", index=False)

    summary = {
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "train_regions": sorted({sample["region"] for sample in train_samples}),
        "val_regions": sorted({sample["region"] for sample in val_samples}),
        "test_regions": sorted({sample["region"] for sample in test_samples}),
        "seed": args.seed,
        "threshold": args.threshold,
    }
    (metrics_dir / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.dry_run:
        print("Dry run completed.")
        return
    if not train_samples or not val_samples or not test_samples:
        raise ValueError("Empty train/val/test split.")

    _train_ds, train_loader = build_loader(train_samples, DEFAULT_FEATURE_NAMES, batch_size=args.batch_size, num_workers=args.num_workers, is_train=True)
    _val_ds, val_loader = build_loader(val_samples, DEFAULT_FEATURE_NAMES, batch_size=args.batch_size, num_workers=args.num_workers, is_train=False)
    _test_ds, test_loader = build_loader(test_samples, DEFAULT_FEATURE_NAMES, batch_size=args.batch_size, num_workers=args.num_workers, is_train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(DEFAULT_FEATURE_NAMES).to(device)
    criterion = build_criterion(device=device)
    best_ckpt = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else ckpt_dir / "best.pth"

    if not args.eval_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy="linear")
        history = []
        best_f1 = float("-inf")
        best_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler=scheduler)
            val_loss, val_metrics = validate(model, val_loader, criterion, device, threshold=args.threshold)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics}
            history.append(row)
            pd.DataFrame(history).to_csv(save_dir / "history.csv", index=False)
            improved = val_metrics["f1_pos"] > best_f1 + 1e-12 or (abs(val_metrics["f1_pos"] - best_f1) <= 1e-12 and val_loss < best_loss)
            if improved:
                best_f1 = float(val_metrics["f1_pos"])
                best_loss = float(val_loss)
                torch.save(model.state_dict(), best_ckpt)
            print(f"epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_pos_f1={val_metrics['f1_pos']:.4f}")

    model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
    model.to(device)
    val_result = evaluate_split("val", model, criterion, val_loader, device, args.threshold, metrics_dir)
    test_result = evaluate_split("test", model, criterion, test_loader, device, args.threshold, metrics_dir)
    pd.DataFrame(
        [
            {"split": "val", "loss": val_result["loss"], **{f"pixel_{k}": v for k, v in val_result["pixel"].items()}, **{f"object_{k}": v for k, v in val_result["object"].items()}},
            {"split": "test", "loss": test_result["loss"], **{f"pixel_{k}": v for k, v in test_result["pixel"].items()}, **{f"object_{k}": v for k, v in test_result["object"].items()}},
        ]
    ).to_csv(metrics_dir / "summary.csv", index=False)
    if args.save_test_predictions:
        export_predictions(model, test_loader, device, save_dir / "test_predictions", threshold=args.threshold)


if __name__ == "__main__":
    main()
