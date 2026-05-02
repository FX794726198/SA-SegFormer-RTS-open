#!/usr/bin/env python
"""Train a paper baseline model on RGB chips with the fixed 837/179/179 split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sa_segformer_rts.baselines import build_baseline_model  # noqa: E402
from sa_segformer_rts.constants import DEFAULT_FEATURE_NAMES, DEFAULT_SEED, DEFAULT_SPLIT_COUNTS, DEFAULT_THRESHOLD  # noqa: E402
from sa_segformer_rts.data import load_manifest  # noqa: E402
from sa_segformer_rts.train import build_criterion, build_loader, split_fixed_counts, split_from_manifest_column, train_one_epoch, validate  # noqa: E402
from sa_segformer_rts.utils import set_all_seeds  # noqa: E402
from sa_segformer_rts.visualize import export_predictions  # noqa: E402


def parse_args(default_model: str):
    parser = argparse.ArgumentParser(description="Train a paper baseline RTS segmentation model")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--manifest", default="data/manifests/manifest_2023_split_837_179_179.csv")
    parser.add_argument("--output-dir", default="runs/baselines")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--train-count", type=int, default=DEFAULT_SPLIT_COUNTS["train"])
    parser.add_argument("--val-count", type=int, default=DEFAULT_SPLIT_COUNTS["val"])
    parser.add_argument("--test-count", type=int, default=DEFAULT_SPLIT_COUNTS["test"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--include-factors", action="store_true", help="Use RGB plus the 10 factor rasters instead of the paper RGB-only baseline input.")
    parser.add_argument("--rgb-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-test-predictions", action="store_true")
    return parser.parse_args()


def main(default_model: str = "unet"):
    args = parse_args(default_model)
    repo_root = Path(args.repo_root).expanduser().resolve()
    set_all_seeds(args.seed)
    feature_names = DEFAULT_FEATURE_NAMES if args.include_factors and not args.rgb_only else []
    samples = load_manifest(repo_root / args.manifest, repo_root=repo_root)
    if samples and args.split_column in samples[0] and any(str(sample.get(args.split_column, "")).strip() for sample in samples):
        train_samples, val_samples, test_samples = split_from_manifest_column(samples, split_column=args.split_column)
    else:
        train_samples, val_samples, test_samples = split_fixed_counts(
            samples,
            split_counts={"train": args.train_count, "val": args.val_count, "test": args.test_count},
            seed=args.seed,
        )

    run_name = args.run_name or args.model.lower().replace("-", "_")
    save_dir = repo_root / args.output_dir / run_name
    ckpt_dir = save_dir / "checkpoints"
    split_dir = save_dir / "splits"
    metrics_dir = save_dir / "metrics"
    for directory in (ckpt_dir, split_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_samples).to_csv(split_dir / "train.csv", index=False)
    pd.DataFrame(val_samples).to_csv(split_dir / "val.csv", index=False)
    pd.DataFrame(test_samples).to_csv(split_dir / "test.csv", index=False)
    print(f"model={args.model} channels={3 + len(feature_names)} train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    _train_ds, train_loader = build_loader(train_samples, feature_names, tuple(args.image_size), args.batch_size, args.num_workers, is_train=True)
    _val_ds, val_loader = build_loader(val_samples, feature_names, tuple(args.image_size), args.batch_size, args.num_workers, is_train=False)
    _test_ds, test_loader = build_loader(test_samples, feature_names, tuple(args.image_size), args.batch_size, args.num_workers, is_train=False)
    first_batch = next(iter(val_loader))
    print(f"input_shape={tuple(first_batch[0].shape)}")
    if args.dry_run:
        print("Dry run completed.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_baseline_model(args.model, in_channels=3 + len(feature_names), classes=1).to(device)
    criterion = build_criterion(pos_weight=args.pos_weight, device=device)
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
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
        print(f"epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_pos_f1={val_metrics['f1_pos']:.4f}")

    model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location="cpu"))
    model.to(device)
    test_loss, test_metrics = validate(model, test_loader, criterion, device, threshold=args.threshold)
    pd.DataFrame([{"split": "test", "loss": test_loss, **test_metrics}]).to_csv(metrics_dir / "test_metrics.csv", index=False)
    print(f"test_loss={test_loss:.4f} test_pos_f1={test_metrics['f1_pos']:.4f} test_f1={test_metrics['f1_score']:.4f}")
    if args.export_test_predictions:
        export_predictions(model, test_loader, device, save_dir / "test_predictions", threshold=args.threshold)


if __name__ == "__main__":
    main()
