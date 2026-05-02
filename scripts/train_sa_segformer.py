#!/usr/bin/env python
"""Train SA-SegFormer on an RTS manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sa_segformer_rts.constants import DEFAULT_FEATURE_NAMES, DEFAULT_SEED, DEFAULT_SPLIT_COUNTS, DEFAULT_THRESHOLD  # noqa: E402
from sa_segformer_rts.data import load_manifest  # noqa: E402
from sa_segformer_rts.train import build_criterion, build_loader, build_model, split_fixed_counts, split_from_manifest_column, train_one_epoch, validate  # noqa: E402
from sa_segformer_rts.utils import set_all_seeds  # noqa: E402
from sa_segformer_rts.visualize import export_predictions  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train SA-SegFormer for RTS segmentation")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--manifest", default="data/manifests/manifest_2023_split_837_179_179.csv")
    parser.add_argument("--output-dir", default="runs/sa_segformer")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--train-count", type=int, default=DEFAULT_SPLIT_COUNTS["train"])
    parser.add_argument("--val-count", type=int, default=DEFAULT_SPLIT_COUNTS["val"])
    parser.add_argument("--test-count", type=int, default=DEFAULT_SPLIT_COUNTS["test"])
    parser.add_argument("--require-exact-total", action="store_true")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--backbone", default="mit_b0")
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--sa-heads", type=int, default=4)
    parser.add_argument("--sa-downsample", type=int, default=8)
    parser.add_argument("--no-decoder-sa", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-val-predictions", action="store_true")
    parser.add_argument("--export-test-predictions", action="store_true")
    return parser.parse_args()


def cap(samples, max_count):
    return samples[:max_count] if max_count and max_count > 0 else samples


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    set_all_seeds(args.seed)

    samples = load_manifest(repo_root / args.manifest, repo_root=repo_root)
    if samples and args.split_column in samples[0] and any(str(sample.get(args.split_column, "")).strip() for sample in samples):
        train_samples, val_samples, test_samples = split_from_manifest_column(samples, split_column=args.split_column)
    else:
        train_samples, val_samples, test_samples = split_fixed_counts(
            samples,
            split_counts={"train": args.train_count, "val": args.val_count, "test": args.test_count},
            seed=args.seed,
            require_exact_total=args.require_exact_total,
        )
    train_samples = cap(train_samples, args.max_train_samples)
    val_samples = cap(val_samples, args.max_val_samples)

    save_dir = repo_root / args.output_dir / args.run_name
    ckpt_dir = save_dir / "checkpoints"
    split_dir = save_dir / "splits"
    metrics_dir = save_dir / "metrics"
    pred_dir = save_dir / "val_predictions"
    for directory in (ckpt_dir, split_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_samples).to_csv(split_dir / "train.csv", index=False)
    pd.DataFrame(val_samples).to_csv(split_dir / "val.csv", index=False)
    pd.DataFrame(test_samples).to_csv(split_dir / "test.csv", index=False)
    print(f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)} save_dir={save_dir}")

    train_ds, train_loader = build_loader(
        train_samples,
        feature_names=DEFAULT_FEATURE_NAMES,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=True,
    )
    val_ds, val_loader = build_loader(
        val_samples,
        feature_names=DEFAULT_FEATURE_NAMES,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
    )
    _test_ds, test_loader = build_loader(
        test_samples,
        feature_names=DEFAULT_FEATURE_NAMES,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
    )
    first_batch = next(iter(val_loader))
    print(f"input_shape={tuple(first_batch[0].shape)} mask_shape={tuple(first_batch[1].shape)}")

    if args.dry_run:
        print("Dry run completed.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        feature_names=DEFAULT_FEATURE_NAMES,
        backbone=args.backbone,
        decoder_dim=args.decoder_dim,
        use_decoder_sa=not args.no_decoder_sa,
        sa_heads=args.sa_heads,
        sa_downsample=args.sa_downsample,
    ).to(device)
    criterion = build_criterion(pos_weight=args.pos_weight, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="linear",
    )

    history = []
    best_f1 = float("-inf")
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler=scheduler)
        val_loss, metrics = validate(model, val_loader, criterion, device, threshold=args.threshold)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
        history.append(row)
        pd.DataFrame(history).to_csv(save_dir / "history.csv", index=False)

        improved = metrics["f1_pos"] > best_f1 + 1e-12 or (abs(metrics["f1_pos"] - best_f1) <= 1e-12 and val_loss < best_loss)
        if improved:
            best_f1 = float(metrics["f1_pos"])
            best_loss = float(val_loss)
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
        torch.save(model.state_dict(), ckpt_dir / "last.pth")

        print(
            f"epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_pos_f1={metrics['f1_pos']:.4f} val_f1={metrics['f1_score']:.4f}"
        )

    if args.export_val_predictions:
        model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location="cpu"))
        model.to(device)
        export_predictions(model, val_loader, device, pred_dir, threshold=args.threshold)
        print(f"Exported validation predictions to {pred_dir}")
    test_loss, test_metrics = validate(model, test_loader, criterion, device, threshold=args.threshold)
    pd.DataFrame([{"split": "test", "loss": test_loss, **test_metrics}]).to_csv(metrics_dir / "test_metrics.csv", index=False)
    print(f"test_loss={test_loss:.4f} test_pos_f1={test_metrics['f1_pos']:.4f} test_f1={test_metrics['f1_score']:.4f}")
    if args.export_test_predictions:
        export_predictions(model, test_loader, device, save_dir / "test_predictions", threshold=args.threshold)


if __name__ == "__main__":
    main()
