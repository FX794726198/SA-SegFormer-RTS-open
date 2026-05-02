#!/usr/bin/env python
"""Cross-year temporal transfer evaluation for FusionSA-SegFormer."""

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
    parser = argparse.ArgumentParser(description="FusionSA-SegFormer cross-year temporal transfer evaluation")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--source-manifest", default="data/manifests/manifest_2024.csv")
    parser.add_argument("--target-manifest", default="data/manifests/manifest_2023.csv")
    parser.add_argument("--output-dir", default="runs/temporal_transfer")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-tag", default="2024")
    parser.add_argument("--target-tag", default="2023")
    parser.add_argument("--regions", default="中1区域数据集,北2区域数据集,北3区域数据集")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Temporal transfer keeps the original 8:2 shared-key split.")
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
    parser.add_argument("--save-source-test-predictions", action="store_true")
    parser.add_argument("--save-target-test-predictions", action="store_true")
    return parser.parse_args()


def build_index(samples, regions):
    include = set(regions) if regions else None
    index = {}
    for sample in samples:
        if include is not None and sample["region"] not in include:
            continue
        key = (sample["region"], sample["stem"])
        index[key] = sample
    return index


def split_shared_keys_8_2(shared_keys, train_ratio, seed):
    grouped = {}
    for key in shared_keys:
        grouped.setdefault(key[0], []).append(key)
    rng = np.random.default_rng(seed)
    train_keys, test_keys = [], []
    for region, keys in sorted(grouped.items()):
        keys = list(keys)
        rng.shuffle(keys)
        total = len(keys)
        n_train = int(round(total * train_ratio))
        n_train = min(max(1, n_train), total - 1)
        train_keys.extend(keys[:n_train])
        test_keys.extend(keys[n_train:])
    return train_keys, test_keys


def samples_from_keys(index, keys):
    return [index[key] for key in keys]


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
    regions = parse_regions(args.regions)
    source_samples = load_manifest(repo_root / args.source_manifest, repo_root=repo_root)
    target_samples = load_manifest(repo_root / args.target_manifest, repo_root=repo_root)
    source_index = build_index(source_samples, regions)
    target_index = build_index(target_samples, regions)
    shared_keys = sorted(set(source_index) & set(target_index))
    if not shared_keys:
        raise RuntimeError("No shared (region, stem) keys between source and target manifests.")

    train_keys, test_keys = split_shared_keys_8_2(shared_keys, args.train_ratio, args.seed)
    source_train = samples_from_keys(source_index, train_keys)
    source_val = samples_from_keys(source_index, test_keys)
    source_test = samples_from_keys(source_index, test_keys)
    target_test = samples_from_keys(target_index, test_keys)

    save_dir = repo_root / args.output_dir / args.run_name
    split_dir = save_dir / "splits"
    metrics_dir = save_dir / "metrics"
    ckpt_dir = save_dir / "checkpoints"
    for directory in (split_dir, metrics_dir, ckpt_dir):
        directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_train).to_csv(split_dir / "source_train.csv", index=False)
    pd.DataFrame(source_val).to_csv(split_dir / "source_val.csv", index=False)
    pd.DataFrame(source_test).to_csv(split_dir / "source_test.csv", index=False)
    pd.DataFrame(target_test).to_csv(split_dir / "target_test.csv", index=False)
    pd.DataFrame([{"region": region, "stem": stem} for region, stem in shared_keys]).to_csv(split_dir / "shared_pairs.csv", index=False)

    summary = {
        "source_tag": args.source_tag,
        "target_tag": args.target_tag,
        "regions": regions,
        "shared_count": len(shared_keys),
        "source_train_count": len(source_train),
        "source_val_count": len(source_val),
        "source_test_count": len(source_test),
        "target_test_count": len(target_test),
        "seed": args.seed,
        "threshold": args.threshold,
        "train_ratio": args.train_ratio,
    }
    (metrics_dir / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.dry_run:
        print("Dry run completed.")
        return

    _train_ds, train_loader = build_loader(source_train, DEFAULT_FEATURE_NAMES, batch_size=args.batch_size, num_workers=args.num_workers, is_train=True)
    _val_ds, val_loader = build_loader(source_val, DEFAULT_FEATURE_NAMES, batch_size=args.batch_size, num_workers=args.num_workers, is_train=False)
    _source_test_ds, source_test_loader = build_loader(source_test, DEFAULT_FEATURE_NAMES, batch_size=args.batch_size, num_workers=args.num_workers, is_train=False)
    _target_test_ds, target_test_loader = build_loader(target_test, DEFAULT_FEATURE_NAMES, batch_size=args.batch_size, num_workers=args.num_workers, is_train=False)

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
    results = {
        f"{args.source_tag}_val": evaluate_split(f"{args.source_tag}_val", model, criterion, val_loader, device, args.threshold, metrics_dir),
        f"{args.source_tag}_test": evaluate_split(f"{args.source_tag}_test", model, criterion, source_test_loader, device, args.threshold, metrics_dir),
        f"{args.target_tag}_test": evaluate_split(f"{args.target_tag}_test", model, criterion, target_test_loader, device, args.threshold, metrics_dir),
    }
    rows = []
    for split_name, result in results.items():
        rows.append({"split": split_name, "loss": result["loss"], **{f"pixel_{key}": value for key, value in result["pixel"].items()}, **{f"object_{key}": value for key, value in result["object"].items()}})
    pd.DataFrame(rows).to_csv(metrics_dir / "summary.csv", index=False)

    if args.save_source_test_predictions:
        export_predictions(model, source_test_loader, device, save_dir / f"{args.source_tag}_test_predictions", threshold=args.threshold)
    if args.save_target_test_predictions:
        export_predictions(model, target_test_loader, device, save_dir / f"{args.target_tag}_test_predictions", threshold=args.threshold)


if __name__ == "__main__":
    main()
