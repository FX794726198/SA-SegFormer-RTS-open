#!/usr/bin/env python
"""Run FusionSA-SegFormer inference from a checkpoint and manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sa_segformer_rts.constants import DEFAULT_FEATURE_NAMES, DEFAULT_THRESHOLD  # noqa: E402
from sa_segformer_rts.data import load_manifest  # noqa: E402
from sa_segformer_rts.train import build_loader, build_model  # noqa: E402
from sa_segformer_rts.visualize import export_predictions  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="FusionSA-SegFormer inference")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--manifest", default="data/manifests/manifest_2023_split_837_179_179.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="runs/inference")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--backbone", default="mit_b0")
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--sa-heads", type=int, default=4)
    parser.add_argument("--sa-downsample", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--prediction-max-preview-side", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    samples = load_manifest(repo_root / args.manifest, repo_root=repo_root)
    if args.max_samples:
        samples = samples[: args.max_samples]
    _dataset, loader = build_loader(
        samples,
        feature_names=DEFAULT_FEATURE_NAMES,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
        weighted_sampling=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        feature_names=DEFAULT_FEATURE_NAMES,
        backbone=args.backbone,
        decoder_dim=args.decoder_dim,
        sa_heads=args.sa_heads,
        sa_downsample=args.sa_downsample,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)
    export_predictions(
        model,
        loader,
        device,
        repo_root / args.output_dir,
        threshold=args.threshold,
        max_preview_side=args.prediction_max_preview_side,
    )
    print(f"Saved predictions to {repo_root / args.output_dir}")


if __name__ == "__main__":
    main()
