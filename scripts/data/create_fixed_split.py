#!/usr/bin/env python
"""Create the required 837/179/179 split from a manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sa_segformer_rts.constants import DEFAULT_SEED, DEFAULT_SPLIT_COUNTS  # noqa: E402
from sa_segformer_rts.data import load_manifest  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Create a fixed 837/179/179 split manifest")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--manifest", default="data/manifests/manifest_2024.csv")
    parser.add_argument("--output", default="data/manifests/manifest_2024_split_837_179_179.csv")
    parser.add_argument("--train-count", type=int, default=DEFAULT_SPLIT_COUNTS["train"])
    parser.add_argument("--val-count", type=int, default=DEFAULT_SPLIT_COUNTS["val"])
    parser.add_argument("--test-count", type=int, default=DEFAULT_SPLIT_COUNTS["test"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--require-exact-total", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    samples = load_manifest(repo_root / args.manifest, repo_root=repo_root)
    counts = {"train": args.train_count, "val": args.val_count, "test": args.test_count}
    train, val, test = split_fixed_counts_light(samples, split_counts=counts, seed=args.seed, require_exact_total=args.require_exact_total)
    rows = [*train, *val, *test]
    df = pd.DataFrame(rows)
    path_columns = [column for column in df.columns if column.endswith("_path") or column in ["DEM", "EVI", "FTI", "LST", "NBR", "NDMI", "NDVI", "TCB", "TCG", "TCW"]]
    for column in path_columns:
        df[column] = df[column].map(lambda value: _relative(value, repo_root))
    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(df["split"].value_counts().reindex(["train", "val", "test"]).to_string())


def _relative(value, root: Path):
    if not isinstance(value, str) or not value:
        return ""
    path = Path(value)
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def split_fixed_counts_light(samples, split_counts, seed, require_exact_total=False):
    import numpy as np

    total_required = sum(split_counts.values())
    if len(samples) < total_required:
        raise ValueError(f"Need at least {total_required} samples for fixed split, got {len(samples)}.")
    if require_exact_total and len(samples) != total_required:
        raise ValueError(f"Expected exactly {total_required} samples, got {len(samples)}.")
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(samples))
    rng.shuffle(idxs)
    selected = [dict(samples[int(idx)]) for idx in idxs[:total_required]]
    offset = 0
    parts = {}
    for split_name in ("train", "val", "test"):
        count = split_counts[split_name]
        parts[split_name] = selected[offset : offset + count]
        for sample in parts[split_name]:
            sample["split"] = split_name
        offset += count
    return parts["train"], parts["val"], parts["test"]


if __name__ == "__main__":
    main()
