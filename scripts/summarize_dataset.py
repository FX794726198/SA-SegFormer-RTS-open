#!/usr/bin/env python
"""Summarize released RTS manifests and data completeness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sa_segformer_rts.constants import DEFAULT_FEATURE_NAMES  # noqa: E402
from sa_segformer_rts.data import load_manifest  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize SA-SegFormer RTS dataset manifests")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--manifests", nargs="*", default=["data/manifests/manifest_2024.csv", "data/manifests/manifest_2023.csv"])
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    rows = []
    for manifest_name in args.manifests:
        manifest_path = repo_root / manifest_name
        samples = load_manifest(manifest_path, repo_root=repo_root)
        df = pd.DataFrame(samples)
        year = str(df["year"].iloc[0]) if len(df) else manifest_path.stem
        exists = {column: df[column].map(lambda value: Path(value).is_file()).mean() for column in ["optical_path", "mask_path", *DEFAULT_FEATURE_NAMES]}
        rows.append(
            {
                "manifest": str(manifest_path.relative_to(repo_root)),
                "year": year,
                "samples": len(df),
                "regions": df["region"].nunique() if len(df) else 0,
                "complete_factor_rate": float((df[DEFAULT_FEATURE_NAMES] != "").all(axis=1).mean()) if len(df) else 0.0,
                **{f"{key}_exists_rate": float(value) for key, value in exists.items()},
            }
        )
        print(f"\n=== {year} ({manifest_path.relative_to(repo_root)}) ===")
        print(f"samples: {len(df)}")
        print(f"regions: {df['region'].nunique() if len(df) else 0}")
        if len(df):
            print(df.groupby("region").size().to_string())
            print("file existence rates:")
            for key, value in exists.items():
                print(f"  {key}: {value:.3f}")

    summary = pd.DataFrame(rows)
    out_path = repo_root / "data" / "manifests" / "dataset_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
