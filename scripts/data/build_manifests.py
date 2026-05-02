#!/usr/bin/env python
"""Build CSV manifests for the released RTS datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sa_segformer_rts.data import build_manifest  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Build SA-SegFormer RTS dataset manifests")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--factors-2024", default="data/2024/factors")
    parser.add_argument("--optical-2024", default="data/2024/optical")
    parser.add_argument("--factors-2023", default="data/2023/factors")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    manifest_dir = repo_root / args.manifest_dir

    jobs = [
        ("2024", repo_root / args.factors_2024, repo_root / args.optical_2024, manifest_dir / "manifest_2024.csv"),
        ("2023", repo_root / args.factors_2023, None, manifest_dir / "manifest_2023.csv"),
    ]

    for year, factor_root, optical_root, out_path in jobs:
        manifest = build_manifest(
            out_path=out_path,
            factor_root=factor_root,
            optical_root=optical_root,
            year=year,
            relative_to=repo_root,
        )
        regions = manifest["region"].nunique()
        complete = (manifest[["DEM", "EVI", "FTI", "LST", "NBR", "NDMI", "NDVI", "TCB", "TCG", "TCW"]] != "").all(axis=1).sum()
        print(f"{year}: wrote {out_path} with {len(manifest)} samples, {regions} regions, {complete} complete factor stacks")


if __name__ == "__main__":
    main()
