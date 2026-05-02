#!/bin/bash
#SBATCH --job-name=rts-baseline
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-unet}"
MANIFEST="${MANIFEST:-data/manifests/manifest_2023_split_837_179_179.csv}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"

cd "${REPO_ROOT}"
mkdir -p slurm runs
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" scripts/train_baseline.py \
  --model "${MODEL}" \
  --manifest "${MANIFEST}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"
