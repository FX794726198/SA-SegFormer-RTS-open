#!/bin/bash
#SBATCH --job-name=rts-temporal
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME}"

cd "${REPO_ROOT}"
mkdir -p slurm runs
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" scripts/eval_temporal_transfer.py \
  --run-name "${RUN_NAME}" \
  --source-manifest "${SOURCE_MANIFEST:-data/manifests/manifest_2024_split_837_179_179.csv}" \
  --target-manifest "${TARGET_MANIFEST:-data/manifests/manifest_2023.csv}" \
  --regions "${REGIONS:-中1区域数据集,北2区域数据集,北3区域数据集}" \
  --epochs "${EPOCHS:-200}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-0}"
