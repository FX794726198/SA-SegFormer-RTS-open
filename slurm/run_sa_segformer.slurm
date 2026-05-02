#!/bin/bash
#SBATCH --job-name=sa-segformer-rts
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST="${MANIFEST:-data/manifests/manifest_2023_split_837_179_179.csv}"
RUN_NAME="${RUN_NAME:-sa_segformer_837_179_179}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"

cd "${REPO_ROOT}"
mkdir -p slurm runs
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" scripts/train_sa_segformer.py \
  --manifest "${MANIFEST}" \
  --run-name "${RUN_NAME}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"
