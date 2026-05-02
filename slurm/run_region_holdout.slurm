#!/bin/bash
#SBATCH --job-name=rts-region
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TEST_REGIONS="${TEST_REGIONS:?Set TEST_REGIONS, e.g. 北1区域数据集}"
RUN_NAME="${RUN_NAME:-region_holdout}"
MANIFEST="${MANIFEST:-data/manifests/manifest_2023.csv}"

cd "${REPO_ROOT}"
mkdir -p slurm runs
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" scripts/eval_region_holdout.py \
  --manifest "${MANIFEST}" \
  --test-regions "${TEST_REGIONS}" \
  --run-name "${RUN_NAME}" \
  --epochs "${EPOCHS:-200}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-0}"
