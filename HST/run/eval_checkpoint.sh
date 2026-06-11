#!/bin/bash
#SBATCH --job-name=hst_eval
#SBATCH --output=/home/ghonkoop/codebase/summarizer/logs/runs/eval_%j.out
#SBATCH --error=/home/ghonkoop/codebase/summarizer/logs/runs/eval_%j.err
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=80G

# Optional environment overrides:
#   CONFIG                config path relative to summarizer/
#   CHECKPOINT_PATH       checkpoint to evaluate
#   EVAL_RUN_NAME         run name to use during evaluation
#   EVAL_NODE_BATCH_SIZE  chunk size for eval node aggregation
#   THRESHOLD_LOWER       optional threshold search lower bound
#   THRESHOLD_UPPER       optional threshold search upper bound
#   THRESHOLD_N_TRIALS    optional threshold search trial count
#   THRESHOLD_METRIC      optional threshold search metric
#   PYTHON_BIN            python executable

set -euo pipefail

SCRIPT_DIR="/home/ghonkoop/codebase/summarizer"

# CONFIG="${CONFIG:-configs/aml/single/HIM/him-2.yaml}"
CONFIG="${CONFIG:-configs/aml/pna/HIS/his-stats-1.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/scratch-shared/ghonkoop/checkpoints/aml/hi-small/HIS-STATS-PNA-m160-l3-norm-posw040-seeds42-44_seed_43/final_checkpoint.pth}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-LIM-1-eval}"
EVAL_NODE_BATCH_SIZE="${EVAL_NODE_BATCH_SIZE:-1024}"
THRESHOLD_LOWER="${THRESHOLD_LOWER:-0.05}"
THRESHOLD_UPPER="${THRESHOLD_UPPER:-0.95}"
THRESHOLD_N_TRIALS="${THRESHOLD_N_TRIALS:-50}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

echo "======================================================"
echo "  HST Checkpoint Evaluation"
echo "======================================================"
echo "  Job ID               : ${SLURM_JOB_ID:-local}"
echo "  Config               : ${CONFIG}"
echo "  Checkpoint           : ${CHECKPOINT_PATH}"
echo "  Eval run name        : ${EVAL_RUN_NAME}"
echo "  Eval node batch size : ${EVAL_NODE_BATCH_SIZE}"
echo "  Threshold lower      : ${THRESHOLD_LOWER:-<config>}"
echo "  Threshold upper      : ${THRESHOLD_UPPER:-<config>}"
echo "  Threshold n trials   : ${THRESHOLD_N_TRIALS:-<config>}"
echo "  Threshold metric     : ${THRESHOLD_METRIC:-<config>}"
echo "  Python               : ${PYTHON_BIN}"
echo "  Started at           : $(date)"
echo "======================================================"

module purge
module load 2025
module load Anaconda3/2025.06-1

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source /sw/arch/RHEL9/EB_production/2025/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
set +u
conda activate HST
set -u

cd "${SCRIPT_DIR}"

EXTRA_ARGS=()
if [[ -n "${THRESHOLD_LOWER}" ]]; then
  EXTRA_ARGS+=(--threshold-lower "${THRESHOLD_LOWER}")
fi
if [[ -n "${THRESHOLD_UPPER}" ]]; then
  EXTRA_ARGS+=(--threshold-upper "${THRESHOLD_UPPER}")
fi
if [[ -n "${THRESHOLD_N_TRIALS}" ]]; then
  EXTRA_ARGS+=(--threshold-n-trials "${THRESHOLD_N_TRIALS}")
fi
if [[ -n "${THRESHOLD_METRIC}" ]]; then
  EXTRA_ARGS+=(--threshold-metric "${THRESHOLD_METRIC}")
fi

"${PYTHON_BIN}" eval_checkpoint.py \
  --config "${CONFIG}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --run-name "${EVAL_RUN_NAME}" \
  --eval-node-batch-size "${EVAL_NODE_BATCH_SIZE}" \
  --disable-wandb \
  "${EXTRA_ARGS[@]}"

echo ""
echo "======================================================"
echo "  Finished at          : $(date)"
echo "======================================================"
