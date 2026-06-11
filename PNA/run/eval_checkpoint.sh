#!/bin/bash
#SBATCH --job-name=pna_eval
#SBATCH --output=/home/ghonkoop/repos/PNA/logs/eval/eval_%j.out
#SBATCH --error=/home/ghonkoop/repos/PNA/logs/eval/eval_%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=40G

set -euo pipefail

SCRIPT_DIR="/home/ghonkoop/repos/PNA"
LOG_DIR="${SCRIPT_DIR}/logs/eval"
mkdir -p "${LOG_DIR}"

CONFIG="${CONFIG:-configs/aml/HIS.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

echo "======================================================"
echo "  PNA Checkpoint Eval"
echo "======================================================"
echo "  Config     : ${CONFIG}"
echo "  Checkpoint : ${CHECKPOINT:-config default}"
echo "======================================================"

module purge
module load 2025
module load Anaconda3/2025.06-1

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

source /sw/arch/RHEL9/EB_production/2025/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
set +u
conda activate HST
set -u

cd "${SCRIPT_DIR}"
if [[ -n "${CHECKPOINT}" ]]; then
  "${PYTHON_BIN}" eval_checkpoint.py --config "${CONFIG}" --checkpoint "${CHECKPOINT}"
else
  "${PYTHON_BIN}" eval_checkpoint.py --config "${CONFIG}"
fi
