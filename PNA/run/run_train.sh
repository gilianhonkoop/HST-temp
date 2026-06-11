#!/bin/bash
#SBATCH --job-name=pna_train
#SBATCH --output=/home/ghonkoop/repos/PNA/logs/runs/train_%j.out
#SBATCH --error=/home/ghonkoop/repos/PNA/logs/runs/train_%j.err
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=100G

set -euo pipefail

SCRIPT_DIR="/home/ghonkoop/repos/PNA"
LOG_DIR="${SCRIPT_DIR}/logs/runs"
mkdir -p "${LOG_DIR}"

# CONFIG="${CONFIG:-configs/aml/HIS-4.yaml}"
CONFIG="${CONFIG:-configs/aml/LIS-4.yaml}"

# CONFIG="${CONFIG:-configs/saml-d/SAMLD.yaml}"
RESUME="${RESUME:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

RESUME_FLAG=""
if [[ "${RESUME}" == "1" || "${RESUME,,}" == "true" ]]; then
  RESUME_FLAG="--resume"
fi

echo "======================================================"
echo "  PNA Training"
echo "======================================================"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  Config       : ${CONFIG}"
echo "  Resume       : ${RESUME}"
echo "  Python       : ${PYTHON_BIN}"
echo "  Started at   : $(date)"
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
"${PYTHON_BIN}" train.py --config "${CONFIG}" ${RESUME_FLAG}

echo ""
echo "======================================================"
echo "  Finished at  : $(date)"
echo "======================================================"
