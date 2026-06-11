#!/bin/bash
#SBATCH --job-name=pna_optuna
#SBATCH --output=/home/ghonkoop/repos/PNA/logs/optuna/optuna_%j.out
#SBATCH --error=/home/ghonkoop/repos/PNA/logs/optuna/optuna_%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=80G

set -euo pipefail

SCRIPT_DIR="/home/ghonkoop/repos/PNA"
LOG_DIR="${SCRIPT_DIR}/logs/optuna"
mkdir -p "${LOG_DIR}"

# CONFIG="${CONFIG:-configs/aml/HIS.yaml}"
# SEARCH_CONFIG="${SEARCH_CONFIG:-configs/hparams/pna_search.yaml}"

CONFIG="${CONFIG:-configs/aml/LIS.yaml}"
SEARCH_CONFIG="${SEARCH_CONFIG:-configs/hparams/lis_pna_search.yaml}"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

echo "======================================================"
echo "  PNA Optuna Search"
echo "======================================================"
echo "  Job ID        : ${SLURM_JOB_ID:-local}"
echo "  Config        : ${CONFIG}"
echo "  Search config : ${SEARCH_CONFIG}"
echo "  Python        : ${PYTHON_BIN}"
echo "  Started at    : $(date)"
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
"${PYTHON_BIN}" optuna_search.py --config "${CONFIG}" --search-config "${SEARCH_CONFIG}"

echo ""
echo "======================================================"
echo "  Finished at   : $(date)"
echo "======================================================"
