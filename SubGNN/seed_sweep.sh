#!/bin/bash
#SBATCH --job-name=subgnn_aml_seeds
#SBATCH --output=/home/ghonkoop/repos/SubGNN/logs/fixed_seeds_%j.out
#SBATCH --error=/home/ghonkoop/repos/SubGNN/logs/fixed_seeds_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=100G

set -euo pipefail

SCRIPT_DIR="/home/ghonkoop/repos/SubGNN/SubGNN"
LOG_DIR="/home/ghonkoop/repos/SubGNN/logs"
mkdir -p "${LOG_DIR}"

DATASET_CONFIG="${1:-${DATASET_CONFIG:-aml_hi_small}}"
case "${DATASET_CONFIG}" in
  aml_hi_small)
    DATASET_LABEL="AML HI-Small"
    DEFAULT_CONFIG_PATH="config_files/aml_HI_Small/subgnn_structure_baseline.json"
    ;;
  samld)
    DATASET_LABEL="SAMLD"
    DEFAULT_CONFIG_PATH="config_files/SAMLD/samld_config.json"
    ;;
  *)
    echo "Usage: sbatch $0 {aml_hi_small|samld}" >&2
    exit 2
    ;;
esac

CONFIG_PATH="${CONFIG_PATH:-${DEFAULT_CONFIG_PATH}}"
SEEDS="${SEEDS:-42,43,44}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

echo "======================================================"
echo "  SubGNN ${DATASET_LABEL} Fixed-Seed Evaluation"
echo "======================================================"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  Config       : ${CONFIG_PATH}"
echo "  Seeds        : ${SEEDS}"
echo "  Python       : ${PYTHON_BIN}"
echo "  Started at   : $(date)"
echo "======================================================"

module purge
module load 2025
module load Anaconda3/2025.06-1

source /sw/arch/RHEL9/EB_production/2025/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
set +u
conda activate SubGNN
set -u

cd "${SCRIPT_DIR}"
"${PYTHON_BIN}" run_fixed_seeds.py -config_path "${CONFIG_PATH}" -seeds "${SEEDS}"

echo ""
echo "======================================================"
echo "  Finished at  : $(date)"
echo "======================================================"
