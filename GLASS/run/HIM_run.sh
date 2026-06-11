#!/bin/bash
#SBATCH --job-name=glass_aml
#SBATCH --output=/home/ghonkoop/repos/GLASS/logs/train_%j.out
#SBATCH --error=/home/ghonkoop/repos/GLASS/logs/train_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=70G

set -euo pipefail

SCRIPT_DIR="/home/ghonkoop/repos/GLASS"
LOG_DIR="/home/ghonkoop/repos/GLASS/logs"
mkdir -p "${LOG_DIR}"

DATASET_NAME="HI-Medium"
CONFIG_PATH="config/HIM/him-4.yml"
AML_BASE_PATH="/home/ghonkoop/data/aml/${DATASET_NAME}"
PYTHON_BIN="python"
USE_WANDB="1"
WANDB_PROJECT="glass"
WANDB_ENTITY="hierarchical-subgraph-transformer"

echo "======================================================"
echo "  GLASS AML Training"
echo "======================================================"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  Dataset      : ${DATASET_NAME}"
echo "  Config path  : ${CONFIG_PATH}"
echo "  AML path     : ${AML_BASE_PATH}"
echo "  WandB        : ${USE_WANDB} (${WANDB_PROJECT})"
echo "  Started at   : $(date)"
echo "======================================================"

module purge
module load 2025
module load Anaconda3/2025.06-1

source /sw/arch/RHEL9/EB_production/2025/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
set +u
conda activate GLASS_H100
set -u

export AML_BASE_PATH
export USE_WANDB
export WANDB_PROJECT
export WANDB_ENTITY

cd "${SCRIPT_DIR}"
"${PYTHON_BIN}" GLASSTest.py \
  --dataset "${DATASET_NAME}" \
  --config "${CONFIG_PATH}" \
  --use_deg \
  --use_seed \
  --use_maxzeroone \
  --repeat 1 \
  --device 0

echo ""
echo "======================================================"
echo "  Finished at  : $(date)"
echo "======================================================"
