#!/bin/bash
#SBATCH --job-name=s2n_samld
#SBATCH --output=/home/ghonkoop/repos/S2N/logs/train_%j.out
#SBATCH --error=/home/ghonkoop/repos/S2N/logs/train_%j.err
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=100G

set -euo pipefail

WANDB_ENTITY="hierarchical-subgraph-transformer"
WANDB_PROJECT="S2N"
WANDB_NAME="SAMLD_1"

SCRIPT_DIR="/home/ghonkoop/repos/S2N/S2N"
LOG_DIR="/home/ghonkoop/repos/S2N/logs"
mkdir -p "${LOG_DIR}"

echo "======================================================"
echo "  S2N SAMLD Training"
echo "======================================================"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  Started at   : $(date)"
echo "======================================================"

module purge
module load 2025
module load Anaconda3/2025.06-1

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

source /sw/arch/RHEL9/EB_production/2025/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh

set +u
conda activate s2n_env
set -u

if [ -f "/home/ghonkoop/repos/S2N/.env" ]; then
    set -a
    source /home/ghonkoop/repos/S2N/.env
    set +a
    echo "Loaded WANDB_API_KEY from .env"
else
    echo "Warning: .env file not found at /home/ghonkoop/repos/S2N/.env"
fi

cd "${SCRIPT_DIR}"

REPO_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${REPO_DIR}:${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1
export CUDA_LAUNCH_BLOCKING=1

"${HOME}/.conda/envs/s2n_env/bin/python" run_main.py \
    +trainer.num_nodes=1 \
    trainer.devices="[0]" \
    datamodule=s2n/samld/for-gcn2 \
    model=gcn2/s2n/for-samld \
    experiment=s2n_aml \
    logger.wandb.entity="${WANDB_ENTITY}" \
    logger.wandb.project="${WANDB_PROJECT}" \
    logger.wandb.name="${WANDB_NAME}"

echo ""
echo "======================================================"
echo "  Finished at  : $(date)"
echo "======================================================"
