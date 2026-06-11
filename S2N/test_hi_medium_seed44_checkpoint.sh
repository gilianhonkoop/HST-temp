#!/bin/bash
#SBATCH --job-name=s2n_him_test
#SBATCH --output=/home/ghonkoop/repos/S2N/logs/test/aml_ckpt_test_%j.out
#SBATCH --error=/home/ghonkoop/repos/S2N/logs/test/aml_ckpt_test_%j.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=100G

set -euo pipefail

SCRIPT_DIR="/home/ghonkoop/repos/S2N/S2N"
REPO_DIR="/home/ghonkoop/repos/S2N"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/ghonkoop/repos/S2N/logs/GCN-S2N/2026-06-05-23-16-29/checkpoints/epoch_epoch=144.ckpt}"
METRICS_OUTPUT="${METRICS_OUTPUT:-/home/ghonkoop/repos/S2N/logs/hi_medium_seed44_test_metrics.json}"

echo "======================================================"
echo "  S2N checkpoint test"
echo "======================================================"
echo "  Job ID      : ${SLURM_JOB_ID:-local}"
echo "  Checkpoint  : ${CHECKPOINT_PATH}"
echo "  Output      : ${METRICS_OUTPUT}"
echo "  Started at  : $(date)"
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

if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    source "${REPO_DIR}/.env"
    set +a
    echo "Loaded WANDB_API_KEY from .env"
else
    echo "Warning: .env file not found at ${REPO_DIR}/.env"
fi

cd "${SCRIPT_DIR}"

export PYTHONPATH="${REPO_DIR}:${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1
export CUDA_LAUNCH_BLOCKING=1

"${HOME}/.conda/envs/s2n_env/bin/python" test_checkpoint.py \
    +trainer.num_nodes=1 \
    trainer.devices="[0]" \
    datamodule=s2n/aml_hi_medium/for-gcn \
    model=gcn/s2n/for-amlhi \
    experiment=s2n_aml \
    seed=44 \
    logger.wandb.entity=hierarchical-subgraph-transformer \
    logger.wandb.project=S2N \
    logger.wandb.name=HIM-GCN2_seed44_test \
    "+checkpoint_path=\"${CHECKPOINT_PATH}\"" \
    "+metrics_output=\"${METRICS_OUTPUT}\""

echo ""
echo "======================================================"
echo "  Finished at : $(date)"
echo "======================================================"
