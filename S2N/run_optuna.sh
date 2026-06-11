#!/bin/bash
#SBATCH --job-name=s2n_amlhi
#SBATCH --output=/home/ghonkoop/repos/S2N/logs/train_%j.out
#SBATCH --error=/home/ghonkoop/repos/S2N/logs/train_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=100G

# ---------------------------------------------------------------------------
# S2N Model Runner for AMLHI
# ---------------------------------------------------------------------------

set -euo pipefail

# --- W&B Configuration ---
WANDB_ENTITY="hierarchical-subgraph-transformer"
WANDB_PROJECT="S2N"
WANDB_NAME="HIS_optuna_1"
TOP_K_WANDB_RUNS="${TOP_K_WANDB_RUNS:-3}"
# -------------------------

SCRIPT_DIR="/home/ghonkoop/repos/S2N/S2N"
LOG_DIR="/home/ghonkoop/repos/S2N/logs"
mkdir -p "${LOG_DIR}"

echo "======================================================"
echo "  S2N Training"
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

# Load .env file to export WANDB_API_KEY
if [ -f "/home/ghonkoop/repos/S2N/.env" ]; then
    set -a
    source /home/ghonkoop/repos/S2N/.env
    set +a
    echo "Loaded WANDB_API_KEY from .env"
else
    echo "Warning: .env file not found at /home/ghonkoop/repos/S2N/.env"
fi

cd "${SCRIPT_DIR}"

# Ensure Python can import local modules after Hydra changes working dir (safe with -u)
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${REPO_DIR}:${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# Help debugging Hydra target import errors if any
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1

# (Dependencies are preinstalled in the conda environment)
export CUDA_LAUNCH_BLOCKING=1

"${HOME}/.conda/envs/s2n_env/bin/python" run_main.py -m \
    +trainer.num_nodes=1 \
    trainer.devices="[0]" \
    datamodule=s2n/amlhi/for-gcn2 \
    model=gcn2/s2n/for-amlhi \
    experiment=s2n_aml \
    logger.wandb.entity="${WANDB_ENTITY}" \
    logger.wandb.project="${WANDB_PROJECT}" \
    logger.wandb.name="${WANDB_NAME}" \
    +logger.wandb.mode="offline" \
    hparams_search=sgn_optuna

echo ""
echo "======================================================"
echo "  Optuna Sweep Finished!"
echo "======================================================"

echo "Locating newest log directory to sync top ${TOP_K_WANDB_RUNS} runs to WandB..."
# Identify the most recent sweep log directory
RECENT_LOG_DIR=$(ls -td /home/ghonkoop/repos/S2N/logs_multi/*/*/ | head -n 1)
WANDB_OFFLINE_DIR="/home/ghonkoop/repos/S2N/logs_wandb/wandb/"

"${HOME}/.conda/envs/s2n_env/bin/python" sync_top_k_wandb.py \
    --multi-run-dir "${RECENT_LOG_DIR}" \
    --wandb-dir "${WANDB_OFFLINE_DIR}" \
    --top-k "${TOP_K_WANDB_RUNS}"

echo ""
echo "======================================================"
echo "  Finished at  : $(date)"
echo "======================================================"
