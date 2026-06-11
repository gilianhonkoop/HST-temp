#!/bin/bash
#SBATCH --job-name=s2n_amlhi
#SBATCH --output=/home/ghonkoop/repos/S2N/logs/train_%j.out
#SBATCH --error=/home/ghonkoop/repos/S2N/logs/train_%j.err
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=25G

# ---------------------------------------------------------------------------
# S2N Model Runner for AMLHI
# ---------------------------------------------------------------------------

set -euo pipefail


SCRIPT_DIR="/home/ghonkoop/repos/S2N/"
LOG_DIR="/home/ghonkoop/repos/S2N/logs"

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

cd "${SCRIPT_DIR}"

export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1
export CUDA_LAUNCH_BLOCKING=1

"${HOME}/.conda/envs/s2n_env/bin/python" post_threshold_search.py

echo ""
echo "======================================================"
echo "  Finished at  : $(date)"
echo "======================================================"
