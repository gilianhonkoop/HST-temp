#!/bin/bash
#SBATCH --job-name=hst_optuna
#SBATCH --output=/home/ghonkoop/codebase/summarizer/logs/optuna/train_%j.out
#SBATCH --error=/home/ghonkoop/codebase/summarizer/logs/optuna/train_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=75G
# SBATCH --array=1-4  # Launch 4 parallel jobs

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

source /sw/arch/RHEL9/EB_production/2025/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
set +u
conda activate HST
set -u

SCRIPT_DIR="/home/ghonkoop/codebase/summarizer"
cd "${SCRIPT_DIR}"

echo "======================================================"
echo "  Started at   : $(date)"
echo "======================================================"

python /home/ghonkoop/codebase/summarizer/optuna_search.py \
  --config-name=hparams_search.yaml \
  train.device=cuda \
  hydra.sweeper.n_trials=15

echo ""
echo "======================================================"
echo "  Finished at  : $(date)"
echo "======================================================"
