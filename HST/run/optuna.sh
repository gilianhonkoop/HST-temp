#!/bin/bash
#SBATCH --job-name=hst_optuna
#SBATCH --output=/home/ghonkoop/codebase/summarizer/logs/optuna/train_%j.out
#SBATCH --error=/home/ghonkoop/codebase/summarizer/logs/optuna/train_%j.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=80G


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

# CONFIG_NAME="${CONFIG_NAME:-LIS_hp_3.yaml}"
CONFIG_NAME="${CONFIG_NAME:-SAMLD_hp.yaml}"
N_TRIALS="${N_TRIALS:-2}"

echo "======================================================"
echo "  Started at   : $(date)"
echo "  Config Name  : ${CONFIG_NAME}"
echo "  Top-k Mode   : load best_checkpoint.pth, no retrain"
echo "  N Trials     : ${N_TRIALS:-config default}"
echo "======================================================"

N_TRIALS_ARG=()
if [[ -n "${N_TRIALS}" ]]; then
  N_TRIALS_ARG=("hydra.sweeper.n_trials=${N_TRIALS}")
fi

python /home/ghonkoop/codebase/summarizer/optuna_search.py \
  --config-name=${CONFIG_NAME} \
  train.device=cuda \
  "${N_TRIALS_ARG[@]}"

echo ""
echo "======================================================"
echo "  Finished at  : $(date)"
echo "======================================================"
