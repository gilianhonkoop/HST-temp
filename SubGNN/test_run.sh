#!/bin/bash
#SBATCH --job-name=subgnn_aml_test
#SBATCH --output=/home/ghonkoop/repos/SubGNN/logs/test_%j.out
#SBATCH --error=/home/ghonkoop/repos/SubGNN/logs/test_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=64G

set -euo pipefail

SCRIPT_DIR="/home/ghonkoop/repos/SubGNN/SubGNN"
LOG_DIR="/home/ghonkoop/repos/SubGNN/logs"
mkdir -p "${LOG_DIR}"

DATA_DIR="${DATA_DIR:-/home/ghonkoop/data/aml/HI-Small}"
RESTORE_PATH="${RESTORE_PATH:-/home/ghonkoop/repos/SubGNN/tensorboard/aml-HI-Small-SubGNN/version_4280293}"
RESTORE_CKPT="${RESTORE_CKPT:-epoch=8-val_micro_f1=0.97-val_acc=0.97-val_auroc=0.93.ckpt}"
WANDB_PROJECT="${WANDB_PROJECT:-ml-detection}"
WANDB_RUN_ID="${WANDB_RUN_ID:-abd0byir}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

echo "======================================================"
echo "  SubGNN AML Test"
echo "======================================================"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  Data dir     : ${DATA_DIR}"
echo "  Restore path : ${RESTORE_PATH}"
echo "  Restore ckpt : ${RESTORE_CKPT}"
echo "  W&B project  : ${WANDB_PROJECT}"
echo "  W&B run id   : ${WANDB_RUN_ID}"
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

WANDB_ARGS=()
if [[ -n "${WANDB_RUN_ID}" ]]; then
  WANDB_ARGS+=( -wandb_project "${WANDB_PROJECT}" -wandb_run_id "${WANDB_RUN_ID}" )
fi

"${PYTHON_BIN}" test.py \
  -data_dir "${DATA_DIR}" \
  -tb_dir "tensorboard" \
  -tb_name "aml-HI-Small-SubGNN" \
  -restoreModelPath "${RESTORE_PATH}" \
  -restoreModelName "${RESTORE_CKPT}" \
  "${WANDB_ARGS[@]}"

echo ""
echo "======================================================"
echo "  Finished at  : $(date)"
echo "======================================================"
