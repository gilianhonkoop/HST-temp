#!/bin/bash
#SBATCH --job-name=hst_train
#SBATCH --output=/home/ghonkoop/codebase/summarizer/logs/runs/train_%j.out
#SBATCH --error=/home/ghonkoop/codebase/summarizer/logs/runs/train_%j.err
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=60G

# ---------------------------------------------------------------------------
# Optional environment overrides:
#   CONFIG          config path (relative to summarizer/) default: configs/elliptic.yaml
#   RESUME          1/0 resume from save_dir/run_name/last_checkpoint.pth default: 0
#   PYTHON_BIN      python executable                       default: python3.11
#
# Example:
#   sbatch --export=ALL,CONFIG=configs/elliptic.yaml /home/ghonkoop/codebase/summarizer/run/run_train.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="/home/ghonkoop/codebase/summarizer"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# CONFIG="${CONFIG:-configs/aml/single/HIM/him-3.yaml}"
# CONFIG="${CONFIG:-configs/aml/single/HIS/his-5.yaml}"
# CONFIG="${CONFIG:-configs/aml/single/LIM/lim-4.yaml}"
# CONFIG="${CONFIG:-configs/aml/single/LIS/lis-14.yaml}"
# CONFIG="${CONFIG:-configs/saml-d/saml-d-4.yaml}"


# CONFIG="${CONFIG:-configs/aml/pna/HIM/him-1.yaml}"
CONFIG="${CONFIG:-configs/aml/pna/HIS/his-5.yaml}"
# CONFIG="${CONFIG:-configs/aml/pna/LIM/lim-1.yaml}"
# CONFIG="${CONFIG:-configs/aml/pna/LIS/lis-2.yaml}"
# CONFIG="${CONFIG:-configs/saml-d/pna/saml-d-1.yaml}"


RESUME="${RESUME:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

RESUME_FLAG=""
if [[ "${RESUME}" == "1" || "${RESUME,,}" == "true" ]]; then
  RESUME_FLAG="--resume"
fi

echo "======================================================"
echo "  HST Training"
echo "======================================================"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  Config       : ${CONFIG}"
echo "  Resume       : ${RESUME}"
echo "  Python       : ${PYTHON_BIN}"
echo "  Started at   : $(date)"
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

# If wandb is enabled in config, WANDB_API_KEY should be exported in your shell/environment.
"${PYTHON_BIN}" train.py --config "${CONFIG}" ${RESUME_FLAG}

echo ""
echo "======================================================"
echo "  Finished at  : $(date)"
echo "======================================================"
