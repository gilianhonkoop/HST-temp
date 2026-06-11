#!/bin/bash
#SBATCH --job-name=hst_threshold_search
#SBATCH --output=/home/ghonkoop/codebase/summarizer/logs/threshold/train_%j.out
#SBATCH --error=/home/ghonkoop/codebase/summarizer/logs/threshold/train_%j.err
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=50G

#!/usr/bin/env bash
set -euo pipefail

# RUN_DIR=${1:?"Usage: $0 <run_dir> [ckpt_name] [metric] [lower] [upper] [n_trials]"}
RUN_DIR="/scratch-shared/ghonkoop/checkpoints/aml/hi-small/aml-HIS-2/"
CKPT_NAME=${2:-best_checkpoint.pth}
METRIC=${3:-val/illicit_f1}
LOWER=${4:-0.01}
UPPER=${5:-0.35}
N_TRIALS=${6:-40}

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


python /home/ghonkoop/codebase/summarizer/post_threshold_search.py \
  --run-dir "$RUN_DIR" \
  --config "/home/ghonkoop/codebase/summarizer/configs/aml/single/hi-small-2.yaml" \
  --ckpt-name "$CKPT_NAME" \
  --threshold-metric "$METRIC" \
  --lower "$LOWER" \
  --upper "$UPPER" \
  --n-trials "$N_TRIALS"
