#!/bin/bash
#SBATCH --job-name=s2n_aml_gcn
#SBATCH --output=/home/ghonkoop/repos/S2N/logs/sweep/aml_sweep_%j.out
#SBATCH --error=/home/ghonkoop/repos/S2N/logs/sweep/aml_sweep_%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=100G

set -euo pipefail

DATASET_CONFIG="${1:-}"

case "${DATASET_CONFIG}" in
  aml_hi_small)
    DATASET_LABEL="AML_HI_SMALL"
    WANDB_DATASET_SHORT="HIS"
    DEFAULT_DATAMODULE_RUN_CONFIG="for-gcn"
    DEFAULT_MODEL_CONFIG="gatconv/s2n/for-amlhi"
    DEFAULT_WANDB_MODEL_SHORT="GATCONV"
    DEFAULT_EXTRA_OVERRIDES="datamodule.s2n_original_edge_attr_mode=append model.layer_kwargs.edge_attr_mode=full datamodule.s2n_original_edge_attr_normalize=zscore"
    ;;
  aml_li_small)
    DATASET_LABEL="AML_LI_SMALL"
    WANDB_DATASET_SHORT="LIS"
    DEFAULT_DATAMODULE_RUN_CONFIG="for-gcn"
    DEFAULT_MODEL_CONFIG="gatconv/s2n/for-amlhi"
    DEFAULT_WANDB_MODEL_SHORT="GATCONV"
    DEFAULT_EXTRA_OVERRIDES="datamodule.s2n_original_edge_attr_mode=append model.layer_kwargs.edge_attr_mode=full datamodule.s2n_original_edge_attr_normalize=zscore"
    ;;
  aml_hi_medium)
    DATASET_LABEL="AML_HI_MEDIUM"
    WANDB_DATASET_SHORT="HIM"
    DEFAULT_DATAMODULE_RUN_CONFIG="for-gcn"
    DEFAULT_MODEL_CONFIG="gatconv/s2n/for-amlhi"
    DEFAULT_WANDB_MODEL_SHORT="GATCONV"
    DEFAULT_EXTRA_OVERRIDES="datamodule.s2n_original_edge_attr_mode=append model.layer_kwargs.edge_attr_mode=full datamodule.s2n_original_edge_attr_normalize=zscore"
    ;;
  aml_li_medium)
    DATASET_LABEL="AML_LI_MEDIUM"
    WANDB_DATASET_SHORT="LIM"
    DEFAULT_DATAMODULE_RUN_CONFIG="for-gcn"
    DEFAULT_MODEL_CONFIG="gatconv/s2n/for-amlhi"
    DEFAULT_WANDB_MODEL_SHORT="GATCONV"
    DEFAULT_EXTRA_OVERRIDES="datamodule.s2n_original_edge_attr_mode=append model.layer_kwargs.edge_attr_mode=full datamodule.s2n_original_edge_attr_normalize=zscore"
    ;;
  samld)
    DATASET_LABEL="SAMLD"
    WANDB_DATASET_SHORT="SAMLD"
    DEFAULT_DATAMODULE_RUN_CONFIG="for-gcn2"
    DEFAULT_MODEL_CONFIG="gcn2/s2n/for-samld"
    DEFAULT_WANDB_MODEL_SHORT="GCN2"
    DEFAULT_EXTRA_OVERRIDES=""
    ;;
  *)
    echo "Usage: sbatch $0 {aml_hi_small|aml_li_small|aml_hi_medium|aml_li_medium|samld}" >&2
    exit 2
    ;;
esac

WANDB_ENTITY="${WANDB_ENTITY:-hierarchical-subgraph-transformer}"
WANDB_PROJECT="${WANDB_PROJECT:-S2N}"

# MODEL_CONFIG="${MODEL_CONFIG:-gcn/s2n/for-amlhi}"
# WANDB_MODEL_SHORT="${WANDB_MODEL_SHORT:-GCN2}"

# MODEL_CONFIG="${MODEL_CONFIG:-gat/s2n/for-amlhi}"
# WANDB_MODEL_SHORT="${WANDB_MODEL_SHORT:-GATv2}"

MODEL_CONFIG="${MODEL_CONFIG:-${DEFAULT_MODEL_CONFIG}}"
WANDB_MODEL_SHORT="${WANDB_MODEL_SHORT:-${DEFAULT_WANDB_MODEL_SHORT}}"

DATAMODULE_RUN_CONFIG="${DATAMODULE_RUN_CONFIG:-${DEFAULT_DATAMODULE_RUN_CONFIG}}"
SEEDS=(${SEEDS:-45 46})

# EXTRA_OVERRIDES=(${EXTRA_OVERRIDES:-})
# EXTRA_OVERRIDES=(${EXTRA_OVERRIDES:-datamodule.s2n_original_edge_attr_mode=append model.layer_kwargs.edge_attr_mode=weight datamodule.s2n_original_edge_attr_normalize=zscore})
# EXTRA_OVERRIDES=(${EXTRA_OVERRIDES:-datamodule.s2n_original_edge_attr_mode=append model.layer_kwargs.edge_attr_mode=features datamodule.s2n_original_edge_attr_normalize=zscore})
EXTRA_OVERRIDES=(${EXTRA_OVERRIDES:-${DEFAULT_EXTRA_OVERRIDES}})

SCRIPT_DIR="/home/ghonkoop/repos/S2N/S2N"
LOG_DIR="/home/ghonkoop/repos/S2N/logs"
mkdir -p "${LOG_DIR}"

echo "======================================================"
echo "  S2N seed run"
echo "======================================================"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  Dataset      : ${DATASET_LABEL}"
echo "  Model config : ${MODEL_CONFIG}"
echo "  Seeds        : ${SEEDS[*]}"
echo "  Extra args   : ${EXTRA_OVERRIDES[*]:-(none)}"
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

SEED_START="${SEEDS[0]}"
NUM_AVERAGING="${#SEEDS[@]}"
WANDB_NAME_TEMPLATE="${WANDB_DATASET_SHORT}-${WANDB_MODEL_SHORT}__seed__"

for IDX in "${!SEEDS[@]}"; do
    EXPECTED_SEED="$((SEED_START + IDX))"
    if [ "${SEEDS[$IDX]}" != "${EXPECTED_SEED}" ]; then
        echo "Error: run_main.py averages consecutive seeds only. Got SEEDS=${SEEDS[*]}." >&2
        exit 2
    fi
done

echo ""
echo "------------------------------------------------------"
echo "  Running ${DATASET_LABEL} seed sweep"
echo "  W&B name template: ${WANDB_NAME_TEMPLATE}"
echo "------------------------------------------------------"

"${HOME}/.conda/envs/s2n_env/bin/python" run_main.py \
    +trainer.num_nodes=1 \
    trainer.devices="[0]" \
    datamodule="s2n/${DATASET_CONFIG}/${DATAMODULE_RUN_CONFIG}" \
    model="${MODEL_CONFIG}" \
    experiment=s2n_aml \
    seed="${SEED_START}" \
    num_averaging="${NUM_AVERAGING}" \
    logger.wandb.entity="${WANDB_ENTITY}" \
    logger.wandb.project="${WANDB_PROJECT}" \
    logger.wandb.name="${WANDB_NAME_TEMPLATE}" \
    "${EXTRA_OVERRIDES[@]}"

echo ""
echo "======================================================"
echo "  Finished at  : $(date)"
echo "======================================================"
