#!/bin/bash
#SBATCH --job-name=saml_d_build_subgraphs
#SBATCH --output=/home/ghonkoop/codebase/builder/logs/build_saml_d_subgraphs_%j.out
#SBATCH --error=/home/ghonkoop/codebase/builder/logs/build_saml_d_subgraphs_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --export=ALL

# ---------------------------------------------------------------------------
# Optional environment overrides:
#   INPUT_CSV           default: /home/ghonkoop/data/saml-d/SAML-D.csv
#   OUTPUT_DIR          default: /home/ghonkoop/data/saml-d/usable
#   CHUNKSIZE           default: 1000000
#   DROP_SINGLETONS     (0|1)  default: 1
#   NO_NORMALIZE        (0|1)  default: 0
#   NO_LICIT_SUBGRAPHS  (0|1)  default: 0
#   LICIT_RATIO         float  default: 25.0
#   LICIT_SEED          int    default: 42
#   LICIT_BFS_PROB      float  default: 0.5
#   MAX_COMPONENT_SIZE  int    default: 3500
#   DATASET_NAME        string default: SAML-D
#   NODE_FEATURE_MODE   (none|constant|activity) default: none
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="/home/ghonkoop/codebase/builder"
LOG_DIR="${SCRIPT_DIR}/logs"

INPUT_CSV="${INPUT_CSV:-/home/ghonkoop/data/saml-d/raw/SAML-D.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ghonkoop/data/saml-d/usable}"
CHUNKSIZE="${CHUNKSIZE:-1000000}"
DROP_SINGLETONS="${DROP_SINGLETONS:-1}"
NO_NORMALIZE="${NO_NORMALIZE:-0}"
NO_LICIT_SUBGRAPHS="${NO_LICIT_SUBGRAPHS:-0}"
LICIT_RATIO="${LICIT_RATIO:-25.0}"
LICIT_SEED="${LICIT_SEED:-42}"
LICIT_BFS_PROB="${LICIT_BFS_PROB:-0.5}"
MAX_COMPONENT_SIZE="${MAX_COMPONENT_SIZE:-500}"
DATASET_NAME="${DATASET_NAME:-SAML-D}"
NODE_FEATURE_MODE="${NODE_FEATURE_MODE:-none}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

DROP_SINGLETONS_FLAG=""
if [[ "${DROP_SINGLETONS}" == "1" || "${DROP_SINGLETONS,,}" == "true" ]]; then
    DROP_SINGLETONS_FLAG="--drop-singletons"
fi

NO_NORMALIZE_FLAG=""
if [[ "${NO_NORMALIZE}" == "1" || "${NO_NORMALIZE,,}" == "true" ]]; then
    NO_NORMALIZE_FLAG="--no-normalize"
fi

NO_LICIT_SUBGRAPHS_FLAG=""
if [[ "${NO_LICIT_SUBGRAPHS}" == "1" || "${NO_LICIT_SUBGRAPHS,,}" == "true" ]]; then
    NO_LICIT_SUBGRAPHS_FLAG="--no-licit-subgraphs"
fi

echo "======================================================"
echo "  SAML-D AML-style subgraph builder"
echo "======================================================"
echo "  Job ID         : ${SLURM_JOB_ID:-local}"
echo "  Input CSV      : ${INPUT_CSV}"
echo "  Output dir     : ${OUTPUT_DIR}"
echo "  Dataset name   : ${DATASET_NAME}"
echo "  Chunksize      : ${CHUNKSIZE}"
echo "  Drop singletons: ${DROP_SINGLETONS}"
echo "  No normalize   : ${NO_NORMALIZE}"
echo "  No licit stage : ${NO_LICIT_SUBGRAPHS}"
echo "  Licit ratio    : ${LICIT_RATIO}"
echo "  Licit seed     : ${LICIT_SEED}"
echo "  Licit BFS prob : ${LICIT_BFS_PROB}"
echo "  Max comp size  : ${MAX_COMPONENT_SIZE}"
echo "  Node features  : ${NODE_FEATURE_MODE}"
echo "  Started at     : $(date)"
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

python3.11 "${SCRIPT_DIR}/build_saml_d_subgraphs.py" \
    --input-csv "${INPUT_CSV}" \
    --output-dir "${OUTPUT_DIR}" \
    --dataset-name "${DATASET_NAME}" \
    --node-feature-mode "${NODE_FEATURE_MODE}" \
    --chunksize "${CHUNKSIZE}" \
    ${DROP_SINGLETONS_FLAG} \
    ${NO_NORMALIZE_FLAG} \
    ${NO_LICIT_SUBGRAPHS_FLAG} \
    --licit-ratio "${LICIT_RATIO}" \
    --licit-seed "${LICIT_SEED}" \
    --licit-bfs-prob "${LICIT_BFS_PROB}" \
    --max-component-size "${MAX_COMPONENT_SIZE}"

echo ""
echo "======================================================"
echo "  Finished at : $(date)"
echo "  Output      : ${OUTPUT_DIR}"
echo "======================================================"
