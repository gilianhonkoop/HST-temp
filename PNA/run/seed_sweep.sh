#!/bin/bash
#SBATCH --job-name=pna_seed
#SBATCH --output=/home/ghonkoop/repos/PNA/logs/seed/seed_%j.out
#SBATCH --error=/home/ghonkoop/repos/PNA/logs/seed/seed_%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --export=ALL
#SBATCH --mem=60G

set -euo pipefail

SCRIPT_DIR="/home/ghonkoop/repos/PNA"
LOG_DIR="${SCRIPT_DIR}/logs/seed"
mkdir -p "${LOG_DIR}"

CONFIG="${CONFIG:-configs/aml/LIM-2.yaml}"
# CONFIG="${CONFIG:-configs/saml-d/SAMLD.yaml}"

SEEDS=(${SEEDS:-42 43 44 45 46})
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
RESUME="${RESUME:-0}"

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

print_rule() {
  printf '%s\n' "================================================================================"
}

print_thin_rule() {
  printf '%s\n' "--------------------------------------------------------------------------------"
}

echo "======================================================"
echo "  PNA Seed sweep"
echo "======================================================"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  Config       : ${CONFIG}"
echo "  Resume       : ${RESUME}"
echo "  Python       : ${PYTHON_BIN}"
echo "  Started at   : $(date)"
echo "======================================================"
echo ""

METRICS_FILES=()
RUN_NAMES=()
FAILED=0
TOTAL_SEEDS="${#SEEDS[@]}"
SEED_INDEX=0

for SEED in "${SEEDS[@]}"; do
  SEED_INDEX=$((SEED_INDEX + 1))
  TMP_CONFIG="$(mktemp /tmp/pna_seed_${SEED}_XXXX.yaml)"
  SEED_INFO="$("${PYTHON_BIN}" - "${CONFIG}" "${TMP_CONFIG}" "${SEED}" <<'PY'
import sys
import os
import yaml

src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, "r") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("train", {})["seed"] = seed
logging_cfg = cfg.setdefault("logging", {})
base = logging_cfg.get("run_name", "PNA")
run_name = f"{base}_seed_{seed}"
logging_cfg["run_name"] = run_name
save_dir = str(logging_cfg.get("save_dir", "outputs"))
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"{run_name}|{os.path.join(save_dir, run_name)}")
PY
)"
  RUN_NAME="${SEED_INFO%%|*}"
  RUN_DIR="${SEED_INFO#*|}"
  METRICS_PATH="${RUN_DIR}/test_metrics.json"
  RUN_NAMES+=("${RUN_NAME}")
  METRICS_FILES+=("${METRICS_PATH}")

  print_rule
  printf '  Seed %s/%s\n' "${SEED_INDEX}" "${TOTAL_SEEDS}"
  print_rule
  printf '  Seed         : %s\n' "${SEED}"
  printf '  Run name     : %s\n' "${RUN_NAME}"
  printf '  Temp config  : %s\n' "${TMP_CONFIG}"
  printf '  Metrics file : %s\n' "${METRICS_PATH}"
  printf '  Started at   : %s\n' "$(date)"
  print_thin_rule
  echo ""

  if "${PYTHON_BIN}" train.py --config "${TMP_CONFIG}"; then
    echo ""
    print_thin_rule
    printf '  Seed %s completed at %s\n' "${SEED}" "$(date)"
    printf '  Metrics file: %s\n' "${METRICS_PATH}"
    print_rule
    echo ""
  else
    STATUS=$?
    FAILED=1
    echo ""
    print_thin_rule
    printf '  Seed %s FAILED with exit code %s at %s\n' "${SEED}" "${STATUS}" "$(date)"
    printf '  Metrics file, if any: %s\n' "${METRICS_PATH}"
    print_rule
    echo ""
  fi

  rm -f "${TMP_CONFIG}"
done

print_rule
echo "  PNA Seed Sweep Test Summary"
print_rule
"${PYTHON_BIN}" - "${SEEDS[@]}" -- "${RUN_NAMES[@]}" -- "${METRICS_FILES[@]}" <<'PY'
import json
import math
import os
import statistics
import sys

args = sys.argv[1:]
first_sep = args.index("--")
second_sep = args.index("--", first_sep + 1)
seeds = args[:first_sep]
run_names = args[first_sep + 1:second_sep]
metric_paths = args[second_sep + 1:]

rows = []
for seed, run_name, path in zip(seeds, run_names, metric_paths):
    row = {"seed": seed, "run_name": run_name, "metrics_path": path, "status": "missing"}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                metrics = json.load(f)
            row.update(metrics)
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = f"error: {exc}"
    rows.append(row)

def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.6f}"
    return str(value)

print("")
print("Per-seed test metrics")
print("-" * 120)
print(
    f"{'seed':>6}  {'status':<10}  {'run_name':<28}  "
    f"{'loss':>10}  {'il_prec':>10}  {'il_rec':>10}  {'il_f1':>10}  "
    f"{'il_pr_auc':>10}  {'il_roc_auc':>11}  {'lic_f1':>10}"
)
print("-" * 120)
for row in rows:
    print(
        f"{row['seed']:>6}  {row['status']:<10.10}  {row['run_name']:<28.28}  "
        f"{fmt(row.get('test/loss')):>10}  "
        f"{fmt(row.get('test/illicit_precision')):>10}  "
        f"{fmt(row.get('test/illicit_recall')):>10}  "
        f"{fmt(row.get('test/illicit_f1')):>10}  "
        f"{fmt(row.get('test/illicit_pr_auc')):>10}  "
        f"{fmt(row.get('test/illicit_roc_auc')):>11}  "
        f"{fmt(row.get('test/licit_f1')):>10}"
    )
print("-" * 120)

ok_rows = [row for row in rows if row.get("status") == "ok"]
metric_keys = sorted(
    {
        key
        for row in ok_rows
        for key, value in row.items()
        if key.startswith("test/") and isinstance(value, (int, float)) and math.isfinite(float(value))
    }
)

print("")
print("Mean +/- std over completed seeds")
print("-" * 72)
print(f"{'metric':<34}  {'mean':>12}  {'std':>12}  {'n':>4}")
print("-" * 72)
for key in metric_keys:
    values = [float(row[key]) for row in ok_rows if isinstance(row.get(key), (int, float))]
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f"{key:<34}  {mean:>12.6f}  {std:>12.6f}  {len(values):>4}")
print("-" * 72)

missing = [row for row in rows if row.get("status") != "ok"]
if missing:
    print("")
    print("Missing or failed seed outputs")
    print("-" * 72)
    for row in missing:
        print(f"seed={row['seed']} status={row['status']} metrics={row['metrics_path']}")
    print("-" * 72)

print("")
PY

echo ""
echo "======================================================"
echo "  Finished at  : $(date)"
echo "======================================================"

if [[ "${FAILED}" != "0" ]]; then
  exit 1
fi
