#!/bin/bash
set -euo pipefail

# === EDIT THESE ===
SPLIT_JSON="./data/splits/vel_norm.json"
CONFIG_DIR="./model_configs"   # directory with your pre-generated configs
RUN_BASE="trajexp"   # base name for run
NUM_FOLDS=5

# ==================
# Validations
# ==================
if [ ! -f "$SPLIT_JSON" ]; then
    echo "ERROR: Split file not found: $SPLIT_JSON"
    exit 1
fi

if [ ! -f "train.slurm" ]; then
    echo "ERROR: train.slurm not found in current directory"
    exit 1
fi

if [ ! -d "$CONFIG_DIR" ]; then
    echo "ERROR: Config directory not found: $CONFIG_DIR"
    exit 1
fi

echo "Starting job submission from configs in: $CONFIG_DIR"
echo "Using split file: $SPLIT_JSON"
echo "Number of folds: $NUM_FOLDS"

# === Dirs & manifest ===
mkdir -p logs/slurm logs/meta
MANIFEST="logs/manifest.csv"
if [[ ! -f "$MANIFEST" ]]; then
  echo "timestamp,job_id,run_name,fold,config_path,split_json,stdout_path,stderr_path,partition,account" > "$MANIFEST"
fi

# === Helpers ===
# Turn a path (relative to submit dir) into a safe name fragment
sanitize_name() {
  local p="$1"
  # rel path for readability
  local rel
  rel="$(python - <<'PY'
import os,sys
p=sys.argv[1]
root=os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())
print(os.path.relpath(p, root))
PY
"$p")"
  # strip extension; replace / space . - with _
  rel="${rel%.*}"
  rel="${rel//\//_}"
  rel="${rel// /_}"
  rel="${rel//./_}"
  rel="${rel//-/_}"
  printf "%s" "$rel"
}


# ==================
# Gather configs
# ==================
mapfile -t CONFIGS < <(find "$CONFIG_DIR" -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.json" \) | sort)

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "ERROR: No configs found in $CONFIG_DIR"
    exit 1
fi

TOTAL_JOBS=$(( ${#CONFIGS[@]} * NUM_FOLDS ))

echo "Total jobs to submit: $TOTAL_JOBS"
echo "Press Enter to continue or Ctrl+C to cancel..."
read

JOB_COUNT=0
for cfg in "${CONFIGS[@]}"; do
  cfg_name=$(basename "${cfg%.*}")  # filename without extension
  for FOLD in $(seq 0 $((NUM_FOLDS - 1))); do
    RUN_NAME="${RUN_BASE}_${cfg_name}_f${FOLD}"
    JOB_COUNT=$((JOB_COUNT + 1))
    echo "[$JOB_COUNT/$TOTAL_JOBS] Submitting ${RUN_NAME}"

    if sbatch train.slurm \
      "${SPLIT_JSON}" \
      "${FOLD}" \
      "${RUN_NAME}" \
      "${cfg}"; then
      echo "  ✓ Job submitted successfully"
    else
      echo "  ✗ Failed to submit job"
      exit 1
    fi
  done
done

echo ""
echo "✓ All $TOTAL_JOBS jobs submitted successfully!"
echo "Check job status with: squeue -u $USER"
