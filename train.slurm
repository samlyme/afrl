#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=tesla
#SBATCH --gres=gpu:1

# Usage:
# sbatch train.slurm <SPLIT_JSON> <FOLD> <RUN_NAME> <CONFIG_PATH>
# Optional env:
#   PYTHON=python
#   EXTRA_ARGS="--seed 42"
#   CONDA_ENV=yourenv
#   VENV_PATH=/path/to/venv

set -euo pipefail

# -----------------------------
# Parse and validate arguments
# -----------------------------
if [[ $# -ne 4 ]]; then
  echo "ERROR: expected 4 arguments: <SPLIT_JSON> <FOLD> <RUN_NAME> <CONFIG_PATH>" >&2
  exit 2
fi

SPLIT_JSON="$1"
FOLD="$2"
RUN_NAME="$3"
CONFIG_PATH="$4"

[[ -f "$SPLIT_JSON" ]] || { echo "ERROR: split not found: $SPLIT_JSON" >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "ERROR: config not found: $CONFIG_PATH" >&2; exit 1; }

# Make squeue/logs show the run name
export SLURM_JOB_NAME="$RUN_NAME"

# -----------------------------
# Environment setup (optional)
# -----------------------------
PYTHON="${PYTHON:-python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" || true
  conda activate "$CONDA_ENV" || echo "WARN: failed to activate conda env '$CONDA_ENV'"
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate" || echo "WARN: failed to activate venv '$VENV_PATH'"
fi

mkdir -p logs

echo "========== SLURM =========="
echo "Job ID:        ${SLURM_JOB_ID:-N/A}"
echo "Job Name:      $RUN_NAME"
echo "Node:          ${SLURMD_NODENAME:-N/A}"
echo "Gres:          ${SLURM_JOB_GRES:-N/A}"
echo "CPUs/Task:     ${SLURM_CPUS_PER_TASK:-N/A}"
echo "==========================="
echo "Split JSON:    $SPLIT_JSON"
echo "Fold:          $FOLD"
echo "Config:        $CONFIG_PATH"
echo "Python:        $PYTHON"
echo "Extra args:    $EXTRA_ARGS"
echo "Launch dir:    ${SLURM_SUBMIT_DIR:-$PWD}"
echo "==========================="

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# -----------------------------
# Per-run app logs (created at run time)
# -----------------------------
RUN_DIR="logs/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"
APP_OUT="$RUN_DIR/app.stdout.log"
APP_ERR="$RUN_DIR/app.stderr.log"

# Ensure immediate flushing from Python and line-buffered tee
export PYTHONUNBUFFERED=1

# Duplicate stdout/stderr: keep SLURM logs AND write app logs
# (If stdbuf isn't available on your cluster, remove it; you'll just have coarser flush.)
exec > >(stdbuf -oL tee -a "$APP_OUT")
exec 2> >(stdbuf -oL tee -a "$APP_ERR" >&2)

# -----------------------------
# Run
# -----------------------------
set -x
$PYTHON -u -m src.main "$RUN_NAME" -s "$SPLIT_JSON" -f "$FOLD" -c "$CONFIG_PATH" $EXTRA_ARGS
set +x

echo "App logs:"
echo "  stdout -> $APP_OUT"
echo "  stderr -> $APP_ERR"
echo "SLURM logs:"
echo "  out -> logs/${RUN_NAME}.${SLURM_JOB_ID:-<jid>}.out"
echo "  err -> logs/${RUN_NAME}.${SLURM_JOB_ID:-<jid>}.err"
echo "Done: $RUN_NAME"
