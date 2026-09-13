#!/bin/bash
# Run Experiment 7 (task generalization) in parallel across GPUs.
#
# Tier 1: 4 tasks × 2 primes × 2 alphas × 2 modes = 32 cells (50k budget)
# Tier 2: 4 tasks × 3 primes × 2 alphas × 2 modes = 48 cells (100k budget)
# Skips existing runs from exp1/exp2 and already-completed exp7 cells.
#
# Usage:
#   chmod +x run_exp7_parallel.sh
#   ./run_exp7_parallel.sh

set -e

TASKS=(addition multiplication division x2_plus_y2)
MODES=(cent fl)

OUTPUT_DIR="results/exp7_tasks"
NUM_GPUS=8
MAX_PARALLEL=8

# Kill all children on Ctrl+C
trap 'echo "Interrupted — killing jobs..."; kill $(jobs -p) 2>/dev/null; wait; exit 1' INT TERM

source venv/bin/activate
mkdir -p logs "${OUTPUT_DIR}/centralized" "${OUTPUT_DIR}/fl_iid"

total=0
failed=0
skipped=0
completed=0

# Track which GPU each PID is using
declare -A pid_gpu
free_gpus=()
for ((i=0; i<NUM_GPUS; i++)); do
  free_gpus+=($i)
done

wait_for_slot() {
  while [ ${#free_gpus[@]} -eq 0 ]; do
    if wait -n; then
      : # success
    else
      failed=$((failed + 1))
    fi
    completed=$((completed + 1))

    for pid in "${!pid_gpu[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        free_gpus+=(${pid_gpu[$pid]})
        unset pid_gpu[$pid]
      fi
    done
    echo "  Slot freed. Running: ${#pid_gpu[@]}/$MAX_PARALLEL  Completed: $completed/$total"
  done
}

launch() {
  local task=$1
  local p=$2
  local alpha=$3
  local mode=$4
  local budget=$5

  local logfile="logs/exp7_${task}_p${p}_a${alpha}_${mode}.log"

  # Skip check: centralized
  if [ "$mode" == "cent" ]; then
    local exp1_file="results/exp1_boundary/history_${task}_gd_p${p}_N256_a${alpha}_s42.json"
    local exp7_file="${OUTPUT_DIR}/centralized/history_${task}_gd_p${p}_N256_a${alpha}_s42.json"
    if [ -f "$exp1_file" ] || [ -f "$exp7_file" ]; then
      echo "  SKIP cent ${task} p=${p} α=${alpha} (exists)"
      skipped=$((skipped + 1))
      return
    fi
  fi

  # Skip check: FL
  if [ "$mode" == "fl" ]; then
    local tag="fed_${task}_gd_p${p}_N256_a${alpha}_K10_le5_ft1.0_iid_s42"
    local exp2_file="results/exp2_aggregation/fl_iid/history_${tag}.json"
    local exp7_file="${OUTPUT_DIR}/fl_iid/history_${tag}.json"
    if [ -f "$exp2_file" ] || [ -f "$exp7_file" ]; then
      echo "  SKIP fl   ${task} p=${p} α=${alpha} (exists)"
      skipped=$((skipped + 1))
      return
    fi
  fi

  wait_for_slot

  local gpu=${free_gpus[0]}
  free_gpus=("${free_gpus[@]:1}")

  echo "  GPU $gpu: ${task} p=${p} α=${alpha} ${mode} (budget=${budget})"
  CUDA_VISIBLE_DEVICES=$gpu python -u -m experiments.exp7_tasks \
    --task "$task" --p "$p" --alpha "$alpha" --mode "$mode" --budget "$budget" \
    > "$logfile" 2>&1 &
  local pid=$!
  pid_gpu[$pid]=$gpu
  total=$((total + 1))
}

wait_remaining() {
  if [ ${#pid_gpu[@]} -gt 0 ]; then
    echo "Waiting for ${#pid_gpu[@]} remaining jobs..."
    for pid in "${!pid_gpu[@]}"; do
      if ! wait $pid; then
        failed=$((failed + 1))
      fi
      completed=$((completed + 1))
    done
    pid_gpu=()
    free_gpus=()
    for ((i=0; i<NUM_GPUS; i++)); do
      free_gpus+=($i)
    done
  fi
}

# === Tier 1: original sweep (near phase boundary) ===
echo ""
echo "=========================================="
echo "  Tier 1: α ∈ {0.25, 0.3}, p ∈ {53, 97}, budget=50k"
echo "=========================================="

for task in "${TASKS[@]}"; do
  for p in 53 97; do
    for alpha in 0.25 0.3; do
      for mode in "${MODES[@]}"; do
        launch "$task" "$p" "$alpha" "$mode" 50000
      done
    done
  done
done

wait_remaining

# === Tier 2: easier settings (more data, bigger budget, larger prime) ===
echo ""
echo "=========================================="
echo "  Tier 2: α ∈ {0.4, 0.5}, p ∈ {53, 97, 113}, budget=100k"
echo "=========================================="

for task in "${TASKS[@]}"; do
  for p in 53 97 113; do
    for alpha in 0.4 0.5; do
      for mode in "${MODES[@]}"; do
        launch "$task" "$p" "$alpha" "$mode" 100000
      done
    done
  done
done

wait_remaining

# === Summary ===
echo ""
echo "=========================================="
echo "  Exp 7 done!"
echo "  Launched: $total  Skipped: $skipped  Failed: $failed"
echo "=========================================="

if [ $failed -gt 0 ]; then
  echo "Check logs/exp7_*.log for failed runs."
  exit 1
fi
