#!/bin/bash
# Run Experiment 5 (algorithm rescue) across GPUs.
#
# 3 settings × 11 algorithms = 33 cells
# Each cell = 3 seeds × 1 FL run
#
# Usage:
#   chmod +x run_exp5_parallel.sh
#   ./run_exp5_parallel.sh

set -e

SETTINGS=(H1 H2 H3)
ALGORITHMS=(
  "FedAvg"
  "FedProx-0.001"
  "FedProx-0.01"
  "FedProx-0.1"
  "FedProx-1.0"
  "FedAdam-0.01"
  "FedAdam-0.1"
  "FedAdam-1.0"
  "FedAvg+WD-0.01"
  "FedAvg+WD-0.1"
  "FedAvg+WD-1.0"
)
T_MAX=50000
OUTPUT_DIR="results/exp5_algorithms"
NUM_GPUS=6
MAX_PARALLEL=6

# Kill all children on Ctrl+C
trap 'echo "Interrupted — killing jobs..."; kill $(jobs -p) 2>/dev/null; wait; exit 1' INT TERM

source venv/bin/activate
mkdir -p logs

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
      :
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

launch_and_track() {
  local cmd="$1"
  local logfile="$2"
  local result_file="$3"

  if [ -f "$result_file" ]; then
    echo "  SKIP (result exists): $(basename $result_file)"
    skipped=$((skipped + 1))
    return
  fi

  wait_for_slot

  local gpu=${free_gpus[0]}
  free_gpus=("${free_gpus[@]:1}")

  echo "  GPU $gpu: $(basename $logfile)"
  CUDA_VISIBLE_DEVICES=$gpu $cmd > "$logfile" 2>&1 &
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

# === Launch all (setting, algorithm) cells ===
echo ""
echo "=========================================="
echo "  Exp 5: Algorithm Rescue"
echo "=========================================="

for setting in "${SETTINGS[@]}"; do
  echo ""
  echo "--- Setting: $setting ---"
  for algo in "${ALGORITHMS[@]}"; do
    launch_and_track \
      "python -u run_experiment.py exp5 --setting $setting --algorithm $algo --t_max $T_MAX --output_dir $OUTPUT_DIR" \
      "logs/exp5_${setting}_${algo}.log" \
      "${OUTPUT_DIR}/exp5_${setting}_${algo}.json"
  done
done
wait_remaining

# === Summary ===
echo ""
echo "=========================================="
echo "  All done!"
echo "  Launched: $total  Skipped: $skipped  Failed: $failed"
echo "=========================================="

if [ $failed -gt 0 ]; then
  echo "Check logs/ for failed runs."
  exit 1
fi
