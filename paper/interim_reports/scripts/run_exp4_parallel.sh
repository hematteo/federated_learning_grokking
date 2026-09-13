#!/bin/bash
# Run Experiment 4 (a, b, c) in batches across GPUs.
#
# 4a: 3 alphas × 4 E × 2 het = 24 cells
# 4b: 3 alphas × 4 f × 2 het = 24 cells
# 4c: 3 alphas × 4 E          = 12 cells
# Total: 60 cells
#
# Usage:
#   chmod +x run_exp4_parallel.sh
#   ./run_exp4_parallel.sh

set -e

ALPHAS=(0.25 0.3 0.5)
E_VALUES=(5 10 25 50)
F_VALUES=(0.2 0.4 0.6 1.0)
HETS=(iid noniid)
E_VALUES_4C=(1 5 10 25)
T_MAX=50000
OUTPUT_DIR="results/exp4_optimization"
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

# Track which GPU each PID is using: pid_gpu[pid]=gpu_id
declare -A pid_gpu
# Track free GPUs as a simple list
free_gpus=()
for ((i=0; i<NUM_GPUS; i++)); do
  free_gpus+=($i)
done

wait_for_slot() {
  # Wait until a GPU slot is free
  while [ ${#free_gpus[@]} -eq 0 ]; do
    # wait -n: wait for ANY one child to finish (bash 4.3+)
    if wait -n; then
      : # success
    else
      failed=$((failed + 1))
    fi
    completed=$((completed + 1))

    # Find which PID(s) finished and reclaim their GPU
    local new_pids=()
    for pid in "${!pid_gpu[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        free_gpus+=(${pid_gpu[$pid]})
        unset pid_gpu[$pid]
      else
        new_pids+=($pid)
      fi
    done
    echo "  Slot freed. Running: ${#pid_gpu[@]}/$MAX_PARALLEL  Completed: $completed/$total"
  done
}

launch_and_track() {
  local cmd="$1"
  local logfile="$2"
  local result_file="$3"

  # Skip if result already exists
  if [ -f "$result_file" ]; then
    echo "  SKIP (result exists): $(basename $result_file)"
    skipped=$((skipped + 1))
    return
  fi

  # Wait for a free GPU slot
  wait_for_slot

  # Grab first free GPU
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

# === Exp 4a: Drift accumulation × heterogeneity ===
echo ""
echo "=========================================="
echo "  Exp 4a: Drift × heterogeneity"
echo "=========================================="
for alpha in "${ALPHAS[@]}"; do
  for E in "${E_VALUES[@]}"; do
    for het in "${HETS[@]}"; do
      launch_and_track \
        "python -u run_experiment.py exp4a --alpha $alpha --E $E --het $het --k 10 --t_max $T_MAX --output_dir $OUTPUT_DIR" \
        "logs/exp4a_a${alpha}_E${E}_${het}.log" \
        "${OUTPUT_DIR}/exp4a_a${alpha}_E${E}_${het}.json"
    done
  done
done
wait_remaining

# === Exp 4b: Partial participation × heterogeneity ===
echo ""
echo "=========================================="
echo "  Exp 4b: Participation × heterogeneity"
echo "=========================================="
for alpha in "${ALPHAS[@]}"; do
  for f in "${F_VALUES[@]}"; do
    for het in "${HETS[@]}"; do
      launch_and_track \
        "python -u run_experiment.py exp4b --alpha $alpha --frac $f --het $het --k 10 --t_max $T_MAX --output_dir $OUTPUT_DIR" \
        "logs/exp4b_a${alpha}_f${f}_${het}.log" \
        "${OUTPUT_DIR}/exp4b_a${alpha}_f${f}_${het}.json"
    done
  done
done
wait_remaining

# === Exp 4c: Compute vs communication ===
echo ""
echo "=========================================="
echo "  Exp 4c: Compute vs communication"
echo "=========================================="
for alpha in "${ALPHAS[@]}"; do
  for E in "${E_VALUES_4C[@]}"; do
    launch_and_track \
      "python -u run_experiment.py exp4c --alpha $alpha --E $E --k 10 --output_dir $OUTPUT_DIR" \
      "logs/exp4c_a${alpha}_E${E}.log" \
      "${OUTPUT_DIR}/exp4c_a${alpha}_E${E}.json"
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
