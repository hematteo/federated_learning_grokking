#!/bin/bash
# Run Experiment 3 (a, a_kval, b) in batches across GPUs.
#
# 3a:     5 alphas × 6 dir_alphas = 30 cells at K=10
# 3a_kval: 3 alphas × 3 dir_alphas =  9 cells at K=20
# 3b:     5 alphas × 2 partitions  = 10 cells at K=10 (skip IID, reuse from 3a dir=1000)
# Total: 49 cells
#
# Usage:
#   chmod +x run_exp3_parallel.sh
#   ./run_exp3_parallel.sh

set -e

ALPHAS=(0.2 0.25 0.3 0.35 0.5)
DIR_ALPHAS=(0.01 0.1 0.5 1.0 10.0 1000.0)
KVAL_ALPHAS=(0.25 0.3 0.5)
KVAL_DIR_ALPHAS=(0.1 1.0 1000.0)
PARTITIONS=(operand target)  # skip IID — reuse 3a dir=1000
K_PRIMARY=10
K_SECONDARY=20
T_MAX=50000
OUTPUT_DIR="results/exp3_heterogeneity"
NUM_GPUS=6
MAX_PARALLEL=6  # Keep low to avoid OOM — each Flower/Ray process is memory-heavy

# Kill all children on Ctrl+C
trap 'echo "Interrupted — killing jobs..."; kill $(jobs -p) 2>/dev/null; wait; exit 1' INT TERM

source venv/bin/activate
mkdir -p logs

gpu=0
pids=()
total=0
failed=0
skipped=0

launch_and_track() {
  local cmd="$1"
  local logfile="$2"

  # Skip if result already exists
  if [ -f "$3" ]; then
    echo "  SKIP (result exists): $logfile"
    skipped=$((skipped + 1))
    return
  fi

  echo "  GPU $gpu: $logfile"
  CUDA_VISIBLE_DEVICES=$gpu $cmd > "$logfile" 2>&1 &
  pids+=($!)
  gpu=$(( (gpu + 1) % NUM_GPUS ))
  total=$((total + 1))

  # When batch is full, wait for it to finish
  if [ ${#pids[@]} -ge $MAX_PARALLEL ]; then
    echo "Batch of ${#pids[@]} running — waiting..."
    for pid in "${pids[@]}"; do
      if ! wait $pid; then
        failed=$((failed + 1))
      fi
    done
    echo "Batch done. ($total launched so far)"
    pids=()
  fi
}

wait_remaining() {
  if [ ${#pids[@]} -gt 0 ]; then
    echo "Final batch of ${#pids[@]} running — waiting..."
    for pid in "${pids[@]}"; do
      if ! wait $pid; then
        failed=$((failed + 1))
      fi
    done
    pids=()
  fi
}

# === Exp 3a: Dirichlet sweep at K_primary ===
echo ""
echo "=========================================="
echo "  Exp 3a: Dirichlet sweep (K=$K_PRIMARY)"
echo "=========================================="
for alpha in "${ALPHAS[@]}"; do
  for dir_alpha in "${DIR_ALPHAS[@]}"; do
    result_file="${OUTPUT_DIR}/exp3a_a${alpha}_dir${dir_alpha}_K${K_PRIMARY}.json"
    launch_and_track \
      "python -u run_experiment.py exp3a --alpha $alpha --dir_alpha $dir_alpha --k $K_PRIMARY --t_max $T_MAX --output_dir $OUTPUT_DIR" \
      "logs/exp3a_a${alpha}_dir${dir_alpha}.log" \
      "$result_file"
  done
done
wait_remaining

# === Exp 3a K-validation at K_secondary ===
echo ""
echo "=========================================="
echo "  Exp 3a K-val: (K=$K_SECONDARY)"
echo "=========================================="
for alpha in "${KVAL_ALPHAS[@]}"; do
  for dir_alpha in "${KVAL_DIR_ALPHAS[@]}"; do
    result_file="${OUTPUT_DIR}/exp3a_a${alpha}_dir${dir_alpha}_K${K_SECONDARY}.json"
    launch_and_track \
      "python -u run_experiment.py exp3a_kval --alpha $alpha --dir_alpha $dir_alpha --k $K_SECONDARY --t_max $T_MAX --output_dir $OUTPUT_DIR" \
      "logs/exp3a_kval_a${alpha}_dir${dir_alpha}.log" \
      "$result_file"
  done
done
wait_remaining

# === Exp 3b: Structured partitions (operand, target) at K_primary ===
# IID is skipped — equivalent to 3a with dir_alpha=1000
echo ""
echo "=========================================="
echo "  Exp 3b: Structured partitions (K=$K_PRIMARY)"
echo "=========================================="
for alpha in "${ALPHAS[@]}"; do
  for partition in "${PARTITIONS[@]}"; do
    result_file="${OUTPUT_DIR}/exp3b_a${alpha}_${partition}_K${K_PRIMARY}.json"
    launch_and_track \
      "python -u run_experiment.py exp3b --alpha $alpha --partition $partition --k $K_PRIMARY --t_max $T_MAX --output_dir $OUTPUT_DIR" \
      "logs/exp3b_a${alpha}_${partition}.log" \
      "$result_file"
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
