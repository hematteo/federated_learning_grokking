#!/bin/bash
# Run Experiment 2 in batches across 6 GPUs.
# Runs MAX_PARALLEL jobs at a time, waiting for each batch to finish
# before launching the next.
#
# Usage:
#   chmod +x run_exp2_parallel.sh
#   ./run_exp2_parallel.sh

set -e

ALPHAS=(0.20 0.25 0.30 0.35 0.50)
KS=(2 5 10 20 50 97)
T_MAX=50000
OUTPUT_DIR="results/exp2_aggregation"
NUM_GPUS=5
MAX_PARALLEL=5

# Kill all children on Ctrl+C
trap 'echo "Interrupted — killing jobs..."; kill $(jobs -p) 2>/dev/null; wait; exit 1' INT TERM

source venv/bin/activate
mkdir -p logs

gpu=0
pids=()
total=0
failed=0

for K in "${KS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    echo "Launching alpha=$alpha K=$K on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -u run_experiment.py exp2 \
      --alpha $alpha --K $K --t_max $T_MAX --output_dir $OUTPUT_DIR \
      > "logs/exp2_a${alpha}_K${K}.log" 2>&1 &
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
      echo "Batch done. ($((total)) / 30 launched so far)"
      pids=()
    fi
  done
done

# Wait for any remaining jobs
if [ ${#pids[@]} -gt 0 ]; then
  echo "Final batch of ${#pids[@]} running — waiting..."
  for pid in "${pids[@]}"; do
    if ! wait $pid; then
      failed=$((failed + 1))
    fi
  done
fi

if [ $failed -eq 0 ]; then
  echo "All $total jobs completed successfully."
else
  echo "$failed/$total jobs failed. Check logs for details."
  exit 1
fi
