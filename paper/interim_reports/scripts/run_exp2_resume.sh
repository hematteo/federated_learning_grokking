#!/bin/bash
# Resume Experiment 2 — only runs cells that are missing a result JSON.
# Runs MAX_PARALLEL jobs at a time. Reduce this if you hit OOM.
#
# Usage:
#   chmod +x run_exp2_resume.sh
#   tmux new -s exp2
#   ./run_exp2_resume.sh

set -e

ALPHAS=(0.2 0.25 0.3 0.35 0.5)
KS=(2 5 10 20 50 97)
T_MAX=50000
OUTPUT_DIR="results/exp2_aggregation"
NUM_GPUS=6
MAX_PARALLEL=3   # Conservative — high-K FL runs use lots of memory

trap 'echo "Interrupted — killing jobs..."; kill $(jobs -p) 2>/dev/null; wait; exit 1' INT TERM

source venv/bin/activate
mkdir -p logs

gpu=0
pids=()
total=0
skipped=0
failed=0

for K in "${KS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    # Skip cells that already have results
    result_file="${OUTPUT_DIR}/exp2_a${alpha}_K${K}.json"
    if [ -f "$result_file" ]; then
      echo "SKIP: alpha=$alpha K=$K (result exists)"
      skipped=$((skipped + 1))
      continue
    fi

    echo "Launching alpha=$alpha K=$K on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -u run_experiment.py exp2 \
      --alpha "$alpha" --K "$K" --t_max $T_MAX --output_dir $OUTPUT_DIR \
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
      echo "Batch done."
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

echo ""
echo "=== Summary ==="
echo "Skipped (already done): $skipped"
echo "Launched: $total"
echo "Failed: $failed"

if [ $failed -gt 0 ]; then
  echo "Check logs for details."
  exit 1
fi
