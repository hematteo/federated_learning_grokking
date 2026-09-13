#!/bin/bash
# Re-run FedAdam and WD cells to get distinct history files
set -e
source venv/bin/activate

ALGOS=("FedAdam-0.01" "FedAdam-0.1" "FedAdam-1.0" "FedAvg+WD-0.01" "FedAvg+WD-0.1" "FedAvg+WD-1.0")
SETTINGS=("H1" "H2" "H3")
T_MAX=50000
OUTPUT_DIR="results/exp5_algorithms"

gpu=0
pids=()

for setting in "${SETTINGS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    echo "GPU $gpu: $setting $algo"
    CUDA_VISIBLE_DEVICES=$gpu python -u run_experiment.py exp5 --setting $setting --algorithm "$algo" --t_max $T_MAX --output_dir $OUTPUT_DIR > "logs/exp5_rerun_${setting}_${algo}.log" 2>&1 &
    pids+=($!)
    gpu=$(( (gpu + 1) % 6 ))  # use GPUs 0-5
    
    # Wait if we have 6 running
    if [ ${#pids[@]} -ge 6 ]; then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
    fi
  done
done

wait
echo "All done!"
