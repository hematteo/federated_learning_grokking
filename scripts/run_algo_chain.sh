#!/usr/bin/env bash
# The 2026-09-09 chain on cam-gpu-acs (gxp-l4-0): Phase 1 of the algorithm plan,
# the B wd=0 control at alpha=0.70, the anchor's H2 mechanism block, then the
# droppable tail (C's calibration, B's decay-band extension, exp2's second
# alpha). Sequential so each manifest gets the whole slot pool; every launch is
# idempotent, so re-running this script resumes wherever it stopped.
#
#   setsid nohup bash scripts/run_algo_chain.sh > logs/sweeps/algo_chain_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#
# GPUs 1 and 2 at four runs each -- the pool the 2-4 Sep sweep used. Every cell
# here is K <= 20, where four concurrent runs cost nothing per run (PROGRESS,
# "Concurrency on one card").
set -u
cd "$(dirname "$0")/.."
GPUS="${GPUS:-1,2}"
PER_GPU="${PER_GPU:-4}"
PY=venv/bin/python
mkdir -p logs/sweeps

run_manifest () {
  local m=$1
  echo "=== $(date -Is) launching $m"
  $PY -u scripts/launch_sweep.py "manifests/$m.jsonl" --gpus "$GPUS" --per-gpu "$PER_GPU" \
      > "logs/sweeps/${m}_$(date +%Y%m%d_%H%M%S).log" 2>&1
  echo "=== $(date -Is) $m exited rc=$?"
  $PY scripts/collect_runs.py > /dev/null 2>&1 || true
}

run_manifest t6_algo_calibration      # Phase 1: A, B, D, E  (~110 slot-h)
run_manifest x_b_wd_zero_fl           # decay-clock control at alpha=0.70 (~13)
run_manifest x_h2_mechanism           # Phase 4: checkpointed arms + damped FedAvg (~55)
run_manifest x_scaffold_rerun         # H1/H3 SCAFFOLD with server-side c_i (~7)
run_manifest t6_algo_calibration_c    # Phase 1: C, withheld until reproducible (~140)
run_manifest x_b_decay_band_long      # B's censored decay rungs (~31)
run_manifest t2_aggregation_alpha2    # exp2's second alpha, 60 outstanding (~75)
echo "=== $(date -Is) chain complete"
