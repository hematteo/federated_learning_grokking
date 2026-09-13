#!/bin/bash
#SBATCH -J fedgrok_algo
#SBATCH --partition=turing
#SBATCH --requeue
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:4
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# The algorithm campaign on CaMLSys (RUNS_TODO entry 0; plan in
# plans/exp5-algorithms-across-setups.md). Moved here 2026-09-11 after a
# warning that dev-gpu-acs is for interactive work only; 93 of
# t6_algo_calibration's 201 runs are already banked from that box and dedup.
#
# RESUME MODEL. launch_sweep.py skips any spec whose RESULT JSON exists, so
# this script is idempotent: preemption-requeue, a chained follow-up job, or a
# manual resubmit all continue where the last one stopped. There is no mid-run
# resume, so every individual run must fit inside one 12 h allocation -- the
# longest cell here is setup C at ~4 h, and manifests are ordered longest-first.
#
# DISK IS THE BINDING CONSTRAINT, not compute. /nfs-share/mh2274 is at ~189 GB
# of a ~200 GB per-user quota (~11 GB headroom) and hitting it does not just
# fail this job, it crashes every other job of this user writing at that
# instant. So:
#   - per-run stdout (8-22 MB each, ~4 GB over the campaign) goes to /dev/shm
#     via --logs-root and is DISCARDED, except for runs that FAIL, which are
#     copied back below. Result JSONs (~5 KB) are the evidence and stay.
#   - the venv, uv cache and every Ray temp/object-store file are on /dev/shm.
#   - the job refuses to start if headroom is under 5 GB.
#
# GPU. 4 of ngongotaha's 7 RTX 2080 Ti, leaving 3 for others; the partition is
# preemptible with --requeue so this yields rather than blocks. 11 GB cards, so
# FEDGROK_GPU_CLIENT_CAP=4 (~2.5 GB per run, 2 runs per card) against the 8 used
# on the 23 GB L4s. Placement only -- results are bit-identical.

set -uo pipefail
export PATH=/nfs-share/$USER/bin:$PATH
export PYTHONUNBUFFERED=1
PROJ=/nfs-share/$USER/federated_learning_grokking
cd "$PROJ" || exit 1

HEADROOM_GB=$(( 200 - $(du -sm /nfs-share/$USER 2>/dev/null | cut -f1) / 1024 ))
echo "quota headroom: ~${HEADROOM_GB} GB"
if [ "$HEADROOM_GB" -lt 5 ]; then
    echo "ABORT: under 5 GB of quota headroom; free space before running." >&2
    exit 1
fi

SCRATCH=/dev/shm/${USER}_fedgrok_${SLURM_JOB_ID:-$$}
mkdir -p "$SCRATCH/tmp" "$SCRATCH/ray" "$SCRATCH/runlogs" logs/sweeps
trap 'rm -rf "$SCRATCH"' EXIT
export TMPDIR="$SCRATCH/tmp"
export UV_CACHE_DIR="$SCRATCH/uv_cache"
export RAY_TMPDIR="$SCRATCH/ray"
export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.01
export FEDGROK_GPU_CLIENT_CAP=4
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) $(date -Is) ==="
echo "cpus=$(nproc)  shm_free=$(df -h /dev/shm | awk 'NR==2{print $4}')"
nvidia-smi -L

# ── Environment ──────────────────────────────────────────────────────────────
# flwr is PINNED TO 1.30, not the 1.27 of camlsys_paper_axes.sh. The 93 runs of
# this campaign already banked ran on flwr 1.30 (cam-gpu-acs), and Flower owns
# the client-sampling RNG -- so within THIS campaign 1.30 is the consistent
# choice, and 1.27 would be the version confound. 1.30 was validated against the
# 1.27 corpus by the E=1 FedAvg identity rung (RESULTS 19).
VENV="$SCRATCH/venv"
uv venv --python 3.12 "$VENV" >/dev/null || { echo "uv venv failed"; exit 1; }
uv pip install --python "$VENV/bin/python" -e ".[dev]" "flwr[simulation]==1.30.*" \
    >"$SCRATCH/pip.log" 2>&1 || { echo "pip install failed:"; tail -30 "$SCRATCH/pip.log"; exit 1; }
PY="$VENV/bin/python"
"$PY" - <<'PYEOF' || exit 1
import torch, flwr
print(f"torch {torch.__version__}  flwr {flwr.__version__}  "
      f"cuda={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}")
assert torch.cuda.is_available(), "no CUDA"
PYEOF

# Gate on the tests, as the previous CaMLSys submission did. The FL half needs
# flwr, which only exists in this venv.
echo "=== test gate $(date -Is)"
"$PY" -m pytest tests/ -q -p no:cacheprovider -W ignore > "$SCRATCH/pytest.log" 2>&1
if [ $? -ne 0 ]; then echo "TESTS FAILED, not launching:"; tail -25 "$SCRATCH/pytest.log"; exit 1; fi
tail -1 "$SCRATCH/pytest.log"

# ── The chain, in the order RUNS_TODO entry 0 fixes ──────────────────────────
run_manifest () {
    local m=$1
    [ -f "manifests/$m.jsonl" ] || { echo "skip $m (no manifest)"; return; }
    echo "=== $(date -Is) launching $m"
    "$PY" -u scripts/launch_sweep.py "manifests/$m.jsonl" \
        --gpus 0,1,2,3 --per-gpu 2 --logs-root "$SCRATCH/runlogs" \
        2>&1 | tee "logs/sweeps/${m}_${SLURM_JOB_ID:-local}.log" | grep -E "FAIL|Sweep complete|specs,"
    "$PY" scripts/collect_runs.py >/dev/null 2>&1 || true
}

run_manifest t6_algo_calibration      # 108 of 201 left (93 banked on cam-gpu-acs)
run_manifest x_b_wd_zero_fl           # decay-clock control at alpha=0.70
run_manifest x_h2_mechanism           # checkpointed arms + damped FedAvg
run_manifest x_scaffold_rerun         # H1/H3 with server-side c_i
run_manifest t6_algo_comparison       # Phase 2, setups present in CALIBRATED
run_manifest t6_algo_calibration_c    # C, withheld until reproducible
run_manifest x_b_decay_band_long      # B's censored decay rungs
run_manifest t2_aggregation_alpha2    # exp2's second alpha

# Keep ONLY the logs of runs that failed -- those are the ones worth the quota.
mkdir -p logs/failed
for L in "$SCRATCH"/runlogs/*.log; do
    [ -e "$L" ] || continue
    id=$(basename "$L" .log)
    [ -f "results/data/runs/$id.json" ] || cp "$L" "logs/failed/$id.log"
done
echo "=== chain pass complete $(date -Is); failed-run logs kept: $(ls logs/failed 2>/dev/null | wc -l)"
