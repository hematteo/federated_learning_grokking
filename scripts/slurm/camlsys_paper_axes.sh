#!/bin/bash
#SBATCH -J fedgrok_paper_axes
#SBATCH --partition=ampere
#SBATCH --requeue
#SBATCH --cpus-per-task=44
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# The paper's three open FL axes on CaMLSys (RUNS_TODO.md entries 4-6):
#   t5_local_epochs -> t3a_dirichlet_setups -> t5_participation, cheapest first.
#
# Provenance: working tree synced from local branch v2-multisetup @ e20e133
# plus the uncommitted 2026-08-31 correctness pass and these three manifests.
# Result JSONs carry the full config; run ids are content hashes.
#
# RESUME MODEL. launch_sweep.py skips any spec whose result JSON exists, so this
# script is idempotent: a requeue (preemption) or a chained follow-up job just
# re-invokes it and continues. A run in flight when the job dies is re-run from
# scratch -- there is no mid-run resume -- so each individual run must fit
# inside one 12 h allocation. Manifests are ordered longest-first internally.
#
# DISK. /nfs-share is at ~180 GB of a ~200 GB per-user quota, and mauao's /tmp
# is 99% full (~30 GB). Nothing large may land on either: the venv, uv cache
# and every Ray temp/object-store file go to /dev/shm (378 GB, RAM-backed) and
# are removed on exit. Only histories, checkpoints and result JSONs (~10 GB
# for the whole campaign) go to /nfs-share.

set -uo pipefail
export PATH=/nfs-share/$USER/bin:$PATH
export PYTHONUNBUFFERED=1
PROJ=/nfs-share/$USER/federated_learning_grokking
cd "$PROJ"

SCRATCH=/dev/shm/${USER}_fedgrok_${SLURM_JOB_ID:-$$}
mkdir -p "$SCRATCH/tmp" "$SCRATCH/ray" logs/sweeps
trap 'rm -rf "$SCRATCH"' EXIT
export TMPDIR="$SCRATCH/tmp"
export UV_CACHE_DIR="$SCRATCH/uv_cache"
export RAY_TMPDIR="$SCRATCH/ray"
# 16 concurrent Ray instances would each claim 30% of the cgroup's memory for
# their object store and spill to /tmp when /dev/shm ran out. Payload per round
# is ~1 MB x K, so 1% (2 GB at --mem=200G) is generous.
export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.01
# VRAM: at most 8 clients per run hold a CUDA context (README, VRAM section).
# Placement only -- results are bit-identical. 4 runs x 8 x ~0.5 GB per A40.
export FEDGROK_GPU_CLIENT_CAP=8
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) $(date -Is) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  cpus=$(nproc)  shm_free=$(df -h /dev/shm | awk 'NR==2{print $4}')"
nvidia-smi -L

# ── Environment: fresh venv on /dev/shm, from pyproject (pinned pairs) ────────
VENV="$SCRATCH/venv"
uv venv --python 3.12 "$VENV" >/dev/null || { echo "uv venv failed"; exit 1; }
# flwr is PINNED to the 1.27 line the 1,529 banked runs were produced with
# (PROGRESS.md, Env). pyproject only lower-bounds it and an unpinned resolve
# gave 1.36 on 2026-09-02 -- Flower owns the client-sampling RNG that the
# partial-participation arms depend on, so a newer Flower under new rungs and
# an older one under their banked controls would be a version confound.
uv pip install --python "$VENV/bin/python" -e ".[dev]" "flwr[simulation]==1.27.*" >"$SCRATCH/pip.log" 2>&1 \
    || { echo "pip install failed:"; tail -30 "$SCRATCH/pip.log"; exit 1; }
PY="$VENV/bin/python"
"$PY" - <<'PYEOF' || exit 1
import torch, flwr, sys
n = torch.cuda.device_count()
print(f"torch {torch.__version__}  flwr {flwr.__version__}  cuda={torch.cuda.is_available()}  gpus={n}")
sys.exit(0 if (torch.cuda.is_available() and n >= 1) else 1)
PYEOF

# ── Final check: the full suite, ON THE NODE (flwr is not installed locally) ──
# Gated by a sentinel so requeues and chained jobs do not spend ~10 min repeating
# it; delete logs/.tests_passed to force a re-run after any code change.
if [ ! -f logs/.tests_passed ]; then
    echo "=== pytest $(date -Is) ==="
    if "$PY" -m pytest tests/ -q -p no:cacheprovider 2>&1 | tee logs/pytest_${SLURM_JOB_ID:-local}.log | tail -8 \
       && grep -qE "^[0-9]+ passed" logs/pytest_${SLURM_JOB_ID:-local}.log \
       && ! grep -qE "[0-9]+ (failed|error)" logs/pytest_${SLURM_JOB_ID:-local}.log; then
        touch logs/.tests_passed
    else
        echo "TESTS FAILED -- not launching any sweep"; exit 1
    fi
fi

# ── The sweeps ────────────────────────────────────────────────────────────────
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
overall=0
for m in t5_local_epochs t3a_dirichlet_setups t5_participation; do
    echo "=== sweep $m  $(date -Is) ==="
    "$PY" scripts/launch_sweep.py "manifests/$m.jsonl" --gpus "$GPUS" --per-gpu 4 \
        2>&1 | tee -a "logs/sweeps/${m}.log"
    rc=${PIPESTATUS[0]}
    echo "=== sweep $m exit=$rc  $(date -Is) ==="
    [ "$rc" -ne 0 ] && overall=$rc
done

"$PY" scripts/collect_runs.py 2>&1 | tail -3
echo "=== done $(date -Is) exit=$overall ==="
exit $overall
