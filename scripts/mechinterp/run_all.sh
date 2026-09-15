#!/usr/bin/env bash
# Chain the mechanistic-interpretability suite. Every stage is resumable, so rerunning
# this script after an interruption repeats only what is missing.
#
#   setsid nohup bash scripts/mechinterp/run_all.sh > logs/mechinterp/run_all.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=venv/bin/python
L=logs/mechinterp
mkdir -p "$L"

wait_for() {                       # wait until no process matches the pattern
  while pgrep -f "$1" > /dev/null; do sleep 60; done
}
stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "waiting for checkpoint shards"
wait_for "python -u scripts/mechinterp/checkpoints.py --shard"
stamp "checkpoint pass: resume (runs the old code skipped)"
$PY -u scripts/mechinterp/checkpoints.py > "$L/checkpoints_resume.log" 2>&1
$PY scripts/mechinterp/checkpoints.py --collect

stamp "spectra (2 shards)"
$PY -u scripts/mechinterp/spectra.py --shard 0/2 > "$L/spectra_0.log" 2>&1 &
$PY -u scripts/mechinterp/spectra.py --shard 1/2 > "$L/spectra_1.log" 2>&1 &
wait
$PY scripts/mechinterp/spectra.py --collect

stamp "waiting for client pass"
wait_for "python -u scripts/mechinterp/clients.py"
if grep -q FAILED "$L/clients.log"; then
  stamp "client pass had failures (runs without spec.json): rerunning"
  $PY -u scripts/mechinterp/clients.py > "$L/clients_rerun.log" 2>&1
fi

stamp "interim findings"
$PY scripts/mechinterp/findings.py > "$L/findings_interim.log" 2>&1

stamp "waiting for twin training"
wait_for "python -u scripts/mechinterp/twins.py train"
stamp "twin comparison"
$PY -u scripts/mechinterp/twins.py compare > "$L/twins_compare.log" 2>&1
stamp "landscape"
$PY -u scripts/mechinterp/landscape.py > "$L/landscape.log" 2>&1

stamp "final findings"
$PY scripts/mechinterp/findings.py > "$L/findings.log" 2>&1
stamp "done"
