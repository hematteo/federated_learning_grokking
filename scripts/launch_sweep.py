"""Launch a manifest of runs across GPUs, one run per slot, idempotent.

Each run is a fresh `python -m fedgrok.run` subprocess with CUDA_VISIBLE_DEVICES
pinned to one device — this is required because the device must be chosen before
torch/Ray import, and because Flower's simulation does not cleanly support two
runs in one process. Runs whose result JSON already exists are skipped, so an
interrupted sweep resumes by re-invoking with the same manifest.

    # generate + run
    python scripts/launch_sweep.py manifests/t0_wd_grid.jsonl
    python scripts/launch_sweep.py manifests/t0_wd_grid.jsonl --gpus 0,2,4,5,6,7 --per-gpu 2
    python scripts/launch_sweep.py manifests/t0_wd_grid.jsonl --dry-run

After a sweep, collect the per-run JSONs into the tidy CSV:
    python scripts/collect_runs.py
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Repo root on the path so `fedgrok` imports whether or not it is pip-installed.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fedgrok.manifest import load_manifest, run_id
from fedgrok.run import result_path, DEFAULT_RESULTS_DIR


def detect_free_gpus(mem_threshold_mb=1000):
    """Return indices of GPUs using less than `mem_threshold_mb` of memory.

    Lets a sweep avoid devices another job is already using (per PROGRESS.md,
    some indices routinely have other work on them).
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    free = []
    for line in out.strip().splitlines():
        idx, used = (x.strip() for x in line.split(","))
        if int(used) < mem_threshold_mb:
            free.append(int(idx))
    return free


def is_done(spec, results_root):
    return os.path.exists(result_path(spec, results_root))


def launch(specs, gpus, per_gpu, results_root, histories_root, poll_s=1.0):
    """Run `specs` across `gpus`, `per_gpu` concurrent runs per device."""
    # A slot is one (gpu, lane). Slots are the unit of concurrency.
    slots = [gpu for gpu in gpus for _ in range(per_gpu)]
    n_slots = len(slots)

    pending = list(specs)
    running = {}  # index into slots -> (Popen, spec, start_time)
    done = failed = 0
    total = len(pending)
    t_start = time.time()

    def free_slot():
        for i in range(n_slots):
            if i not in running:
                return i
        return None

    while pending or running:
        # Fill free slots.
        while pending and (slot := free_slot()) is not None:
            spec = pending.pop(0)
            gpu = slots[slot]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [sys.executable, "-m", "fedgrok.run",
                   "--spec", json.dumps(spec),
                   "--results-root", results_root,
                   "--histories-root", histories_root]
            log_path = os.path.join(results_root, spec["id"] + ".log")
            os.makedirs(results_root, exist_ok=True)
            logf = open(log_path, "w")
            proc = subprocess.Popen(cmd, env=env, stdout=logf,
                                    stderr=subprocess.STDOUT)
            running[slot] = (proc, spec, time.time(), logf)
            print(f"[gpu {gpu}] start {spec['id']}  "
                  f"({total - len(pending) - len(running) + 1}.. of {total})")

        # Reap finished slots.
        for slot, (proc, spec, t0, logf) in list(running.items()):
            if proc.poll() is None:
                continue
            logf.close()
            dt = time.time() - t0
            ok = proc.returncode == 0 and is_done(spec, results_root)
            if ok:
                done += 1
                tag = "done"
            else:
                failed += 1
                tag = f"FAIL rc={proc.returncode} (see {spec['id']}.log)"
            print(f"[gpu {slots[slot]}] {tag} {spec['id']}  "
                  f"{dt:.0f}s  [{done} ok / {failed} fail / {total}]")
            del running[slot]

        if pending or running:
            time.sleep(poll_s)

    elapsed = time.time() - t_start
    print(f"\nSweep complete: {done} ok, {failed} failed, {total} total, "
          f"{elapsed / 60:.1f} min wall.")
    return done, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSONL manifest of run specs")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU indices (default: auto-detect free)")
    parser.add_argument("--per-gpu", type=int, default=2,
                        help="Concurrent runs per GPU (default 2)")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--histories-root", default="results/runs")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run even if a result JSON already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run and exit")
    args = parser.parse_args()

    specs = load_manifest(args.manifest)
    for spec in specs:
        spec["id"] = run_id(spec)

    if args.gpus is not None:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    else:
        gpus = detect_free_gpus()
        if not gpus:
            print("No free GPUs detected; pass --gpus explicitly (or -1 for CPU).")
            sys.exit(1)

    todo = specs if args.rerun else [s for s in specs
                                     if not is_done(s, args.results_root)]
    skipped = len(specs) - len(todo)

    print(f"Manifest: {args.manifest}")
    print(f"  {len(specs)} specs, {skipped} already done, {len(todo)} to run")
    print(f"  GPUs {gpus} x {args.per_gpu} = {len(gpus) * args.per_gpu} slots")

    if args.dry_run:
        for spec in todo[:20]:
            print("  would run:", spec["id"], spec.get("mode"), spec.get("task"),
                  {k: spec[k] for k in ("p", "alpha", "local_epochs", "partition",
                                        "seed", "strategy") if k in spec})
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        return

    if not todo:
        print("Nothing to do.")
        return

    _done, failed = launch(todo, gpus, args.per_gpu, args.results_root,
                           args.histories_root)
    # Exit non-zero if anything failed. A sweep is normally detached with
    # `setsid nohup ... &`, so the status line in the log is the only signal --
    # and exiting 0 after 60 failed runs reads as a clean sweep to anything
    # checking programmatically.
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
