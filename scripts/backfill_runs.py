"""Repair banked result JSONs that predate a schema addition.

A result row is written once, when its run finishes, so it records the schema of
the day. Adding a field to `run.py` therefore leaves every earlier run with a
hole -- and because `collect_runs.py` writes `restval=""`, that hole reaches the
CSV as an empty cell rather than an error. Two live examples:

  checkpoint_client_weights   The 20 t2_boundary runs each saved 20 per-client
                              weight matrices (400 files, on disk now). Their
                              manifest sets the flag `true`. Their rows say
                              nothing, so the only data that can test the
                              frequency-consensus hypothesis is unselectable by
                              query -- you have to already know which runs to look
                              at, which defeats the point of the table.

  t_memo / delay              Added because `t_grok` alone cannot tell "did not
                              generalise" from "did not train". Every banked run
                              predates it.

WHERE THE TRUTH LIVES. Not in one place, which is why this is a script and not a
flag on the collector:

  config fields   the MANIFEST. `run.py` only started dropping a `spec.json`
                  beside each history in c196d1b, so runs older than that have no
                  per-run spec on disk -- but their id is a content hash of the
                  spec, so the manifest that produced them is authoritative and
                  verifiable (the id must re-hash to itself).
  outcome fields  the run's own HISTORY json, which has always been written.

WHAT THIS WILL NOT DO. It never recomputes a value that is already present, and
it never recomputes `t_grok` at all. Banked t_grok values are the evidentiary
record; `delay` is derived from the recorded t_grok rather than from a fresh
computation, so a row's grokking verdict cannot move as a side effect of adding a
column. Runs whose history is missing are reported and skipped, not guessed at.

    python scripts/backfill_runs.py --dry-run
    python scripts/backfill_runs.py
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fedgrok.analysis.grokking_metrics import (                # noqa: E402
    compute_t_memo, compute_t_first_cross, count_post_cross_dips,
)
from fedgrok.data.registry import grok_threshold             # noqa: E402
from fedgrok.manifest import build_config, load_manifest, run_id  # noqa: E402
from fedgrok.run import _write_json_atomic                     # noqa: E402

MANIFEST_DIR = os.path.join(os.path.dirname(__file__), "..", "manifests")

# Fields skipped when filling from the resolved config: paths and flags, not
# measurements. Mirrors run.py's own `_skip`.
SKIP_CONFIG = {"output_dir", "save_weights"}


def manifest_index(manifest_dir=MANIFEST_DIR):
    """id -> spec, over every manifest, verifying each id re-hashes to itself.

    The self-hash check is the guard that makes the manifest usable as the
    authority: if a spec no longer produces the id it is filed under, the file
    has been edited since the run and its config no longer describes what ran.
    """
    index, suspect = {}, []
    for path in sorted(glob.glob(os.path.join(manifest_dir, "*.jsonl"))):
        for spec in load_manifest(path):
            claimed = spec["id"]
            bare = {k: v for k, v in spec.items() if k != "id"}
            if run_id(bare) != claimed:
                suspect.append((os.path.basename(path), claimed))
                continue
            index.setdefault(claimed, spec)
    return index, suspect


def _history(run_id_, histories_root="results/runs"):
    """The run's history dict, or None if it was not kept."""
    paths = glob.glob(os.path.join(histories_root, run_id_, "history_*.json"))
    if not paths:
        return None
    with open(paths[0]) as handle:
        return json.load(handle)


def _fill_config(row, spec):
    """Config fields the row is missing, taken from the resolved dataclass."""
    import dataclasses
    cfg = build_config(spec)
    added = {}
    for key, value in dataclasses.asdict(cfg).items():
        if key.startswith("_") or key in SKIP_CONFIG:
            continue
        if key not in row or row[key] is None:
            added[key] = value
    # grok_threshold is DERIVED from the dataset, not a Config field, so the
    # loop above never supplies it. Without it `_fill_outcomes` silently skips
    # t_first_cross -- which is how 109 rows ended up recording grokked=True
    # with no crossing time, and a run that groks at 7,600 plots as one that
    # never reached the bar.
    if row.get("grok_threshold") in (None, ""):
        added["grok_threshold"] = grok_threshold(cfg)
    return added


def _fill_outcomes(row, history, bar=None):
    """t_memo, delay and peak_train_acc from the history and the RECORDED t_grok.

    `bar` is the grok threshold resolved for this run, passed in because it may
    have been filled on THIS pass (see `_fill_config`) and so is not yet on the
    row. Falls back to the recorded value.
    """
    if history is None:
        return {}
    steps = history.get("total_steps", history.get("epoch", []))
    train_accs = history.get("train_acc", [])
    if not steps or not train_accs:
        return {}

    out = {}
    if "peak_train_acc" not in row:
        # Finite points only, matching extract_grokking_results: max() over a
        # list containing NaN returns a position-dependent answer, because every
        # comparison with NaN is False.
        out["peak_train_acc"] = max(
            (a for a in train_accs if math.isfinite(a)), default=0.0)

    # t_first_cross / post_grok_dips need the bar the run was scored at, which is
    # recorded per row precisely because it varies by dataset.
    test_accs = history.get("test_acc", [])
    bar = row.get("grok_threshold") or bar
    if test_accs and bar not in (None, ""):
        bar = float(bar)
        if "t_first_cross" not in row:
            out["t_first_cross"] = compute_t_first_cross(steps, test_accs, bar)
        if "post_grok_dips" not in row:
            out["post_grok_dips"] = count_post_cross_dips(steps, test_accs, bar)
    if "t_memo" in row:
        return out

    t_memo = compute_t_memo(steps, train_accs)
    # "inf" is how _write_json_atomic serialises infinity; a banked censored run
    # reads it back as that string, not as a float.
    t_grok = row.get("t_grok")
    t_grok = float("inf") if t_grok in ("inf", None) else float(t_grok)
    delay = (t_grok - t_memo) if (t_grok != float("inf")
                                  and t_memo != float("inf")) else float("inf")
    out.update({"t_memo": t_memo, "delay": delay})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="results/data/runs")
    parser.add_argument("--histories-root", default="results/runs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    args = parser.parse_args()

    index, suspect = manifest_index()
    if suspect:
        print(f"WARNING: {len(suspect)} manifest line(s) do not re-hash to their "
              f"own id and are excluded, e.g. {suspect[:3]}")

    paths = sorted(glob.glob(os.path.join(args.runs_dir, "*.json")))
    changed = unmatched = no_history = 0
    added_keys = {}

    for path in paths:
        with open(path) as handle:
            row = json.load(handle)
        run = row.get("id") or os.path.splitext(os.path.basename(path))[0]

        additions = {}
        spec = index.get(run)
        if spec is None:
            unmatched += 1
        else:
            additions.update(_fill_config(row, spec))

        history = _history(run, args.histories_root)
        if history is None:
            no_history += 1
        additions.update(_fill_outcomes(row, history,
                                        bar=additions.get("grok_threshold")))

        if not additions:
            continue
        changed += 1
        for key in additions:
            added_keys[key] = added_keys.get(key, 0) + 1
        if not args.dry_run:
            row.update(additions)
            _write_json_atomic(path, row)

    verb = "would update" if args.dry_run else "updated"
    print(f"{len(paths)} result rows scanned; {verb} {changed}")
    for key, count in sorted(added_keys.items(), key=lambda kv: -kv[1]):
        print(f"  {key:32s} +{count}")
    if unmatched:
        print(f"  NOTE: {unmatched} row(s) matched no manifest spec -- config "
              f"fields left as recorded (ad-hoc runs, or a manifest since edited)")
    if no_history:
        print(f"  NOTE: {no_history} row(s) have no history on disk -- outcome "
              f"fields left as recorded")


if __name__ == "__main__":
    main()
