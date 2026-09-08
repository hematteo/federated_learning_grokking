"""Rewrite banked result JSONs so they are valid JSON.

Python's `json` emits bare `NaN` / `Infinity` tokens for non-finite floats. Those
are not in the JSON grammar, so a file containing them round-trips fine in Python
and is rejected by jq, JavaScript, Go, Rust and R. `fedgrok.run._write_json_atomic`
serialises non-finite floats as the strings "nan" / "inf" / "-inf" instead; it
handled "inf" from the start but not "nan", so every run whose `final_ipr` is NaN
-- i.e. every non-modular run, IPR being GrokNet-specific -- was written
unparseable. 1,038 of 1,529 banked files at the time of writing.

This applies the current serialiser to files already on disk. It is idempotent
(a repaired file re-reads as the string "nan" and is written back unchanged) and
it does not touch any value: only the encoding of non-finite floats changes.

`results/data/runs_v2.csv` does NOT change, because `str(float('nan'))` and
`str("nan")` are both "nan" -- but regenerate it with collect_runs.py anyway to
confirm that.

    python scripts/repair_result_json.py --dry-run
    python scripts/repair_result_json.py
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fedgrok.run import _write_json_atomic


def _has_bare_nonfinite(text: str) -> bool:
    """Does this file rely on Python's non-standard JSON extensions?

    Detected by re-parsing with strict constants rather than by scanning the
    text, so a run whose *string* payload happens to contain "NaN" is not
    misread as broken.
    """

    def _reject(token):
        raise ValueError(token)

    try:
        json.loads(text, parse_constant=_reject)
        return False
    except ValueError:
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="results/data/runs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing.",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.runs_dir, "*.json")))
    if not paths:
        print(f"No result JSONs under {args.runs_dir}")
        return 1

    broken = []
    for path in paths:
        with open(path) as handle:
            text = handle.read()
        if _has_bare_nonfinite(text):
            broken.append(path)

    print(f"{len(paths)} result JSON(s); {len(broken)} contain bare NaN/Infinity.")
    if not broken:
        print("Nothing to repair.")
        return 0

    if args.dry_run:
        for path in broken[:5]:
            print(f"  would repair {path}")
        if len(broken) > 5:
            print(f"  ... and {len(broken) - 5} more")
        return 0

    for path in broken:
        with open(path) as handle:
            row = json.load(handle)
        _write_json_atomic(path, row)

    # Re-check, so the script reports the state it actually left behind rather
    # than the state it intended to.
    still = [p for p in broken if _has_bare_nonfinite(open(p).read())]
    print(f"Repaired {len(broken) - len(still)} file(s).")
    if still:
        print(f"  FAILED on {len(still)}, e.g. {still[:3]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
