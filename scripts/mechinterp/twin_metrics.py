"""Checkpoint metrics for the centralised twins, on the same footing as the banked runs.

    venv/bin/python scripts/mechinterp/twin_metrics.py            # -> results/mechinterp/twins/ckpt/<id>.json
    venv/bin/python scripts/mechinterp/twin_metrics.py --collect  # -> results/mechinterp/twin_checkpoint_metrics.csv

Runs checkpoints.analyse_run on every twin (results/mechinterp/twins/), so a federated
run and the centralised model trained from its exact initial weights can be placed in
the same order-parameter space. The twins' rows (twins/rows/<id>.json) carry their own
t_memo / t_first_cross; the CSV has the checkpoint_metrics.csv columns plus `twin = 1`.
"""
import argparse
import glob
import json
import math
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(__file__))
import common as C                                             # noqa: E402
import checkpoints as K                                        # noqa: E402
import twins as T                                              # noqa: E402

OUTDIR = os.path.join(T.TW, "ckpt")


def rows():
    out = {}
    for f in glob.glob(os.path.join(T.TW, "rows", "*.json")):
        r = {k: ("" if v is None else str(v)) for k, v in json.load(open(f)).items()}
        r["axis"] = "twin"
        out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    C.RUNS = os.path.join(T.TW, "runs")          # spec / history / checkpoints resolve under the twins
    rs = rows()
    if a.collect:
        out = []
        for f in sorted(glob.glob(os.path.join(OUTDIR, "*.json"))):
            j = json.load(open(f))
            base = {**j["run"], "twin": 1}
            tm, tf = C.fnum(base.get("t_memo")), C.fnum(base.get("t_first_cross"))
            for rec in j["checkpoints"]:
                r = {**base, **{k: (math.nan if v is None else v) for k, v in rec.items()}}
                r["step_over_t_memo"] = rec["step"] / tm if math.isfinite(tm) and tm > 0 else math.nan
                r["step_over_t_cross"] = rec["step"] / tf if math.isfinite(tf) and tf > 0 else math.nan
                out.append(r)
        print("  wrote", C.write_csv(os.path.join(C.OUT, "twin_checkpoint_metrics.csv"), out), len(out), "rows")
        return
    os.makedirs(OUTDIR, exist_ok=True)
    for k, (rid, row) in enumerate(sorted(rs.items())):
        path = os.path.join(OUTDIR, f"{rid}.json")
        if os.path.exists(path):
            continue
        try:
            res = K.analyse_run(rid, row)
            if res is not None:
                C.write_json(path, res)
                print(f"[{k + 1}/{len(rs)}] {rid} {row['setup']} {len(res['checkpoints'])} ckpts {res['seconds']}s",
                      flush=True)
        except Exception:
            print(f"[{k + 1}/{len(rs)}] {rid} FAILED\n{traceback.format_exc()}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
