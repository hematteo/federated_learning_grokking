"""Spectra of each federated run and its centralised twin at their last common step.

    venv/bin/python scripts/mechinterp/twin_endpoints.py   # -> results/mechinterp/twin_endpoint_spectra.csv

twin_trajectories.csv reduces "same circuit?" to scalars (key-frequency Jaccard, dominant
irrep). This keeps the vectors so figures can show which frequencies / irreps each model
uses. One row per (run, model in {fed, twin}, component):
  modular   component = frequency w, value = clock energy share of exp(i w (a+b-c))
            in the logits (normalised to sum 1 over frequencies)
  s5        component = irrep, value = isotypic energy fraction of the input block
The pair is taken at the last row of twin_trajectories.csv for that run (the twin
checkpoint within 2.5% of the federated step).
"""
import collections
import csv
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import common as C                                             # noqa: E402
import checkpoints as K                                        # noqa: E402
import twins as T                                              # noqa: E402
from fedgrok.metrics import irreps                             # noqa: E402


def main():
    rows = C.load_rows()
    last = {}
    for r in csv.DictReader(open(os.path.join(C.OUT, "twin_trajectories.csv"))):
        if r["id"] not in last or float(r["step"]) > float(last[r["id"]]["step"]):
            last[r["id"]] = r
    out = []
    for k, (rid, tr) in enumerate(sorted(last.items())):
        try:
            cfg = C.config(rid)
            d = C.data(cfg)
            step, tstep = float(tr["step"]), int(tr["twin_step"])
            path = [p for s, _, _, p in C.checkpoints(rid) if abs(s - step) < 1e-6][0]
            tpath = dict(T.twin_ckpts(tr["twin_id"]))[tstep]
            base = {**C.descriptors(rows[rid]), "axis": rows[rid]["axis"], "step": step, "twin_step": tstep,
                    "twin_id": tr["twin_id"]}
            for who, p in (("fed", path), ("twin", tpath)):
                m = C.load_model(cfg, p)
                if cfg.dataset == "modular":
                    G = d["G"]
                    gp = K.grid_parts(m, d)
                    _, e, _ = K.clock_decomposition(gp["logits"].reshape(G, G, -1), G)
                    e = e / max(e.sum(), 1e-30)
                    for w, v in enumerate(e, start=1):
                        out.append({**base, "model": who, "component": w, "value": float(v)})
                    del gp
                elif cfg.dataset == "s5":
                    fr = irreps.fractions(T.input_block_for_irreps(m))
                    for name, v in fr.items():
                        out.append({**base, "model": who, "component": name, "value": float(v)})
                del m
            print(f"[{k + 1}/{len(last)}] {rid}", flush=True)
        except Exception:
            print(f"[{k + 1}/{len(last)}] {rid} FAILED\n{traceback.format_exc()}", flush=True)
        torch.cuda.empty_cache()
    print("  wrote", C.write_csv(os.path.join(C.OUT, "twin_endpoint_spectra.csv"), out), len(out), "rows")


if __name__ == "__main__":
    main()
