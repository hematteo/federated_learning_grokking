"""Loss along straight lines in weight space between checkpoints.

    venv/bin/python scripts/mechinterp/landscape.py         # -> results/mechinterp/landscape_paths.csv
                                                            #    results/mechinterp/landscape_barriers.csv

Three paths per run, each evaluated at 11 evenly spaced points theta(l) = (1-l) theta_a + l theta_b:

  memo_to_end   the checkpoint nearest t_memo to the last checkpoint: is the generalising
                solution reachable from the memorising one without crossing a train-loss
                barrier, and how does test accuracy change along the way?
  init_to_end   the rebuilt initial weights to the last checkpoint
  fed_to_twin   federated runs only: the last federated checkpoint to the centralised
                twin's checkpoint at the same step (twins.py) -- linear mode connectivity
                between a federated and a centralised solution from the same initialisation

landscape_paths.csv: one row per (run, path, l) with train/test loss and accuracy.
landscape_barriers.csv: one row per (run, path) with
  barrier_train_loss  max over l of train loss minus the linear interpolation of the endpoint
                      losses (Frankle et al.'s instability measure; ~0 = linearly connected)
  barrier_rel         the barrier over the mean endpoint train loss
  min_test_acc_path   lowest test accuracy along the path, and at which l
  test_acc_mid        test accuracy at l = 0.5
"""
import glob
import math
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import common as C                                               # noqa: E402
import checkpoints as K                                          # noqa: E402
import twins as T                                                # noqa: E402
from fedgrok.core.registry import build_loss                     # noqa: E402

LAMBDAS = np.linspace(0, 1, 11)


def evaluate(model, cfg, d):
    loss_fn = build_loss(cfg).loss_fn
    out = {}
    with torch.no_grad():
        for split, x, y, t in (("tr", d["xtr"], d["ytr"], d["ttr"]), ("te", d["xte"], d["yte"], d["tte"])):
            L = K.forward_parts(model, x)["logits"]
            out[f"loss_{split}"] = float(loss_fn(L, t))
            out[f"acc_{split}"] = float((L.argmax(1) == y).float().mean() * 100)
    return out


def path(cfg, d, sa, sb):
    model = C.init_model(cfg)
    rows = []
    for lam in LAMBDAS:
        state = {k: (1 - lam) * sa[k].to(C.DEVICE) + lam * sb[k].to(C.DEVICE) for k in sa}
        model.load_state_dict(state)
        rows.append({"lambda": float(lam), **evaluate(model, cfg, d)})
    return rows


def barrier(rows):
    tl = np.array([r["loss_tr"] for r in rows])
    lin = tl[0] + (tl[-1] - tl[0]) * LAMBDAS
    b = float(np.max(tl - lin))
    ta = np.array([r["acc_te"] for r in rows])
    j = int(np.argmin(ta))
    return {"barrier_train_loss": b, "barrier_rel": b / max((tl[0] + tl[-1]) / 2, 1e-12),
            "min_test_acc_path": float(ta[j]), "min_test_acc_lambda": float(LAMBDAS[j]),
            "test_acc_mid": float(ta[5]), "test_acc_start": float(ta[0]), "test_acc_end": float(ta[-1]),
            "train_loss_start": float(tl[0]), "train_loss_end": float(tl[-1])}


def main():
    rows = C.load_rows()
    ids = sorted(rid for rid in rows if glob.glob(os.path.join(C.RUNS, rid, "checkpoints", "ckpt_*.pt")))
    twins = T.twin_index()
    paths, bars = [], []
    for k, rid in enumerate(ids):
        r = rows[rid]
        try:
            cfg = C.config(rid)
            d = C.data(cfg)
            cks = C.checkpoints(rid)
            if len(cks) < 2:
                continue
            end = torch.load(cks[-1][3], map_location="cpu")
            todo = []
            tm = C.fnum(r["t_memo"])
            if math.isfinite(tm):
                steps = np.array([c[0] for c in cks])
                j = int(np.argmin(np.abs(steps - tm)))
                if j < len(cks) - 1 and 0.5 <= steps[j] / tm <= 2.0:
                    todo.append(("memo_to_end", torch.load(cks[j][3], map_location="cpu"), end, cks[j][0], cks[-1][0]))
            init = {k2: v.detach().cpu() for k2, v in C.init_model(cfg).state_dict().items()}
            todo.append(("init_to_end", init, end, 0.0, cks[-1][0]))
            if r["mode"] == "federated":
                tw = twins.get(T.family_key(r) + (int(C.fnum(r["seed"])),))
                if tw is not None:
                    tck = T.twin_ckpts(tw["id"])
                    if tck:
                        # The last step both have: twins can stop before the federated budget.
                        ts = np.array([t for t, _ in tck])
                        fs = np.array([c[0] for c in cks])
                        common = min(fs[-1], ts[-1])
                        jf = int(np.argmin(np.abs(fs - common)))
                        jt = int(np.argmin(np.abs(ts - fs[jf])))
                        if abs(ts[jt] - fs[jf]) / max(fs[jf], 1) <= 0.025:
                            todo.append(("fed_to_twin", torch.load(cks[jf][3], map_location="cpu"),
                                         torch.load(tck[jt][1], map_location="cpu"), float(fs[jf]), float(ts[jt])))
            for name, sa, sb, step_a, step_b in todo:
                pr = path(cfg, d, sa, sb)
                base = {**C.descriptors(r), "path": name, "step_a": step_a, "step_b": step_b}
                paths += [{**base, **p} for p in pr]
                bars.append({**base, **barrier(pr)})
            if k % 25 == 0:
                print(f"  {k}/{len(ids)}", flush=True)
        except Exception:
            print(f"  {rid} FAILED\n{traceback.format_exc()}", flush=True)
        torch.cuda.empty_cache()
    C.write_csv(os.path.join(C.OUT, "landscape_paths.csv"), paths)
    C.write_csv(os.path.join(C.OUT, "landscape_barriers.csv"), bars)
    print("  wrote landscape tables:", len(bars), "paths")


if __name__ == "__main__":
    main()
