"""Centralised twins: the same model from the same initial weights, trained on all the data.

    venv/bin/python scripts/mechinterp/twins.py train      # -> results/mechinterp/twins/
    venv/bin/python scripts/mechinterp/twins.py compare    # -> results/mechinterp/twin_trajectories.csv

WHY. The banked centralised baselines carry no checkpoints, so "is the federated
circuit the centralised circuit reached later?" could only be asked of order
parameters, never of weights. Initial weights depend on the seed alone (common.py),
so a centralised run trained now with the federated run's seed, width, data
fraction, decay and optimiser starts from EXACTLY the federated run's initial
weights. Every difference between the two trajectories is then federation.

TWIN CELLS: one per family of federated runs with saved weights -- (setup, data
fraction, weight decay, width) -- with the family's largest budget, checkpoints on a
spacing that divides the federated checkpoint steps, and the seeds that family used.
They are written under results/mechinterp/twins/ and are NOT banked into the run
table (checkpoint_every is inside the run id, so they would not collide with the
banked baselines, but they are an analysis instrument, not a result).

COMPARE: for every federated checkpoint of every run in a family, against the twin
with the same seed at the nearest checkpoint step (within 2.5%):
  tw_step_gap            |federated step - twin step| / federated step
  tw_rel_dist            ||theta_FL - theta_C|| / ||theta_C|| over all weight matrices
  tw_cos                 cosine between the two flattened weight vectors
  tw_first_rel_dist      the same for the input layer (W1 / W_E / first MLP layer)
  tw_pred_agree_te       fraction of held-out examples with the same predicted class
  tw_pred_agree_grid     the same over the complete input grid (grid tasks)
  tw_cka_te              linear CKA of the penultimate representations on held-out data
  tw_clock_key_jaccard   modular: overlap of the two models' clock key-frequency sets
  tw_clock_energy_cos    modular: cosine of the per-frequency clock-energy vectors
  tw_irrep_same_dominant S5: do the input blocks share their dominant irrep
  tw_irrep_frac_cos      S5: cosine of the input blocks' irrep energy profiles
plus the twin's own test accuracy at that step (tw_twin_acc_te).
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import common as C                                             # noqa: E402
import checkpoints as K                                        # noqa: E402
from fedgrok.metrics import irreps                             # noqa: E402

TW = os.path.join(C.OUT, "twins")
FED_ONLY = {"num_clients", "num_rounds", "local_epochs", "fraction_train", "partition",
            "dirichlet_alpha", "proximal_mu", "strategy", "server_lr", "server_momentum", "tau",
            "feddyn_alpha", "track_client_drift", "persist_local_opt_state", "eval_every",
            "checkpoint_client_weights", "output_dir"}

# (setup, fraction key, weight_decay, width) -> (budget steps, checkpoint_every, log_every, seeds)
CELLS = {
    ("A", "0.3", "0.0", "256"): (50_000, 1_000, 100, [42, 123, 456]),
    ("A", "0.25", "0.0", "256"): (100_000, 2_000, 100, [42, 123, 456, 789, 1011]),
    ("A'", "0.2", "0.1", "256"): (100_000, 2_000, 100, [42, 123, 456]),
    ("B", "0.3", "0.1", "128"): (200_000, 4_000, 200, [42, 123, 456]),
    ("B", "0.3", "1.0", "128"): (100_000, 2_000, 100, [42, 123, 456]),
    ("B", "0.4", "1.0", "128"): (100_000, 2_000, 100, [42, 123, 456]),
    # C at half its federated budget: federated C cells cross at 8k-30k steps, so 100k still
    # covers crossing several times over, and it halves the most expensive twins.
    ("C", "0.4", "1.0", "256"): (100_000, 4_000, 200, [42, 123, 456]),
    ("C", "0.5", "1.0", "256"): (100_000, 4_000, 200, [42, 123, 456]),
    ("D", "0.3", "1.0", "256"): (250_000, 5_000, 500, [42, 123, 456]),
    ("E", "n2000", "0.1", "200"): (40_000, 1_000, 100, [42, 123, 456]),
}


# Measured ms per centralised epoch from the banked baselines, for balancing --shard.
EST_MS = {"A": 0.9, "A'": 0.85, "B": 8.8, "C": 22.0, "D": 1.5, "E": 17.0}


def family_key(r):
    frac = r["alpha"] if r["dataset"] != "mnist" else "n" + r["n_train"]
    return (r["setup"], frac, r["weight_decay"], r["hidden_width"])


def fed_runs_by_family(rows):
    fam = {}
    for rid, r in rows.items():
        if r["mode"] == "federated" and glob.glob(os.path.join(C.RUNS, rid, "checkpoints", "ckpt_*.pt")):
            fam.setdefault(family_key(r), []).append(rid)
    return fam


def twin_spec(fed_spec, budget, ckpt_every, log_every, seed):
    from fedgrok.manifest import TAG_KEYS
    s = {k: v for k, v in fed_spec.items() if k not in FED_ONLY and k not in TAG_KEYS}
    s.update({"mode": "centralized", "seed": seed, "epochs": budget, "log_every": log_every,
              "checkpoint_every": ckpt_every})
    return s


def jobs(shard=None):
    """(key, seed) pairs, cheapest first; with shard "i/n", greedily balanced by estimated cost."""
    js = [(key, seed) for key, (b, ck, lg, seeds) in CELLS.items() for seed in seeds]
    cost = {j: CELLS[j[0]][0] * EST_MS[j[0][0]] * (2.0 if j[0] == ("B", "0.4", "1.0", "128") else 1.0)
            for j in js}
    js.sort(key=lambda j: cost[j])
    if not shard:
        return js
    i, n = (int(x) for x in shard.split("/"))
    loads, assign = [0.0] * n, {}
    for j in sorted(js, key=lambda j: -cost[j]):
        k = loads.index(min(loads))
        assign[j] = k
        loads[k] += cost[j]
    return [j for j in js if assign[j] == i]


def train(shard=None):
    from fedgrok.run import run_spec
    rows = C.load_rows()
    fam = fed_runs_by_family(rows)
    for key, seed in jobs(shard):
        if key not in fam:
            print("  no federated family", key)
            continue
        base = C.spec(sorted(fam[key])[0])
        budget, ck, lg, _ = CELLS[key]
        if True:
            sp = twin_spec(base, budget, ck, lg, seed)
            sp["setup"] = key[0]
            fed_cfg = C.build_config({k: v for k, v in base.items() if k != "output_dir"})
            tw_cfg = C.build_config(sp)
            shared = ("dataset", "model", "loss", "optimizer", "lr", "weight_decay", "alpha", "p", "task",
                      "hidden_width", "n_heads", "d_mlp", "n_layers", "init_scale", "activation",
                      "batch_size", "n_train", "n_test", "group_n", "momentum")
            bad = [f for f in shared if getattr(fed_cfg, f) != getattr(tw_cfg, f)]
            if bad:
                raise RuntimeError(f"twin {key} differs from its federated family on {bad}")
            from fedgrok.manifest import run_id
            rid = run_id(sp)
            if os.path.exists(os.path.join(TW, "rows", rid + ".json")):
                continue
            print(f"  twin {key} seed {seed} -> {rid}", flush=True)
            try:
                run_spec(sp, results_root=os.path.join(TW, "rows"), histories_root=os.path.join(TW, "runs"))
            except Exception:
                print(traceback.format_exc(), flush=True)


def twin_index():
    out = {}
    for f in glob.glob(os.path.join(TW, "rows", "*.json")):
        row = json.load(open(f))
        frac = str(row["alpha"]) if row["dataset"] != "mnist" else "n" + str(row["n_train"])
        key = (row.get("setup"), frac, str(row["weight_decay"]), str(row["hidden_width"]), int(row["seed"]))
        out[key] = row
    return out


def twin_ckpts(rid):
    out = []
    for f in glob.glob(os.path.join(TW, "runs", rid, "checkpoints", "ckpt_epoch*.pt")):
        out.append((int(re.search(r"epoch(\d+)", f).group(1)), f))
    return sorted(out)


def linear_cka(X, Y):
    X = X.double() - X.double().mean(0)
    Y = Y.double() - Y.double().mean(0)
    xy = float((X.t() @ Y).norm() ** 2)
    xx = float((X.t() @ X).norm())
    yy = float((Y.t() @ Y).norm())
    return xy / (xx * yy) if xx > 0 and yy > 0 else math.nan


def flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters() if p.dim() >= 2])


def first_layer(model):
    if hasattr(model, "W1"):
        return model.W1.detach()
    if hasattr(model, "W_E"):
        return model.W_E.detach()
    return model.layers[0].weight.detach()


def input_block_for_irreps(model):
    if hasattr(model, "W1"):
        return model.W1.detach()[:, :model.W1.shape[1] // 2]
    return model.W_E.detach().t()


def compare():
    rows = C.load_rows()
    fam = fed_runs_by_family(rows)
    twins = twin_index()
    out = []
    for key, rids in sorted(fam.items()):
        if key not in CELLS:
            continue
        for rid in sorted(rids):
            r = rows[rid]
            tw = twins.get(key + (int(C.fnum(r["seed"])),))
            if tw is None:
                continue
            try:
                cfg = C.config(rid)
                d = C.data(cfg)
                tck = twin_ckpts(tw["id"])
                if not tck:
                    continue
                tsteps = np.array([t for t, _ in tck])
                for step, unit, n, path in C.checkpoints(rid):
                    j = int(np.argmin(np.abs(tsteps - step)))
                    gap = abs(tsteps[j] - step) / max(step, 1)
                    if gap > 0.025:
                        continue
                    mf = C.load_model(cfg, path)
                    mc = C.load_model(cfg, tck[j][1])
                    rec = {**C.descriptors(r), "step": step, "twin_step": int(tsteps[j]), "tw_step_gap": gap,
                           "twin_id": tw["id"]}
                    a, b = flat(mf), flat(mc)
                    rec["tw_rel_dist"] = float((a - b).norm() / b.norm())
                    rec["tw_cos"] = float((a @ b) / (a.norm() * b.norm()))
                    fa, fb = first_layer(mf), first_layer(mc)
                    rec["tw_first_rel_dist"] = float((fa - fb).norm() / fb.norm())
                    pf, pc = K.forward_parts(mf, d["xte"]), K.forward_parts(mc, d["xte"])
                    rec["tw_pred_agree_te"] = float((pf["logits"].argmax(1) == pc["logits"].argmax(1)).float().mean())
                    rec["tw_twin_acc_te"] = float((pc["logits"].argmax(1) == d["yte"]).float().mean() * 100)
                    rec["tw_fed_acc_te"] = float((pf["logits"].argmax(1) == d["yte"]).float().mean() * 100)
                    rec["tw_cka_te"] = linear_cka(pf["feat"], pc["feat"])
                    if "G" in d:
                        gf, gc = K.grid_parts(mf, d), K.grid_parts(mc, d)
                        rec["tw_pred_agree_grid"] = float((gf["logits"].argmax(1) == gc["logits"].argmax(1)).float().mean())
                        if cfg.dataset == "modular":
                            G = d["G"]
                            _, ef, _ = K.clock_decomposition(gf["logits"].reshape(G, G, -1), G)
                            _, ec, _ = K.clock_decomposition(gc["logits"].reshape(G, G, -1), G)
                            kf, kc = set(K.key_set(ef)), set(K.key_set(ec))
                            rec["tw_clock_key_jaccard"] = len(kf & kc) / len(kf | kc)
                            rec["tw_clock_energy_cos"] = float(ef @ ec / (np.linalg.norm(ef) * np.linalg.norm(ec) + 1e-30))
                        del gf, gc
                    if cfg.dataset == "s5":
                        ff = irreps.fractions(input_block_for_irreps(mf))
                        fc = irreps.fractions(input_block_for_irreps(mc))
                        rec["tw_irrep_same_dominant"] = float(max(ff, key=ff.get) == max(fc, key=fc.get))
                        va, vb = np.array(list(ff.values())), np.array([fc[k] for k in ff])
                        rec["tw_irrep_frac_cos"] = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
                    out.append(rec)
                    del mf, mc, pf, pc
                print(f"  {rid} {key} done", flush=True)
            except Exception:
                print(f"  {rid} FAILED\n{traceback.format_exc()}", flush=True)
    print("  wrote", C.write_csv(os.path.join(C.OUT, "twin_trajectories.csv"), out), len(out), "rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["train", "compare"])
    ap.add_argument("--shard")
    a = ap.parse_args()
    train(a.shard) if a.what == "train" else compare()


if __name__ == "__main__":
    main()
