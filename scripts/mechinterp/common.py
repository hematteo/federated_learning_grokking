"""Shared plumbing for the mechanistic-interpretability suite.

Everything here is read-only against the banked runs: the run table, each run's
spec.json, its history and its checkpoints. Nothing trains except twins.py.

Two facts the suite rests on, both checked rather than assumed:

  * A run's initial weights are reproducible from its seed alone. Both training
    loops call torch.manual_seed(cfg.seed) and then build the model; dataset
    construction uses numpy RandomState or a private torch.Generator and never
    touches the global stream. The logged step-0 weight norms of centralised and
    federated runs with the same seed and width are identical on every setup,
    so `init_model` rebuilds the exact starting point, and `init_check` verifies
    it per run against the history.
  * A checkpoint is a bare state_dict; the spec.json beside the history says
    which architecture and data split it belongs to.
"""
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fedgrok.manifest import build_config                  # noqa: E402
from fedgrok.core.registry import build_model, build_loss   # noqa: E402
from fedgrok.data.registry import (build_dataset, dataset_dims, dataset_grid,  # noqa: E402
                                   has_grid)

CSV = os.path.join(ROOT, "results", "data", "runs_v2.csv")
RUNS = os.path.join(ROOT, "results", "runs")
OUT = os.path.join(ROOT, "results", "mechinterp")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Columns carried from the run table into every output row, so each table can be
# grouped by experiment axis without a join.
DESCRIPTORS = ["id", "setup", "mode", "group", "arm", "dataset", "model", "optimizer", "loss",
               "alpha", "n_train", "hidden_width", "weight_decay", "lr", "seed", "num_clients",
               "local_epochs", "fraction_train", "partition", "dirichlet_alpha", "strategy",
               "server_lr", "proximal_mu", "num_rounds", "epochs", "grok_threshold",
               "t_memo", "t_first_cross", "t_grok", "grokked", "peak_train_acc", "final_acc",
               "steps_run"]

# Experiment axis a group belongs to, for summaries.
AXIS = {
    "aggregation": "K", "aggregation_alpha2": "K", "setup_k_ladder": "K", "k_fixed_total": "K",
    "k_collapse_budget": "K", "k_collapse_wd": "K", "boundary": "K", "k50_ladder": "K",
    "local_epochs": "E", "probe": "E", "e50_long": "E", "adam_restart": "E",
    "participation": "f", "participation_setups": "f",
    "dirichlet_setups": "dirichlet", "dirichlet_band": "dirichlet", "size_control": "dirichlet",
    "partitions": "partition", "algorithms": "algorithm", "server_lr_cal": "algorithm",
    "d_internals": "centralised_alpha",
}


def fnum(v):
    if v in (None, "", "None"):
        return math.nan
    if v in ("inf", "Infinity"):
        return math.inf
    try:
        return float(v)
    except ValueError:
        return math.nan


def infer_setup(r):
    if r.get("setup"):
        return r["setup"]
    return {("modular", "groknet", "gd"): "A", ("modular", "groknet", "adamw"): "A'",
            ("modular", "transformer", "adamw"): "B", ("s5", "transformer", "adamw"): "C",
            ("s5", "groknet", "adamw"): "D", ("mnist", "mlp", "adamw"): "E"}.get(
        (r.get("dataset"), r.get("model"), r.get("optimizer")), "?")


def load_rows():
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        r["setup"] = infer_setup(r)
        r["axis"] = AXIS.get(r["group"], "other") if r["mode"] == "federated" else "centralised"
    return {r["id"]: r for r in rows}


def descriptors(r):
    d = {k: r.get(k, "") for k in DESCRIPTORS}
    d["axis"] = r.get("axis", "")
    return d


_MANIFEST_SPECS = None


def spec(rid):
    """The run's spec: its spec.json, or -- for runs banked before spec.json was
    written -- the manifest line whose content hash IS the run id."""
    path = os.path.join(RUNS, rid, "spec.json")
    if os.path.exists(path):
        return json.load(open(path))
    global _MANIFEST_SPECS
    if _MANIFEST_SPECS is None:
        from fedgrok.manifest import load_manifest, run_id
        _MANIFEST_SPECS = {}
        for m in sorted(glob.glob(os.path.join(ROOT, "manifests", "*.jsonl"))):
            for sp in load_manifest(m):
                _MANIFEST_SPECS.setdefault(run_id(sp), sp)
    if rid not in _MANIFEST_SPECS:
        raise FileNotFoundError(f"{rid}: no spec.json and no manifest line hashes to it")
    return _MANIFEST_SPECS[rid]


def config(rid):
    s = dict(spec(rid))
    s.pop("output_dir", None)
    return build_config(s)


def history(rid):
    f = glob.glob(os.path.join(RUNS, rid, "history_*.json"))
    return json.load(open(f[0])) if f else None


def steps_of(h):
    return h.get("total_steps") or h.get("epoch") or []


# ── checkpoints ──────────────────────────────────────────────────────────────

_CK = re.compile(r"ckpt_(round|epoch)(\d+)\.pt$")


def checkpoints(rid, h=None):
    """[(step, unit, index, path)] sorted by step. Rounds are converted to the
    run's total_steps via its history (checkpoint rounds are always eval rounds)."""
    out = []
    files = glob.glob(os.path.join(RUNS, rid, "checkpoints", "ckpt_*.pt"))
    if not files:
        return out
    h = h if h is not None else history(rid)
    r2s = {}
    if h and "round" in h:
        r2s = {int(rn): float(st) for rn, st in zip(h["round"], h["total_steps"])}
    for f in files:
        m = _CK.search(f)
        if not m:
            continue
        unit, n = m.group(1), int(m.group(2))
        if unit == "epoch":
            step = float(n)
        else:
            if n not in r2s:
                continue
            step = r2s[n]
        out.append((step, unit, n, f))
    return sorted(out)


def select_checkpoints(cks, row, max_n=24):
    """At most max_n checkpoints: always first and last, the ones nearest to each
    event time the run reached, and the rest spread log-uniformly in step."""
    if len(cks) <= max_n:
        return list(cks)
    steps = np.array([c[0] for c in cks])
    keep = {0, len(cks) - 1}
    for key in ("t_memo", "t_first_cross", "t_grok"):
        t = fnum(row.get(key))
        if math.isfinite(t):
            keep.add(int(np.argmin(np.abs(steps - t))))
    target = np.logspace(np.log10(max(steps[0], 1.0)), np.log10(steps[-1]), max_n)
    for t in target:
        if len(keep) >= max_n:
            break
        keep.add(int(np.argmin(np.abs(steps - t))))
    i = 0
    while len(keep) < max_n and i < len(cks):          # fill any gaps left by collisions
        keep.add(i)
        i += max(1, len(cks) // max_n)
    return [cks[i] for i in sorted(keep)]


# ── models and data ──────────────────────────────────────────────────────────

def load_model(cfg, path_or_state):
    model = build_model(cfg)
    state = torch.load(path_or_state, map_location="cpu") if isinstance(path_or_state, str) \
        else path_or_state
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def init_model(cfg):
    torch.manual_seed(int(cfg.seed))
    return build_model(cfg).to(DEVICE).eval()


def init_check(model0, h):
    """|logged step-0 norm - rebuilt norm| (NaN if the history lacks it)."""
    mats = [p.detach() for p in model0.parameters() if p.dim() >= 2]
    total = float(sum(float(p.norm()) ** 2 for p in mats) ** 0.5)
    for key, val in (("weight_norm_total", total),
                     ("weight_norm_layer1", float(model0.W1.detach().norm()) if hasattr(model0, "W1") else None)):
        logged = (h or {}).get(key)
        if logged and val is not None and logged[0] is not None and math.isfinite(logged[0]):
            return abs(logged[0] - val)
    return math.nan


_DATA = {}


def data(cfg):
    """Train/test tensors on DEVICE plus loss targets, cached by data identity."""
    key = (cfg.dataset, cfg.p, cfg.task, cfg.alpha, cfg.seed, cfg.group_n, cfg.n_train, cfg.n_test,
           cfg.loss)
    if key not in _DATA:
        if len(_DATA) > 6:
            _DATA.clear()
        xtr, ytr, xte, yte = build_dataset(cfg)
        n_classes = dataset_dims(cfg)[1]
        loss = build_loss(cfg)
        d = {"xtr": xtr.to(DEVICE), "ytr": ytr.to(DEVICE).long(), "xte": xte.to(DEVICE),
             "yte": yte.to(DEVICE).long(), "n_classes": n_classes}
        d["ttr"] = loss.prepare_target(d["ytr"], n_classes)
        d["tte"] = loss.prepare_target(d["yte"], n_classes)
        if has_grid(cfg):
            x, labels, _ = dataset_grid(cfg)
            x = torch.as_tensor(x, dtype=torch.float32)
            G = x.shape[1] // 2
            ia, ib = x[:, :G].argmax(1), x[:, G:].argmax(1)
            order = torch.argsort(ia * G + ib)             # grid rows in (a, b) raster order
            d["xgrid"] = x[order].to(DEVICE)
            d["ygrid"] = torch.as_tensor(labels)[order].long().to(DEVICE)
            d["G"] = G

            def to_ab(xs):
                return xs[:, :G].argmax(1), xs[:, G:].argmax(1)
            d["tr_ab"], d["te_ab"] = to_ab(d["xtr"]), to_ab(d["xte"])
        _DATA[key] = d
    return _DATA[key]


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def san(o):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, (np.floating,)):
            return san(float(o))
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, dict):
            return {k: san(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [san(v) for v in o]
        return o
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(san(obj), fh)
    os.replace(tmp, path)


def write_csv(path, rows):
    """Union-of-keys CSV, descriptor columns first."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys, seen = [], set()
    for k in DESCRIPTORS + ["axis"]:
        if any(k in r for r in rows):
            keys.append(k); seen.add(k)
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(v, float) and not math.isfinite(v)) else v)
                        for k, v in r.items()})
    return path


def shard_filter(ids, shard):
    if not shard:
        return ids
    i, n = (int(x) for x in shard.split("/"))
    return [rid for j, rid in enumerate(sorted(ids)) if j % n == i]
