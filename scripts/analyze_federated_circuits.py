"""Regenerate RESULTS.md section 24 from banked histories and checkpoints. No training.

    venv/bin/python scripts/analyze_federated_circuits.py masking     # 24.1  T alone vs the model, from histories (setup D)
    venv/bin/python scripts/analyze_federated_circuits.py decompose   # 24.2-3  exact A / T / B ablations on D checkpoints
    venv/bin/python scripts/analyze_federated_circuits.py additive    # 24.4  least-squares single-operand split, C (and D as the check)
    venv/bin/python scripts/analyze_federated_circuits.py embedding   # 24.4  irrep profile of C's token embedding, global and per client
    venv/bin/python scripts/analyze_federated_circuits.py all [--json DIR]

THE QUESTION. When federation stalls a model that memorised, is the compositional
circuit absent, or present and overridden? Setup D answers it exactly: its logit
is A[c,a] + 2T[c,a,b] + B[c,b] (metrics/quadratic_circuits), and only T can
compose, so scoring T on its own is an ablation with nothing to fit. The
transformer gets the same ablation approximately through metrics/additive.

WHY THE CLIENT-SIDE PROBES ARE NOT USED. The histories carry client_circ_* series,
but GrokClient evaluates them on each client's own TRAINING shard, where T is
fit to 100% by construction; the global series are on the held-out pairs. The
two are not comparable, and the per-client snapshots hold only the first-operand
block, so the client story below is limited to what the global model shows.
"""
import argparse, csv, glob, json, os, re, statistics, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedgrok.manifest import build_config
from fedgrok.core.registry import build_model
from fedgrok.data.registry import build_dataset
from fedgrok.metrics import quadratic_circuits as qc, irreps
from fedgrok.metrics.additive import additive_report
from fedgrok.metrics.fourier import compute_accuracy

torch.set_grad_enabled(False)
CSV = "results/data/runs_v2.csv"
BAR_D = 85.0


def rows():
    return list(csv.DictReader(open(CSV)))


def sel(setup, **kw):
    return [r for r in rows() if r["setup"] == setup
            and all(str(r.get(k)) == str(v) for k, v in kw.items())]


def history(r):
    fs = glob.glob(f"results/runs/{r['id']}/history_*.json")
    return json.load(open(fs[0])) if fs else None


def ckpts(rid, kind="ckpt_"):
    fs = glob.glob(f"results/runs/{rid}/checkpoints/{kind}*.pt")
    return sorted((int(re.search(r"(\d+)\.pt", f).group(1)), f) for f in fs)


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def fmt(v, d=0):
    return "—" if v is None else f"{v:,.{d}f}"


# ── the cells every subcommand reads ─────────────────────────────────────────
D_CELLS = [
    ("centralised α 0.30",     dict(mode="centralized", group="aggregation", alpha="0.3")),
    ("iid K=2, E=5",           dict(mode="federated", group="aggregation", num_clients="2")),
    ("iid K=5, E=5",           dict(mode="federated", group="aggregation", num_clients="5")),
    ("iid K=10, E=5",          dict(mode="federated", group="aggregation", num_clients="10")),
    ("iid K=20, E=5",          dict(mode="federated", group="aggregation", num_clients="20")),
    ("iid K=50, E=5",          dict(mode="federated", group="aggregation", num_clients="50")),
    ("iid K=10, E=10",         dict(mode="federated", group="local_epochs", local_epochs="10")),
    ("iid K=10, E=25",         dict(mode="federated", group="local_epochs", local_epochs="25")),
    ("iid K=10, E=50",         dict(mode="federated", group="local_epochs", local_epochs="50")),
    ("iid K=10, E=50, 2M",     dict(mode="federated", group="e50_long")),
    ("Dirichlet 1.0, K=10",    dict(mode="federated", group="dirichlet_setups", dirichlet_alpha="1.0")),
    ("Dirichlet 1000, K=10",   dict(mode="federated", group="dirichlet_setups", dirichlet_alpha="1000.0")),
    ("coset K=5",              dict(mode="federated", group="partitions", partition="coset")),
    ("operand K=10",           dict(mode="federated", group="partitions", partition="operand", num_clients="10")),
    ("target K=10",            dict(mode="federated", group="partitions", partition="target", num_clients="10")),
]
C_CELLS = [
    ("iid K=5 (ladder, α 0.5)",  dict(group="setup_k_ladder", num_clients="5")),
    ("iid K=10 (ladder, α 0.5)", dict(group="setup_k_ladder", num_clients="10")),
    ("iid K=50 (ladder, α 0.5)", dict(group="setup_k_ladder", num_clients="50")),
    ("coset K=5",                dict(group="partitions", partition="coset")),
    ("Dirichlet K=10",           dict(group="partitions", partition="dirichlet", num_clients="10")),
    ("operand K=10",             dict(group="partitions", partition="operand", num_clients="10")),
    ("target K=10",              dict(group="partitions", partition="target", num_clients="10")),
    ("Dirichlet K=50",           dict(group="partitions", partition="dirichlet", num_clients="50")),
    ("operand K=50",             dict(group="partitions", partition="operand", num_clients="50")),
    ("target K=50",              dict(group="partitions", partition="target", num_clients="50")),
]


def masking():
    """24.1 -- first step T alone reaches the bar, against the model's first crossing."""
    print("24.1  setup D: step at which T alone (marginals stripped) first reaches 85% held-out "
          "accuracy, against the full model\n")
    print(f"{'cell':<24}{'n':>3}{'held':>6}{'T≥85':>6}{'t_T':>10}{'t_full':>10}"
          f"{'final test':>12}{'final T':>9}{'T share':>9}")
    out = {}
    for name, kw in D_CELLS:
        recs = []
        for r in sel("D", **kw):
            h = history(r)
            if not h or "circ_acc_interaction" not in h:
                continue
            steps = h.get("total_steps") or h.get("epoch")
            T, test = h["circ_acc_interaction"], h["test_acc"]
            first = lambda v: next((s for s, x in zip(steps, v) if x is not None and x >= BAR_D), None)
            recs.append(dict(tT=first(T), tF=first(test), test=test[-1], T=T[-1],
                             S=h["circ_share_interaction"][-1], held=r["grokked"] == "True"))
        if not recs:
            continue
        out[name] = recs
        print(f"{name:<24}{len(recs):>3}{sum(x['held'] for x in recs):>4}/{len(recs)}"
              f"{sum(x['tT'] is not None for x in recs):>4}/{len(recs)}"
              f"{fmt(med([x['tT'] for x in recs])):>10}{fmt(med([x['tF'] for x in recs])):>10}"
              f"{med([x['test'] for x in recs]):>12.1f}{med([x['T'] for x in recs]):>9.1f}"
              f"{med([x['S'] for x in recs]):>9.2f}")
    return out


_data = {}


def _test_set(cfg):
    key = (cfg.dataset, cfg.group_n, cfg.alpha, cfg.seed)
    if key not in _data:
        _, _, x, y = build_dataset(cfg)
        _data[key] = (x, y)
    return _data[key]


def _cenergy(t):
    c = t - t.mean(dim=1, keepdim=True)
    return float((c ** 2).sum(dim=1).mean())


def decompose():
    """24.2-24.3 -- exact A / T / B ablations on setup D's checkpoints."""
    out = {}
    for name, kw in D_CELLS:
        series = []
        for r in sel("D", **kw):
            cks = ckpts(r["id"])
            if not cks:
                continue
            cfg = build_config(json.load(open(f"results/runs/{r['id']}/spec.json")))
            model = build_model(cfg)
            x_test, y_test = _test_set(cfg)
            E = float(r["local_epochs"] or 0)
            pts = []
            for rnd, f in cks:
                model.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
                a, T, b = qc.decompose(model, x_test)
                full = a + T + b
                U, V = qc.operand_blocks(model)
                eF = _cenergy(full) or 1.0
                pts.append(dict(
                    step=rnd * E if r["mode"] == "federated" else rnd,
                    acc=compute_accuracy(full, y_test), accT=compute_accuracy(T, y_test),
                    acc_noA=compute_accuracy(T + b, y_test), acc_noB=compute_accuracy(a + T, y_test),
                    accM=compute_accuracy(a + b, y_test),
                    eA=_cenergy(a) / eF, eB=_cenergy(b) / eF, eT=_cenergy(T) / eF,
                    sU=irreps.structure_score(U), sV=irreps.structure_score(V),
                    units=qc.interaction_units(model)))
            series.append(dict(id=r["id"], seed=r["seed"], pts=pts))
        if series:
            out[name] = series

    print("\n24.2  setup D, FINAL checkpoint, medians over runs. Accuracy with a term removed; "
          "e_* = class-centred logit energy of the term / that of the full logit\n")
    print(f"{'cell':<24}{'n':>3}{'model':>7}{'T alone':>9}{'no A':>7}{'no B':>7}{'A+B':>6}"
          f"{'e_A':>7}{'e_B':>6}{'e_T':>6}{'s_U':>6}{'s_V':>6}{'units':>7}")
    for name, series in out.items():
        last = [s["pts"][-1] for s in series]
        g = lambda k: med([p[k] for p in last])
        print(f"{name:<24}{len(last):>3}{g('acc'):>7.1f}{g('accT'):>9.1f}{g('acc_noA'):>7.1f}"
              f"{g('acc_noB'):>7.1f}{g('accM'):>6.1f}{g('eA'):>7.2f}{g('eB'):>6.2f}{g('eT'):>6.2f}"
              f"{g('sU'):>6.2f}{g('sV'):>6.2f}{g('units'):>7.0f}")
    for name in ("iid K=10, E=50, 2M", "coset K=5"):
        if name not in out:
            continue
        print(f"\n24.3  {name}: trajectory, medians over runs")
        print(f"{'step':>12}{'model':>7}{'T alone':>9}{'no A':>7}{'no B':>7}{'e_A':>6}{'e_B':>6}{'e_T':>6}{'s_U':>6}")
        series = out[name]
        for i in range(min(len(s["pts"]) for s in series)):
            pts = [s["pts"][i] for s in series]
            g = lambda k: med([p[k] for p in pts])
            print(f"{g('step'):>12,.0f}{g('acc'):>7.1f}{g('accT'):>9.1f}{g('acc_noA'):>7.1f}"
                  f"{g('acc_noB'):>7.1f}{g('eA'):>6.2f}{g('eB'):>6.2f}{g('eT'):>6.2f}{g('sU'):>6.2f}")
    return out


def additive():
    """24.4 -- least-squares single-operand split at the final checkpoint; C, and D as the check."""
    cells = [("C " + n, "C", kw) for n, kw in C_CELLS] + [
        ("D iid K=10, E=5 (check)", "D", dict(mode="federated", group="aggregation", num_clients="10")),
        ("D coset K=5 (check)", "D", dict(mode="federated", group="partitions", partition="coset")),
    ]
    print("\n24.4  additive split L = mu + f[a] + g[b] + R[a,b] on the full grid; held-out accuracy "
          "at the final checkpoint, medians over runs\n")
    print(f"{'cell':<30}{'n':>3}{'held':>6}{'model':>7}{'R alone':>9}{'no f':>7}{'no g':>7}{'f+g':>6}"
          f"{'e_f':>7}{'e_g':>6}{'e_R':>6}")
    out = {}
    for name, setup, kw in cells:
        recs = []
        for r in sel(setup, **kw):
            cks = ckpts(r["id"])
            if not cks:
                continue
            cfg = build_config(json.load(open(f"results/runs/{r['id']}/spec.json")))
            model = build_model(cfg)
            model.load_state_dict(torch.load(cks[-1][1], map_location="cpu", weights_only=True))
            rep = additive_report(model, cfg)
            rep["held"] = r["grokked"] == "True"
            recs.append(rep)
        if not recs:
            continue
        out[name] = recs
        g = lambda k: med([x[k] for x in recs])
        print(f"{name:<30}{len(recs):>3}{sum(x['held'] for x in recs):>4}/{len(recs)}"
              f"{g('acc_full'):>7.1f}{g('acc_interaction'):>9.1f}{g('acc_no_a'):>7.1f}"
              f"{g('acc_no_b'):>7.1f}{g('acc_additive'):>6.1f}{g('share_a'):>7.2f}"
              f"{g('share_b'):>6.2f}{g('share_interaction'):>6.2f}")
    return out


def embedding():
    """24.4 -- irrep energy profile of C's token embedding W_E, global and per client."""
    names = irreps.IRREP_NAMES
    base = dict(zip(names, irreps.random_baseline_fractions(5)))
    out = {}
    for name, kw in C_CELLS:
        recs = []
        for r in sel("C", **kw):
            cks = ckpts(r["id"], "ckpt_round")
            if not cks:
                continue
            clients = dict(ckpts(r["id"], "client_w1_round"))
            rnd, f = cks[-1]
            sd = torch.load(f, map_location="cpu", weights_only=True)
            WE = sd["W_E"].T.float()
            rec = dict(structure=irreps.structure_score(WE), shares=irreps.fractions(WE),
                       held=r["grokked"] == "True")
            if rnd in clients:
                W = torch.as_tensor(np.stack(torch.load(clients[rnd], map_location="cpu",
                                                        weights_only=False))).float()
                dev = W - W.mean(0, keepdim=True)
                rec["client_dev"] = float((dev ** 2).sum() / (W ** 2).sum())
            recs.append(rec)
        if recs:
            out[name] = recs
    print("\n24.4  setup C, dominant irrep of W_E per run at the final checkpoint "
          "(a grokked C model composes in ONE irrep; iid leaves the choice to the seed, coset fixes [4,1])\n")
    print(f"{'cell':<26}{'run':>4}{'held':>5}{'struct':>8}   dominant (share)   second (share)")
    for name, recs in out.items():
        for i, x in enumerate(recs):
            top = sorted(x["shares"].items(), key=lambda kv: -kv[1])
            print(f"{name:<26}{i + 1:>4}{('yes' if x['held'] else 'no'):>5}{x['structure']:>8.3f}"
                  f"   [{top[0][0]}] {top[0][1]:.3f}        [{top[1][0]}] {top[1][1]:.3f}")
    print("\n24.4  setup C, irrep energy shares of W_E at the final checkpoint (medians over runs -- "
          "these need not sum to one when runs disagree on the irrep); "
          "'dev' = energy of client embeddings' deviation from their mean\n")
    print(f"{'cell':<26}{'held':>5}{'struct':>8}" + "".join(f"{n:>8}" for n in names) + f"{'dev':>7}")
    print(f"{'random':<26}{'':>5}{0.0:>8.3f}" + "".join(f"{base[n]:>8.3f}" for n in names))
    for name, recs in out.items():
        print(f"{name:<26}{sum(x['held'] for x in recs):>3}/{len(recs)}"
              f"{med([x['structure'] for x in recs]):>8.3f}"
              + "".join(f"{med([x['shares'][n] for x in recs]):>8.3f}" for n in names)
              + f"{med([x.get('client_dev') for x in recs]) if any('client_dev' in x for x in recs) else float('nan'):>7.3f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["masking", "decompose", "additive", "embedding", "all"])
    ap.add_argument("--json", help="directory to write each subcommand's raw numbers into")
    a = ap.parse_args()
    todo = ["masking", "decompose", "additive", "embedding"] if a.what == "all" else [a.what]
    for what in todo:
        res = globals()[what]()
        if a.json:
            os.makedirs(a.json, exist_ok=True)
            json.dump(res, open(os.path.join(a.json, f"{what}.json"), "w"))
