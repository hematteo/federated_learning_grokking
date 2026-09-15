"""Two figures on what federation does, measured against same-initialisation centralised twins.

    venv/bin/python scripts/plotting/federation_effect_figures.py
        -> figures/mechinterp/fed_effect_route.png/.pdf
           figures/mechinterp/fed_effect_destination.png/.pdf
           results/mechinterp/FEDERATION_EFFECT.md   (the numbers behind both)

Every federated run on an IID client-count ladder is paired with the centralised twin
trained from its exact initial weights (same seed, data fraction, decay, optimiser;
scripts/mechinterp/twins.py). Ratios are federated / twin for the same seed, medians
over seeds. Inputs: checkpoint_metrics.csv, twin_checkpoint_metrics.csv
(twin_metrics.py), twin_trajectories.csv and landscape_paths.csv.

ROUTE        (a) memorisation clock and (b) generalisation clock relative to the twin, vs K;
             (c-e) trajectories through (representation structure, test accuracy) for A, B
             and D, twin in black; (f) the structure at first crossing relative to the
             twin's at its own first crossing, vs K.
DESTINATION  at the end of training: (a) weight distance to the twin, (b) test accuracy along
             the straight line to the twin at low and high K, (c) whether the same circuit is
             used (clock key frequencies on B, dominant irrep on C and D), vs K.
"""
import collections
import csv
import importlib.util
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("mf", os.path.join(_HERE, "mechinterp_figures.py"))
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)
FD, num, q, fvals, SCOL, SLAB = mf.FD, mf.num, mf.q, mf.fvals, mf.SCOL, mf.SLAB

OUT_MD = os.path.join(mf.MI, "FEDERATION_EFFECT.md")
LINES = []
# The IID K ladder used for each setup here (B at wd 0.1, where its K ladder has checkpoints at crossing).
LADDER = {"A": [("0.3", "0.0"), ("0.25", "0.0")], "A'": [("0.2", "0.1")], "B": [("0.3", "0.1"), ("0.3", "1.0")],
          "C": [("0.4", "1.0"), ("0.5", "1.0")], "D": [("0.3", "1.0")], "E": [("0.5", "0.1")]}


def say(line):
    LINES.append(line)


def frac_key(r):
    return str(float(r["alpha"])) if r["dataset"] != "mnist" else "n" + str(r["n_train"])


def pair_key(r):
    return (r["setup"], frac_key(r), str(float(r["weight_decay"])), str(int(float(r["seed"]))))


def on_ladder(r, fams=None):
    if r["mode"] != "federated" or r["axis"] != "K" or r["partition"] != "iid" or r["id"] in mf.DIVERGED:
        return False
    fams = fams or LADDER[r["setup"]]
    return any(mf._eq(r["alpha"], a) and mf._eq(r["weight_decay"], wd) for a, wd in fams)


def load():
    fed_runs, fed_m = mf.ckpt()
    tw_rows = list(csv.DictReader(open(os.path.join(mf.MI, "twin_checkpoint_metrics.csv"))))
    tw_runs, tw_m = FD.run_moments_from_ckpts(tw_rows)
    twins = {pair_key(m["init"]): (tw_runs[rid], m) for rid, m in tw_m.items()}
    pairs = []
    for rid, m in fed_m.items():
        r = m["init"]
        if on_ladder(r) and pair_key(r) in twins:
            pairs.append((fed_runs[rid], m, *twins[pair_key(r)]))
    return pairs


def ratio_by_k(pairs, setup, fn, fams=None):
    """{K: [fn(fed, twin)]} for one setup's ladder(s)."""
    out = collections.defaultdict(list)
    for fr, fm, tr, tm in pairs:
        r = fm["init"]
        if r["setup"] == setup and (fams is None or on_ladder(r, fams)):
            v = fn(fr, fm, tr, tm)
            if v is not None:
                out[int(num(r["num_clients"]))].append(v)
    return out


def median_line(ax, byk, color, marker="o", label=None, ls="-", ms=4, cap=None):
    ks = sorted(byk)
    xs, ys = [], []
    for k in ks:
        v = np.array(byk[k], float)
        fin = v[np.isfinite(v)]
        for i, y in enumerate(v):
            y_plot = cap if (not np.isfinite(y) and cap) else y
            if np.isfinite(y_plot):
                ax.scatter(mf.jitter(k, i, len(v)), y_plot, s=10, color=color, alpha=0.45, lw=0,
                           marker="^" if not np.isfinite(y) else marker, zorder=3)
        if fin.size * 2 > v.size:                       # median defined only if most runs reached the event
            xs.append(k)
            ys.append(float(np.median(v)) if np.isfinite(np.median(v)) else float(np.median(fin)))
    ax.plot(xs, ys, color=color, marker=marker, ms=ms, lw=1.6, ls=ls, label=label, zorder=4,
            markeredgecolor="white", markeredgewidth=0.4)
    return dict(zip(xs, ys))


def event_ratio(key):
    def fn(fr, fm, tr, tm):
        a, b = num(fm["init"][key]), num(tm["init"][key])
        if not (math.isfinite(b) and b > 0):
            return None
        return a / b if math.isfinite(a) else math.inf
    return fn


def moment_ratio(metric, moment="cross"):
    def fn(fr, fm, tr, tm):
        if moment not in fm or moment not in tm:
            return None
        a, b = num(fm[moment][metric]), num(tm[moment][metric])
        return a / b if math.isfinite(a) and math.isfinite(b) and b != 0 else None
    return fn


# ═════════════════════════════════════════════════════════════════════════════

PORTRAIT = [("A", ("0.25", "0.0"), "unit_interaction_share", "hidden units: two-operand interaction share",
             "units combine the operands"),
            ("B", ("0.3", "0.1"), "w_WU_stable_rank", "unembedding W_U stable rank", "W_U rank"),
            ("D", ("0.3", "1.0"), "w_W2_stable_rank", "output layer W2 stable rank", "W2 rank")]


def route(pairs):
    fig = plt.figure(figsize=(11.5, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    say("## Figure 1: the route (federated / same-seed centralised twin)")
    say("")
    # (a) (b) clocks
    for j, (key, lab, L) in enumerate((("t_memo", "memorisation time  t_memo", "a"),
                                       ("t_first_cross", "generalisation time  t_first_cross", "b"))):
        ax = fig.add_subplot(gs[0, j])
        for s in mf.SETUPS:
            byk = ratio_by_k(pairs, s, event_ratio(key))
            if not byk:
                continue
            med = median_line(ax, byk, SCOL[s], label=SLAB[s], cap=150 if j == 1 else 700)
            say(f"- {s} `{key}` fed/twin by K: " + ", ".join(
                f"K{k} ×{fmt_ratio(np.median(byk[k]))} (reached {np.isfinite(byk[k]).sum()}/{len(byk[k])})"
                for k in sorted(byk)))
        ax.axhline(1, color="k", lw=0.7, ls=":")
        ax.set_yscale("log")
        ax.text(0.99, 0.99, "▲ = never reached within budget", transform=ax.transAxes, ha="right", va="top",
                fontsize=6, color="0.35")
        mf.kaxis(ax)
        ax.set_ylabel(f"{lab}\nfederated ÷ same-init centralised twin")
        ax.set_title(("(a) memorisation clock" if j == 0 else "(b) generalisation clock"), fontsize=8.5, loc="left")
        if j == 0:
            ax.legend(fontsize=6, frameon=False, loc="upper left")
    say("")
    # (c) distortion at crossing
    ax = fig.add_subplot(gs[0, 2])
    DIST = [("A", ("0.25", "0.0"), "unit_interaction_share", "A: two-operand unit share", "-"),
            ("A", ("0.25", "0.0"), "sharp_lr_lambda", "A: sharpness lr·λ", "--"),
            ("B", ("0.3", "0.1"), "w_WU_stable_rank", "B: W_U stable rank", "-"),
            ("B", ("0.3", "0.1"), "fout_dc_share", "B: W_U class-prior (DC) share", "--"),
            ("D", ("0.3", "1.0"), "w_W2_stable_rank", "D: W2 stable rank", "-"),
            ("D", ("0.3", "1.0"), "fn_loss_tr", "D: train loss", "--")]
    for s, fam, metric, lab, ls in DIST:
        byk = ratio_by_k(pairs, s, moment_ratio(metric), fams=[fam])
        if byk:
            median_line(ax, byk, SCOL[s], label=lab, ls=ls, marker="o" if ls == "-" else "s")
            say(f"- {s} (α {fam[0]}, wd {fam[1]}) `{metric}` at first crossing, fed/twin: " + ", ".join(
                f"K{k} ×{fmt_ratio(np.median(byk[k]))} (n={len(byk[k])})" for k in sorted(byk)))
    ax.axhline(1, color="k", lw=0.7, ls=":")
    ax.set_yscale("log")
    mf.kaxis(ax)
    ax.set_ylabel("value at first crossing\nfederated ÷ twin (at its own crossing)")
    ax.set_title("(c) the state in which the model generalises", fontsize=8.5, loc="left")
    ax.legend(fontsize=5.8, frameon=False, loc="upper left", ncol=1)
    say("")
    # (d-f) portraits
    for j, (s, fam, metric, lab, short) in enumerate(PORTRAIT):
        ax = fig.add_subplot(gs[1, j])
        seen_twin = set()
        for fr, fm, tr, tm in sorted(pairs, key=lambda p: -num(p[1]["init"]["num_clients"])):
            r = fm["init"]
            if r["setup"] != s or not on_ladder(r, [fam]):
                continue
            x, y = fvals(fr, metric), fvals(fr, "fn_acc_te")
            ok = np.isfinite(x) & np.isfinite(y)
            ax.plot(x[ok], y[ok], color=mf.kcolor(r["num_clients"]), lw=1.0, alpha=0.85, zorder=3)
            ax.scatter(x[ok][-1:], y[ok][-1:], color=mf.kcolor(r["num_clients"]), s=12, zorder=4)
            tid = tm["init"]["id"]
            if tid not in seen_twin:
                seen_twin.add(tid)
                xt, yt = fvals(tr, metric), fvals(tr, "fn_acc_te")
                okt = np.isfinite(xt) & np.isfinite(yt)
                ax.plot(xt[okt], yt[okt], color="k", lw=2.0, alpha=0.9, zorder=5)
            for mom, mk in (("cross", "*"),):
                if mom in fm:
                    ax.scatter(num(fm[mom][metric]), num(fm[mom]["fn_acc_te"]), marker=mk, s=40,
                               color=mf.kcolor(r["num_clients"]), edgecolor="k", lw=0.4, zorder=6)
        if metric != "unit_interaction_share":
            ax.set_xscale("log")
        ax.set_ylim(-3, 103)
        ax.set_xlabel(lab)
        ax.set_ylabel("test accuracy (%)")
        ax.set_title(f"({'def'[j]}) {SLAB[s]}" + (f", wd {fam[1]}" if s == "B" else "") +
                     ": route to generalisation", fontsize=8.5, loc="left", color=SCOL[s])
    ax.legend(handles=[Line2D([], [], color="k", lw=2, label="centralised twin (same init)"),
                       Line2D([], [], color="0.5", lw=1, label="federated, colour = K"),
                       Line2D([], [], marker="*", ls="", color="0.5", markeredgecolor="k", ms=8,
                              label="federated first crossing")], fontsize=6, frameon=False, loc="lower left")
    mf.k_colorbar(fig, fig.axes[3:])
    fig.suptitle("What federation does on the way: it stretches the clocks, and at high K the model reaches "
                 "generalisation by a different route (IID client ladders, each run vs its same-init centralised twin)",
                 fontsize=9)
    mf.save(fig, "fed_effect_route")


def fmt_ratio(v):
    return "∞" if not np.isfinite(v) else (f"{v:.2f}" if v < 10 else f"{v:.0f}")


# ═════════════════════════════════════════════════════════════════════════════

def destination(pairs):
    last = mf.twin_last()
    paths = collections.defaultdict(list)
    for r in mf.table("landscape_paths.csv"):
        if r["path"] == "fed_to_twin" and r["id"] not in mf.DIVERGED:
            paths[r["id"]].append(r)
    say("## Figure 2: the destination (end of training, federated vs twin)")
    say("")
    fig = plt.figure(figsize=(11.5, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 6)
    # (a) distance
    ax = fig.add_subplot(gs[0, 0:2])
    for s in mf.SETUPS:
        recs = [r for r in last.values() if on_ladder(r, LADDER[s][:1]) and r["setup"] == s]
        byk = collections.defaultdict(list)
        for r in recs:
            byk[int(num(r["num_clients"]))].append(num(r["tw_rel_dist"]))
        if byk:
            median_line(ax, byk, SCOL[s], label=SLAB[s])
            if s == "B":
                ax.plot([], [], " ", label="(B: wd 1.0 ladder in this figure)")
            say(f"- {s} relative distance to twin at end: " + ", ".join(
                f"K{k} {np.median(byk[k]):.2f}" for k in sorted(byk)))
    ax.set_yscale("log")
    mf.kaxis(ax)
    ax.set_ylabel("‖θ_FL − θ_twin‖ / ‖θ_twin‖")
    ax.set_title("(a) how far the federated model ends from its twin", fontsize=8.5, loc="left")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    # (b) CKA
    ax = fig.add_subplot(gs[0, 2:4])
    for s in mf.SETUPS:
        recs = [r for r in last.values() if on_ladder(r, LADDER[s][:1]) and r["setup"] == s]
        byk = collections.defaultdict(list)
        for r in recs:
            byk[int(num(r["num_clients"]))].append(num(r["tw_cka_te"]))
        if byk:
            median_line(ax, byk, SCOL[s])
            say(f"- {s} CKA with twin at end: " + ", ".join(f"K{k} {np.median(byk[k]):.2f}" for k in sorted(byk)))
    mf.kaxis(ax)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("linear CKA of held-out representations")
    ax.set_title("(b) do they represent the data the same way?", fontsize=8.5, loc="left")
    # (c) same circuit
    ax = fig.add_subplot(gs[0, 4:6])
    for s, metric, lab, mk in (("A'", "tw_clock_key_jaccard", "A′: clock key-frequency Jaccard", "o"),
                               ("B", "tw_clock_key_jaccard", "B: clock key-frequency Jaccard", "o"),
                               ("C", "tw_irrep_same_dominant", "C: same dominant irrep", "s"),
                               ("D", "tw_irrep_same_dominant", "D: same dominant irrep", "s")):
        recs = [r for r in last.values() if on_ladder(r, LADDER[s][:1] if s != "B" else [("0.3", "1.0")])
                and r["setup"] == s]
        byk = collections.defaultdict(list)
        for r in recs:
            byk[int(num(r["num_clients"]))].append(num(r[metric]))
        if byk:
            median_line(ax, {k: [np.mean(v)] if "same" in metric else v for k, v in byk.items()}, SCOL[s],
                        marker=mk, label=lab)
            say(f"- {s} `{metric}` at end: " + ", ".join(
                f"K{k} {np.mean(byk[k]) if 'same' in metric else np.median(byk[k]):.2f}" for k in sorted(byk)))
    mf.kaxis(ax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("overlap with the twin's circuit\n(fraction of runs for irreps)")
    ax.set_title("(c) do they use the same circuit?  (A omitted: its clock is spread\nevenly over all 48 frequencies, so overlap is uninformative)", fontsize=7.5, loc="left")
    ax.legend(fontsize=6, frameon=False, loc="lower left")

    say("")
    # (d-i) straight line to the twin, low vs high K per setup
    for j, s in enumerate(mf.SETUPS):
        ax = fig.add_subplot(gs[1, j])
        fam = LADDER[s][:1] if s != "B" else [("0.3", "1.0")]
        ks = sorted({int(num(r["num_clients"])) for r in last.values() if r["setup"] == s and on_ladder(r, fam)})
        for k, ls in ((ks[0], ":"), (ks[-1], "-")):
            curves = [sorted(v, key=lambda r: num(r["lambda"])) for rid, v in paths.items()
                      if v[0]["setup"] == s and on_ladder(v[0], fam) and int(num(v[0]["num_clients"])) == k]
            if not curves:
                continue
            lam = fvals(curves[0], "lambda")
            M = np.array([fvals(c, "acc_te") for c in curves if len(c) == len(lam)])
            ax.plot(lam, np.median(M, 0), color=mf.kcolor(k), lw=1.8, ls=ls, label=f"K{k}")
            ax.fill_between(lam, M.min(0), M.max(0), color=mf.kcolor(k), alpha=0.15, lw=0)
            say(f"- {s} K{k}: test acc along straight line fed → twin, min over λ (median over runs) "
                f"{np.median(M.min(1)):.1f}%, at λ = 0.5 {np.median(M[:, len(lam) // 2]):.1f}% (n = {len(M)})")
        ax.set_ylim(-3, 103)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["fed", "½", "twin"])
        ax.set_title(("(d) " if j == 0 else "") + s + (" (wd 1.0)" if s == "B" else ""), fontsize=8.5, color=SCOL[s], loc="left" if j == 0 else "center")
        if j == 0:
            ax.set_ylabel("test accuracy (%) on the\nstraight line federated → twin")
        else:
            ax.set_yticklabels([])
        ax.legend(fontsize=6, frameon=False, loc="lower center")
    fig.suptitle("Where federation ends up: GD (A) lands in its twin's basin and circuit; AdamW setups land in a "
                 "different solution, and the transformers in a different circuit, even at K = 2", fontsize=9)
    mf.save(fig, "fed_effect_destination")


def main():
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
    pairs = load()
    say("# What federation does, measured against same-initialisation centralised twins")
    say("")
    say(f"Generated by `scripts/plotting/federation_effect_figures.py`. {len(pairs)} federated runs on IID client "
        f"ladders, each paired with the centralised twin from its exact initial weights. Ratios: federated / twin, "
        f"same seed, median over seeds; ∞ = never reached within budget. Steps are each run's own step count.")
    say("")
    route(pairs)
    destination(pairs)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print("  wrote", OUT_MD)


if __name__ == "__main__":
    main()
