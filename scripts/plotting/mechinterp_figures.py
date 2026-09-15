"""Mechanistic-interpretability figures and the key-statistics bulletin.

    venv/bin/python scripts/plotting/mechinterp_figures.py            # all figures + bulletin
    venv/bin/python scripts/plotting/mechinterp_figures.py mi02 mi08  # a subset

Reads results/mechinterp/ (scripts/mechinterp/, see its README) and writes
figures/mechinterp/mi*.png/.pdf and results/mechinterp/BULLETIN.md. Every number in
the bulletin is recomputed here from the tables, not copied.

Moments: `memo` / `cross` are the checkpoints nearest t_memo / t_first_cross, kept only
within x0.67-x1.5 of the event (findings.run_moments_from_ckpts); `end` is the last
checkpoint. Twin comparisons at `end` use the last federated checkpoint that has a twin
checkpoint within 2.5% of its step. The six setup-A runs that diverged to NaN are
excluded everywhere except mi01.

  Diverged runs     mi01  starvation runs: memorise, then loss and norm blow up to NaN
  Twins             mi02  federated vs same-init centralised twin at the end, against K
                    mi03  distance to the twin and CKA through training
                    mi04  clock spectra of federated model and twin (A, A', B)
                    mi05  S5 irrep profiles of federated model and twin (C, D)
                    mi06  twin similarity by data partition
  Landscape         mi07  test accuracy / train loss along linear paths
                    mi08  path midpoints per setup
  High-K geometry   mi09  B: unembedding collapse at first crossing
                    mi10  D: output-layer collapse at first crossing
                    mi11  last-layer rank against K, all setups
                    mi12  singular-value spectra at first crossing, low vs high K
                    mi13  A: unit structure, sharpness and layer norms against K
                    mi14  A: the same against Dirichlet heterogeneity
  Delay             mi15  A: structure at memorisation predicts the delay
                    mi16  strongest memorisation-time predictors per setup
  Order             mi17  which order parameters move before first crossing
                    mi18  order parameters aligned on first crossing
                    mi19  D: the circuit forms long before the model generalises
  Spectra           mi20  clock-frequency energy through training (heatmaps)
                    mi21  S5 irrep energy through training
                    mi22  Fourier structure through training, coloured by K
  Weights           mi23  norms, Gini and rank through training, coloured by K
                    mi24  Gini coefficients at memo / cross / end against K
  Clients           mi25  client deviations: independent, zero-sum
"""
import collections
import csv
import glob
import importlib.util
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("pf", os.path.join(_HERE, "paper_figures.py"))
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)
sys.path.insert(0, os.path.join(_HERE, "..", "mechinterp"))
import common as C                                             # noqa: E402
import findings as FD                                          # noqa: E402

MI = C.OUT
OUT = "figures/mechinterp"
BULLETIN = os.path.join(MI, "BULLETIN.md")
SETUPS = ["A", "A'", "B", "C", "D", "E"]
SCOL = {**pf.COL, "A'": "#17becf"}
SLAB = {"A": "A  (MLP, Z97, GD)", "A'": "A′ (MLP, Z97, AdamW)", "B": "B  (transformer, Z113)",
        "C": "C  (transformer, S5)", "D": "D  (MLP, S5)", "E": "E  (MLP, MNIST)"}
DIVERGED = {"fede_addition_6d6f4e0bfdbf", "fede_addition_7fe78f36bfcf", "fede_addition_c3bbe3356ae0",
            "fede_addition_d1cb32e8055d", "fede_addition_e0d5064eab5c", "fede_addition_e30a92766b18"}
FIRST = {"A": "W1", "A'": "W1", "B": "WE", "C": "WE", "D": "W1", "E": "L0"}
LAST = {"A": "W2", "A'": "W2", "B": "WU", "C": "WU", "D": "W2", "E": "L2"}
# The IID K ladder per setup used for "against K" panels: (alpha, weight decay)
KFAM = {"A": ("0.3", "0.0"), "A'": ("0.2", "0.1"), "B": ("0.3", "1.0"), "C": ("0.4", "1.0"),
        "D": ("0.3", "1.0"), "E": ("0.5", "0.1")}
KTICKS = [2, 5, 10, 20, 50, 97]
STATS = collections.OrderedDict()
TREND = {}          # bulletin sections -> list of lines


def note(section, line):
    STATS.setdefault(section, []).append(line)


HEAD = []


def head(key, line):
    HEAD.append((key, line))


# ── data ─────────────────────────────────────────────────────────────────────

_CACHE = {}


def table(name):
    if name not in _CACHE:
        _CACHE[name] = list(csv.DictReader(open(os.path.join(MI, name))))
    return _CACHE[name]


num = C.fnum


def ckpt():
    """(runs: id -> sorted checkpoint records, moments: id -> {init, memo, cross, end})"""
    if "ck" not in _CACHE:
        rows = [r for r in table("checkpoint_metrics.csv") if r["id"] not in DIVERGED]
        _CACHE["ck"] = FD.run_moments_from_ckpts(rows)
    return _CACHE["ck"]


def twin_last():
    if "twl" not in _CACHE:
        last = {}
        for r in table("twin_trajectories.csv"):
            if r["id"] in DIVERGED:
                continue
            if r["id"] not in last or num(r["step"]) > num(last[r["id"]]["step"]):
                last[r["id"]] = r
        _CACHE["twl"] = last
    return _CACHE["twl"]


def match(r, setup=None, **kw):
    if setup is not None and r["setup"] != setup:
        return False
    for k, v in kw.items():
        vals = v if isinstance(v, (list, tuple, set)) else [v]
        if not any(_eq(r.get(k, ""), x) for x in vals):
            return False
    return True


def _eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def k_ladder(r, s):
    a, wd = KFAM[s]
    return (match(r, s, mode="federated", axis="K", partition="iid", alpha=a, weight_decay=wd)
            and not (s == "B" and r["group"] == "k_collapse_wd"))


def kcolor(K):
    if not math.isfinite(num(K)):           # centralised
        return (0.55, 0.55, 0.55, 1.0)
    return plt.cm.viridis(Normalize(math.log(2), math.log(97))(math.log(max(float(K), 2))) * 0.92)


def fvals(recs, key):
    return np.array([num(r.get(key, "")) for r in recs], float)


def q(v, p):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, p)) if v.size else math.nan


def spearman(x, y):
    return FD.spearman(x, y)


def fmt(v, d=2):
    if v is None or not isinstance(v, (int, float, np.floating)) or not math.isfinite(v):
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) < 0.01 and v != 0:
        return f"{v:.1e}"
    return f"{v:.{d}f}" if abs(v) < 100 else f"{v:.0f}"


# ── plotting helpers ─────────────────────────────────────────────────────────

def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png/.pdf")


def kaxis(ax, ks=KTICKS):
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("clients K")


def jitter(x, i, n, width=0.06):
    return x * math.exp((i - (n - 1) / 2) * width) if n > 1 else x


def strip_median(ax, xs, ys, color, marker="o", label=None, logx=True, lw=1.4, ms=3.2, alpha=0.55):
    """Per-run points (jittered) and the median per x as a line."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if not xs.size:
        return
    ux = np.unique(xs)
    for x in ux:
        sel = ys[xs == x]
        px = [jitter(x, i, len(sel)) if logx else x + (i - (len(sel) - 1) / 2) * 0.01 for i in range(len(sel))]
        ax.scatter(px, sel, s=ms ** 2, color=color, alpha=alpha, marker=marker, lw=0, zorder=3)
    med = [np.median(ys[xs == x]) for x in ux]
    ax.plot(ux, med, color=color, lw=lw, marker=marker, ms=ms + 0.8, label=label, zorder=4,
            markeredgecolor="white", markeredgewidth=0.4)


def letter(ax, L):
    ax.text(-0.13, 1.06, L, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def setup_legend(fig, setups, **kw):
    hs = [Line2D([], [], color=SCOL[s], marker="o", lw=1.4, ms=4, label=SLAB[s]) for s in setups]
    fig.legend(handles=hs, **{"loc": "lower center", "ncol": len(setups), "frameon": False,
                              "bbox_to_anchor": (0.5, -0.04), **kw})


def k_colorbar(fig, axes, label="clients K"):
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=LogNorm(2, 97 / 0.92))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_ticks(KTICKS)
    cb.set_ticklabels([str(k) for k in KTICKS])
    cb.minorticks_off()
    cb.set_label(label)
    return cb


def history_arrays(rid):
    h = C.history(rid)
    steps = np.array([num(x) for x in h.get("total_steps", h.get("round", []))], float)
    out = {"step": steps}
    for k, v in h.items():
        if isinstance(v, list) and len(v) == len(steps):
            out[k] = np.array([num(x) if x is not None else math.nan for x in v], float)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# mi01  diverged starvation runs
# ═════════════════════════════════════════════════════════════════════════════

def mi01():
    rows = C.load_rows()
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9), constrained_layout=True)
    cmap = plt.cm.autumn
    onsets = []
    for i, rid in enumerate(sorted(DIVERGED, key=lambda r: rows[r]["num_clients"] + rows[r]["partition"])):
        r = rows[rid]
        h = history_arrays(rid)
        st = h["step"]
        bad = ~np.isfinite(h["train_loss"]) | (h["train_loss"] > 1e3)
        j = int(np.argmax(bad)) if bad.any() else len(st) - 1
        onsets.append((rid, st[j], h["train_loss"][max(j - 1, 0)], np.nanmax(h["train_loss"][: j + 1]),
                       h["weight_norm_total"][max(j - 1, 0)], np.nanmax(h["weight_norm_total"][: j + 2])))
        col = cmap(i / 6 * 0.8)
        lab = f"K{r['num_clients']} {r['partition'].replace('dirichlet_sizes', 'size ctl').replace('dirichlet', 'Dir')} 0.01, s{r['seed']}"
        keep = st <= st[j] * 1.02
        for ax, key in zip(axs, ("train_acc", "train_loss", "weight_norm_total")):
            ax.plot(st[keep], h[key][keep], color=col, lw=1.1, label=lab)
            ax.scatter([st[j]], [np.nanmax(h[key][max(j - 1, 0): j + 1]) if key != "train_acc" else 0],
                       marker="X", s=36, color=col, zorder=5, edgecolor="k", lw=0.4)
    sibs = [r for r in rows.values() if r["group"] in ("dirichlet_band", "size_control") and r["id"] not in DIVERGED
            and match(r, "A", num_clients=["20", "50"])]
    for r in sibs[:24]:
        try:
            h = history_arrays(r["id"])
        except Exception:
            continue
        for ax, key in zip(axs, ("train_acc", "train_loss", "weight_norm_total")):
            ax.plot(h["step"], h[key], color="0.75", lw=0.6, zorder=1)
    axs[0].set_ylabel("train accuracy (%)")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("train loss (MSE)")
    axs[2].set_ylabel("total weight norm")
    for ax, L in zip(axs, "abc"):
        ax.set_xscale("log")
        ax.set_xlabel("steps")
        letter(ax, L)
    axs[0].legend(fontsize=6, loc="center left", frameon=False)
    axs[2].plot([], [], color="0.75", lw=0.6, label="non-diverged runs, same groups (K 20/50)")
    axs[2].plot([], [], "X", color="0.3", label="first non-finite / blow-up evaluation")
    axs[2].legend(fontsize=6, loc="upper left", frameon=False)
    fig.suptitle("Setup A starvation runs (Dirichlet 0.01) do not stall: they memorise, then diverge to NaN",
                 fontsize=9)
    save(fig, "mi01_starvation_divergence")
    s = "1. Starvation runs diverge (setup A)"
    steps = sorted(o[1] for o in onsets)
    note(s, f"Runs that diverged to NaN: **6** (Dirichlet α 0.01 or its shard-size control, K = 20/50); recorded "
            f"as censored (`grokked = False`) in the run table.")
    note(s, f"Blow-up step: {', '.join(fmt(x) for x in steps)}.")
    note(s, f"Train loss just before blow-up: median {fmt(q([o[2] for o in onsets], .5), 4)}; "
            f"peak logged: {fmt(max(o[3] for o in onsets))} (non-finite afterwards).")
    note(s, f"Total weight norm before → at blow-up: median {fmt(q([o[4] for o in onsets], .5))} → "
            f"{fmt(q([o[5] for o in onsets], .5))}.")
    runs = collections.defaultdict(list)
    for r in table("checkpoint_metrics.csv"):
        if r["setup"] == "A" and r["group"] in ("dirichlet_band", "size_control"):
            runs[r["id"]].append(r)
    lastd = [fvals(sorted(v, key=lambda r: num(r["step"])), "sharp_lr_lambda") for k, v in runs.items() if k in DIVERGED]
    lastd = [x[np.isfinite(x)][-1] for x in lastd if np.isfinite(x).any()]
    sib = [np.nanmax(fvals(v, "sharp_lr_lambda")) for k, v in runs.items() if k not in DIVERGED
           and match(v[0], alpha="0.25")]
    note(s, f"Global-model sharpness does not explain the blow-up: last finite lr·λ_max of the diverged runs "
            f"{fmt(min(lastd))}–{fmt(max(lastd))}, while non-diverged runs of the same α 0.25 groups peak at "
            f"{fmt(q(sib, .25))}–{fmt(max(sib))} (median {fmt(q(sib, .5))}) without diverging. Client-side (local-shard) "
            f"curvature was not measured.")
    head(9, f"**Six setup-A 'starvation' runs diverged to NaN; they were not censored.** Each memorised, "
            f"then blew up (at step {fmt(steps[0])}–{fmt(steps[-1])}). Their last finite global lr·λ_max "
            f"({fmt(min(lastd))}–{fmt(max(lastd))}) is below that of stable sibling runs (median {fmt(q(sib, .5))}), "
            f"so global sharpness does not explain it (mi01).")


# ═════════════════════════════════════════════════════════════════════════════
# twins
# ═════════════════════════════════════════════════════════════════════════════

def landscape_mid(path):
    out = {}
    for r in table("landscape_barriers.csv"):
        if r["path"] == path and r["id"] not in DIVERGED:
            out[r["id"]] = r
    return out


def mi02():
    last = twin_last()
    ftt = landscape_mid("fed_to_twin")
    panels = [("tw_rel_dist", "relative weight distance\n‖θ_FL − θ_C‖ / ‖θ_C‖", None),
              ("tw_cka_te", "linear CKA of representations", (-0.03, 1.03)),
              ("tw_pred_agree_te", "held-out predictions that agree", (0.4, 1.02)),
              ("circuit", "same circuit: clock key-frequency\nJaccard (Z_p) / irrep-profile cosine (S5)", (-0.03, 1.03)),
              ("tw_first_rel_dist", "relative distance, input layer", None),
              ("mid", "test acc. at midpoint of the\nstraight line federated → twin (%)", (-3, 103))]
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.0), constrained_layout=True)
    s_sec = "2. Federated vs same-initialisation centralised twin (end of training)"
    for s in SETUPS:
        recs = [r for r in last.values() if k_ladder(r, s)]
        if not recs:
            continue
        K = fvals(recs, "num_clients")
        for ax, (key, lab, lim) in zip(axs.flat, panels):
            if key == "circuit":
                y = fvals(recs, "tw_clock_key_jaccard") if s in ("A", "A'", "B") else fvals(recs, "tw_irrep_frac_cos")
                mk = "o" if s in ("A", "A'", "B") else "s"
            elif key == "mid":
                y = np.array([num(ftt[r["id"]]["test_acc_mid"]) if r["id"] in ftt else math.nan for r in recs])
                mk = "o"
            else:
                y, mk = fvals(recs, key), "o"
            strip_median(ax, K, y, SCOL[s], marker=mk)
        ks = sorted(set(K))
        lo = [r for r in recs if num(r["num_clients"]) == ks[0]]
        hi = [r for r in recs if num(r["num_clients"]) == ks[-1]]
        circ = "tw_clock_key_jaccard" if s in ("A", "A'", "B") else "tw_irrep_frac_cos"
        mids = lambda rs: q([num(ftt[r["id"]]["test_acc_mid"]) for r in rs if r["id"] in ftt], .5)
        same = ""
        if s in ("C", "D"):
            same = (f"; same dominant irrep K{int(ks[0])} {fmt(q(fvals(lo, 'tw_irrep_same_dominant'), .5))} → "
                    f"K{int(ks[-1])} {fmt(q(fvals(hi, 'tw_irrep_same_dominant'), .5))}")
        note(s_sec, f"**{s}** (α {KFAM[s][0]}, wd {KFAM[s][1]}, K {int(ks[0])} → {int(ks[-1])}, n = {len(recs)}): "
                    f"rel. distance {fmt(q(fvals(lo, 'tw_rel_dist'), .5))} → {fmt(q(fvals(hi, 'tw_rel_dist'), .5))}; "
                    f"CKA {fmt(q(fvals(lo, 'tw_cka_te'), .5))} → {fmt(q(fvals(hi, 'tw_cka_te'), .5))}; "
                    f"prediction agreement {fmt(q(fvals(lo, 'tw_pred_agree_te'), .5))} → {fmt(q(fvals(hi, 'tw_pred_agree_te'), .5))}"
                    + (f"; {'key-freq Jaccard' if s in ('A', 'A' + chr(39), 'B') else 'irrep cosine'} "
                       f"{fmt(q(fvals(lo, circ), .5))} → {fmt(q(fvals(hi, circ), .5))}" if s != "E" else "")
                    + same
                    + f"; midpoint test acc. to twin {fmt(mids(lo), 1)}% → {fmt(mids(hi), 1)}%.")
    A_ = [r for r in last.values() if k_ladder(r, "A")]
    tr = [r for r in last.values() if k_ladder(r, "B") or k_ladder(r, "C")]
    head(1, f"**GD federation stays in the centralised basin; AdamW federation does not.** Setup A vs its "
            f"same-init twin: relative distance {fmt(q(fvals([r for r in A_ if _eq(r['num_clients'], 2)], 'tw_rel_dist'), .5))} "
            f"(K2) → {fmt(q(fvals([r for r in A_ if _eq(r['num_clients'], 50)], 'tw_rel_dist'), .5))} (K50), test "
            f"accuracy at the midpoint of the straight line to the twin {fmt(q([num(ftt[r['id']]['test_acc_mid']) for r in A_ if r['id'] in ftt], .5), 1)}%. "
            f"Transformers B and C: midpoint {fmt(q([num(ftt[r['id']]['test_acc_mid']) for r in tr if r['id'] in ftt], .5), 1)}% "
            f"(n = {len(tr)}), CKA with the twin {fmt(q(fvals([r for r in tr if num(r['num_clients']) <= 2], 'tw_cka_te'), .5))} "
            f"at K2 and {fmt(q(fvals([r for r in tr if num(r['num_clients']) >= 20], 'tw_cka_te'), .5))} at K ≥ 20 (mi02, mi07).")
    for s in ("B", "D"):
        hi = [r for r in last.values() if k_ladder(r, s) and _eq(r["num_clients"], 50)]
        note(s_sec, f"Caveat, {s} at K50: {sum(math.isfinite(num(r['t_first_cross'])) for r in hi)} of {len(hi)} runs "
                    f"ever crossed the bar, so low prediction agreement there compares a non-generalised model with "
                    f"a generalised twin.")
    for ax, (key, lab, lim), L in zip(axs.flat, panels, "abcdef"):
        kaxis(ax)
        ax.set_ylabel(lab)
        if lim:
            ax.set_ylim(*lim)
        letter(ax, L)
    axs[0, 0].set_yscale("log")
    axs[1, 1].set_yscale("log")
    setup_legend(fig, SETUPS, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Same initial weights, trained federated vs centralised: GD (A) stays in the centralised basin; "
                 "AdamW setups end elsewhere", fontsize=9)
    save(fig, "mi02_twin_similarity_vs_K")


def mi03():
    tw = [r for r in table("twin_trajectories.csv") if r["id"] not in DIVERGED]
    fig, axs = plt.subplots(3, 6, figsize=(15, 7.0), sharex="col", constrained_layout=True)
    for j, s in enumerate(SETUPS):
        by = collections.defaultdict(list)
        for r in tw:
            if k_ladder(r, s):
                by[r["id"]].append(r)
        for rid, recs in sorted(by.items(), key=lambda kv: num(kv[1][0]["num_clients"])):
            recs.sort(key=lambda r: num(r["step"]))
            st = fvals(recs, "step")
            col = kcolor(recs[0]["num_clients"])
            for i, key in enumerate(("tw_rel_dist", "tw_cka_te", "tw_twin_acc_te")):
                y = fvals(recs, key)
                ax = axs[i, j]
                if key == "tw_twin_acc_te":
                    ax.plot(st, fvals(recs, "tw_fed_acc_te"), color=col, lw=0.9, alpha=0.8)
                    ax.plot(st, y, color="k", lw=0.6, ls=":", alpha=0.5)
                else:
                    ax.plot(st, y, color=col, lw=0.9, alpha=0.8)
            tc = num(recs[0]["t_first_cross"])
            if math.isfinite(tc):
                axs[0, j].axvline(tc, color=col, lw=0.5, ls="--", alpha=0.4)
        axs[0, j].set_title(SLAB[s], fontsize=8, color=SCOL[s])
        axs[0, j].set_yscale("log")
        axs[1, j].set_ylim(-0.03, 1.03)
        axs[2, j].set_ylim(-3, 103)
        axs[2, j].set_xlabel("steps")
        for i in range(3):
            axs[i, j].set_xscale("log")
    axs[0, 0].set_ylabel("rel. distance to twin")
    axs[1, 0].set_ylabel("CKA with twin")
    axs[2, 0].set_ylabel("test accuracy (%)\nfederated (colour), twin (dotted)")
    k_colorbar(fig, axs[:, -1])
    fig.suptitle("Distance to the same-initialisation centralised twin through training (dashed: each run's first "
                 "crossing)", fontsize=9)
    save(fig, "mi03_twin_trajectories")


def endpoint_spectra():
    if "tes" not in _CACHE:
        d = collections.defaultdict(lambda: collections.defaultdict(dict))
        meta = {}
        for r in table("twin_endpoint_spectra.csv"):
            d[r["id"]][r["model"]][r["component"]] = num(r["value"])
            meta[r["id"]] = r
        _CACHE["tes"] = (d, meta)
    return _CACHE["tes"]


def mi04():
    d, meta = endpoint_spectra()
    last = twin_last()
    cells = [("A", "50"), ("A'", "2"), ("A'", "50"), ("B", "2"), ("B", "20")]
    fig, axs = plt.subplots(len(cells), 3, figsize=(10.5, 9.4), constrained_layout=True)
    for i, (s, K) in enumerate(cells):
        rids = sorted([rid for rid, m in meta.items() if k_ladder(m, s) and _eq(m["num_clients"], K)],
                      key=lambda r: num(meta[r]["seed"]))[:3]
        for j in range(3):
            ax = axs[i, j]
            if j >= len(rids):
                ax.axis("off")
                continue
            rid = rids[j]
            fed, twin = d[rid]["fed"], d[rid]["twin"]
            w = np.array(sorted(int(k) for k in fed))
            ef = np.array([fed[str(k)] for k in w])
            et = np.array([twin[str(k)] for k in w])
            ax.bar(w, ef, width=0.9, color=SCOL[s], lw=0)
            ax.bar(w, -et, width=0.9, color="0.35", lw=0)
            ax.axhline(0, color="k", lw=0.5)
            m = max(ef.max(), et.max()) * 1.1
            ax.set_ylim(-m, m)
            ax.set_yticks([-m / 1.1, 0, m / 1.1])
            ax.set_yticklabels([f"{m / 1.1:.2f}", "0", f"{m / 1.1:.2f}"])
            jac = num(last[rid]["tw_clock_key_jaccard"]) if rid in last else math.nan
            ax.set_title(f"{s}, K = {K}, seed {meta[rid]['seed']} · step {fmt(num(meta[rid]['step']))} · "
                         f"key-freq Jaccard {fmt(jac)}", fontsize=7)
            if i == len(cells) - 1:
                ax.set_xlabel("frequency w")
            if j == 0:
                ax.set_ylabel("clock energy share")
    fig.suptitle("Which Fourier frequencies carry the logits' clock cos(w(a+b−c)): federated (colour, up) vs same-init "
                 "centralised twin (grey, down).\nOn A the clock is spread evenly over all 48 frequencies, so key-frequency overlap is uninformative there", fontsize=9)
    save(fig, "mi04_twin_clock_spectra")


IRREPS = ["5", "41", "32", "311", "221", "2111", "11111"]


def mi05():
    d, meta = endpoint_spectra()
    fig, axs = plt.subplots(1, 4, figsize=(10.5, 4.6), constrained_layout=True,
                            gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    for c, s in enumerate(("C", "D")):
        rids = sorted([rid for rid, m in meta.items() if k_ladder(m, s)],
                      key=lambda r: (num(meta[r]["num_clients"]), num(meta[r]["seed"])))
        F = np.array([[d[r]["fed"].get(x, math.nan) for x in IRREPS] for r in rids])
        T = np.array([[d[r]["twin"].get(x, math.nan) for x in IRREPS] for r in rids])
        labs = [f"K{meta[r]['num_clients']} s{meta[r]['seed']}" for r in rids]
        for k, (M, name) in enumerate(((F, "federated"), (T, "centralised twin"))):
            ax = axs[2 * c + k]
            im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=max(0.6, np.nanmax(F)))
            ax.set_xticks(range(len(IRREPS)))
            ax.set_xticklabels(IRREPS, rotation=60, fontsize=6.5)
            ax.set_yticks(range(len(labs)))
            ax.set_yticklabels(labs if k == 0 else [], fontsize=5.5)
            ax.set_title(f"{s}: {name}", fontsize=8, color=SCOL[s])
            ax.set_xlabel("irrep of S5 (input block)")
            for i in range(len(rids)):
                jm = int(np.nanargmax(M[i]))
                ax.plot(jm, i, "o", ms=2.2, color="cyan")
    last = twin_last()
    for s in ("C", "D"):
        for lo, hi_ in ((2, 10), (20, 50)):
            rs = [r for r in last.values() if k_ladder(r, s) and lo <= num(r["num_clients"]) <= hi_]
            v = fvals(rs, "tw_irrep_same_dominant")
            note("2. Federated vs same-initialisation centralised twin (end of training)",
                 f"**{s}** K{lo}–{hi_}: federated input block has the twin's dominant irrep in "
                 f"{100 * np.nanmean(v):.0f}% of {np.isfinite(v).sum()} runs.")
    cr = [r for r in last.values() if k_ladder(r, "C")]
    lo_ = fvals([r for r in cr if num(r["num_clients"]) <= 10], "tw_irrep_same_dominant")
    hi2 = fvals([r for r in cr if num(r["num_clients"]) >= 20], "tw_irrep_same_dominant")
    head(2, f"**Which S5 irrep the transformer (C) uses is set by the initialisation, until federation overrides "
            f"it.** Federated and same-init centralised models share the dominant irrep in {100 * np.nanmean(lo_):.0f}% "
            f"of runs at K ≤ 10 (three seeds, three different irreps) but {100 * np.nanmean(hi2):.0f}% at K ≥ 20 (mi05).")
    fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01, label="isotypic energy fraction")
    fig.suptitle("S5 setups: irrep energy of the input weights at the end, federated vs same-init twin "
                 "(dot: dominant irrep)", fontsize=9)
    save(fig, "mi05_twin_irreps")


PARTS = ["iid", "dirichlet", "operand", "target", "label_block", "coset"]
PARTLAB = {"iid": "IID", "dirichlet": "Dir", "operand": "op.", "target": "label", "label_block": "label",
           "coset": "coset"}


def mi06():
    last = twin_last()
    metrics = [("tw_rel_dist", "rel. distance to twin"), ("tw_cka_te", "CKA with twin"),
               ("circuit", "key-freq Jaccard (Z_p) / irrep cosine (S5)")]
    fig, axs = plt.subplots(3, 6, figsize=(15, 6.6), constrained_layout=True)
    s_sec = "6. Data partitions (twin similarity at end)"
    for j, s in enumerate(SETUPS):
        a, wd = KFAM[s]
        recs = [r for r in last.values() if r["setup"] == s and r["mode"] == "federated"
                and (r["axis"] == "partition" or (k_ladder(r, s)))]
        Ks = sorted({int(num(r["num_clients"])) for r in recs if r["axis"] == "partition"})
        xs, xl = [], []
        pos = 0
        for K in Ks:
            for p in PARTS:
                sel = [r for r in recs if int(num(r["num_clients"])) == K and r["partition"] == p
                       and (r["axis"] == "partition" or p == "iid")]
                if p == "iid":
                    sel = [r for r in sel if k_ladder(r, s)]
                if not sel:
                    continue
                for i, (key, lab) in enumerate(metrics):
                    if key == "circuit":
                        y = fvals(sel, "tw_clock_key_jaccard" if s in ("A", "A'", "B") else "tw_irrep_frac_cos")
                    else:
                        y = fvals(sel, key)
                    col = {"iid": "0.35", "label_block": pf.PART["target"]}.get(p) or pf.PART.get(p, "0.5")
                    axs[i, j].scatter(pos + np.linspace(-0.15, 0.15, len(y)), y, s=12, color=col, lw=0, zorder=3)
                    axs[i, j].plot([pos - 0.3, pos + 0.3], [np.nanmedian(y)] * 2, color=col, lw=1.6)
                if p in ("target", "label_block", "operand"):
                    note(s_sec, f"**{s}** K{K} {PARTLAB[p]}: rel. distance {fmt(q(fvals(sel, 'tw_rel_dist'), .5))}, "
                                f"CKA {fmt(q(fvals(sel, 'tw_cka_te'), .5))} (n = {len(sel)}).")
                xs.append(pos)
                xl.append(f"{PARTLAB[p]}\nK{K}")
                pos += 1
            pos += 0.6
        for i in range(3):
            axs[i, j].set_xticks(xs)
            axs[i, j].set_xticklabels(xl if i == 2 else [], fontsize=5.5, rotation=90)
            axs[i, j].grid(axis="x", visible=False)
        axs[0, j].set_title(SLAB[s], fontsize=8, color=SCOL[s])
        axs[1, j].set_ylim(-0.03, 1.03)
        axs[2, j].set_ylim(-0.03, 1.03)
    for i, (key, lab) in enumerate(metrics):
        axs[i, 0].set_ylabel(lab)
    axs[2, 5].text(0.5, 0.5, "no circuit\nmeasure (MNIST)", transform=axs[2, 5].transAxes, ha="center",
                   va="center", fontsize=7, color="0.5")
    fig.suptitle("Twin similarity at the end by data partition: incoherent (label) shards move GD furthest from "
                 "the centralised solution", fontsize=9)
    save(fig, "mi06_twin_partitions")


# ═════════════════════════════════════════════════════════════════════════════
# landscape
# ═════════════════════════════════════════════════════════════════════════════

PATHS = [("init_to_end", "initialisation → end"), ("memo_to_end", "memorisation → end"),
         ("fed_to_twin", "federated → centralised twin")]


def mi07():
    paths = collections.defaultdict(list)
    for r in table("landscape_paths.csv"):
        if r["mode"] == "federated" and r["id"] not in DIVERGED:
            paths[(r["setup"], r["path"], r["id"])].append(r)
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.6), constrained_layout=True, sharex=True)
    for j, (p, plab) in enumerate(PATHS):
        for s in SETUPS:
            curves = [sorted(v, key=lambda r: num(r["lambda"])) for (ss, pp, _), v in paths.items() if ss == s and pp == p]
            if not curves:
                continue
            lam = fvals(curves[0], "lambda")
            for i, key in enumerate(("acc_te", "loss_tr")):
                M = np.array([fvals(c, key) for c in curves if len(c) == len(lam)])
                med = np.nanmedian(M, 0)
                axs[i, j].plot(lam, med, color=SCOL[s], lw=1.5, label=f"{s} (n={len(M)})")
                axs[i, j].fill_between(lam, np.nanquantile(M, .25, 0), np.nanquantile(M, .75, 0), color=SCOL[s],
                                       alpha=0.15, lw=0)
        axs[0, j].set_title(plab, fontsize=8)
        axs[0, j].set_ylim(-3, 103)
        axs[1, j].set_yscale("log")
        axs[1, j].set_xlabel("interpolation λ  (0 = start, 1 = end)")
        axs[0, j].legend(fontsize=6, frameon=False, loc="lower left" if j else "upper left")
        letter(axs[0, j], "abc"[j])
    axs[0, 0].set_ylabel("test accuracy (%)")
    axs[1, 0].set_ylabel("train loss")
    fig.suptitle("Linear interpolation, federated runs (median, IQR): MLP solutions are linearly connected, "
                 "transformer solutions are not", fontsize=9)
    save(fig, "mi07_linear_paths")


def mi08():
    bars = collections.defaultdict(list)
    for r in table("landscape_barriers.csv"):
        if r["mode"] == "federated" and r["id"] not in DIVERGED:
            bars[(r["setup"], r["path"])].append(r)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0), constrained_layout=True, sharey=True)
    s_sec = "5. Linear connectivity (federated runs, test accuracy at the path midpoint)"
    for j, (p, plab) in enumerate(PATHS):
        ax = axs[j]
        line = []
        for i, s in enumerate(SETUPS):
            v = fvals(bars.get((s, p), []), "test_acc_mid")
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            ax.scatter(i + np.random.default_rng(0).uniform(-0.25, 0.25, v.size), v, s=7, color=SCOL[s], alpha=0.5,
                       lw=0)
            ax.plot([i - 0.32, i + 0.32], [np.median(v)] * 2, color="k", lw=1.6)
            ax.text(i, 104, f"{np.median(v):.0f}", ha="center", fontsize=6.5)
            line.append(f"{s} {np.median(v):.1f}% (n={v.size})")
        note(s_sec, f"{plab}: " + "; ".join(line) + ".")
        if p == "memo_to_end":
            head(6, f"**Memorising and generalising weights are linearly connected on MLPs, not transformers.** Test "
                    f"accuracy at the midpoint of memorisation → end: " + "; ".join(line) + " (mi07, mi08).")
        ax.set_xticks(range(len(SETUPS)))
        ax.set_xticklabels(SETUPS)
        ax.set_ylim(-3, 110)
        ax.set_title(plab, fontsize=8)
        ax.grid(axis="x", visible=False)
        letter(ax, "abc"[j])
    axs[0].set_ylabel("test accuracy at λ = 0.5 (%)")
    fig.suptitle("Midpoint of each linear path (black: median, number above)", fontsize=9)
    save(fig, "mi08_path_midpoints")


# ═════════════════════════════════════════════════════════════════════════════
# high-K geometry
# ═════════════════════════════════════════════════════════════════════════════

def at_moment(pred, moment):
    runs, moments = ckpt()
    return [m[moment] for rid, m in moments.items() if moment in m and pred(m["init"])]


def trend_panels(name, title, families, metrics, section, logy=(), ncols=3, moment="cross", xkey="num_clients",
                 xaxis=None, xlabel=None):
    """families: [(label, predicate, colour, marker)]; metrics: [(column, ylabel)]"""
    n = len(metrics)
    rows_ = math.ceil(n / ncols)
    fig, axs = plt.subplots(rows_, ncols, figsize=(3.5 * ncols, 2.7 * rows_), constrained_layout=True, squeeze=False)
    for (flab, pred, col, mk) in families:
        recs = at_moment(pred, moment)
        if not recs:
            continue
        x = fvals(recs, xkey)
        if xaxis == "dir":
            x = x.copy()
        for ax, (key, lab) in zip(axs.flat, metrics):
            y = fvals(recs, key)
            strip_median(ax, x, y, col, marker=mk, label=flab)
            rho, p, nn = spearman(np.log(x) if xaxis != "lin" else x, y)
            xs = sorted(set(x[np.isfinite(y)]))
            if len(xs) >= 2:
                lo = q(y[x == xs[0]], .5)
                hi = q(y[x == xs[-1]], .5)
                xl = "K" if xkey == "num_clients" else "α_dir " if xkey == "dirichlet_alpha" else xkey
                fx = (lambda v: f"{int(v)}") if xkey == "num_clients" else (lambda v: f"{v:g}")
                TREND[(flab, key)] = (xs[0], xs[-1], lo, hi, rho, nn)
                note(section, f"{flab} · `{key}` at {moment}: {xl}{fx(xs[0])} → {xl}{fx(xs[-1])}: "
                              f"{fmt(lo, 3)} → {fmt(hi, 3)} (Spearman ρ = {fmt(rho)}, n = {nn}).")
    for ax, (key, lab), L in zip(axs.flat, metrics, "abcdefghijkl"):
        if xaxis == "dir":
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_xlabel(xlabel or "Dirichlet α_dir (IID ← → non-IID)")
        else:
            kaxis(ax)
        ax.set_ylabel(lab)
        if key in logy:
            ax.set_yscale("log")
        letter(ax, L)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=6.5, frameon=False)
    fig.suptitle(title, fontsize=9)
    save(fig, name)


def mi09():
    fams = [("B, α 0.3, wd 0.1", lambda r: match(r, "B", mode="federated", axis="K", partition="iid", alpha="0.3",
                                                  weight_decay="0.1"), SCOL["B"], "o"),
            ("B, α 0.3, wd 1.0", lambda r: k_ladder(r, "B"), "#8c4a00", "s"),
            ("B, α 0.4, wd 1.0", lambda r: match(r, "B", mode="federated", axis="K", partition="iid", alpha="0.4",
                                                  weight_decay="1.0"), "#f5b041", "^")]
    metrics = [("w_WU_eff_rank", "unembedding W_U effective rank"), ("w_WU_stable_rank", "W_U stable rank"),
               ("w_WU_fro", "W_U Frobenius norm"), ("fout_dc_share", "W_U Fourier power at DC (class prior)"),
               ("fn_loss_tr", "train loss"), ("attn_to_a_mean", "attention to first operand")]
    _mi09(fams, metrics)


def _mi09(fams, metrics):
    trend_panels("mi09_B_unembedding_collapse", "Setup B at first crossing: high-K transformers generalise with an "
                 "unembedding collapsed onto the class-prior direction", fams, metrics,
                 "3. High-K geometry at first crossing", logy=("w_WU_eff_rank", "w_WU_stable_rank", "fn_loss_tr"))
    a, b = TREND[("B, α 0.3, wd 0.1", "w_WU_eff_rank")], TREND[("B, α 0.3, wd 0.1", "fout_dc_share")]
    d, e = TREND[("D, α 0.3", "w_W2_stable_rank")] if ("D, α 0.3", "w_W2_stable_rank") in TREND else (0,) * 6, None
    head(3, f"**High-K transformers generalise with a collapsed unembedding.** Setup B (wd 0.1) at first crossing, "
            f"K{int(a[0])} → K{int(a[1])}: W_U effective rank {fmt(a[2], 1)} → {fmt(a[3], 1)} (ρ = {fmt(a[4])}), share of "
            f"its Fourier power at DC (class prior) {fmt(b[2])} → {fmt(b[3])} (mi09).")


def mi10():
    fams = [("D, α 0.3", lambda r: k_ladder(r, "D"), SCOL["D"], "o")]
    metrics = [("w_W2_spec", "W2 spectral norm"), ("w_W2_stable_rank", "W2 stable rank"),
               ("rep_pr_te", "representation participation ratio"), ("fn_loss_tr", "train loss"),
               ("sharp_lambda_max", "sharpness λ_max"), ("w_W1_stable_rank", "W1 stable rank")]
    _mi10(fams, metrics)


def _mi10(fams, metrics):
    trend_panels("mi10_D_output_collapse", "Setup D at first crossing: the output layer collapses to rank ~2 while "
                 "train loss and sharpness rise", fams, metrics, "3. High-K geometry at first crossing",
                 logy=("w_W2_stable_rank", "fn_loss_tr", "sharp_lambda_max"))
    a, b = TREND[("D, α 0.3", "w_W2_stable_rank")], TREND[("D, α 0.3", "fn_loss_tr")]
    head(4, f"**The S5 MLP (D) does the same in its output layer.** At first crossing, K{int(a[0])} → K{int(a[1])}: "
            f"W2 stable rank {fmt(a[2], 1)} → {fmt(a[3], 1)} (ρ = {fmt(a[4])}) while train loss at crossing rises "
            f"{fmt(b[2])} → {fmt(b[3])}; no K50 run crossed (mi10).")


def mi11():
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1), constrained_layout=True)
    for s in SETUPS:
        recs = at_moment(lambda r: k_ladder(r, s), "cross")
        if not recs:
            continue
        K = fvals(recs, "num_clients")
        for ax, key in zip(axs, (f"w_{LAST[s]}_stable_rank", f"w_{LAST[s]}_eff_rank", f"w_{FIRST[s]}_eff_rank")):
            y = fvals(recs, key)
            base = q(y[K == np.nanmin(K[np.isfinite(y)])], .5) if np.isfinite(y).any() else math.nan
            strip_median(ax, K, y / base, SCOL[s])
    for ax, lab, L in zip(axs, ("last layer: stable rank", "last layer: effective rank", "first layer: effective rank"),
                          "abc"):
        kaxis(ax)
        ax.set_yscale("log")
        ax.axhline(1, color="0.5", lw=0.6, ls=":")
        ax.set_ylabel(lab + "\n(relative to the lowest K)")
        letter(ax, L)
    setup_legend(fig, SETUPS, bbox_to_anchor=(0.5, -0.12), ncol=6)
    fig.suptitle("Rank of each model's layers at first crossing, against K (IID ladders)", fontsize=9)
    save(fig, "mi11_layer_rank_vs_K")


def spectra_npz(rid):
    f = os.path.join(MI, "spectra", f"{rid}.npz")
    return np.load(f, allow_pickle=False) if os.path.exists(f) else None


def nearest_idx(steps, t):
    return int(np.argmin(np.abs(np.log(steps.clip(1)) - math.log(max(t, 1)))))


def mi12():
    runs, moments = ckpt()
    rows = C.load_rows()
    cells = [("A", "W1", "W2"), ("B", "WE", "WU"), ("D", "W1", "W2"), ("C", "WE", "WU")]
    fig, axs = plt.subplots(2, 4, figsize=(12, 5.4), constrained_layout=True)
    for j, (s, mf, ml) in enumerate(cells):
        for rid, m in moments.items():
            r = m["init"]
            if not (match(r, "B", mode="federated", axis="K", partition="iid", alpha="0.3", weight_decay="0.1")
                    if s == "B" else k_ladder(r, s)):
                continue
            z = spectra_npz(rid)
            if z is None:
                continue
            tc = num(r["t_first_cross"])
            t = tc if math.isfinite(tc) else num(z["steps"][-1])
            i = nearest_idx(z["steps"], t)
            for row, M in enumerate((mf, ml)):
                sv = z[f"sv_{M}"][i]
                sv = sv / sv.sum()
                axs[row, j].plot(np.arange(1, sv.size + 1), sv, color=kcolor(r["num_clients"]), lw=0.9, alpha=0.85,
                                 ls="-" if math.isfinite(tc) else "--")
        for row, M in enumerate((mf, ml)):
            ax = axs[row, j]
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{s}{' (wd 0.1)' if s == 'B' else ''}: {M} ({'first' if row == 0 else 'last'} layer)", fontsize=8, color=SCOL[s])
            ax.set_xlabel("singular value index")
        axs[0, j].set_ylim(bottom=1e-4)
        axs[1, j].set_ylim(bottom=1e-4)
    axs[0, 0].set_ylabel("σ_i / Σσ")
    axs[1, 0].set_ylabel("σ_i / Σσ")
    k_colorbar(fig, axs[:, -1])
    fig.suptitle("Singular-value spectra at first crossing (dashed: never crossed, last checkpoint), coloured by K",
                 fontsize=9)
    save(fig, "mi12_sv_spectra_at_cross")


def mi13():
    fams = [("A, α 0.30 (K 2–50)", lambda r: k_ladder(r, "A"), SCOL["A"], "o"),
            ("A, α 0.25 (K 20–97)", lambda r: match(r, "A", mode="federated", axis="K", partition="iid", alpha="0.25"),
             "#0b3d91", "s"),
            ("A, α 0.25, operand, K 97", lambda r: match(r, "A", mode="federated", axis="K", partition="operand",
                                                          alpha="0.25"), "#e67e22", "D")]
    metrics = [("unit_fourier_marginal_share", "hidden units: single-operand\nFourier share"),
               ("unit_interaction_share", "hidden units: two-operand\ninteraction share"),
               ("fin_uv_agree_strong", "strong neurons whose U and V\nblocks share a frequency"),
               ("sharp_lr_lambda", "lr · λ_max  (GD edge = 2)"),
               ("w_W1_fro", "‖W1‖_F"), ("w_W2_fro", "‖W2‖_F"),
               ("rep_nc1_tr", "NC1 (within / between variability)"), ("clock_nfreq90", "frequencies holding 90%\nof clock energy"),
               ("unit_sum_freq_concentration", "unit sum-frequency concentration")]
    _mi13(fams, metrics)


def _mi13(fams, metrics):
    trend_panels("mi13_A_structure_vs_K", "Setup A at first crossing against K: units lose two-operand structure, "
                 "sharpness rises, W1 grows and W2 shrinks", fams, metrics, "3. High-K geometry at first crossing",
                 logy=("rep_nc1_tr",))
    a = TREND[("A, α 0.25 (K 20–97)", "unit_interaction_share")]
    b = TREND[("A, α 0.25 (K 20–97)", "rep_nc1_tr")]
    head(7, f"**Many-client GD reaches the bar with units that barely combine the operands.** Setup A, α 0.25, at "
            f"first crossing K{int(a[0])} → K{int(a[1])}: two-operand interaction share of hidden-unit variance "
            f"{fmt(a[2])} → {fmt(a[3])}, NC1 {fmt(b[2])} → {fmt(b[3])} (mi13).")


def mi14():
    fams = [("A, α 0.30, K 10", lambda r: match(r, "A", mode="federated", axis="dirichlet", alpha="0.3",
                                                   partition="dirichlet"), SCOL["A"], "o"),
            ("A, α 0.25, K 20", lambda r: match(r, "A", mode="federated", axis="dirichlet", alpha="0.25",
                                                   num_clients="20", partition="dirichlet"), "#0b3d91", "s"),
            ("A, α 0.25, K 50", lambda r: match(r, "A", mode="federated", axis="dirichlet", alpha="0.25",
                                                   num_clients="50", partition="dirichlet"), "#6c3483", "^")]
    metrics = [("sharp_lr_lambda", "lr · λ_max"), ("unit_fourier_marginal_share", "single-operand Fourier share"),
               ("unit_interaction_share", "two-operand interaction share"), ("w_W1_fro", "‖W1‖_F"),
               ("fn_loss_tr", "train loss"), ("fin_uv_agree_strong", "U/V frequency agreement")]
    trend_panels("mi14_A_structure_vs_dirichlet", "Setup A at first crossing against label skew (Dirichlet): the "
                 "same signatures as more clients", fams, metrics, "4. Heterogeneity at first crossing (setup A)",
                 xkey="dirichlet_alpha", xaxis="dir", logy=("fn_loss_tr",))


# ═════════════════════════════════════════════════════════════════════════════
# delay predictors
# ═════════════════════════════════════════════════════════════════════════════

def roc(pos, neg):
    thr = np.unique(np.concatenate([pos, neg]))[::-1]
    tpr = [np.mean(pos >= t) for t in thr]
    fpr = [np.mean(neg >= t) for t in thr]
    return np.r_[0, fpr, 1], np.r_[0, tpr, 1]


def mi15():
    runs, moments = ckpt()
    metrics = [("unit_sum_freq_concentration", "unit sum-frequency concentration"),
               ("unit_fourier_sum_share", "unit Fourier (a+b) sum share"),
               ("rep_between_within_te", "held-out between/within class variance")]
    ms = [m for m in moments.values() if m["init"]["setup"] == "A" and "memo" in m]
    crossed = [m for m in ms if math.isfinite(num(m["init"]["t_first_cross"]))
               and num(m["init"]["t_first_cross"]) > num(m["init"]["t_memo"])]
    censored = [m for m in ms if not math.isfinite(num(m["init"]["t_first_cross"]))]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.3), constrained_layout=True)
    s_sec = "7. What predicts the delay at memorisation (setup A)"
    note(s_sec, f"Runs with a checkpoint at memorisation: {len(ms)} ({len(crossed)} crossed, {len(censored)} never "
                f"crossed; the 6 diverged runs and runs crossing no later than memorising excluded).")
    alpha_mark = {"0.3": "o", "0.25": "s"}
    for ax, (key, lab) in zip(axs, metrics):
        x = np.array([num(m["memo"][key]) for m in crossed])
        dly = np.array([num(m["init"]["t_first_cross"]) - num(m["init"]["t_memo"]) for m in crossed])
        K = [m["init"]["num_clients"] for m in crossed]
        for a, mk in alpha_mark.items():
            sel = np.array([_eq(m["init"]["alpha"], a) for m in crossed])
            ax.scatter(x[sel], dly[sel], c=[kcolor(k) for k, s_ in zip(K, sel) if s_], marker=mk, s=18,
                       edgecolor="k", lw=0.25, zorder=3)
        xc = np.array([num(m["memo"][key]) for m in censored])
        yc = np.array([num(m["init"]["steps_run"]) - num(m["init"]["t_memo"]) for m in censored])
        ax.scatter(xc, yc, marker="x", c=[kcolor(m["init"]["num_clients"]) for m in censored], s=26, lw=1.1, zorder=3)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel(lab + " at t_memo (log)")
        rho, p, n = spearman(x, np.log(dly))
        # within-condition: residuals from each cell's median (cells with >= 2 crossed runs)
        cells = collections.defaultdict(list)
        for m, xi, di in zip(crossed, x, dly):
            if math.isfinite(xi):
                cells[FD.cell(m["init"])].append((xi, math.log(di)))
        rx, ry = [], []
        for v in cells.values():
            if len(v) >= 2:
                mx, my = np.median([a for a, _ in v]), np.median([b for _, b in v])
                rx += [a - mx for a, _ in v]
                ry += [b - my for _, b in v]
        rwc, _, nwc = spearman(rx, ry)
        au = FD.auc(x, xc)
        within = [m for m in crossed if k_ladder(m["init"], "A")]
        rw, _, nw = spearman([num(m["memo"][key]) for m in within],
                             [math.log(num(m["init"]["t_first_cross"]) - num(m["init"]["t_memo"])) for m in within])
        ax.set_title(f"ρ = {rho:.2f} (n={n}), AUC = {max(au, 1 - au):.2f}\nwithin condition ρ = {rwc:.2f} (n={nwc})",
                     fontsize=7)
        note(s_sec, f"`{key}`: Spearman ρ with log delay = {fmt(rho)} (n = {n}); AUC crossed vs never = "
                    f"{fmt(1 - au if au < 0.5 else au)} ({len(crossed)} vs {len(censored)}); within the α 0.30 IID K "
                    f"ladder ρ = {fmt(rw)} (n = {nw}; K varies along it); **within condition** (deviations from each cell's median, "
                    f"{sum(len(v) >= 2 for v in cells.values())} cells) ρ = {fmt(rwc)} (n = {nwc}).")
        if key == "unit_sum_freq_concentration":
            head(5, f"**On A the delay is largely fixed at memorisation.** Sum-frequency concentration of the hidden "
                    f"units at t_memo vs log delay: ρ = {fmt(rho)} (n = {n}); separates runs that never cross, AUC "
                    f"{fmt(max(au, 1 - au))}; within a condition (deviation from cell median) ρ = {fmt(rwc)} "
                    f"(n = {nwc}) (mi15).")
        pos = np.array([num(m["memo"][key]) for m in crossed])
        pos, neg = pos[np.isfinite(pos)], xc[np.isfinite(xc)]
        if au < 0.5:
            pos, neg = -pos, -neg
        fpr, tpr = roc(pos, neg)
        axs[3].plot(fpr, tpr, lw=1.3, label=f"{lab} ({max(au, 1 - au):.2f})")
    axs[0].set_ylabel("delay  t_first_cross − t_memo (steps)\n× = never crossed (budget − t_memo)")
    axs[3].plot([0, 1], [0, 1], color="0.6", lw=0.6, ls=":")
    axs[3].set_xlabel("false positive rate")
    axs[3].set_ylabel("true positive rate (crossed)")
    axs[3].legend(fontsize=5.8, frameon=False, loc="lower right")
    axs[3].set_title("crossed vs never crossed", fontsize=7.5)
    k_colorbar(fig, axs[:3])
    axs[1].legend(handles=[Line2D([], [], marker="o", ls="", color="0.4", label="α 0.30"),
                           Line2D([], [], marker="s", ls="", color="0.4", label="α 0.25")], fontsize=6, frameon=False)
    fig.suptitle("Setup A: sum-frequency structure in the hidden units at memorisation separates slow from fast "
                 "conditions (colour: K; square: α 0.25), and still ranks runs within a condition", fontsize=9)
    save(fig, "mi15_A_delay_predictors")


FAMCOL = {"w_": "#7f8c8d", "rep": "#8e44ad", "uni": "#2980b9", "add": "#16a085", "cir": "#27ae60", "fin": "#d35400",
          "fU_": "#d35400", "fV_": "#d35400", "fou": "#d35400", "clo": "#c0392b", "irr": "#c0392b", "cos": "#e67e22",
          "att": "#f1c40f", "sha": "#2c3e50", "fn_": "#95a5a6"}


def mi16():
    dp = table("findings/delay_predictors.csv")
    setups = ["A", "A'", "B", "C", "E"]
    fig, axs = plt.subplots(1, 5, figsize=(15, 4.2), constrained_layout=True)
    for ax, s in zip(axs, setups):
        rs = [r for r in dp if r["setup"] == s and math.isfinite(num(r["rho_memo_vs_log_delay"]))
              and not r["metric"].startswith(("fn_acc", "fn_loss", "fn_margin", "fn_entropy", "fn_gap"))]
        rs.sort(key=lambda r: -abs(num(r["rho_memo_vs_log_delay"])))
        rs = rs[:12][::-1]
        y = np.arange(len(rs))
        v = fvals(rs, "rho_memo_vs_log_delay")
        ax.barh(y, v, color=[FAMCOL.get(r["metric"][:3], "0.5") for r in rs])
        ax.set_yticks(y)
        ax.set_yticklabels([r["metric"] for r in rs], fontsize=5.8)
        for yi, r in zip(y, rs):
            au = num(r["auc_memo_crossed_vs_not"])
            if math.isfinite(au):
                ax.text(0.02 if num(r["rho_memo_vs_log_delay"]) < 0 else -0.02, yi, f"AUC {max(au, 1 - au):.2f}",
                        va="center", ha="left" if num(r["rho_memo_vs_log_delay"]) < 0 else "right", fontsize=5.3)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlim(-1.05, 1.05)
        n = rs[0]["n_crossed"] if rs else "?"
        ax.set_title(f"{SLAB[s]}\nn crossed = {n}", fontsize=7.5, color=SCOL[s])
        ax.set_xlabel("Spearman ρ, metric at t_memo vs log delay")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Strongest memorisation-time predictors of the delay per setup (training-performance metrics "
                 "removed; D has no crossed run with a checkpoint near t_memo). Small n on A′, C, E.", fontsize=9)
    save(fig, "mi16_predictor_ranking")


# ═════════════════════════════════════════════════════════════════════════════
# order
# ═════════════════════════════════════════════════════════════════════════════

HSERIES = {"test_loss": "test loss", "weight_norm_layer1": "first-layer norm", "weight_norm_total": "total norm",
           "ipr": "input Fourier IPR", "embed_ipr": "embedding Fourier IPR", "coset_accuracy": "coset accuracy",
           "circ_share_interaction": "circuit interaction share", "circ_acc_interaction": "circuit accuracy",
           "irrep_structure_u": "irrep structure (U)"}
CKLEAD = {"A": ["unit_sum_freq_concentration", "unit_fourier_sum_share", "w_W1_fro", "fU_ipr", "clock_share",
                "clock_nfreq90", "sharp_lambda_max", "rep_eff_rank_te", "w_W1_sv_gini", "w_W2_fro"],
          "D": ["circ_acc_interaction_te", "irr_U_structure", "coset_a_n_acc_te", "coset_s_nm1_acc_te", "rep_pr_te",
                "w_W2_stable_rank", "w_W1_fro", "sharp_lambda_max", "unit_interaction_share"],
          "B": ["clock_nfreq90", "clock_share", "w_WU_eff_rank", "fin_ipr", "attn_to_a_mean", "w_WE_fro"]}


def mi17():
    hl = [r for r in table("findings/history_leads.csv") if r["axis"] == "all"]
    ll = table("findings/lead_lag_ckpt.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.1]})
    s_sec = "8. What moves before first crossing (lead as a fraction of the delay; + = before)"
    y, labs, ax = 0, [], axs[0]
    for s in SETUPS:
        for r in [r for r in hl if r["setup"] == s]:
            med, lo, hi = num(r["median_lead_frac_delay"]), num(r["q25"]), num(r["q75"])
            ax.plot([lo, hi], [y, y], color=SCOL[s], lw=2.2, alpha=0.5, solid_capstyle="butt")
            ax.plot(med, y, "o", color=SCOL[s], ms=5, markeredgecolor="k", markeredgewidth=0.3)
            ax.text(1.42, y, f"{100 * num(r['frac_leading']):.0f}%  n={r['n']}", va="center", fontsize=5.8)
            labs.append(f"{s}: {HSERIES.get(r['series'], r['series'])}")
            note(s_sec, f"**{s}** {HSERIES.get(r['series'], r['series'])} (logged history): median lead "
                        f"{fmt(med)} (IQR {fmt(lo)} to {fmt(hi)}), leads in {100 * num(r['frac_leading']):.0f}% of "
                        f"{r['n']} runs.")
            y += 1
        y += 0.6
    g = {(r["setup"], r["series"]): r for r in hl}
    head(8, f"**Norm moves first, Fourier sparsity follows.** Setup A: first-layer norm completes half its change before "
            f"first crossing in {100 * num(g[('A', 'weight_norm_layer1')]['frac_leading']):.0f}% of runs; input "
            f"Fourier IPR in only {100 * num(g[('A', 'ipr')]['frac_leading']):.0f}% (it sharpens after the model "
            f"generalises). E: weight norm leads in {100 * num(g[('E', 'weight_norm_total')]['frac_leading']):.0f}% (mi17, mi18).")
    ax.set_yticks([i + 0.6 * k for i, k in zip(range(len(labs)), _group_offsets(hl))])
    ax.set_yticklabels(labs, fontsize=6.3)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(-1.2, 1.4)
    ax.set_xlabel("lead of half-change time over t_first_cross (fraction of delay)")
    ax.set_title("(a) logged histories, all runs  (right: % of runs leading)", fontsize=8)
    ax.grid(axis="y", visible=False)
    ax = axs[1]
    y, labs = 0, []
    for s, mets in CKLEAD.items():
        recs = {r["metric"]: r for r in ll if r["setup"] == s}
        for mname in mets:
            if mname not in recs:
                continue
            r = recs[mname]
            v = num(r["median_lead_frac_delay"])
            ax.barh(y, max(min(v, 1.5), -1.5), color=SCOL[s], alpha=0.8)
            ax.text(1.55, y, f"{100 * num(r['frac_runs_leading']):.0f}%  n={r['n_runs']}", va="center", fontsize=5.8)
            labs.append((y, f"{s}: {mname}"))
            note(s_sec, f"**{s}** `{mname}` (checkpoints): median lead {fmt(v)}, leads in "
                        f"{100 * num(r['frac_runs_leading']):.0f}% of {r['n_runs']} runs.")
            y += 1
        y += 0.8
    ax.set_yticks([a for a, _ in labs])
    ax.set_yticklabels([b for _, b in labs], fontsize=6.3)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(-1.5, 1.95)
    ax.set_xlabel("median lead (fraction of delay, clipped at ±1.5)")
    ax.set_title("(b) checkpoint metrics, A, D and B", fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.suptitle("Order: which quantities complete half their change before the model first crosses the test bar",
                 fontsize=9)
    save(fig, "mi17_lead_lag")


def _group_offsets(hl):
    out, k = [], 0
    for s in SETUPS:
        for _ in [r for r in hl if r["setup"] == s]:
            out.append(k)
        k += 1
    return out


XSER = {"A": ["test_acc", "weight_norm_layer1", "ipr", "train_loss"],
        "A'": ["test_acc", "weight_norm_layer1", "ipr", "train_loss"],
        "B": ["test_acc", "test_loss", "embed_ipr", "weight_norm_total"],
        "C": ["test_acc", "coset_accuracy", "weight_norm_total", "test_loss"],
        "D": ["test_acc", "circ_acc_interaction", "irrep_structure_u", "coset_accuracy", "weight_norm_layer1"],
        "E": ["test_acc", "weight_norm_total", "test_loss", "train_loss"]}
XCOL = {"test_acc": "k", "weight_norm_layer1": "#2980b9", "weight_norm_total": "#2980b9", "ipr": "#c0392b",
        "embed_ipr": "#c0392b", "train_loss": "#95a5a6", "test_loss": "#7f8c8d", "coset_accuracy": "#e67e22",
        "circ_acc_interaction": "#27ae60", "irrep_structure_u": "#8e44ad"}


def mi18():
    ser = collections.defaultdict(list)
    for r in table("history_series_crossaligned.csv"):
        if r["id"] not in DIVERGED:
            ser[r["id"]].append(r)
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.0), constrained_layout=True, sharex=True)
    for ax, s in zip(axs.flat, SETUPS):
        runs = [v for v in ser.values() if v[0]["setup"] == s]
        grid = np.logspace(np.log10(0.02), np.log10(20), 48)
        for key in XSER[s]:
            M = []
            for recs in runs:
                x = fvals(recs, "x")
                y = fvals(recs, key)
                if np.isfinite(y).sum() < 10:
                    continue
                if "loss" in key:
                    y = np.log10(np.clip(y, 1e-12, None))
                lo, hi = np.nanmin(y), np.nanmax(y)
                if hi - lo < 1e-12:
                    continue
                row = np.full(48, np.nan)
                idx = np.array([int(np.argmin(np.abs(grid - xx))) for xx in x])
                row[idx] = (y - lo) / (hi - lo)
                M.append(row)
            if not M:
                continue
            M = np.array(M)
            ok = np.isfinite(M).sum(0) >= max(3, len(M) // 3)
            med = np.where(ok, np.nanmedian(M, 0), np.nan)
            lab = HSERIES.get(key, key.replace("_", " ")) + ("" if "loss" not in key else " (log)")
            ax.plot(grid, med, color=XCOL.get(key, "0.5"), lw=1.5 if key != "test_acc" else 2.0,
                    label=f"{lab} (n={len(M)})")
            ax.fill_between(grid, np.where(ok, np.nanquantile(M, .25, 0), np.nan),
                            np.where(ok, np.nanquantile(M, .75, 0), np.nan), color=XCOL.get(key, "0.5"), alpha=0.12,
                            lw=0)
        ax.axvline(1, color="k", lw=0.6, ls="--")
        ax.set_xscale("log")
        ax.set_title(SLAB[s], fontsize=8, color=SCOL[s])
        ax.legend(fontsize=5.5, frameon=True, framealpha=0.85, loc="center right" if s in ("B", "C") else "center left")
        ax.set_ylim(-0.05, 1.05)
    for ax in axs[1]:
        ax.set_xlabel("step / t_first_cross")
    for ax in axs[:, 0]:
        ax.set_ylabel("min–max normalised per run\n(median, IQR)")
    fig.suptitle("Order parameters aligned on each run's first crossing (dashed), all runs that crossed", fontsize=9)
    save(fig, "mi18_crossaligned_order_parameters")


def t_half(steps, y):
    ok = np.isfinite(y)
    if ok.sum() < 5:
        return math.nan
    s, v = steps[ok], y[ok]
    lo, hi = v[0], np.nanmax(v)
    if hi - lo < 1e-9:
        return math.nan
    idx = np.nonzero(v >= lo + 0.5 * (hi - lo))[0]
    return float(s[idx[0]]) if idx.size else math.nan


def mi19():
    rows = C.load_rows()
    ser = collections.defaultdict(list)
    for r in table("history_series_crossaligned.csv"):
        if r["setup"] == "D":
            ser[r["id"]].append(r)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    pts = collections.defaultdict(list)
    for rid, recs in ser.items():
        recs.sort(key=lambda r: num(r["x"]))
        st = fvals(recs, "step")
        tc, tm = num(rows[rid]["t_first_cross"]), num(rows[rid]["t_memo"])
        for key in ("circ_acc_interaction", "irrep_structure_u", "coset_accuracy"):
            th = t_half(st, fvals(recs, key))
            if math.isfinite(th):
                pts[key].append((th, tc, tm, rows[rid]["mode"], rows[rid]["num_clients"]))
    ex = []
    for rid, recs in sorted(ser.items()):
        r = rows[rid]
        if (r["mode"] == "federated" and k_ladder(r, "D") and not any(e[1]["num_clients"] == r["num_clients"] for e in ex)
                and np.isfinite(fvals(recs, "circ_acc_interaction")).sum() > 10):
            ex.append((sorted(recs, key=lambda q_: num(q_["x"])), r))
    ax = axs[0]
    for recs, r in sorted(ex, key=lambda e: num(e[1]["num_clients"])):
        col = kcolor(r["num_clients"])
        st = fvals(recs, "step")
        ca = fvals(recs, "circ_acc_interaction")
        ax.plot(st, fvals(recs, "test_acc") / 100, color=col, lw=1.3, label=f"K{r['num_clients']}")
        ax.plot(st, ca / (100 if np.nanmax(ca) > 1.5 else 1), color=col, lw=1.0, ls="--")
    ax.plot([], [], color="0.4", ls="-", label="test accuracy")
    ax.plot([], [], color="0.4", ls="--", label="circuit accuracy")
    ax.set_xscale("log")
    ax.set_xlabel("steps")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=5.8, frameon=False)
    ax.set_title("(a) example federated runs (α 0.3 K ladder, one seed each)", fontsize=7.5)
    s_sec = "9. Setup D: circuit before generalisation"
    for ax, key, lab, L in ((axs[1], "circ_acc_interaction", "circuit accuracy", "b"),
                            (axs[2], "irrep_structure_u", "irrep structure (U)", "c")):
        P = pts[key]
        for mode, mk in (("centralized", "s"), ("federated", "o")):
            sel = [p for p in P if p[3] == mode]
            ax.scatter([p[1] for p in sel], [p[0] for p in sel], marker=mk, s=13, lw=0.2, edgecolor="k",
                       c=[kcolor(p[4]) for p in sel], alpha=0.8, label=f"{mode} (n={len(sel)})")
        lim = [1e3, 5e5]
        ax.plot(lim, lim, color="k", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("t_first_cross")
        ax.set_ylabel(f"half-rise time of {lab}")
        ratio = np.array([p[0] / p[1] for p in P])
        ax.set_title(f"({L}) {lab}: median t½ / t_cross = {np.median(ratio):.2f}", fontsize=8)
        ax.legend(fontsize=6, frameon=False)
        for mode in ("centralized", "federated"):
            rr = np.array([p[0] / p[1] for p in P if p[3] == mode])
            if rr.size:
                note(s_sec, f"{lab}, {mode}: half-rise before first crossing in {100 * np.mean(rr < 1):.0f}% of "
                            f"{rr.size} runs; median t½ / t_first_cross = {fmt(float(np.median(rr)))} "
                            f"(IQR {fmt(q(rr, .25))}–{fmt(q(rr, .75))}).")
    P = pts["circ_acc_interaction"]
    rc = np.array([p[0] / p[1] for p in P if p[3] == "centralized"])
    rf = np.array([p[0] / p[1] for p in P if p[3] == "federated"])
    head(4.5, f"**On D the circuit is ready long before the model generalises.** Circuit accuracy reaches half its "
              f"rise before first crossing in {100 * np.mean(np.r_[rc, rf] < 1):.0f}% of {rc.size + rf.size} runs, at a "
              f"median {fmt(float(np.median(rc)))} (centralised) and {fmt(float(np.median(rf)))} (federated) of t_first_cross (mi19).")
    fig.suptitle("Setup D: the compositional circuit forms long before the full model generalises (grey: centralised)",
                 fontsize=9)
    save(fig, "mi19_D_circuit_before_generalisation")


# ═════════════════════════════════════════════════════════════════════════════
# spectra through training
# ═════════════════════════════════════════════════════════════════════════════

def pick_run(pred, seed=None):
    runs, moments = ckpt()
    cands = sorted([rid for rid, m in moments.items() if pred(m["init"])
                    and os.path.exists(os.path.join(MI, "spectra", f"{rid}.npz"))],
                   key=lambda r: num(moments[r]["init"]["seed"]))
    if seed is not None:
        s = [r for r in cands if _eq(moments[r]["init"]["seed"], seed)]
        if s:
            return s[0]
    return cands[0] if cands else None


def mi20():
    runs, moments = ckpt()
    cells = [("A", "2"), ("A", "50"), ("A'", "2"), ("A'", "50"), ("B", "2"), ("B", "20")]
    fig, axs = plt.subplots(3, 2, figsize=(10.5, 9.5), constrained_layout=True)
    for ax, (s, K) in zip(axs.flat, cells):
        rid = pick_run(lambda r: k_ladder(r, s) and _eq(r["num_clients"], K), seed=42)
        if rid is None:
            ax.axis("off")
            continue
        z = spectra_npz(rid)
        E = z["clock_energy"]
        E = E / np.clip(E.sum(1, keepdims=True), 1e-30, None)
        st = z["steps"]
        im = ax.imshow(E.T, aspect="auto", origin="lower", cmap="inferno", norm=LogNorm(1e-4, 1),
                       extent=[-0.5, len(st) - 0.5, 0.5, E.shape[1] + 0.5])
        r = moments[rid]["init"]
        for t, ls, lab in ((num(r["t_memo"]), ":", "t_memo"), (num(r["t_first_cross"]), "--", "t_cross")):
            if math.isfinite(t):
                ax.axvline(np.interp(math.log(t), np.log(st.clip(1)), np.arange(len(st))), color="cyan", lw=0.9,
                           ls=ls)
        tick = np.linspace(0, len(st) - 1, min(8, len(st))).astype(int)
        ax.set_xticks(tick)
        ax.set_xticklabels([fmt(st[i]) for i in tick], fontsize=6, rotation=30)
        ax.set_title(f"{SLAB[s]}, K = {K}, seed {r['seed']}", fontsize=8, color=SCOL[s])
        ax.set_ylabel("frequency w")
    for ax in axs[-1]:
        ax.set_xlabel("checkpoint step (cyan: t_memo dotted, t_first_cross dashed)")
    fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01, label="share of clock energy")
    fig.suptitle("Clock-frequency energy of the logits through training: GD spreads over many frequencies; "
                 "the transformer concentrates on a few", fontsize=9)
    save(fig, "mi20_clock_energy_heatmaps")


def mi21():
    runs, moments = ckpt()
    cells = [("D", lambda r: match(r, "D", mode="centralized", alpha="0.3"), "centralised"),
             ("D", lambda r: k_ladder(r, "D") and _eq(r["num_clients"], "2"), "K = 2"),
             ("D", lambda r: k_ladder(r, "D") and _eq(r["num_clients"], "50"), "K = 50"),
             ("C", lambda r: k_ladder(r, "C") and _eq(r["num_clients"], "2"), "K = 2"),
             ("C", lambda r: k_ladder(r, "C") and _eq(r["num_clients"], "10"), "K = 10"),
             ("C", lambda r: k_ladder(r, "C") and _eq(r["num_clients"], "50"), "K = 50")]
    cols = plt.cm.tab10(np.arange(7))
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.6), constrained_layout=True, sharey=True)
    for ax, (s, pred, lab) in zip(axs.flat, cells):
        rid = pick_run(pred, seed=42)
        if rid is None:
            ax.axis("off")
            continue
        z = spectra_npz(rid)
        st = z["steps"].astype(float)
        F = z["irrep_in"]
        names = [str(n) for n in z["irrep_names"]]
        ax.stackplot(st, F.T, colors=cols, labels=names, alpha=0.9)
        r = moments[rid]["init"]
        for t, ls in ((num(r["t_memo"]), ":"), (num(r["t_first_cross"]), "--")):
            if math.isfinite(t):
                ax.axvline(t, color="k", lw=0.9, ls=ls)
        ax.set_xscale("log")
        ax.set_xlim(st[0], st[-1])
        ax.set_ylim(0, 1)
        ax.set_title(f"{s}, {lab}, seed {r['seed']}", fontsize=8, color=SCOL[s])
    axs[0, 0].legend(fontsize=6, ncol=2, loc="upper left", frameon=True, title="irrep")
    for ax in axs[:, 0]:
        ax.set_ylabel("input-weight energy fraction")
    for ax in axs[1]:
        ax.set_xlabel("steps (dotted t_memo, dashed t_first_cross)")
    fig.suptitle("S5 irrep energy of the input weights through training", fontsize=9)
    save(fig, "mi21_irrep_dynamics")


def curves_by_k(name, title, setups, metrics, xkey="step_over_t_cross", logy=(), xlabel="step / t_first_cross"):
    runs, moments = ckpt()
    fig, axs = plt.subplots(len(metrics), len(setups), figsize=(2.6 * len(setups) + 0.6, 2.1 * len(metrics)),
                            constrained_layout=True, squeeze=False, sharex="col")
    for j, s in enumerate(setups):
        for rid, recs in runs.items():
            r0 = recs[0]
            if not k_ladder(r0, s):
                continue
            x = fvals(recs, xkey)
            if not np.isfinite(x).any():
                continue
            for i, (key, lab) in enumerate(metrics):
                k = key.format(first=FIRST[s], last=LAST[s], fin="fU" if s in ("A", "A'") else "fin")
                y = fvals(recs, k)
                ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
                if ok.sum() < 2:
                    continue
                axs[i, j].plot(x[ok], y[ok], color=kcolor(r0["num_clients"]), lw=0.9, alpha=0.8)
        axs[0, j].set_title(SLAB[s], fontsize=8, color=SCOL[s])
        for i, (key, lab) in enumerate(metrics):
            ax = axs[i, j]
            ax.set_xscale("log")
            if xkey == "step_over_t_cross":
                ax.axvline(1, color="k", lw=0.5, ls="--")
            if key in logy:
                ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel(lab.format(first="first", last="last", fin="input"), fontsize=7)
        axs[-1, j].set_xlabel(xlabel)
    k_colorbar(fig, axs[:, -1])
    fig.suptitle(title, fontsize=9)
    save(fig, name)


def mi22():
    curves_by_k("mi22_fourier_structure_dynamics",
                "Fourier structure through training on the modular setups, coloured by K (IID ladders; crossed runs)",
                ["A", "A'", "B"],
                [("{fin}_ipr", "input-weight Fourier IPR"), ("{fin}_nfreq90", "input frequencies for 90% power"),
                 ("clock_share", "clock share of logit energy"), ("clock_nfreq90", "clock frequencies for 90%"),
                 ("clock_restricted_loss_te", "restricted loss (key freqs only)"),
                 ("clock_excluded_loss_te", "excluded loss (key freqs removed)"),
                 ("unit_fourier_sum_share", "unit (a+b) sum-frequency share")],
                logy=("clock_restricted_loss_te", "clock_excluded_loss_te", "{fin}_nfreq90"))


def mi23():
    curves_by_k("mi23_weight_geometry_dynamics",
                "Weight norms, Gini coefficients and ranks through training, coloured by K (IID ladders; crossed runs)",
                SETUPS,
                [("w_all_norm", "total weight norm"), ("w_all_rel_dist_init", "rel. distance from init"),
                 ("w_{first}_sv_gini", "{first} layer: Gini of singular values"),
                 ("w_{first}_row_gini", "{first} layer: Gini of neuron norms"),
                 ("w_{first}_eff_rank", "{first} layer: effective rank"),
                 ("w_{last}_stable_rank", "{last} layer: stable rank"),
                 ("rep_eff_rank_te", "representation effective rank")],
                logy=("w_{last}_stable_rank",))


def mi24():
    runs, moments = ckpt()
    metrics = [("w_{first}_sv_gini", "first layer: Gini(σ)"), ("w_{first}_row_gini", "first layer: Gini(neuron norms)"),
               ("w_{first}_entry_gini", "first layer: Gini(|entries|)"), ("w_{last}_sv_gini", "last layer: Gini(σ)"),
               ("rep_unit_gini_te", "hidden activations: Gini over units")]
    mcol = {"memo": "#3498db", "cross": "#e67e22", "end": "#2c3e50"}
    fig, axs = plt.subplots(len(metrics), 6, figsize=(15, 2.2 * len(metrics)), constrained_layout=True,
                            squeeze=False)
    s_sec = "10. Gini coefficients (IID K ladders)"
    for j, s in enumerate(SETUPS):
        for i, (key, lab) in enumerate(metrics):
            k = key.format(first=FIRST[s], last=LAST[s])
            ax = axs[i, j]
            for mo, col in mcol.items():
                recs = [m[mo] for m in moments.values() if mo in m and k_ladder(m["init"], s)]
                if not recs:
                    continue
                x, y = fvals(recs, "num_clients"), fvals(recs, k)
                strip_median(ax, x, y, col, label=mo, ms=2.6)
                if mo == "end" and np.isfinite(y).any():
                    xs = sorted(set(x[np.isfinite(y)]))
                    rho, _, n = spearman(np.log(x), y)
                    note(s_sec, f"**{s}** `{k}` at end: K{int(xs[0])} {fmt(q(y[x == xs[0]], .5), 3)} → "
                                f"K{int(xs[-1])} {fmt(q(y[x == xs[-1]], .5), 3)} (ρ = {fmt(rho)}, n = {n}).")
            kaxis(ax)
            if i < len(metrics) - 1:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(lab, fontsize=7)
        axs[0, j].set_title(SLAB[s], fontsize=8, color=SCOL[s])
    axs[0, 0].legend(fontsize=6, frameon=False, title="moment", title_fontsize=6)
    fig.suptitle("Gini coefficients (sparsity / inequality) at memorisation, first crossing and the end, against K",
                 fontsize=9)
    save(fig, "mi24_gini_vs_K")


# ═════════════════════════════════════════════════════════════════════════════
# clients
# ═════════════════════════════════════════════════════════════════════════════

def mi25():
    cl = [r for r in table("client_metrics.csv") if r["id"] not in DIVERGED]
    by = collections.defaultdict(list)
    for r in cl:
        by[r["id"]].append(r)
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.3), constrained_layout=True)
    s_sec = "11. Client deviations"
    ax = axs[0]
    for i, s in enumerate(SETUPS):
        v = [num(r["cli_dev_pair_cos"]) * (num(r["cli_n"]) - 1) for r in cl if r["setup"] == s]
        v = np.array([x for x in v if math.isfinite(x)])
        if not v.size:
            continue
        ax.boxplot(v, positions=[i], widths=0.6, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=SCOL[s], alpha=0.5), medianprops=dict(color="k"))
        note(s_sec, f"**{s}**: normalised pairwise cosine (n − 1)·cos median {fmt(np.median(v), 3)} "
                    f"(IQR {fmt(q(v, .25), 3)} to {fmt(q(v, .75), 3)}; {v.size} snapshots).")
    ax.axhline(-1, color="k", lw=0.6, ls=":")
    ax.set_xticks(range(6))
    ax.set_xticklabels(SETUPS)
    ax.set_ylabel("(n − 1) · mean pairwise cosine\nof client deviations")
    ax.set_title("(a) −1 = independent, zero-sum deviations", fontsize=8)
    rows = []
    for rid, recs in by.items():
        r0 = recs[0]
        if r0["setup"] != "A":
            continue
        tm, tc = num(r0["t_memo"]), num(r0["t_first_cross"])
        if not (math.isfinite(tm) and math.isfinite(tc) and tc > tm):
            continue
        dl = [r for r in recs if tm < num(r["step"]) < tc]
        if not dl:
            continue
        raw = np.median(fvals(dl, "cli_dev_pair_cos"))
        n = np.median(fvals(dl, "cli_n"))
        rows.append((raw, raw * (n - 1), tc - tm, r0["num_clients"], np.median(fvals(dl, "cli_rel_spread"))))
    rows = [r for r in rows if all(math.isfinite(x) for x in (r[0], r[1], r[2]))]
    for ax, idx, lab, L in ((axs[1], 0, "raw mean pairwise cosine", "b"), (axs[2], 1, "(n − 1) · cosine", "c"),
                            (axs[3], 4, "relative client spread", "d")):
        x = np.array([r[idx] for r in rows])
        y = np.array([r[2] for r in rows])
        ax.scatter(x, y, c=[kcolor(r[3]) for r in rows], s=16, edgecolor="k", lw=0.25)
        rho, p, n = spearman(x, np.log(y))
        ax.set_yscale("log")
        ax.set_xlabel(lab + " (median during delay)")
        ax.set_title(f"({L}) A: ρ with log delay = {rho:.2f} (n={n})", fontsize=8)
        note(s_sec, f"A, {lab} during the delay vs log delay: ρ = {fmt(rho)} (n = {n}).")
        if idx == 0:
            raw_rho = rho
        if idx == 1:
            head(10, f"**Client disagreement is unstructured.** Normalised pairwise cosine of client deviations ≈ −1 on "
                     f"every setup (independent, zero-sum); on A its correlation with log delay falls from ρ = "
                     f"{fmt(raw_rho)} (raw) to {fmt(rho)} once the client count is removed (mi25).")
        if idx == 4:
            ax.set_xscale("log")
    axs[1].set_ylabel("delay (steps)")
    k_colorbar(fig, axs[1:])
    fig.suptitle("Client deviations from the client mean share no direction; their raw correlation with the delay is "
                 "mostly the number of clients", fontsize=9)
    save(fig, "mi25_client_deviations")


# ═════════════════════════════════════════════════════════════════════════════
# bulletin
# ═════════════════════════════════════════════════════════════════════════════

FIGREF = {"1.": "mi01", "2.": "mi02–mi05", "3.": "mi09–mi13", "4.": "mi14", "5.": "mi07–mi08", "6.": "mi06",
          "7.": "mi15–mi16", "8.": "mi17–mi18", "9.": "mi19", "10.": "mi23–mi24", "11.": "mi25"}


def coverage():
    runs = {r["id"] for r in table("checkpoint_metrics.csv")}
    s = "0. Coverage"
    note(s, f"Checkpoint metrics: {len(runs)} runs, {len(table('checkpoint_metrics.csv')):,} checkpoints.")
    note(s, f"Logged histories: {len(table('history_moments.csv')):,} runs; client snapshots: "
            f"{len({r['id'] for r in table('client_metrics.csv')})} runs ({len(table('client_metrics.csv')):,} rows).")
    tw = table("twin_trajectories.csv")
    note(s, f"Centralised twins: {len({r['twin_id'] for r in tw})} runs compared against "
            f"{len({r['id'] for r in tw})} federated runs ({len(tw):,} matched checkpoints).")
    note(s, f"Linear paths: {len(table('landscape_barriers.csv')):,}. Every figure excludes the 6 diverged runs "
            f"except mi01.")


def write_bulletin():
    order = sorted(STATS, key=lambda k: float(k.split(".")[0]))
    lines = ["# Mechanistic interpretability: key statistics", "",
             "Generated by `scripts/plotting/mechinterp_figures.py` from `results/mechinterp/` (numbers recomputed, "
             "not copied). Figures: `figures/mechinterp/`. Medians over runs unless stated; `→` is lowest → highest "
             "value of the axis. Spearman p-values are omitted (Fisher-z approximations, n small).", "",
             "## Headlines (strongest first)", ""]
    for i, (_, l) in enumerate(sorted(HEAD, key=lambda h: h[0]), start=1):
        lines.append(f"{i}. {l}")
    lines += ["", "Caveats: no same-initialisation noise floor for AdamW twin distances was measured; setup C is "
              "non-deterministic run to run; checkpoint moments are the nearest checkpoint within ×0.67–1.5 of the event; "
              "n per cell is 3–6.", ""]
    for k in order:
        ref = FIGREF.get(k.split(" ")[0], "")
        lines.append(f"## {k}" + (f"  ({ref})" if ref else ""))
        lines.append("")
        seen = set()
        for l in STATS[k]:
            if l not in seen:
                lines.append(f"- {l}")
                seen.add(l)
        lines.append("")
    with open(BULLETIN, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {BULLETIN}")


FIGS = {f.__name__: f for f in (mi01, mi02, mi03, mi04, mi05, mi06, mi07, mi08, mi09, mi10, mi11, mi12, mi13, mi14,
                                mi15, mi16, mi17, mi18, mi19, mi20, mi21, mi22, mi23, mi24, mi25)}


def main():
    want = sys.argv[1:] or list(FIGS)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
    coverage()
    for name in want:
        FIGS[name]()
    if not sys.argv[1:]:
        write_bulletin()


if __name__ == "__main__":
    main()
