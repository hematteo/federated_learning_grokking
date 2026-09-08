"""The paper's main figures, regenerated from the run table and histories.

    python3 scripts/plotting/paper_figures.py               # -> paper/figures/
    python3 scripts/plotting/paper_figures.py --only 1 A1   # a subset

Every number is computed from results/data/runs_v2.csv and results/runs/<id>/
history_*.json; nothing is typed in. Figure-by-figure numbers used in captions
are written to paper/figures/figure_numbers.json. Captions live in
paper/figures.tex.

Style follows the 1 Sep draft's figures: boxed axes, centred descriptive titles,
framed legends, mathtext axis labels, bold panel letters, error bars that span
the runs, a viridis-like palette for the training fraction, IID solid and
non-IID dashed, and a cross for "failed to grok within budget". Marker fill is
the held fraction (hollow when no run held the bar to the end).

Statistics. Per-cell times are Kaplan-Meier medians over runs, right-censoring
runs whose event (memorisation, or first crossing of the bar) did not happen
within the run's own budget; error bars span the finite per-run values. delay =
t_first_cross - t_memo per run, median over runs with both finite. C's numbers
are drawn faded and withheld from any quantitative claim (RESULTS 23).
"""

import argparse
import collections
import csv
import glob
import json
import math
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from fedgrok.analysis.survival import km_median

CSV = "results/data/runs_v2.csv"
HIST = "results/runs"
OUT = "paper/figures"

# The draft's palettes: matplotlib's categorical set for setups, partitions and
# aggregation rules (FedAvg grey, FedAdam green, FedProx blue as in its Fig. 8),
# and its purple / blue / teal / orange for the training fraction.
COL = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728", "E": "#9467bd"}
PART = {
    "operand": "#1f77b4",
    "coset": "#9467bd",
    "target": "#d62728",
    "label_block": "#ff7f0e",
    "dirichlet": "#2ca02c",
}
STRAT = {
    "fedavg": "#7f7f7f",
    "fedadam": "#2ca02c",
    "fedyogi": "#17becf",
    "scaffold": "#9467bd",
    "fedavgm": "#ff7f0e",
    "fedprox": "#1f77b4",
}
ALPHA_COL = {0.25: "#9b59b6", 0.3: "#3498db", 0.35: "#1abc9c", 0.5: "#f39c12"}
NAME = {
    "A": "A: quad-MLP, mod 97, GD",
    "B": "B: transformer, mod 113",
    "C": "C: transformer, $S_5$",
    "D": "D: quad-MLP, $S_5$",
    "E": "E: MLP, MNIST",
}
INK, INK2, MUTED, GRID = "#111111", "#444444", "#888888", "#d9d9d9"
WITHHELD = 0.45  # line alpha for setup C

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "legend.title_fontsize": 7,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": False,
        "axes.titlelocation": "center",
        "axes.titleweight": "normal",
        "axes.titlepad": 5,
        "savefig.dpi": 220,
        "figure.dpi": 100,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

NUMERIC = [
    "alpha",
    "n_train",
    "weight_decay",
    "num_clients",
    "local_epochs",
    "fraction_train",
    "dirichlet_alpha",
    "server_lr",
    "proximal_mu",
    "num_rounds",
    "epochs",
    "hidden_width",
    "t_grok",
    "t_first_cross",
    "t_memo",
    "peak_train_acc",
    "final_acc",
    "steps_run",
    "wall_s",
    "grok_threshold",
    "seed",
    "final_ipr",
    "lr",
]
NUMBERS = {}
TFC = r"$t_{\mathrm{first\,cross}}$"
TMEMO = r"$t_{\mathrm{memo}}$"


def _f(x):
    if x in ("", None):
        return None
    if x == "inf":
        return math.inf
    try:
        return float(x)
    except ValueError:
        return None


def infer_setup(row):
    if row.get("setup"):
        return row["setup"]
    key = (row.get("dataset"), row.get("model"))
    return {
        ("modular", "groknet"): "A",
        ("modular", "transformer"): "B",
        ("s5", "transformer"): "C",
        ("s5", "groknet"): "D",
        ("mnist", "mlp"): "E",
    }.get(key, "?")


def load_rows():
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        r["setup"] = infer_setup(r)
        for k in NUMERIC:
            r[k] = _f(r.get(k))
        r["grokked"] = r.get("grokked") in ("True", "true", "1")
        if r["mode"] == "federated":
            r["budget"] = r["steps_run"] or (
                r["num_rounds"] * r["local_epochs"] * (r["fraction_train"] or 1.0)
            )
        else:
            r["budget"] = r["steps_run"] or r["epochs"]
        r["delay"] = (
            r["t_first_cross"] - r["t_memo"]
            if _finite(r["t_first_cross"]) and _finite(r["t_memo"])
            else None
        )
    return rows


def _finite(x):
    return x is not None and math.isfinite(x)


_HCACHE = {}


def history(run_id):
    if run_id not in _HCACHE:
        hits = glob.glob(os.path.join(HIST, run_id, "history_*.json"))
        _HCACHE[run_id] = json.load(open(hits[0])) if hits else None
    return _HCACHE[run_id]


def sel(rows, **kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            x = r.get(k)
            if callable(v):
                ok = v(x)
            elif isinstance(v, (set, list, tuple)):
                ok = x in v
            else:
                ok = x == v
            if not ok:
                break
        if ok:
            out.append(r)
    return out


def fed(rows, setup, **kw):
    base = dict(
        mode="federated",
        setup=setup,
        strategy="fedavg",
        partition="iid",
        local_epochs=5.0,
        fraction_train=1.0,
    )
    base.update(kw)
    return sel(rows, **base)


# ── statistics ────────────────────────────────────────────────────────────────


def km(vals, budgets):
    d = [v if _finite(v) else b for v, b in zip(vals, budgets)]
    e = [1 if _finite(v) else 0 for v in vals]
    return km_median(d, e) if d else math.inf


def _span(vals):
    return (min(vals), max(vals)) if vals else (math.inf, math.inf)


def stats(rs):
    """KM medians for memorisation and first crossing, median delay, held fraction,
    and the min-max span of the finite per-run values (the error bars)."""
    if not rs:
        return None
    b = [r["budget"] for r in rs]
    delays = [r["delay"] for r in rs if r["delay"] is not None]
    memos = [r["t_memo"] for r in rs if _finite(r["t_memo"])]
    fcs = [r["t_first_cross"] for r in rs if _finite(r["t_first_cross"])]
    st = {
        "n": len(rs),
        "held": sum(r["grokked"] for r in rs) / len(rs),
        "crossed": sum(_finite(r["t_first_cross"]) for r in rs) / len(rs),
        "memo": km([r["t_memo"] for r in rs], b),
        "fc": km([r["t_first_cross"] for r in rs], b),
        "delay": float(np.median(delays)) if delays else math.inf,
        "n_delay": len(delays),
        "peak": float(
            np.median(
                [r["peak_train_acc"] for r in rs if r["peak_train_acc"] is not None]
            )
        ),
        "final": float(
            np.median([r["final_acc"] for r in rs if r["final_acc"] is not None])
        ),
        "budget": max(b),
    }
    st["memo_lo"], st["memo_hi"] = _span(memos)
    st["fc_lo"], st["fc_hi"] = _span(fcs)
    st["delay_lo"], st["delay_hi"] = _span(delays)
    finals = [r["final_acc"] for r in rs if r["final_acc"] is not None]
    st["final_lo"], st["final_hi"] = _span(finals)
    return st


def rounded(st, nd=1):
    return {
        k: (round(v, nd) if _finite(v) else None) if isinstance(v, float) else v
        for k, v in st.items()
    }


def by_cell(rs, *keys):
    cells = collections.defaultdict(list)
    for r in rs:
        key = tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]]
        cells[key].append(r)
    return cells


def best_control(rs):
    """Among candidate control runs, keep the cell with the largest budget."""
    if not rs:
        return []
    cells = by_cell(rs, "num_rounds", "epochs")
    return max(cells.values(), key=lambda c: c[0]["budget"])


def cent_runs(rows, s, alpha, wd=None):
    """The centralised control for a setup at a training fraction (and decay)."""
    kw = dict(mode="centralized", setup=s, alpha=alpha)
    if wd is not None:
        kw["weight_decay"] = wd
    return best_control(
        [
            r
            for r in sel(rows, **kw)
            if r["arm"] != "cent_reduced" and r["group"] != "d_internals"
        ]
    )


# ── drawing helpers ───────────────────────────────────────────────────────────

KS = [1, 2, 5, 10, 20, 50, 97]  # the draft's K axis: equal spacing, "cent" at 1
ES = [1, 5, 10, 25, 50]


def kx(K):
    return KS.index(int(K))


def ex(E):
    return ES.index(int(E))


def cat_axis(ax, labels, start=0):
    n = len(labels)
    ax.set_xticks(range(start, start + n))
    ax.set_xticklabels(labels)
    ax.set_xlim(start - 0.5, start + n - 0.5)


def _human(v, _=None):
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def log_steps(ax, axis="y", lo=None, hi=None, dense=True):
    """Log axis with 1k / 10k / 100k ticks; 1-2-5 ticks when it spans < 3 decades."""
    setter = ax.set_yscale if axis == "y" else ax.set_xscale
    setter("log")
    if lo is not None:
        (ax.set_ylim if axis == "y" else ax.set_xlim)(lo, hi)
    a, b = ax.get_ylim() if axis == "y" else ax.get_xlim()
    ks = range(int(math.floor(math.log10(a))), int(math.ceil(math.log10(b))) + 1)
    decades = [10**k for k in ks if a <= 10**k <= b]
    ticks = decades
    if dense and len(decades) < 3:
        ticks = [10**k * m for k in ks for m in (1, 2, 5) if a <= 10**k * m <= b]
    axis_obj = ax.yaxis if axis == "y" else ax.xaxis
    axis_obj.set_major_locator(FixedLocator(ticks))
    axis_obj.set_major_formatter(FuncFormatter(_human))
    axis_obj.set_minor_formatter(NullFormatter())


def letter(ax, L):
    """The draft's bold panel letter: outside the axes, above the y-axis labels."""
    ax.annotate(
        f"{L})", xy=(0, 1), xycoords="axes fraction", xytext=(-30, 3),
        textcoords="offset points", ha="left", va="bottom", fontsize=9,
        fontweight="bold", annotation_clip=False,
    )


def mark(ax, x, y, color, frac, marker="o", size=5, z=4, alpha=1.0):
    """One point whose fill carries the held fraction; hollow when nothing held."""
    ax.plot(
        [x],
        [y],
        marker=marker,
        ms=size,
        mec=color,
        mfc=color if frac >= 0.999 else "white",
        mew=1.2,
        ls="none",
        zorder=z,
        alpha=alpha,
    )
    if 0 < frac < 0.999:
        ax.plot(
            [x],
            [y],
            marker=marker,
            ms=size * 0.5,
            mec="none",
            mfc=color,
            ls="none",
            zorder=z + 1,
            alpha=alpha,
        )


def censored_x(ax, x, color, y=0.96, size=7, alpha=1.0):
    """A cell with no finite value: the draft's cross, on the top edge."""
    ax.plot(
        [x],
        [y],
        marker="x",
        ms=size,
        mec=color,
        mew=1.5,
        ls="none",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        zorder=6,
        alpha=alpha,
    )


CENS_HANDLE = Line2D(
    [], [], marker="x", color=INK2, ls="none", mew=1.5, ms=7, label="Failed to grok"
)


def series(
    ax,
    xs,
    ys,
    fracs,
    color,
    label=None,
    marker="o",
    lw=1.5,
    ls="-",
    zorder=3,
    alpha=1.0,
    size=5,
    err=None,
):
    """A line with held-fraction markers and, when err=(lo, hi) is given, error
    bars spanning the runs (the draft's convention)."""
    pts = [(x, y, f) for x, y, f in zip(xs, ys, fracs) if _finite(y)]
    if pts:
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=color,
            lw=lw,
            ls=ls,
            zorder=zorder,
            label=label,
            alpha=alpha,
        )
        if err is not None:
            for x, y, lo, hi in zip(xs, ys, err[0], err[1]):
                if _finite(y) and _finite(lo) and _finite(hi):
                    ax.errorbar(
                        [x],
                        [y],
                        yerr=[[max(y - min(lo, y), 0)], [max(max(hi, y) - y, 0)]],
                        color=color,
                        capsize=2.5,
                        elinewidth=0.9,
                        capthick=0.9,
                        ls="none",
                        marker="none",
                        zorder=zorder,
                        alpha=alpha,
                    )
        for x, y, f in pts:
            mark(ax, x, y, color, f, marker, alpha=alpha, size=size)
    return pts


def span(sts, key, scale=1.0):
    return (
        [st[key + "_lo"] / scale for st in sts],
        [st[key + "_hi"] / scale for st in sts],
    )


def setup_handles(setups):
    return [
        Line2D([], [], color=COL[s], lw=1.5, marker="o", label=NAME[s]) for s in setups
    ]


def logx_ticks(ax, values, labels=None):
    ax.set_xscale("log")
    ax.set_xticks(values)
    ax.set_xticklabels(labels or [f"{v:g}" for v in values])
    ax.minorticks_off()


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", facecolor="white"
        )
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png/.pdf")


# ── Fig 1: survival and the two clocks ────────────────────────────────────────

SRC1 = {
    "aggregation",
    "setup_k_ladder",
    "k_collapse_budget",
    "k_fixed_total",
    "boundary",
}


def k_ladder(rows, s, alpha, wd=None):
    """K -> stats for one setup's FedAvg ladder, with the centralised control at K=1."""
    rs = [
        r
        for r in fed(rows, s, alpha=alpha)
        if r["group"] in SRC1 and (wd is None or r["weight_decay"] == wd)
    ]
    out = {}
    cent = cent_runs(rows, s, alpha, wd)
    if cent:
        out[1] = stats(cent)
    for K, runs in by_cell(rs, "num_clients").items():
        out[int(K)] = stats(runs)
    return out


def fig1(rows):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.6),
        gridspec_kw=dict(width_ratios=[1.2, 1, 1], wspace=0.5),
    )
    numbers = {"ratio": {}, "ladders": {}}
    # (a) the draft's Fig. 1: slowdown ratio vs K on the anchor, alpha as the series
    ax = axes[0]
    for alpha in (0.3, 0.25):
        col = ALPHA_COL[alpha]
        runs = [r for r in fed(rows, "A", alpha=alpha) if r["group"] in SRC1]
        base = stats(cent_runs(rows, "A", alpha))["fc"]
        cells = by_cell(runs, "num_clients")
        xs, med, lo, hi, fr = [], [], [], [], []
        for K in sorted(cells):
            if int(K) not in KS:
                continue
            st = stats(cells[K])
            rat = [
                r["t_first_cross"] / base
                for r in cells[K]
                if _finite(r["t_first_cross"])
            ]
            numbers["ratio"][f"alpha={alpha:g} K={int(K)}"] = {
                "median_ratio": round(float(np.median(rat)), 3) if rat else None,
                "min": round(min(rat), 3) if rat else None,
                "max": round(max(rat), 3) if rat else None,
                "crossed": f"{len(rat)}/{st['n']}",
                "held": st["held"],
                "base_fc": base,
            }
            if not rat:
                censored_x(ax, kx(K), col)
                continue
            m = float(np.median(rat))
            xs.append(kx(K))
            med.append(m)
            lo.append(m - min(rat))
            hi.append(max(rat) - m)
            fr.append(st["held"])
            if len(rat) < st["n"]:
                ax.annotate(
                    f"{len(rat)}/{st['n']} crossed",
                    (kx(K), max(rat)),
                    xytext=(-6, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=6.5,
                    color=col,
                )
        ax.errorbar(
            xs,
            med,
            yerr=[lo, hi],
            color=col,
            lw=1.5,
            capsize=3,
            elinewidth=1,
            ls="-",
            marker="none",
            zorder=3,
            label=f"α = {alpha:g}",
        )
        for x, m, f in zip(xs, med, fr):
            mark(ax, x, m, col, f)
    op = sel(rows, group="boundary", partition="operand")
    if op:
        base = stats(cent_runs(rows, "A", 0.25))["fc"]
        st = stats(op)
        rat = [r["t_first_cross"] / base for r in op if _finite(r["t_first_cross"])]
        m = float(np.median(rat))
        xo = kx(97) + 0.22
        ax.errorbar(
            [xo],
            [m],
            yerr=[[m - min(rat)], [max(rat) - m]],
            color=ALPHA_COL[0.25],
            capsize=3,
            elinewidth=1,
            ls="none",
            marker="none",
            zorder=3,
        )
        mark(ax, xo, m, ALPHA_COL[0.25], st["held"], marker="s", size=5.5)
        ax.annotate(
            "operand",
            (xo, m),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=6.5,
            color=ALPHA_COL[0.25],
        )
        numbers["ratio"]["alpha=0.25 K=97 operand"] = {
            "median_ratio": round(m, 3),
            "min": round(min(rat), 3),
            "max": round(max(rat), 3),
            "crossed": f"{len(rat)}/{st['n']}",
            "held": st["held"],
            "fc": st["fc"],
        }
    ax.axhline(1.0, color=MUTED, ls="--", lw=0.9, zorder=2, label="No slowdown")
    cat_axis(ax, ["2", "5", "10", "20", "50", "97"], start=1)
    ax.set_ylim(0.85, 4.3)
    ax.set_xlabel("K (number of clients)")
    ax.set_ylabel(
        r"$t^{\mathrm{FL}}_{\mathrm{first\,cross}}\;/\;t^{\mathrm{cent}}_{\mathrm{first\,cross}}$"
    )
    ax.set_title("Slowdown vs K (A, IID)")
    ax.legend(loc="center left", title="Train fraction α")
    # (b), (c) the two clocks on A, B, D
    for s, alpha, wd in (("A", 0.3, None), ("B", 0.3, 0.1), ("D", 0.3, 1.0)):
        lad = k_ladder(rows, s, alpha, wd)
        Ks = [K for K in sorted(lad) if K in KS]
        sts = [lad[K] for K in Ks]
        xs = [kx(K) for K in Ks]
        fr = [st["held"] for st in sts]
        series(
            axes[1],
            xs,
            [st["memo"] for st in sts],
            fr,
            COL[s],
            label=NAME[s],
            err=span(sts, "memo"),
        )
        series(
            axes[2],
            xs,
            [st["delay"] for st in sts],
            fr,
            COL[s],
            label=NAME[s],
            err=span(sts, "delay"),
        )
        for K in Ks:
            if not _finite(lad[K]["memo"]):
                censored_x(axes[1], kx(K), COL[s])
            if not _finite(lad[K]["delay"]):
                censored_x(axes[2], kx(K), COL[s])
        tag = f"{s} α={alpha:g}" + (f" wd={wd:g}" if wd is not None else "")
        numbers["ladders"][tag] = {K: rounded(lad[K]) for K in Ks}
    for ax in axes[1:]:
        cat_axis(ax, ["cent", "2", "5", "10", "20", "50", "97"])
        ax.set_xlabel("K (number of clients)")
    log_steps(axes[1], "y", 100, 3e5)
    log_steps(axes[2], "y", 200, 3e5)
    axes[1].set_ylabel(TMEMO + " (gradient steps)")
    axes[2].set_ylabel("delay (gradient steps)")
    axes[1].set_title("Time to Memorise vs K")
    axes[2].set_title("Delay vs K")
    axes[1].legend(loc="upper left", handles=setup_handles("ABD") + [CENS_HANDLE])
    for ax, L in zip(axes, "abc"):
        letter(ax, L)
    fig.tight_layout()
    NUMBERS["fig1"] = numbers
    save(fig, "fig1_two_clocks")


# ── Fig 2: local work ─────────────────────────────────────────────────────────


def _e_ladder(rows, s):
    le = sel(rows, group="local_epochs", setup=s)
    if not le:
        return {}
    alpha, wd, ntr = le[0]["alpha"], le[0]["weight_decay"], le[0]["n_train"]
    ctrl = best_control(
        fed(
            rows,
            s,
            alpha=alpha,
            weight_decay=wd,
            n_train=ntr,
            num_clients=10.0,
            group=lambda g: g != "local_epochs",
        )
    )
    cells = by_cell(le + ctrl, "local_epochs")
    return {int(E): stats(rs) for E, rs in sorted(cells.items())}


def fig2(rows):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.5))
    numbers = {}
    lads = {s: _e_ladder(rows, s) for s in "ABCDE"}
    # (a) the draft's Fig. 3 on the anchor, with memorisation added: the gap is the delay
    ax = axes[0]
    la = lads["A"]
    Es = sorted(la)
    sts = [la[E] for E in Es]
    xs = [ex(E) for E in Es]
    fr = [st["held"] for st in sts]
    fc = [st["fc"] / 1e3 for st in sts]
    memo = [st["memo"] / 1e3 for st in sts]
    ax.fill_between(xs, memo, fc, color=COL["A"], alpha=0.10, lw=0, zorder=1)
    series(ax, xs, fc, fr, COL["A"], label="First crossing", err=span(sts, "fc", 1e3))
    series(
        ax,
        xs,
        memo,
        fr,
        COL["A"],
        ls="--",
        marker="s",
        label="Memorisation",
        err=span(sts, "memo", 1e3),
    )
    ax.annotate(
        "delay",
        (xs[-1], (memo[-1] + fc[-1]) / 2),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=7,
        color=COL["A"],
    )
    cat_axis(ax, [str(E) for E in ES])
    ax.set_ylim(0, 26)
    ax.set_ylabel("Gradient steps (×1000)")
    ax.set_title("Grokking Time vs E (A)")
    ax.legend(loc="upper left")
    # (b) memorisation in rounds, five setups
    for s in "ABCDE":
        lad = lads[s]
        if not lad:
            continue
        Es = sorted(lad)
        sts = [lad[E] for E in Es]
        fr = [st["held"] for st in sts]
        al = WITHHELD if s == "C" else 1.0
        lo = [st["memo_lo"] / E for st, E in zip(sts, Es)]
        hi = [st["memo_hi"] / E for st, E in zip(sts, Es)]
        series(
            axes[1],
            [ex(E) for E in Es],
            [
                st["memo"] / E if _finite(st["memo"]) else math.inf
                for st, E in zip(sts, Es)
            ],
            fr,
            COL[s],
            alpha=al,
            label=NAME[s] + (" (withheld)" if s == "C" else ""),
            err=(lo, hi),
        )
        numbers[s] = {E: rounded(lad[E]) for E in Es}
    log_steps(axes[1], "y")
    axes[1].set_ylabel(TMEMO + " / E (rounds)")
    axes[1].set_title("Memorisation Rounds vs E")
    # (c) the delay, four setups (C's is ~0 from E=25 and withheld)
    for s in "ABDE":
        lad = lads[s]
        Es = sorted(lad)
        sts = [lad[E] for E in Es]
        fr = [st["held"] for st in sts]
        series(
            axes[2],
            [ex(E) for E in Es],
            [st["delay"] for st in sts],
            fr,
            COL[s],
            label=NAME[s],
            err=span(sts, "delay"),
        )
        for E in Es:
            if not _finite(lad[E]["delay"]):
                axes[2].plot(
                    [ex(E)],
                    [0.96],
                    marker="*",
                    ms=10,
                    mec=COL[s],
                    mfc=COL[s],
                    ls="none",
                    transform=axes[2].get_xaxis_transform(),
                    clip_on=False,
                    zorder=6,
                )
    log_steps(axes[2], "y")
    axes[2].set_ylabel("delay (gradient steps)")
    axes[2].set_title("Delay vs E")
    fig.legend(
        handles=setup_handles("ABCDE")
        + [
            Line2D([], [], marker="*", color=COL["D"], ls="none", ms=9, label="Fixed point (Fig. 6)"),
            CENS_HANDLE,
        ],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.16),
        fontsize=6.5,
    )
    for ax in axes[1:]:
        cat_axis(ax, [str(E) for E in ES])
    for ax, L in zip(axes, "abc"):
        ax.set_xlabel("Local epochs (E)")
        letter(ax, L)
    fig.tight_layout()
    NUMBERS["fig2"] = numbers
    save(fig, "fig2_local_work")


# ── Fig 3: participation ──────────────────────────────────────────────────────


def _divergence(r):
    """Mean client weight divergence per round before the first crossing.

    A whole-run mean is dominated by the post-grokking tail, whose length is set
    by the budget rather than by the dynamics; the window that bears on the delay
    is the one before the bar is reached. Censored runs use the whole run.
    """
    h = history(r["id"])
    if not h or not h.get("client_weight_divergence"):
        return None
    st = np.asarray(h["total_steps"], dtype=float)
    d = np.asarray(h["client_weight_divergence"], dtype=float)
    cut = r["t_first_cross"] if _finite(r["t_first_cross"]) else math.inf
    m = (st > 0) & (st <= cut) & np.isfinite(d)
    return float(d[m].mean()) if m.any() else None


def _f_ladders(rows):
    panels = [
        (s, sel(rows, group="participation_setups", setup=s), 20.0, "-", NAME[s])
        for s in "ABCDE"
    ]
    panels.append(
        ("A", sel(rows, group="participation", setup="A"), 50.0, "--", "A, K=50")
    )
    out = []
    for s, ps, Kc, ls, tag in panels:
        if not ps:
            continue
        alpha, wd, ntr = ps[0]["alpha"], ps[0]["weight_decay"], ps[0]["n_train"]
        ctrl = best_control(
            fed(
                rows,
                s,
                alpha=alpha,
                weight_decay=wd,
                n_train=ntr,
                num_clients=Kc,
                group=lambda g: g not in ("participation_setups", "participation"),
            )
        )
        cells = by_cell(ps + ctrl, "fraction_train")
        rounds = {}
        for f in sorted(cells):
            rs = cells[f]
            E = rs[0]["local_epochs"]
            conv = lambda t: t / (E * f) if _finite(t) else math.inf
            st = stats(rs)
            st["memo_r"] = km(
                [conv(r["t_memo"]) for r in rs], [conv(r["budget"]) for r in rs]
            )
            st["fc_r"] = km(
                [conv(r["t_first_cross"]) for r in rs],
                [conv(r["budget"]) for r in rs],
            )
            st["fc_r_lo"], st["fc_r_hi"] = _span(
                [conv(r["t_first_cross"]) for r in rs if _finite(r["t_first_cross"])]
            )
            divs = [d for d in (_divergence(r) for r in rs) if d is not None]
            st["div"] = float(np.median(divs)) if divs else math.inf
            st["div_lo"], st["div_hi"] = _span(divs)
            rounds[f] = st
        out.append((s, ls, tag, rounds))
    return out


def fig3(rows):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.55))
    numbers = {}
    for s, ls, tag, rounds in _f_ladders(rows):
        fs = sorted(rounds)
        sts = [rounds[f] for f in fs]
        fr = [st["held"] for st in sts]
        al = WITHHELD if s == "C" else 1.0
        lab = tag + (" (withheld)" if s == "C" else "")
        series(
            axes[0],
            fs,
            [st["fc_r"] for st in sts],
            fr,
            COL[s],
            ls=ls,
            alpha=al,
            label=lab,
            err=span(sts, "fc_r"),
        )
        series(
            axes[1],
            fs,
            [st["div"] for st in sts],
            fr,
            COL[s],
            ls=ls,
            alpha=al,
            label=lab,
            err=span(sts, "div"),
        )
        series(
            axes[2],
            fs,
            [st["final"] for st in sts],
            fr,
            COL[s],
            ls=ls,
            alpha=al,
            label=lab,
            err=span(sts, "final"),
        )
        numbers[tag] = {f: rounded(rounds[f], 4) for f in fs}
    for ax, t in zip(
        axes,
        [
            "First Crossing vs f",
            "Client Divergence vs f",
            "Final Test Accuracy vs f",
        ],
    ):
        ax.set_title(t)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(0.12, 1.1)
        ax.set_xlabel("Participation fraction (f)")
    log_steps(axes[0], "y")
    axes[0].set_ylabel(TFC + " (rounds)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Client divergence (mean before crossing)")
    axes[2].set_ylim(60, 102)
    axes[2].set_ylabel("Test accuracy (%)")
    fig.legend(
        handles=setup_handles("ABCDE")
        + [Line2D([], [], color=COL["A"], ls="--", marker="o", label="A, K=50")],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.2),
        fontsize=6.5,
    )
    for ax, L in zip(axes, "abc"):
        letter(ax, L)
    fig.tight_layout()
    NUMBERS["fig3"] = numbers
    save(fig, "fig3_participation")


# ── Fig 4: heterogeneity and structure ────────────────────────────────────────

COHERENT = {"A": "operand", "A'": "operand", "B": "operand", "C": "coset", "D": "coset"}
DIRX = [0.01, 0.1, 0.5, 1, 10, 1000]
DIRL = ["0.01", "0.1", "0.5", "1", "10", "1000"]
OFF4 = {"B": 0.02, "C": -0.02, "D": -0.05, "E": 0.05}  # vertical offsets, panel (b)
DIRLAB = r"Dirichlet $\alpha_{\mathrm{dir}}$ (non-IID ← → IID)"


def fig4(rows):
    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1, 1.05],
        width_ratios=[1, 1, 1.25],
        hspace=0.8,
        wspace=0.75,
        top=0.95,
        bottom=0.16,
        left=0.1,
        right=0.98,
    )
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    axd = fig.add_subplot(gs[1, :])
    numbers = {"anchor": {}, "held": {}, "starvation": {}, "structure": {}}
    # (a) the draft's Fig. 2 on the anchor: first crossing vs concentration
    k10 = sel(rows, group="dirichlet_setups", setup="A") + sel(
        rows, group="k_fixed_total", setup="A", partition="dirichlet", num_clients=10.0
    )
    arms = [("K=10, α=0.30", k10, ALPHA_COL[0.3], "-", "o")]
    for K, ls, mk in ((20.0, "--", "s"), (50.0, ":", "^")):
        arms.append(
            (
                f"K={int(K)}, α=0.25",
                sel(rows, group="dirichlet_band", setup="A", num_clients=K),
                ALPHA_COL[0.25],
                ls,
                mk,
            )
        )
    for lab, rs, col, ls, mk in arms:
        cells = by_cell(rs, "dirichlet_alpha")
        das = sorted(cells)
        sts = [stats(cells[d]) for d in das]
        series(
            axa,
            das,
            [st["fc"] / 1e3 for st in sts],
            [st["held"] for st in sts],
            col,
            ls=ls,
            marker=mk,
            label=lab,
            err=span(sts, "fc", 1e3),
        )
        for d, st in zip(das, sts):
            if not _finite(st["fc"]):
                censored_x(axa, d, col)
        numbers["anchor"][lab] = {d: rounded(st) for d, st in zip(das, sts)}
    logx_ticks(axa, DIRX, DIRL)
    axa.set_ylim(0, 80)
    axa.set_ylabel(TFC + " (gradient steps ×1000)")
    axa.set_title("Heterogeneity (A)")
    axa.legend(
        loc="upper right",
        fontsize=6,
        handles=[
            Line2D([], [], color=ALPHA_COL[0.3], marker="o", label="K=10, α=0.30"),
            Line2D(
                [], [], color=ALPHA_COL[0.25], marker="s", ls="--", label="K=20, α=0.25"
            ),
            Line2D(
                [], [], color=ALPHA_COL[0.25], marker="^", ls=":", label="K=50, α=0.25"
            ),
            CENS_HANDLE,
        ],
    )
    # (b) the other setups at K=10: fraction crossed, fill = memorised
    for s in "BCDE":
        ds = sel(rows, group="dirichlet_setups", setup=s)
        if not ds:
            continue
        alpha, wd, ntr = ds[0]["alpha"], ds[0]["weight_decay"], ds[0]["n_train"]
        mid = sel(
            rows,
            group="partitions",
            setup=s,
            partition="dirichlet",
            num_clients=10.0,
            alpha=alpha,
            weight_decay=wd,
            n_train=ntr,
        )
        cells = by_cell(ds + mid, "dirichlet_alpha")
        das = sorted(cells)
        st = {d: stats(cells[d]) for d in das}
        al = WITHHELD if s == "C" else 1.0
        series(
            axb,
            das,
            [st[d]["crossed"] + OFF4[s] for d in das],
            [1.0 if st[d]["peak"] >= 99 else 0.0 for d in das],
            COL[s],
            alpha=al,
            label=s + (" (withheld)" if s == "C" else ""),
        )
        numbers["held"][s] = {
            d: {
                "held": st[d]["held"],
                "crossed": st[d]["crossed"],
                "peak": round(st[d]["peak"], 1),
                "fc": (round(st[d]["fc"]) if _finite(st[d]["fc"]) else None),
            }
            for d in das
        }
    logx_ticks(axb, DIRX, DIRL)
    axb.set_ylim(-0.1, 1.12)
    axb.set_yticks([0, 0.5, 1.0])
    axb.set_ylabel("Fraction of runs that crossed")
    axb.set_title("Fraction Crossed (K=10)")
    axb.legend(
        loc="center right",
        fontsize=6,
        handles=[Line2D([], [], color=COL[s], marker="o", label=s) for s in "BCDE"]
        + [
            Line2D([], [], marker="o", color=INK2, ls="none", ms=5, label="memorised"),
            Line2D(
                [],
                [],
                marker="o",
                color=INK2,
                mfc="white",
                ls="none",
                ms=5,
                mew=1.2,
                label="never memorised",
            ),
        ],
    )
    for ax in (axa, axb):
        ax.set_xlabel(DIRLAB)
    # (c) the starvation control on the anchor, run for run
    conc = sel(rows, group="dirichlet_band", dirichlet_alpha=lambda d: d in (0.01, 0.1))
    ctrl = sel(rows, group="size_control")
    ykeys, y = [], 0
    for K in (20.0, 50.0):
        for d in (0.01, 0.1):
            for seed in (42.0, 123.0, 456.0):
                a = [
                    r
                    for r in conc
                    if r["num_clients"] == K
                    and r["dirichlet_alpha"] == d
                    and r["seed"] == seed
                ]
                b = [
                    r
                    for r in ctrl
                    if r["num_clients"] == K
                    and r["dirichlet_alpha"] == d
                    and r["seed"] == seed
                ]
                if not a or not b:
                    continue
                for r, col, mk in (
                    (a[0], PART["target"], "o"),
                    (b[0], PART["dirichlet"], "s"),
                ):
                    t = r["t_first_cross"]
                    if _finite(t):
                        axc.plot(
                            [t],
                            [y],
                            marker=mk,
                            ms=5,
                            mfc=col,
                            mec=col,
                            ls="none",
                            zorder=3,
                        )
                    else:
                        axc.plot(
                            [r["budget"]],
                            [y],
                            marker="x",
                            ms=7,
                            mec=col,
                            mew=1.5,
                            ls="none",
                            zorder=3,
                        )
                ykeys.append((y, f"K{int(K)}, {d:g}, run {int(seed)}"))
                numbers["starvation"][f"K{int(K)} dir{d:g} seed{int(seed)}"] = {
                    "concentrated": a[0]["t_first_cross"]
                    if _finite(a[0]["t_first_cross"])
                    else None,
                    "size_only": b[0]["t_first_cross"]
                    if _finite(b[0]["t_first_cross"])
                    else None,
                }
                y += 1
    axc.axhline(5.5, color=GRID, lw=0.8)
    axc.set_yticks([k for k, _ in ykeys])
    axc.set_yticklabels([lab for _, lab in ykeys], fontsize=5.8)
    log_steps(axc, "x")
    axc.set_xlabel(TFC + " (gradient steps)")
    axc.set_title("Starvation Control (A)")
    axc.invert_yaxis()
    axc.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                color=PART["target"],
                ls="none",
                label="Labels concentrated",
            ),
            Line2D(
                [],
                [],
                marker="s",
                color=PART["dirichlet"],
                ls="none",
                label="Same shard sizes, labels random",
            ),
            Line2D(
                [],
                [],
                marker="x",
                color=INK2,
                ls="none",
                mew=1.5,
                label="Failed to grok (100k budget)",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(-0.5, -0.3),
        ncol=1,
        fontsize=6,
    )
    # (d) partition structure against matched iid baselines
    groups, x, seps, last = [], 0, [], None
    YTOP = 30
    for s in "ABCDE":
        ps = sel(rows, group="partitions", setup=s)
        for K in sorted({r["num_clients"] for r in ps}):
            if s == "C" and K == 50.0:
                continue  # a training failure on every arm, not a partition result
            here = [r for r in ps if r["num_clients"] == K]
            alpha, wd, ntr = (
                here[0]["alpha"],
                here[0]["weight_decay"],
                here[0]["n_train"],
            )
            base = best_control(
                fed(
                    rows,
                    s,
                    alpha=alpha,
                    weight_decay=wd,
                    n_train=ntr,
                    num_clients=K,
                    group=lambda g: (
                        g in ("aggregation", "k_fixed_total", "setup_k_ladder")
                    ),
                )
            )
            if not base:
                continue
            if last is not None and s != last:
                seps.append(x - 0.5)
            last = s
            bst = stats(base)
            parts = by_cell(here, "partition")
            order = [
                p
                for p in ("operand", "coset", "target", "label_block", "dirichlet")
                if p in parts
            ]
            width = 0.8 / max(len(order), 1)
            al = WITHHELD if s == "C" else 1.0
            for i, p in enumerate(order):
                st = stats(parts[p])
                ratio = (
                    st["fc"] / bst["fc"]
                    if _finite(st["fc"]) and _finite(bst["fc"])
                    else math.inf
                )
                xi = x + (i - (len(order) - 1) / 2) * width
                frac_txt = f"{int(st['held'] * st['n'])}/{st['n']}"
                if _finite(ratio):
                    axd.bar(
                        xi,
                        ratio,
                        width=width * 0.92,
                        color=PART[p],
                        zorder=3,
                        alpha=al,
                        edgecolor="white",
                        lw=0.4,
                    )
                    if st["held"] < 0.999:
                        axd.text(
                            xi,
                            ratio,
                            frac_txt,
                            ha="center",
                            va="bottom",
                            fontsize=5.5,
                            color=INK2,
                        )
                else:
                    axd.bar(
                        xi,
                        YTOP,
                        width=width * 0.92,
                        color="white",
                        edgecolor=PART[p],
                        hatch="///",
                        lw=0.6,
                        zorder=3,
                        alpha=al * 0.85,
                    )
                    axd.text(
                        xi,
                        YTOP * 1.08,
                        frac_txt,
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                        color=PART[p],
                        clip_on=False,
                        zorder=4,
                    )
                numbers["structure"][f"{s} K={int(K)} {p}"] = {
                    "ratio": round(ratio, 2) if _finite(ratio) else None,
                    "held": st["held"],
                    "iid_fc": round(bst["fc"]) if _finite(bst["fc"]) else None,
                    "iid_held": bst["held"],
                }
            lab = f"{s}, K={int(K)}"
            if bst["held"] < 0.999:
                lab += f"\n(iid {int(bst['held'] * bst['n'])}/{bst['n']})"
            if s == "C":
                lab += "\nwithheld"
            groups.append((x, lab))
            x += 1
    for sx in seps:
        axd.axvline(sx, color="#bbbbbb", lw=0.7, zorder=1)
    axd.axhline(1.0, color=INK2, lw=0.9, ls="--", zorder=2)
    axd.set_xticks([g for g, _ in groups])
    axd.set_xticklabels([l for _, l in groups], fontsize=6.5)
    axd.set_xlim(-0.6, x - 0.4)
    axd.set_yscale("log")
    axd.set_ylim(0.3, YTOP)
    axd.set_yticks([0.5, 1, 2, 5, 10])
    axd.set_yticklabels(["0.5×", "1×", "2×", "5×", "10×"])
    axd.set_ylabel(TFC + " / matched IID")
    axd.set_title("Partition Structure vs Matched IID Baseline")
    axd.legend(
        handles=[
            Patch(color=PART[p], label=lab)
            for p, lab in (
                ("operand", "Operand (coherent on mod p)"),
                ("coset", "Coset (coherent on $S_5$)"),
                ("target", "Target (incoherent)"),
                ("label_block", "Label block (incoherent)"),
                ("dirichlet", "Dirichlet 0.5 (unstructured)"),
            )
        ]
        + [
            Patch(
                facecolor="white", edgecolor=INK2, hatch="///", label="Failed to grok"
            )
        ],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        fontsize=6.5,
    )
    for ax, L in zip((axa, axb, axc, axd), "abcd"):
        letter(ax, L)
    NUMBERS["fig4"] = numbers
    save(fig, "fig4_heterogeneity_structure")


# ── Fig 5: drift and mitigation ───────────────────────────────────────────────

AXIS_KIND = {
    "participation": "Sampling (free)",
    "participation_setups": "Sampling (free)",
    "size_control": "Shard sizes (free)",
    "dirichlet_band": "Label conflict",
    "dirichlet_setups": "Label conflict",
    "aggregation": "Clients K",
    "k_fixed_total": "K / partition",
    "setup_k_ladder": "Clients K",
    "boundary": "Clients K",
    "local_epochs": "Local epochs E",
    "probe": "Local epochs E",
    "partitions": "Partition",
}
KIND_COL = {
    "Sampling (free)": "#2ca02c",
    "Shard sizes (free)": "#9467bd",
    "Label conflict": "#d62728",
    "Clients K": "#1f77b4",
    "K / partition": "#1f77b4",
    "Local epochs E": "#ff7f0e",
    "Partition": "#e377c2",
}
CELLS5 = [
    ("H1", 0.25, 25.0, "iid"),
    ("H2", 0.25, 25.0, "dirichlet"),
    ("H3", 0.3, 50.0, "dirichlet"),
]
ORDER5 = ["fedavg", "fedadam", "fedyogi", "scaffold", "fedavgm", "fedprox"]
LAB5 = {
    "fedavg": "FedAvg",
    "fedadam": "FedAdam",
    "fedyogi": "FedYogi",
    "scaffold": "SCAFFOLD",
    "fedavgm": "FedAvgM",
    "fedprox": "FedProx (μ=0.01)",
}


def _cell_title(name, alpha, E, part):
    kind = "IID" if part == "iid" else "non-IID"
    return f"{name}: α={alpha:g}, E={int(E)}, {kind}"


def fig5(rows):
    fig = plt.figure(figsize=(7.2, 5.3))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.15, 1],
        hspace=0.5,
        wspace=0.32,
        top=0.95,
        bottom=0.14,
        left=0.1,
        right=0.98,
    )
    axa = fig.add_subplot(gs[0, :])
    axb = [fig.add_subplot(gs[1, i]) for i in range(3)]
    numbers = {"scatter_n": 0, "algorithms": {}}
    # (a) delay vs divergence on the anchor, every FedAvg run with a history
    an = [
        r
        for r in rows
        if r["mode"] == "federated"
        and r["setup"] == "A"
        and r["strategy"] == "fedavg"
        and r["group"] in AXIS_KIND
    ]
    pts = collections.defaultdict(list)
    for r in an:
        d = _divergence(r)
        if d is None or d <= 0:
            continue
        kind = AXIS_KIND[r["group"]]
        if r["group"] == "k_fixed_total" and r["partition"] != "iid":
            kind = "Partition"
        pts[kind].append((d, r["delay"] if r["delay"] is not None else None, r))
    for kind, ps in pts.items():
        fin = [(d, y) for d, y, _ in ps if y is not None]
        cen = [d for d, y, _ in ps if y is None]
        axa.plot(
            [p[0] for p in fin],
            [p[1] for p in fin],
            "o",
            ms=3.8,
            mfc=KIND_COL[kind],
            mec="white",
            mew=0.5,
            alpha=0.9,
            label=f"{kind} (n={len(ps)})",
            zorder=3,
        )
        if cen:
            axa.plot(
                cen,
                [0.96] * len(cen),
                marker="x",
                ms=6,
                mec=KIND_COL[kind],
                mew=1.3,
                ls="none",
                transform=axa.get_xaxis_transform(),
                clip_on=True,
                zorder=4,
            )
        numbers["scatter_n"] += len(ps)
    for strat, mk in (("scaffold", "D"), ("fedprox", "X")):
        rs = sel(rows, group="algorithms", strategy=strat, setup="A")
        for r in rs:
            d = _divergence(r)
            if d is None or d <= 0:
                continue
            if r["delay"] is not None:
                axa.plot(
                    [d],
                    [r["delay"]],
                    marker=mk,
                    ms=6.5,
                    mfc=STRAT[strat],
                    mec="white",
                    ls="none",
                    zorder=5,
                )
            else:
                axa.plot(
                    [d],
                    [0.96],
                    marker=mk,
                    ms=6.5,
                    mfc="white",
                    mec=STRAT[strat],
                    ls="none",
                    transform=axa.get_xaxis_transform(),
                    clip_on=False,
                    zorder=5,
                )
    axa.plot([], [], marker="D", color=STRAT["scaffold"], ls="none", label="SCAFFOLD")
    axa.plot(
        [],
        [],
        marker="X",
        color=STRAT["fedprox"],
        ls="none",
        label="FedProx (hollow = failed to grok)",
    )
    axa.set_xscale("log")
    axa.set_xlim(5e-5, 0.2)
    off = sum(1 for ps in pts.values() for d, _, _ in ps if d > 0.2)
    if off:
        axa.text(
            0.99,
            0.88,
            f"{off} failed run off-scale at divergence ~7×10² (a starved client)",
            transform=axa.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color=INK2,
        )
    log_steps(axa, "y")
    axa.set_xlabel("Mean client divergence per round (before first crossing)")
    axa.set_ylabel("delay (gradient steps)")
    axa.set_title("Delay vs Client Divergence (setup A, FedAvg, every design axis)")
    axa.legend(ncol=2, fontsize=6.3, loc="lower left", title="Design axis")
    # (b-d) the draft's Fig. 8 form: trajectories per hard cell, every method overlaid
    for ax, (name, alpha, E, part) in zip(axb, CELLS5):
        rs = sel(rows, group="algorithms", alpha=alpha, local_epochs=E, partition=part)
        cells = by_cell(rs, "strategy")
        for strat in ORDER5:
            runs = cells.get(strat, [])
            if not runs:
                continue
            st = stats(runs)
            numbers["algorithms"][f"{name} {strat}"] = {
                "fc": round(st["fc"]) if _finite(st["fc"]) else None,
                "held": st["held"],
                "n": st["n"],
            }
            for k, r in enumerate(runs):
                h = history(r["id"])
                if not h:
                    continue
                xs = [x for x in h["total_steps"] if x > 0]
                ys = h["test_acc"][len(h["total_steps"]) - len(xs) :]
                ax.plot(
                    xs,
                    ys,
                    color=STRAT[strat],
                    lw=0.9,
                    alpha=0.85,
                    label=LAB5[strat] if (k == 0 and ax is axb[2]) else None,
                )
        ax.axhline(95, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        log_steps(ax, "x")
        ax.set_ylim(-3, 103)
        ax.set_title(_cell_title(name, alpha, E, part), fontsize=7.5)
        ax.set_xlabel("Gradient steps")
    axb[0].set_ylabel("Test accuracy (%)")
    fig.legend(
        handles=[Line2D([], [], color=STRAT[k], lw=1.5, label=LAB5[k]) for k in ORDER5],
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=6.5,
    )
    for ax, L in zip([axa] + axb, "abcd"):
        letter(ax, L)
    NUMBERS["fig5"] = numbers
    save(fig, "fig5_drift_mitigation")


# ── Fig 6: the fixed point ────────────────────────────────────────────────────


def _traj(ax, rs, xkey, key, color, xscale=1.0, lw=1.1, alpha=0.9, label=None):
    for k, r in enumerate(rs):
        h = history(r["id"])
        if not h or key not in h:
            continue
        xs = np.asarray(h[xkey], dtype=float) / xscale
        ax.plot(
            xs, h[key], color=color, lw=lw, alpha=alpha, label=label if k == 0 else None
        )


def _stationarity(ax, rs, ref_step, xscale):
    """Median over runs of each quantity divided by its value at the reference step."""
    specs = (
        ("train_loss", "#9467bd", "-", "Train loss"),
        ("weight_norm_total", "#2ca02c", "--", "Weight norm"),
        ("mean_client_drift", "#ff7f0e", ":", "Client drift / round"),
    )
    out = {}
    for key, col, ls, lab in specs:
        curves = []
        for r in rs:
            h = history(r["id"])
            if not h or key not in h:
                continue
            st = np.asarray(h["total_steps"], dtype=float)
            v = np.asarray(h[key], dtype=float)
            i = int(np.argmin(np.abs(st - ref_step)))
            m = st >= st[i]
            curves.append((st[m], v[m] / v[i]))
        if not curves:
            continue
        grid = curves[0][0]
        med = np.median(
            np.vstack([np.interp(grid, c[0], c[1]) for c in curves]), axis=0
        )
        ax.plot(grid / xscale, med, color=col, ls=ls, lw=1.3, label=lab)
        out[lab] = {
            "at_end": round(float(med[-1]), 4),
            "min": round(float(med.min()), 4),
            "max": round(float(med.max()), 4),
        }
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_ylim(0.6, 1.25)
    return out


def fig6(rows):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(6.8, 4.4),
        sharex="col",
        gridspec_kw=dict(hspace=0.35, wspace=0.3),
    )
    left = sel(rows, group="e50_long", setup="D")
    right = sel(rows, group="dirichlet_setups", setup="D", dirichlet_alpha=1.0)
    _traj(
        axes[0, 0],
        left,
        "total_steps",
        "test_acc",
        COL["D"],
        xscale=1e6,
        label="D, K=10, E=50, IID",
    )
    _traj(
        axes[0, 1],
        right,
        "total_steps",
        "test_acc",
        COL["D"],
        xscale=1e3,
        label="D, K=10, E=5, Dirichlet 1.0",
    )
    cent = sel(rows, group="central_anchor", setup="D", alpha=0.3)
    for k, r in enumerate(cent):
        h = history(r["id"])
        if not h:
            continue
        for ax, sc in ((axes[0, 0], 1e6), (axes[0, 1], 1e3)):
            ax.plot(
                np.asarray(h["epoch"]) / sc,
                h["test_acc"],
                color=MUTED,
                lw=0.8,
                alpha=0.7,
                label="Centralized D, α=0.30" if k == 0 else None,
            )
    for j in range(2):
        axes[0, j].axhline(85, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        axes[0, j].set_ylim(0, 102)
        axes[0, j].legend(fontsize=6.3, loc="lower right")
    axes[0, 0].axvline(0.25, color=INK2, lw=0.7)
    axes[0, 0].annotate(
        "original 250k budget",
        (0.25, 55),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=6.3,
        color=INK2,
    )
    axes[0, 0].set_ylabel("Test accuracy (%)")
    axes[0, 0].set_title("Test Accuracy, E=50 (2M steps)")
    axes[0, 1].set_title("Test Accuracy, Dirichlet 1.0")
    stat_l = _stationarity(axes[1, 0], left, 250_000, 1e6)
    stat_r = _stationarity(axes[1, 1], right, 100_000, 1e3)
    axes[1, 0].set_title("Normalised at 250k steps")
    axes[1, 1].set_title("Normalised at 100k steps")
    axes[1, 0].set_ylabel("Value / value at reference")
    axes[1, 0].set_xlabel("Gradient steps (millions)")
    axes[1, 1].set_xlabel("Gradient steps (thousands)")
    axes[1, 0].legend(fontsize=6.3, loc="upper left", ncol=1)
    for ax, L in zip(axes.ravel(), "abcd"):
        letter(ax, L)
    num = {"stationarity_left": stat_l, "stationarity_right": stat_r}
    for r in left:
        h = history(r["id"])
        if h:
            n = len(h["total_steps"])
            i250 = min(range(n), key=lambda i: abs(h["total_steps"][i] - 250000))
            num[f"seed {int(r['seed'])}"] = {
                "t_memo": r["t_memo"],
                "test@250k": h["test_acc"][i250],
                "test@2M": h["test_acc"][-1],
                "train_loss@250k": h["train_loss"][i250],
                "train_loss@2M": h["train_loss"][-1],
                "wnorm@250k": h["weight_norm_total"][i250],
                "wnorm@2M": h["weight_norm_total"][-1],
                "drift@250k": h["mean_client_drift"][i250],
                "drift@2M": h["mean_client_drift"][-1],
            }
    NUMBERS["fig6"] = num
    save(fig, "fig6_fixed_point")


# ── Fig 7: mechanism ──────────────────────────────────────────────────────────


def _band(ax, curves, color, label, lw=1.8):
    grid = curves[0][0]
    stack = np.vstack([np.interp(grid, c[0], c[1]) for c in curves])
    ax.fill_between(
        grid, stack.min(axis=0), stack.max(axis=0), color=color, alpha=0.18, lw=0
    )
    med = np.median(stack, axis=0)
    ax.plot(grid, med, color=color, lw=lw, label=label)
    return grid, med


def fig7(rows):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.45))
    b97 = sel(rows, group="boundary", num_clients=97.0)
    arms = by_cell(b97, "partition")
    numbers = {"ipr": {}, "earliest_cross_round": None}
    earliest = math.inf
    for part, col, lab in (
        ("iid", "#7f7f7f", "IID"),
        ("operand", "#1f77b4", "Operand"),
    ):
        curves = []
        for r in arms.get(part, []):
            h = history(r["id"])
            if not h:
                continue
            curves.append((np.asarray(h["round"]), np.asarray(h["ipr"], dtype=float)))
            if _finite(r["t_first_cross"]):
                earliest = min(earliest, r["t_first_cross"] / 5.0)
        if curves:
            grid, med = _band(
                axes[0], curves, col, f"{lab} (median, range of {len(curves)})"
            )
            numbers["ipr"][part] = {
                int(x): round(float(y), 4)
                for x, y in zip(grid, med)
                if x in (2000, 4000, 6000, 8000, 10000, 13000, 16000, 19000, 20000)
            }
    numbers["earliest_cross_round"] = round(earliest) if _finite(earliest) else None
    if _finite(earliest):
        for ax in axes[:2]:
            ax.axvline(earliest, color=INK2, lw=0.7, ls="--")
        axes[0].annotate(
            "earliest first\ncrossing in the cell",
            (earliest, 0.03),
            xycoords=("data", "axes fraction"),
            va="bottom",
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            fontsize=6.3,
            color=INK2,
        )
    axes[0].set_title("Global IPR (K=97, α=0.25)")
    axes[0].legend(fontsize=6.3, loc="upper left")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("IPR (Fourier concentration)")
    groups = {True: [], False: []}
    for r in arms.get("iid", []):
        h = history(r["id"])
        if h:
            groups[r["grokked"]].append(
                (np.asarray(h["round"]), np.asarray(h["ipr"], dtype=float))
            )
    for ok, col, lab in (
        (False, "#7f7f7f", "Never crossed"),
        (True, "#1f77b4", "Crossed later"),
    ):
        if groups[ok]:
            _band(axes[1], groups[ok], col, f"{lab} ({len(groups[ok])})", lw=1.5)
    axes[1].set_title("IID Runs by Outcome")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("IPR (Fourier concentration)")
    axes[1].legend(fontsize=6.3, loc="upper left")
    for ax in axes[:2]:
        ax.xaxis.set_major_locator(FixedLocator([0, 5000, 10000, 15000, 20000]))
        ax.xaxis.set_major_formatter(FuncFormatter(_human))
    di = sel(rows, group="d_internals", alpha=0.4)
    for k, r in enumerate(di):
        h = history(r["id"])
        if not h or "circ_acc_interaction" not in h:
            continue
        ep = np.asarray(h["epoch"], dtype=float)
        m = ep > 0
        axes[2].plot(
            ep[m],
            np.asarray(h["circ_acc_interaction"])[m],
            color="#9467bd",
            lw=1.0,
            alpha=0.85,
            label="Compositional circuit T alone" if k == 0 else None,
        )
        axes[2].plot(
            ep[m],
            np.asarray(h["test_acc"])[m],
            color=COL["D"],
            lw=1.0,
            alpha=0.85,
            label="Full model, test" if k == 0 else None,
        )
    for x, lab, ha, dx in ((1800, "plateau", "right", -3), (3600, "dip", "left", 3)):
        axes[2].axvline(x, color=INK2, lw=0.7, ls="--")
        axes[2].annotate(
            lab,
            (x, 100),
            xytext=(dx, -8),
            ha=ha,
            textcoords="offset points",
            fontsize=6.3,
            color=INK2,
        )
    log_steps(axes[2], "x")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_title("Circuit vs Full Model (D)")
    axes[2].legend(fontsize=6.3, loc="lower right")
    for ax, L in zip(axes, "abc"):
        letter(ax, L)
    fig.tight_layout()
    NUMBERS["fig7"] = numbers
    save(fig, "fig7_mechanism")


# ── Appendix A1: the two clocks on every setup ────────────────────────────────


def figA1(rows):
    setups = ["A", "B", "C", "D", "E"]
    fig, axes = plt.subplots(2, 5, figsize=(7.4, 3.8), sharex=False)
    numbers = {}
    for j, s in enumerate(setups):
        rs = [r for r in fed(rows, s) if r["group"] in SRC1]
        ser = by_cell(rs, "alpha", "weight_decay", "n_train")
        primary = sorted(ser, key=lambda kv: (-kv[0], kv[1]))[0][:2] if ser else None
        for (alpha, wd, ntr), runs in sorted(
            ser.items(), key=lambda kv: (-kv[0][0], kv[0][1])
        ):
            cells = by_cell(runs, "num_clients")
            Ks = sorted(cells)
            if len(Ks) < 2:
                continue
            cent = best_control(
                [
                    r
                    for r in sel(
                        rows,
                        mode="centralized",
                        setup=s,
                        alpha=alpha,
                        weight_decay=wd,
                        n_train=ntr,
                    )
                    if r["arm"] != "cent_reduced" and r["group"] != "d_internals"
                ]
            )
            lad = {}
            if cent:
                lad[1] = stats(cent)
            for K in Ks:
                lad[int(K)] = stats(cells[K])
            Kk = [K for K in sorted(lad) if K in KS]
            sts = [lad[K] for K in Kk]
            tag = (
                (f"α={alpha:g}" + (f", wd={wd:g}" if s in "BCD" else ""))
                if s != "E"
                else f"n={ntr:g}"
            )
            ls = "-" if (len(ser) == 1 or (alpha, wd) == primary) else "--"
            al = WITHHELD if s == "C" else 1.0
            xs = [kx(K) for K in Kk]
            fr = [st["held"] for st in sts]
            series(
                axes[0, j],
                xs,
                [st["memo"] for st in sts],
                fr,
                COL[s],
                ls=ls,
                alpha=al,
                size=4,
                label=tag,
                err=span(sts, "memo"),
            )
            series(
                axes[1, j],
                xs,
                [st["delay"] for st in sts],
                fr,
                COL[s],
                ls=ls,
                alpha=al,
                size=4,
                label=tag,
                err=span(sts, "delay"),
            )
            for K in Kk:
                if not _finite(lad[K]["memo"]):
                    censored_x(
                        axes[0, j],
                        kx(K),
                        COL[s],
                        y=0.96 if ls == "-" else 0.88,
                        size=6,
                        alpha=al,
                    )
                if not _finite(lad[K]["delay"]):
                    censored_x(
                        axes[1, j],
                        kx(K),
                        COL[s],
                        y=0.96 if ls == "-" else 0.88,
                        size=6,
                        alpha=al,
                    )
            numbers[f"{s} {tag}"] = {K: rounded(lad[K]) for K in Kk}
        if s == "A":
            op = sel(rows, group="boundary", partition="operand")
            if op:
                st = stats(op)
                mark(
                    axes[0, j],
                    kx(97),
                    st["memo"],
                    COL[s],
                    st["held"],
                    marker="s",
                    size=5,
                )
                mark(
                    axes[1, j],
                    kx(97),
                    st["delay"],
                    COL[s],
                    st["held"],
                    marker="s",
                    size=5,
                )
                axes[1, j].plot(
                    [], [], marker="s", color=COL[s], ls="none", label="operand, K=97"
                )
                numbers["A operand K=97"] = rounded(st)
        for i in range(2):
            ax = axes[i, j]
            cat_axis(ax, ["cent", "2", "5", "10", "20", "50", "97"])
            ax.tick_params(labelsize=5.5)
            log_steps(ax, "y", *((100, 4e5) if i == 0 else (200, 4e5)))
            ax.legend(fontsize=5, loc="upper left" if i == 0 else "lower right")
        axes[0, j].set_title(
            NAME[s] + (" (withheld)" if s == "C" else ""), color=COL[s], fontsize=6.5
        )
        axes[1, j].set_xlabel("K (number of clients)")
    axes[0, 0].set_ylabel(TMEMO + " (gradient steps)")
    axes[1, 0].set_ylabel("delay (gradient steps)")
    fig.legend(
        handles=[
            Line2D([], [], color=INK2, lw=1.5, label="setup's working point"),
            Line2D(
                [],
                [],
                color=INK2,
                lw=1.5,
                ls="--",
                label="second series (another α or decay)",
            ),
            CENS_HANDLE,
        ],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=6.5,
    )
    fig.tight_layout()
    NUMBERS["figA1"] = numbers
    save(fig, "figA1_two_clocks_all_setups")


# ── Appendix A2: first-crossing time per method ───────────────────────────────


def figA2(rows):
    fig, axb = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.35))
    numbers = {}
    for ax, (name, alpha, E, part) in zip(axb, CELLS5):
        rs = sel(rows, group="algorithms", alpha=alpha, local_epochs=E, partition=part)
        cells = by_cell(rs, "strategy")
        for i, strat in enumerate(ORDER5):
            runs = cells.get(strat, [])
            if not runs:
                continue
            st = stats(runs)
            ts = [r["t_first_cross"] for r in runs]
            fin = [t for t in ts if _finite(t)]
            cen = [r["budget"] for r, t in zip(runs, ts) if not _finite(t)]
            ax.plot(
                fin,
                [i] * len(fin),
                "o",
                ms=4,
                mfc=STRAT[strat],
                mec="white",
                mew=0.5,
                zorder=3,
            )
            ax.plot(
                cen, [i] * len(cen), "x", ms=6.5, mec=STRAT[strat], mew=1.4, zorder=3
            )
            if _finite(st["fc"]):
                ax.plot(
                    [st["fc"]],
                    [i],
                    marker="|",
                    ms=12,
                    mew=2,
                    color=STRAT[strat],
                    zorder=4,
                )
            numbers[f"{name} {strat}"] = {
                "fc": round(st["fc"]) if _finite(st["fc"]) else None,
                "held": st["held"],
                "n": st["n"],
            }
        ax.set_yticks(range(len(ORDER5)))
        ax.set_yticklabels([LAB5[s] for s in ORDER5] if ax is axb[0] else [])
        ax.invert_yaxis()
        log_steps(ax, "x", dense=False)
        ax.set_title(_cell_title(name, alpha, E, part), fontsize=7.5)
    axb[1].set_xlabel(TFC + " (gradient steps; bar = KM median, × = failed to grok)")
    for ax, L in zip(axb, "abc"):
        letter(ax, L)
    fig.tight_layout()
    NUMBERS["figA2"] = numbers
    save(fig, "figA2_algorithms_first_crossing")


FIGS = {
    "1": fig1,
    "2": fig2,
    "3": fig3,
    "4": fig4,
    "5": fig5,
    "6": fig6,
    "7": fig7,
    "A1": figA1,
    "A2": figA2,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    rows = load_rows()
    for n, fn in FIGS.items():
        if args.only and n not in args.only:
            continue
        print(f"fig {n}")
        fn(rows)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "figure_numbers.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(NUMBERS)
    json.dump(
        old,
        open(path, "w"),
        indent=1,
        default=lambda x: (
            None if isinstance(x, float) and not math.isfinite(x) else str(x)
        ),
    )
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
