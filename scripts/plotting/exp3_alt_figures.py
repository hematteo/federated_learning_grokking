"""Alternative views of exp3 (data heterogeneity) to the per-run curve grids.

    venv/bin/python scripts/plotting/exp3_alt_figures.py         # -> figures/exp3/

Same runs as exp3_curves.py (paper_figures.fig4's selection). Three figures,
each aimed at the claim "heterogeneity breaks a different phase per architecture":

  exp3_phase_plane.png   train accuracy (x) against test accuracy (y), one path
                         per run, one column per setup. Row 1 the Dirichlet ladder
                         at K=10, row 2 the partitions (K=10). Where a
                         path ends is the outcome: top-right grokked, right edge
                         below the bar memorised-only, left never memorised.
  exp3_outcome_grid.png  setups x conditions. Each cell is classified by where its
                         runs ended and carries crossed / memorised counts and the
                         median first crossing relative to the matched IID cell.
  exp3_median_bands.png  small multiples, one row per condition: median test
                         accuracy over runs with a min-max band, the IID band
                         repeated in every row, median train accuracy dashed.

Classification (per run, then per cell by majority):
  grokked            test accuracy crossed the bar (t_first_cross finite)
  memorised only     train accuracy reached 99% (peak_train_acc >= 99), no crossing
  not memorised      neither (train accuracy never reached 99%)
Setup C is drawn but its numbers are withheld from quantitative claims (RESULTS 23).
"""
import argparse
import importlib.util
import os
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ec = _load("exp3_curves")
pf = ec.pf

GRAY, INK, INK2, RULE = "#898781", "#111111", "#52514e", "#e1e0d9"
# Status steps from the validated reference palette; each state also carries a
# glyph and a word, so colour never carries it alone.
STATE = {
    "grokked": ("#0ca30c", "✓", "grokked"),
    "memo": ("#fab219", "◐", "memorised only"),
    "never": ("#d03b3b", "✗", "not memorised (train < 99%)"),
}
SETUPS = "ABCDE"


# ── conditions ───────────────────────────────────────────────────────────────

def conditions(rows):
    """{setup: [(key, label, color, runs, iid_runs, K)]} in a fixed row order."""
    out = {}
    for s in SETUPS:
        conds = []
        lad = ec.dirichlet_ladder(rows, s)
        if lad:
            r0, cells, base = lad
            conds.append(("iid", "IID", GRAY, base, base, 10))
            for d in ec.DIR_RUNGS:
                if d in cells:
                    conds.append((f"dir{d:g}", f"Dirichlet {d:g}", ec.DIR_COL[d],
                                  cells[d], base, 10))
        for K, r0, parts, base in ec.partition_columns(rows, s):
            for p in ("operand", "label"):
                if p in parts and K == 10:
                    conds.append((p, ec.PART_LABEL[p],
                                  ec.PART_COL[p], parts[p], base, K))
        out[s] = conds
    return out


ROW_ORDER = (["iid"] + [f"dir{d:g}" for d in ec.DIR_RUNGS]
             + ["operand", "label"])
ROW_LABEL = {"iid": "IID", **{f"dir{d:g}": f"Dirichlet {d:g}" for d in ec.DIR_RUNGS},
             "operand": "operand", "label": "label"}


def run_state(r):
    if pf._finite(r["t_first_cross"]):
        return "grokked"
    if (r["peak_train_acc"] or 0) >= 99:
        return "memo"
    return "never"


def cell_state(runs):
    states = [run_state(r) for r in runs]
    top = max(STATE, key=lambda k: (states.count(k), -list(STATE).index(k)))
    return top, len(set(states)) > 1


def curve(r):
    h = pf.history(r["id"])
    if h is None:
        return None
    steps = h.get("total_steps") or h.get("epoch")
    keep = [i for i, x in enumerate(steps) if x > 0]
    return (np.array([steps[i] for i in keep]), np.array([h["test_acc"][i] for i in keep]),
            np.array([h["train_acc"][i] for i in keep]))


# ── 1. phase plane ───────────────────────────────────────────────────────────

def phase_axes(ax, bar):
    ax.axhspan(bar, 103, xmin=0, xmax=1, color=STATE["grokked"][0], alpha=0.06, lw=0, zorder=0)
    ax.add_patch(Rectangle((99, -2), 4, bar + 2, color=STATE["memo"][0], alpha=0.12,
                           lw=0, zorder=0))
    ax.axhline(bar, color=GRAY, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.axvline(99, color=GRAY, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-2, 103)
    ax.set_ylim(-2, 103)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    ax.set_aspect("equal")
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=7, colors=INK2)


def phase_path(ax, r, color, lw=1.0, z=3):
    c = curve(r)
    if c is None:
        return
    _, te, tr = c
    ls = ":" if ec.dropped_clients(r, pf.history(r["id"])) else "-"
    ax.plot(tr, te, color=color, lw=lw, ls=ls, alpha=0.8, zorder=z, solid_capstyle="round")
    if pf._finite(r["t_first_cross"]):
        ax.plot(tr[-1], te[-1], "o", ms=4.5, color=color, mec="white", mew=0.8, zorder=z + 2)
    else:
        ax.plot(tr[-1], te[-1], "x", ms=6, color=color, mew=1.5, zorder=z + 2)


def plot_phase(conds, out):
    fig, axes = plt.subplots(2, len(SETUPS), figsize=(3.0 * len(SETUPS), 7.4))
    for j, s in enumerate(SETUPS):
        bar = conds[s][0][3][0]["grok_threshold"]
        for i, family in enumerate(("dirichlet", "partition")):
            ax = axes[i, j]
            phase_axes(ax, bar)
            for key, label, col, runs, base, K in conds[s]:
                is_dir = key == "iid" or key.startswith("dir")
                if key == "iid":
                    for r in runs:
                        phase_path(ax, r, GRAY, lw=2.2, z=2)
                    continue
                if (family == "dirichlet") != is_dir:
                    continue
                for r in runs:
                    phase_path(ax, r, col)
            if i == 0:
                ax.set_title(pf.NAME[s] + (" (withheld)" if s == "C" else ""),
                             fontsize=8, loc="left", color=INK2)
            if j == 0:
                ax.set_ylabel(("Dirichlet ladder, K = 10" if i == 0 else "partitions, K = 10")
                              + "\ntest accuracy (%)", fontsize=8, color=INK2)
            if i == 1:
                ax.set_xlabel("train accuracy (%)", fontsize=8, color=INK2)
    # Region names, once, on the anchor's top panel.
    ax = axes[0, 0]
    ax.text(3, 99, "grokked", fontsize=6.5, color=STATE["grokked"][0], va="top")
    ax.text(97, 45, "memorised,\nnot generalised", fontsize=6.5, color="#9a6a00",
            ha="right", va="center")
    ax.text(3, 45, "not\nmemorised", fontsize=6.5, color=STATE["never"][0], va="center")
    present = {k for s in SETUPS for k, *_ in conds[s]}
    h1 = ([Line2D([], [], color=GRAY, lw=2.2, label="IID")]
          + [Line2D([], [], color=ec.DIR_COL[d], lw=1.8, label=f"Dir {d:g}")
             for d in ec.DIR_RUNGS])
    h2 = [Line2D([], [], color=ec.PART_COL[p], lw=1.8, label=ec.PART_LABEL[p])
          for p in ("operand", "label") if p in present]
    h3 = [Line2D([], [], color=INK2, marker="o", ls="none", ms=4.5, label="crossed bar"),
          Line2D([], [], color=INK2, marker="x", ls="none", ms=6, mew=1.5, label="never crossed"),
          Line2D([], [], color=INK2, lw=1.2, ls=":", label="clients dropped")]
    fig.legend(handles=h1 + h2 + h3, loc="lower center", ncol=len(h1 + h2 + h3),
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.4,
               columnspacing=1.0)
    fig.suptitle("exp3 — where heterogeneous runs end up: train against test accuracy, "
                 "one path per run\nFedAvg · E = 5 · dashed lines: the grok bar and 99% train",
                 fontsize=10, x=0.01, ha="left")
    # Manual spacing: tight_layout with equal-aspect axes lets the rows overlap.
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.12, hspace=0.3, wspace=0.2)
    save(fig, out, "exp3_phase_plane")


# ── 2. outcome grid ──────────────────────────────────────────────────────────

def plot_grid(conds, out):
    rows_present = [k for k in ROW_ORDER if any(k == c[0] for s in SETUPS for c in conds[s])]
    fig, ax = plt.subplots(figsize=(2.2 * len(SETUPS) + 1.6, 0.52 * len(rows_present) + 1.3))
    for i, key in enumerate(rows_present):
        for j, s in enumerate(SETUPS):
            match = [c for c in conds[s] if c[0] == key]
            x, y = j, i
            if not match:
                ax.add_patch(Rectangle((x - 0.47, y - 0.44), 0.94, 0.88, facecolor="#f3f2ee",
                                       edgecolor="none"))
                ax.text(x, y, "not run", ha="center", va="center", fontsize=7, color="#a9a8a2")
                continue
            _, _, _, runs, base, K = match[0]
            state, mixed = cell_state(runs)
            col, glyph, _ = STATE[state]
            n = len(runs)
            crossed = sum(run_state(r) == "grokked" for r in runs)
            memo = sum((r["peak_train_acc"] or 0) >= 99 for r in runs)
            ax.add_patch(Rectangle((x - 0.47, y - 0.44), 0.94, 0.88, facecolor=col,
                                   alpha=0.16 if s != "C" else 0.08, edgecolor="none"))
            ax.add_patch(Rectangle((x - 0.47, y - 0.44), 0.05, 0.88, facecolor=col,
                                   edgecolor="none"))
            # Kaplan-Meier medians with censoring at each run's budget -- the
            # statistic paper Fig. 4(d) uses, so a partly censored cell is not a
            # median over its survivors.
            fc, bfc = pf.stats(runs)["fc"], pf.stats(base)["fc"]
            ratio = ""
            if key != "iid" and pf._finite(fc) and pf._finite(bfc):
                ratio = f"  {fc / bfc:.1f}×"
            peak = statistics.median(r["peak_train_acc"] or 0 for r in runs)
            ink = INK if s != "C" else "#8c8b86"
            ax.text(x - 0.36, y - 0.1, f"{glyph} {crossed}/{n} crossed{ratio}", ha="left",
                    va="center", fontsize=7.2, color=ink)
            ax.text(x - 0.36, y + 0.2, f"memorised {memo}/{n} · peak train {peak:.0f}%"
                    + ("  · mixed" if mixed else ""),
                    ha="left", va="center", fontsize=6.3, color=INK2 if s != "C" else "#a9a8a2")
    ax.set_xlim(-0.55, len(SETUPS) - 0.45)
    ax.set_ylim(len(rows_present) - 0.5, -0.5)
    ax.set_xticks(range(len(SETUPS)))
    ax.set_xticklabels([pf.NAME[s] + ("\n(withheld)" if s == "C" else "") for s in SETUPS],
                       fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows_present)))
    ax.set_yticklabels([ROW_LABEL[k] for k in rows_present], fontsize=8)
    ax.tick_params(length=0, colors=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    sep = rows_present.index("operand") - 0.5 if "operand" in rows_present else None
    if sep:
        ax.axhline(sep, color=INK2, lw=0.8)
    handles = [Patch(facecolor=STATE[k][0], alpha=0.5, label=f"{STATE[k][1]} {STATE[k][2]}")
               for k in STATE]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3,
              fontsize=7.5, frameon=False)
    fig.suptitle("exp3 — outcome by setup and heterogeneity condition (K = 10 unless noted)\n"
                 "cell state = majority of runs · ratio = Kaplan–Meier median first crossing / matched IID",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, out, "exp3_outcome_grid")


# ── 3. median + band small multiples ─────────────────────────────────────────

def band(runs, grid):
    te, tr = [], []
    for r in runs:
        c = curve(r)
        if c is None:
            continue
        x, a, b = c
        inside = (grid >= x[0]) & (grid <= x[-1])
        te.append(np.where(inside, np.interp(grid, x, a), np.nan))
        tr.append(np.where(inside, np.interp(grid, x, b), np.nan))
    if not te:
        return None
    te, tr = np.array(te), np.array(tr)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        return (np.nanmedian(te, 0), np.nanmin(te, 0), np.nanmax(te, 0), np.nanmedian(tr, 0))


def plot_bands(conds, out):
    rows_present = [k for k in ROW_ORDER if k != "iid"
                    and any(k == c[0] for s in SETUPS for c in conds[s])]
    fig, axes = plt.subplots(len(rows_present), len(SETUPS),
                             figsize=(2.9 * len(SETUPS), 1.15 * len(rows_present) + 1.2),
                             sharex="col", sharey=True, squeeze=False)
    for j, s in enumerate(SETUPS):
        allx = [curve(r) for c in conds[s] for r in c[3]]
        lo = min(c[0][0] for c in allx if c is not None)
        hi = max(c[0][-1] for c in allx if c is not None)
        grid = np.logspace(np.log10(lo), np.log10(hi), 300)
        bar = conds[s][0][3][0]["grok_threshold"]
        for i, key in enumerate(rows_present):
            ax = axes[i, j]
            match = [c for c in conds[s] if c[0] == key]
            ax.set_xscale("log")
            ax.set_ylim(-2, 122)          # headroom above 100 for the outcome label
            ax.set_yticks([0, 50, 100])
            ax.grid(False)
            ax.axhline(bar, color=GRAY, lw=0.7, ls=(0, (4, 3)), zorder=1)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=6.5, colors=INK2)
            if not match:
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center",
                        va="center", fontsize=7, color="#a9a8a2")
                continue
            _, label, col, runs, base, K = match[0]
            b = band(base, grid)
            if b:
                ax.fill_between(grid, b[1], b[2], color=GRAY, alpha=0.18, lw=0, zorder=1)
                ax.plot(grid, b[0], color=GRAY, lw=1.1, zorder=2)
            m = band(runs, grid)
            if m:
                ax.fill_between(grid, m[1], m[2], color=col, alpha=0.25, lw=0, zorder=3)
                ax.plot(grid, m[0], color=col, lw=1.6, zorder=4)
                ax.plot(grid, m[3], color=col, lw=0.9, ls="--", zorder=4)
            state, mixed = cell_state(runs)
            ax.text(0.02, 113, f"{STATE[state][1]} {sum(run_state(r) == 'grokked' for r in runs)}"
                    f"/{len(runs)} crossed" + (f" · K = {int(K)}" if K != 10 else ""),
                    transform=ax.get_yaxis_transform(), fontsize=6.8, color=INK2, va="center")
        axes[0, j].set_title(pf.NAME[s] + (" (withheld)" if s == "C" else ""), fontsize=8,
                             loc="left", color=INK2)
        axes[-1, j].set_xlabel("gradient steps", fontsize=7.5, color=INK2)
    for i, key in enumerate(rows_present):
        axes[i, 0].set_ylabel(ROW_LABEL[key], fontsize=7.5, color=INK2, rotation=0,
                              ha="right", va="center", labelpad=8)
    handles = [Line2D([], [], color=INK2, lw=1.6, label="median test accuracy"),
               Patch(facecolor=INK2, alpha=0.25, label="min–max over runs"),
               Line2D([], [], color=INK2, lw=0.9, ls="--", label="median train accuracy"),
               Line2D([], [], color=GRAY, lw=1.1, label="matched IID (band + median)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("exp3 — accuracy per heterogeneity condition against matched IID, "
                 "median over runs\nFedAvg · E = 5 · K = 10 unless noted · glyph = cell outcome, "
                 "count = runs that crossed the bar", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.95))
    save(fig, out, "exp3_median_bands")


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}/{name}.png/.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/exp3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    conds = conditions(pf.load_rows())
    plot_phase(conds, a.out)
    plot_grid(conds, a.out)
    plot_bands(conds, a.out)


if __name__ == "__main__":
    main()
