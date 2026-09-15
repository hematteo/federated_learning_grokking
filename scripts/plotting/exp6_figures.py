"""exp6, federated algorithms: figures in paper_figures' style.

    venv/bin/python scripts/plotting/exp6_figures.py         # -> figures/exp6/

`t3_algorithm_comparison`, setup A only (quadratic MLP, mod 97, GD), K = 10,
10,000 rounds, 5 seeds, three hard cells from paper_figures.CELLS5:

  H1  alpha 0.25, E = 25, IID               (250,000 steps)
  H2  alpha 0.25, E = 25, Dirichlet 0.1     (250,000 steps)
  H3  alpha 0.30, E = 50, Dirichlet 0.1     (500,000 steps)

and six aggregation rules, each at its own calibrated server learning rate
(t3_server_lr_calibration): FedAvg, FedAdam (server lr 0.1), FedYogi (0.1),
SCAFFOLD, FedAvgM (server lr 1.0 with momentum), FedProx (mu = 0.01).

  exp6_two_clocks.png          (a) t_memo and (b) t_first_cross relative to FedAvg
  exp6_memorise.png            (a) on its own
  exp6_generalise.png          (b) on its own
  exp6_divergence.png          client weight divergence against steps: smoothed
                               median over seeds per method, range band, dot at
                               the median first crossing
  exp6_curves.png              test (top) and train (bottom) accuracy per run
                               against steps, one column per cell

Ratios are Kaplan-Meier medians over runs (censored at each run's budget) over the
FedAvg cell's; bars span the per-run values. A cross on the top edge: no run
reached the event within budget. Every cell evaluates every 20 rounds, i.e. every
500 steps at E = 25 and 1,000 at E = 50, so the fast methods' times are coarse.
"""
import importlib.util
import math
import os
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("pf", os.path.join(_HERE, "paper_figures.py"))
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)

OUT = "figures/exp6"
CELLS = pf.CELLS5                      # (name, alpha, E, partition)
ORDER = pf.ORDER5                      # fedavg, fedadam, fedyogi, scaffold, fedavgm, fedprox
LAB = pf.LAB5
COL = pf.STRAT
CELL_MARK = {"H1": "o", "H2": "s", "H3": "^"}
CELL_TXT = {"H1": "H1: α 0.25, E 25, IID", "H2": "H2: α 0.25, E 25, Dir 0.1",
            "H3": "H3: α 0.30, E 50, Dir 0.1"}


def cells(rows):
    """{cell name: {strategy: runs}}"""
    out = {}
    for name, alpha, E, part in CELLS:
        rs = pf.sel(rows, group="algorithms", alpha=alpha, local_epochs=E, partition=part)
        out[name] = {s: v for s, v in pf.by_cell(rs, "strategy").items()}
    return out


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png/.pdf")


def method_axis(ax):
    pf.cat_axis(ax, [LAB[s].replace(" (μ=0.01)", " (μ 0.01)") for s in ORDER])
    ax.tick_params(axis="x", labelsize=6.5)
    for t in ax.get_xticklabels():
        t.set_rotation(30)
        t.set_ha("right")
        t.set_rotation_mode("anchor")
    ax.grid(axis="x", visible=False)


PAPER_CELL_TXT = {"H1": r"H1: $\alpha = 0.25$, $E = 25$, IID",
                  "H2": r"H2: $\alpha = 0.25$, $E = 25$, Dirichlet $\alpha_{\mathrm{dir}} = 0.1$",
                  "H3": r"H3: $\alpha = 0.30$, $E = 50$, Dirichlet $\alpha_{\mathrm{dir}} = 0.1$"}


def cell_handles(extra=True, paper=False):
    txt = PAPER_CELL_TXT if paper else CELL_TXT
    h = [Line2D([], [], color=pf.INK2, marker=CELL_MARK[n], ls="none", ms=5, label=txt[n])
         for n, *_ in CELLS]
    if paper:
        h += [Line2D([], [], color=pf.INK2, marker="x", ls="none", ms=6, mew=1.5,
                     label="all runs censored")]
        return h
    if extra:
        h += [Line2D([], [], color=pf.INK2, marker="o", ls="none", ms=5, mfc="white", mew=1.2,
                     label="some runs did not reach it"),
              Line2D([], [], color=pf.INK2, marker="x", ls="none", ms=6, mew=1.5,
                     label="no run reached it within budget")]
    return h


# ── two clocks relative to FedAvg ────────────────────────────────────────────

def clock_panel(ax, data, key):
    field = "t_memo" if key == "memo" else "t_first_cross"
    for c_i, (name, *_rest) in enumerate(CELLS):
        base = pf.stats(data[name]["fedavg"])[key]
        for m_i, s in enumerate(ORDER):
            runs = data[name].get(s)
            if not runs:
                continue
            st = pf.stats(runs)
            x = m_i + (c_i - 1) * 0.22
            frac = sum(pf._finite(r[field]) for r in runs) / len(runs)
            if not (pf._finite(st[key]) and pf._finite(base)):
                ax.plot([x], [0.955], marker="x", ms=6, mec=COL[s], mew=1.5, ls="none",
                        transform=ax.get_xaxis_transform(), clip_on=False, zorder=6)
                continue
            y = st[key] / base
            ax.errorbar([x], [y], yerr=[[max(y - st[key + "_lo"] / base, 0)],
                                         [max(st[key + "_hi"] / base - y, 0)]],
                        color=COL[s], capsize=2, elinewidth=0.8, capthick=0.8, ls="none", zorder=3)
            pf.mark(ax, x, y, COL[s], frac, marker=CELL_MARK[name], size=5)
    ax.axhline(1.0, color=pf.MUTED, ls="--", lw=0.9, zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(0.03, 50)
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}×" for t in ticks])
    ax.minorticks_off()
    method_axis(ax)


def two_clocks(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.38))
    clock_panel(axes[0], data, "memo")
    clock_panel(axes[1], data, "fc")
    axes[0].set_ylabel("Time to memorise\n(relative to FedAvg)")
    axes[1].set_ylabel("Time to first generalise\n(relative to FedAvg)")
    axes[0].set_title("Time to Memorise by Aggregation Rule (Setup A)")
    axes[1].set_title("Time to Generalise by Aggregation Rule (Setup A)")
    for ax, L in zip(axes, "ab"):
        pf.letter(ax, L)
    fig.legend(handles=cell_handles(paper=True), loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2,
               fontsize=6.6, frameon=False, columnspacing=1.6, handletextpad=0.4)
    save(fig, "exp6_two_clocks")
    for key, name, title, ylab in (
            ("memo", "exp6_memorise", "Time to Memorise by Aggregation Rule (A, K = 10)",
             pf.TMEMO + " / FedAvg"),
            ("fc", "exp6_generalise", "Time to Generalise by Aggregation Rule (A, K = 10)",
             pf.TFC + " / FedAvg")):
        fig, ax = plt.subplots(figsize=(4.8, 3.5))
        clock_panel(ax, data, key)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        fig.legend(handles=cell_handles(), loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 0.0), fontsize=6.4)
        fig.tight_layout()
        save(fig, name)


# ── client divergence ────────────────────────────────────────────────────────

SMOOTH = 15          # rolling-median window, in logged rounds


def _smoothed_log_div(r, grid):
    """log10 divergence of one run, rolling-median smoothed, on a shared step grid.

    Per-round divergence swings over an order of magnitude from one logged round
    to the next once a run has grokked, so the raw series is smoothed in log
    space before comparison. Outside the run's own step range the value is NaN."""
    h = pf.history(r["id"])
    if h is None:
        return None
    pts = [(x, d) for x, d in zip(h["total_steps"], h["client_weight_divergence"])
           if x > 0 and d is not None and d > 0]
    if len(pts) < SMOOTH:
        return None
    x = np.array([p[0] for p in pts])
    y = np.log10([p[1] for p in pts])
    half = SMOOTH // 2
    ys = np.array([np.median(y[max(0, i - half):i + half + 1]) for i in range(len(y))])
    inside = (grid >= x[0]) & (grid <= x[-1])
    return np.where(inside, np.interp(grid, x, ys), np.nan)


def divergence(data):
    """Client weight divergence against steps: median over seeds, smoothed, per method.

    Not a per-run average. Averaging before each run's own first crossing (the
    window paper_figures._divergence uses) compares a fast method's first few
    thousand steps with a slow method's first fifty thousand, and averaging the
    whole run mixes in the post-grokking tail, whose length is set by when a
    method grokked. On H2 the two disagree by two orders of magnitude for
    SCAFFOLD. A shared step axis compares methods at the same time.

    Line: median over the 5 seeds of each run's smoothed log divergence. Band:
    min-max over seeds. Dot: the method's Kaplan-Meier median first crossing,
    placed on its line."""
    import warnings
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    for ax, (name, *_rest) in zip(axes, CELLS):
        budget = max(r["budget"] for rs in data[name].values() for r in rs)
        grid = np.logspace(np.log10(500), np.log10(budget), 200)
        for s in ORDER:
            runs = data[name].get(s, [])
            curves = [c for c in (_smoothed_log_div(r, grid) for r in runs) if c is not None]
            if not curves:
                continue
            arr = np.array(curves)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med, lo, hi = (np.nanmedian(arr, 0), np.nanmin(arr, 0), np.nanmax(arr, 0))
            ok = np.isfinite(med)
            ax.fill_between(grid[ok], 10 ** lo[ok], 10 ** hi[ok], color=COL[s], alpha=0.15,
                            lw=0, zorder=2)
            ax.plot(grid[ok], 10 ** med[ok], color=COL[s], lw=1.6, zorder=3)
            fc = pf.stats(runs)["fc"]
            if pf._finite(fc) and grid[ok][0] <= fc <= grid[ok][-1]:
                ax.plot([fc], [10 ** np.interp(fc, grid[ok], med[ok])], "o", ms=5.5,
                        color=COL[s], mec="white", mew=0.9, zorder=5)
        ax.set_yscale("log")
        pf.log_steps(ax, "x", dense=False)
        ax.grid(axis="x", visible=False)
        ax.set_title(CELL_TXT[name], fontsize=7.5)
        ax.set_xlabel("gradient steps")
    axes[0].set_ylabel("client weight divergence per round")
    fig.legend(handles=[Line2D([], [], color=COL[s], lw=1.8, label=LAB[s]) for s in ORDER]
               + [Line2D([], [], color=pf.INK2, marker="o", ls="none", ms=5.5,
                         label="median first crossing")],
               loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=6.6)
    fig.suptitle("Client Divergence over Training by Aggregation Rule (setup A, K = 10)\n"
                 "median over 5 seeds, smoothed; band = range over seeds",
                 fontsize=8.5, y=1.06)
    save(fig, "exp6_divergence")


# ── curves ───────────────────────────────────────────────────────────────────

def curves(data):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharey=True,
                             gridspec_kw=dict(hspace=0.18, wspace=0.1))
    for j, (name, alpha, E, part) in enumerate(CELLS):
        for s in ORDER:
            for r in data[name].get(s, []):
                h = pf.history(r["id"])
                if h is None:
                    continue
                pts = [(x, te, tr) for x, te, tr in zip(h["total_steps"], h["test_acc"],
                                                        h["train_acc"]) if x > 0]
                xs = [p[0] for p in pts]
                axes[0, j].plot(xs, [p[1] for p in pts], color=COL[s], lw=0.9, alpha=0.8)
                axes[1, j].plot(xs, [p[2] for p in pts], color=COL[s], lw=0.9, alpha=0.8)
        for ax, y in ((axes[0, j], 95.0), (axes[1, j], 99.0)):
            ax.axhline(y, color=pf.MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
            pf.log_steps(ax, "x", dense=False)
            ax.set_ylim(-3, 103)
            ax.set_yticks([0, 50, 100])
            ax.grid(axis="x", visible=False)
        axes[0, j].set_title(PAPER_CELL_TXT[name].replace(", Dirichlet", "\nDirichlet").replace(", IID", "\nIID"),
                             fontsize=7.5)
        axes[0, j].tick_params(labelbottom=False)
        axes[1, j].set_xlabel("gradient steps")
    axes[0, 0].set_ylabel("test accuracy (%)")
    axes[1, 0].set_ylabel("train accuracy (%)")
    fig.legend(handles=[Line2D([], [], color=COL[s], lw=1.6, label=LAB[s]) for s in ORDER],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=6.6, frameon=False)
    fig.suptitle("Training Curves per Run by Aggregation Rule (Setup A, K = 10, 5 seeds)",
                 fontsize=9, y=1.0)
    save(fig, "exp6_curves")


def main():
    data = cells(pf.load_rows())
    for name, by in data.items():
        b = pf.stats(by["fedavg"])
        print(f"  {name}: " + "; ".join(
            f"{s} memo {pf.stats(by[s])['memo'] / b['memo']:.2f}x fc "
            + (f"{pf.stats(by[s])['fc'] / b['fc']:.2f}x" if pf._finite(pf.stats(by[s])['fc']) else "inf")
            for s in ORDER if s in by))
    two_clocks(data)
    divergence(data)
    curves(data)


if __name__ == "__main__":
    main()
