"""exp4, local epochs: figures in paper_figures' style.

    venv/bin/python scripts/plotting/exp4_figures.py         # -> figures/exp4/

The E ladder is `t5_local_epochs`: FedAvg, K = 10, IID, E in {5, 10, 25, 50}
(A also E = 1), rounds scaled by 5/E so every rung does the same gradient work.
A also carries E = 100 and 200 from `x_a_high_e`, at twice the budget
(100,000 steps against 50,000), compared on first crossing.
The E = 5 control is the banked K = 10 cell paper_figures._e_ladder picks, so
these are the runs Fig. 2 summarises.

  exp4_two_clocks.png            (a) t_memo and (b) t_first_cross relative to E = 5
  exp4_memorise_vs_E.png         (a) on its own
  exp4_generalise_vs_E.png       (b) on its own
  exp4_accuracy_vs_E.png         final test accuracy per run, median line
  exp4_curves.png                test (top) and train (bottom) accuracy per run
                                 against steps, one column per setup, E on a ramp

Ratios are Kaplan-Meier medians over runs (censored at each run's budget) over
the E = 5 cell's; bars span the per-run values. A cross on the top edge: no run
reached the event. C is faded (RESULTS 23).
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
_spec = importlib.util.spec_from_file_location("ec", os.path.join(_HERE, "exp3_curves.py"))
ec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ec)
pf = ec.pf

SETUPS = "ABCDE"
ES = [1, 5, 10, 25, 50, 100, 200]
# Ordered axis -> one hue light to dark; E = 1, 100 and 200 are setup A only.
E_COL = dict(zip(ES, ["#c9ddf7", "#9cc2f0", "#6fa3e6", "#3f82d8", "#2463b4",
                      "#154a8c", "#0a2f5c"]))
OUT = "figures/exp4"


def ladder(rows, s):
    """{E: runs} for the setup's E ladder, E = 5 from the matched K = 10 control."""
    le = pf.sel(rows, group="local_epochs", setup=s)
    if not le:
        return {}
    r0 = le[0]
    ctrl = pf.best_control(pf.fed(
        rows, s, alpha=r0["alpha"], weight_decay=r0["weight_decay"], n_train=r0["n_train"],
        num_clients=10.0, group=lambda g: g != "local_epochs"))
    return {int(E): rs for E, rs in sorted(pf.by_cell(le + ctrl, "local_epochs").items())}


def fade(s):
    return pf.WITHHELD if s == "C" else 1.0


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png/.pdf")


def setup_handles(extra=True):
    h = [Line2D([], [], color=pf.COL[s], marker="o", lw=1.2, ms=4.5, alpha=fade(s),
                label=pf.NAME[s] + (" (withheld)" if s == "C" else "")) for s in SETUPS]
    if extra:
        h += [Line2D([], [], color=pf.INK2, marker="o", ls="none", ms=4.5, label="all runs reached it"),
              Line2D([], [], color=pf.INK2, marker="o", ls="none", ms=4.5, mfc="white", mew=1.2,
                     label="some runs did not"),
              Line2D([], [], color=pf.INK2, marker="x", ls="none", ms=6, mew=1.5,
                     label="no run reached it")]
    return h


# ── two clocks ───────────────────────────────────────────────────────────────

def clock_panel(ax, lads, key, ylim=(0.2, 40)):
    field = "t_memo" if key == "memo" else "t_first_cross"
    for i, s in enumerate(SETUPS):
        lad = lads[s]
        if 5 not in lad:
            continue
        base = pf.stats(lad[5])[key]
        xs, ys, lo, hi, fr = [], [], [], [], []
        for E, runs in lad.items():
            st = pf.stats(runs)
            x = ES.index(E) + (i - 2) * 0.07
            frac = sum(pf._finite(r[field]) for r in runs) / len(runs)
            if not (pf._finite(st[key]) and pf._finite(base)):
                pf.censored_x(ax, x, pf.COL[s], y=0.955, size=6, alpha=fade(s))
                continue
            xs.append(x); ys.append(st[key] / base)
            lo.append(st[key + "_lo"] / base); hi.append(st[key + "_hi"] / base); fr.append(frac)
        pf.series(ax, xs, ys, fr, pf.COL[s], err=(lo, hi), alpha=fade(s), size=4.5, lw=1.2)
    ax.axhline(1.0, color=pf.MUTED, ls="--", lw=0.9, zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ticks = [t for t in (0.2, 0.5, 1, 2, 5, 10, 20) if ylim[0] <= t <= ylim[1]]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}×" for t in ticks])
    ax.minorticks_off()
    pf.cat_axis(ax, [str(E) for E in ES])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("local epochs E (same total gradient steps)")


def two_clocks(lads):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.3))
    clock_panel(axes[0], lads, "memo")
    clock_panel(axes[1], lads, "fc")
    axes[0].set_ylabel(pf.TMEMO + " / E = 5")
    axes[1].set_ylabel(pf.TFC + " / E = 5")
    axes[0].set_title("Time to Memorise vs Local Epochs")
    axes[1].set_title("Time to Generalise vs Local Epochs")
    for ax, L in zip(axes, "ab"):
        pf.letter(ax, L)
    fig.legend(handles=setup_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=4, fontsize=6.6)
    save(fig, "exp4_two_clocks")
    for key, name, title, ylab in (
            ("memo", "exp4_memorise_vs_E", "Time to Memorise vs Local Epochs (K = 10)",
             pf.TMEMO + " / E = 5"),
            ("fc", "exp4_generalise_vs_E", "Time to Generalise vs Local Epochs (K = 10)",
             pf.TFC + " / E = 5")):
        fig, ax = plt.subplots(figsize=(4.6, 3.5))
        clock_panel(ax, lads, key)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        fig.legend(handles=setup_handles(), loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 0.0), fontsize=6.6)
        fig.tight_layout()
        save(fig, name)


# ── accuracy reached ─────────────────────────────────────────────────────────

def accuracy_vs_E(lads):
    """Final test accuracy per run against E. Peak train accuracy is not drawn:
    it is 100% on every rung of every setup but C's E = 50."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    rng = random.Random(0)
    for i, s in enumerate(SETUPS):
        xs, meds = [], []
        for E, runs in lads[s].items():
            x = ES.index(E) + (i - 2) * 0.1
            vals = [r["final_acc"] for r in runs if r["final_acc"] is not None]
            for v in vals:
                ax.plot(x + rng.uniform(-0.03, 0.03), v, "o", ms=2.6, color=pf.COL[s],
                        alpha=0.5 * fade(s), mec="none", zorder=3)
            xs.append(x); meds.append(float(np.median(vals)))
        ax.plot(xs, meds, color=pf.COL[s], lw=1.3, marker="o", ms=4.5, alpha=fade(s), zorder=4)
    # Each setup's grok bar, labelled in the right margin.
    for y, lab in ((95, "bar A, B"), (90, "bar E"), (85, "bar C, D")):
        ax.axhline(y, color=pf.MUTED, ls=":", lw=0.8, zorder=1)
        ax.text(1.01, y, lab, transform=ax.get_yaxis_transform(), fontsize=6,
                color=pf.MUTED, va="center", clip_on=False)
    pf.cat_axis(ax, [str(E) for E in ES])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(60, 102)
    ax.set_ylabel("final test accuracy (%)")
    ax.set_xlabel("local epochs E (same total gradient steps)")
    ax.set_title("Final Test Accuracy vs Local Epochs (K = 10)")
    fig.legend(handles=setup_handles(extra=False)
               + [Line2D([], [], color=pf.INK2, marker="o", lw=1.3, ms=4.5, label="median"),
                  Line2D([], [], color=pf.INK2, marker="o", ls="none", ms=2.6, alpha=0.5,
                         label="individual runs")],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6.6)
    fig.tight_layout()
    save(fig, "exp4_accuracy_vs_E")


# ── curves ───────────────────────────────────────────────────────────────────

def curves(lads):
    fig, axes = plt.subplots(2, 5, figsize=(10.5, 4.6), sharex="col", sharey=True,
                             gridspec_kw=dict(hspace=0.15, wspace=0.12))
    for j, s in enumerate(SETUPS):
        lad = lads[s]
        groups = [(E_COL[E], lad[E], 3 + k, 1.0) for k, E in enumerate(sorted(lad))]
        flagged = ec.draw_groups(axes[0, j], axes[1, j], groups)
        if flagged:
            print(f"  {s}: clients dropped in {flagged}")
        r0 = next(iter(lad.values()))[0]
        axes[0, j].set_title(pf.NAME[s] + (" (withheld)" if s == "C" else "")
                             + f"\n{ec.context(r0)} · K = 10", fontsize=7.5)
        axes[1, j].set_xlabel("gradient steps", fontsize=7.5)
        for ax in axes[:, j]:
            for t in list(ax.texts):     # margin labels collide with the next column
                t.remove()
        for ax in axes[:, j]:
            ax.tick_params(labelsize=6.5)
    axes[0, 0].set_ylabel("test accuracy (%)", fontsize=8)
    axes[1, 0].set_ylabel("train accuracy (%)", fontsize=8)
    handles = [Line2D([], [], color=E_COL[E], lw=1.8, label=f"E = {E}") for E in ES]
    handles += ec.MARK_HANDLES[:2]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=9,
               fontsize=7)
    fig.suptitle("Training Curves per Run across Local Epochs (same total gradient steps)",
                 fontsize=9.5, y=1.0)
    save(fig, "exp4_curves")


def main():
    rows = pf.load_rows()
    lads = {s: ladder(rows, s) for s in SETUPS}
    for s in SETUPS:
        print(f"  {s}: " + ", ".join(
            f"E={E}: {sum(pf._finite(r['t_first_cross']) for r in rs)}/{len(rs)} crossed, "
            f"memo {pf.stats(rs)['memo']:.0f}, fc {pf.stats(rs)['fc']:.0f}"
            for E, rs in lads[s].items()))
    two_clocks(lads)
    accuracy_vs_E(lads)
    curves(lads)


if __name__ == "__main__":
    main()
