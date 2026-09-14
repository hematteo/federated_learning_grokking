"""exp3b, data partitions: three views in paper_figures' style.

    venv/bin/python scripts/plotting/exp3_partition_plots.py      # -> figures/exp3/

  exp3b_accuracy_reached.png   how far each partition gets: final test accuracy
                               (top) and peak train accuracy (bottom) per run, per
                               setup, every K the partition campaign ran. Unlike a
                               time ratio this is defined for runs that never
                               grok, which is most structured cells.
  exp3b_first_cross_vs_K.png   first-crossing time against K per setup, one line
                               per partition with IID for reference.
  exp3b_curves_by_partition.png  held-out accuracy per run against steps, a row
                               per setup and a column per partition (K = 10),
                               IID in gray behind.

Runs are exp3_curves.partition_columns' (paper_figures.fig4(d)'s selection).
Setup C is faded (RESULTS 23).
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
IID_COL = "#7f7f7f"
ORDER = ["iid", "operand", "label", "dirichlet"]
TICK = {"iid": "IID", "operand": "operand", "label": "label",
        "dirichlet": "Dir 0.5"}
LEGEND = {"iid": "IID", "operand": "operand (coherent, mod p)",
          "label": "label (incoherent; E: contiguous blocks)",
          "dirichlet": "Dirichlet 0.5 (unstructured)"}
K_MARK = ["o", "s", "^"]
OUT = "figures/exp3"


def colour(p):
    return IID_COL if p == "iid" else ec.PART_COL[p]


def columns(rows, s):
    """[(K, {partition: runs})] with IID folded in as a partition."""
    out = []
    for K, _, parts, base in ec.partition_columns(rows, s):
        cell = {p: parts[p] for p in ORDER if p in parts}
        if base:
            cell["iid"] = base
        out.append((int(K), cell))
    return out


def fade(s):
    return pf.WITHHELD if s == "C" else 1.0


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png/.pdf")


def legend(fig, present, extra=(), y=-0.02, ncol=4):
    handles = [Line2D([], [], color=colour(p), marker="o", ls="none", ms=5, label=LEGEND[p])
               for p in ORDER if p in present] + list(extra)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               fontsize=6.6)


# ── 1. accuracy reached ──────────────────────────────────────────────────────

def accuracy_reached(rows):
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 4.6), sharey=True,
                             gridspec_kw=dict(hspace=0.55, wspace=0.12))
    rng = random.Random(0)
    present = set()
    for j, s in enumerate(SETUPS):
        cols = columns(rows, s)
        cats = [p for p in ORDER if any(p in c for _, c in cols)]
        present |= set(cats)
        nk = len(cols)
        for i, (field, ref) in enumerate((("final_acc", None), ("peak_train_acc", 99.0))):
            ax = axes[i, j]
            bar = next(iter(cols[0][1].values()))[0]["grok_threshold"]
            ax.axhline(bar if ref is None else ref, color=pf.MUTED, ls="--", lw=0.8, zorder=1)
            for k_i, (K, cell) in enumerate(cols):
                off = (k_i - (nk - 1) / 2) * 0.32
                for p, runs in cell.items():
                    x = cats.index(p) + off
                    vals = [r[field] for r in runs if r[field] is not None]
                    for v in vals:
                        ax.plot(x + rng.uniform(-0.06, 0.06), v, marker=K_MARK[k_i], ms=3.2,
                                color=colour(p), mec="white", mew=0.3, ls="none", zorder=3,
                                alpha=fade(s))
                    if vals:
                        m = float(np.median(vals))
                        ax.plot([x - 0.13, x + 0.13], [m, m], color=colour(p), lw=1.6,
                                zorder=4, alpha=fade(s))
            pf.cat_axis(ax, [TICK[p] for p in cats])
            ax.tick_params(axis="x", labelsize=6)
            for t in ax.get_xticklabels():
                t.set_rotation(40)
                t.set_ha("right")
                t.set_rotation_mode("anchor")
            ax.grid(axis="x", visible=False)
            ax.set_ylim(-3, 103)
            if i == 0:
                ks = "  ".join(f"{K}{m}" for (K, _), m in zip(cols, ["●", "■", "▲"]))
                ax.set_title(f"{s}{' (withheld)' if s == 'C' else ''}\nK: {ks}", fontsize=7)
    axes[0, 0].set_ylabel("final test accuracy (%)")
    axes[1, 0].set_ylabel("peak train accuracy (%)")
    legend(fig, present, ncol=3, y=-0.02)
    fig.suptitle("How Far Each Partition Gets", fontsize=9, y=1.02)
    save(fig, "exp3b_accuracy_reached")


# ── 2. first crossing against K ──────────────────────────────────────────────

def first_cross_vs_K(rows):
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.5), sharey=True,
                             gridspec_kw=dict(wspace=0.12))
    present = set()
    for ax, s in zip(axes, SETUPS):
        cols = columns(rows, s)
        Ks = [K for K, _ in cols]
        parts = [p for p in ORDER if any(p in c for _, c in cols)]
        present |= set(parts)
        for p_i, p in enumerate(parts):
            xs, ys, lo, hi, fr = [], [], [], [], []
            for K, cell in cols:
                if p not in cell:
                    continue
                st = pf.stats(cell[p])
                x = Ks.index(K) + (p_i - (len(parts) - 1) / 2) * 0.07
                if not pf._finite(st["fc"]):
                    pf.censored_x(ax, x, colour(p), y=0.955, size=5.5, alpha=fade(s))
                    continue
                xs.append(x); ys.append(st["fc"]); lo.append(st["fc_lo"]); hi.append(st["fc_hi"])
                fr.append(st["crossed"])
            pf.series(ax, xs, ys, fr, colour(p), err=(lo, hi), alpha=fade(s), size=4,
                      lw=1.8 if p == "iid" else 1.2)
        pf.cat_axis(ax, [str(K) for K in Ks])
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("K")
        ax.set_title(s + (" (withheld)" if s == "C" else ""), fontsize=7.5)
    pf.log_steps(axes[0], "y", 1e3, 5e5)
    axes[0].set_ylabel(pf.TFC + " (gradient steps)")
    legend(fig, present, ncol=4, y=-0.08, extra=[
        Line2D([], [], marker="x", color=pf.INK2, ls="none", mew=1.5, ms=6,
               label="no run crossed")])
    fig.suptitle("First Crossing against Client Count, by Partition", fontsize=9, y=1.04)
    save(fig, "exp3b_first_cross_vs_K")


# ── 3. curves, organised by partition ────────────────────────────────────────

def curves_by_partition(rows):
    parts = ["operand", "label", "dirichlet"]
    fig, axes = plt.subplots(5, len(parts), figsize=(7.2, 6.4), sharex="row", sharey=True,
                             gridspec_kw=dict(hspace=0.45, wspace=0.1))
    for i, s in enumerate(SETUPS):
        cols = dict(columns(rows, s))
        for j, p in enumerate(parts):
            ax = axes[i, j]
            K = 10
            cell = cols.get(K, {})
            if p not in cell:
                ax.set_axis_off()
                continue
            for runs, col, lw, z in ((cell.get("iid", []), IID_COL, 1.6, 2),
                                     (cell[p], colour(p), 0.9, 3)):
                for r in runs:
                    h = pf.history(r["id"])
                    if h is None:
                        continue
                    st = h.get("total_steps") or h.get("epoch")
                    pts = [(x, y) for x, y in zip(st, h["test_acc"]) if x > 0]
                    ax.plot([a for a, _ in pts], [b for _, b in pts], color=col, lw=lw,
                            alpha=0.75 * fade(s), zorder=z)
            ax.axhline(r["grok_threshold"], color=pf.MUTED, ls="--", lw=0.7, zorder=1)
            ax.set_xscale("log")
            ax.set_ylim(-3, 103)
            ax.set_yticks([0, 50, 100])
            ax.tick_params(labelsize=6)
            crossed = sum(pf._finite(r["t_first_cross"]) for r in cell[p])
            ax.text(0.03, 0.62, f"{crossed}/{len(cell[p])}\ncrossed", transform=ax.transAxes,
                    fontsize=5.8, color=pf.INK2, va="center")
        axes[i, 0].set_ylabel(f"{s}{' (withheld)' if s == 'C' else ''}\ntest acc. (%)",
                              fontsize=7)
        if not axes[i, 0].axison:
            # First column empty on this row: label the first drawn panel instead,
            # and give it back the tick labels sharey hid.
            first = next(ax for ax in axes[i] if ax.axison)
            first.set_ylabel(f"{s}\ntest acc. (%)", fontsize=7)
            first.tick_params(labelleft=True)
    for j, p in enumerate(parts):
        axes[0, j].set_title(TICK[p].replace("\n", " "), fontsize=7.5)
        for ax in axes[:, j][::-1]:
            if ax.axison:
                ax.set_xlabel("gradient steps", fontsize=6.5)
                break
    legend(fig, set(parts) | {"iid"}, ncol=3, y=0.04)
    fig.suptitle("Held-out Accuracy per Run, by Partition (K = 10)",
                 fontsize=9, y=0.94)
    save(fig, "exp3b_curves_by_partition")


def main():
    rows = pf.load_rows()
    accuracy_reached(rows)
    first_cross_vs_K(rows)
    curves_by_partition(rows)


if __name__ == "__main__":
    main()
