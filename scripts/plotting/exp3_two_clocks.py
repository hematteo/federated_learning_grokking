"""exp3 as the two clocks: which phase does data heterogeneity slow, per setup?

    venv/bin/python scripts/plotting/exp3_two_clocks.py      # -> figures/exp3/

One figure in paper_figures' style (boxed axes, setup colours, marker fill =
fraction of runs, a cross on the top edge = no run reached the event):

  (a) t_memo / matched IID          against Dirichlet concentration, K=10
  (b) t_first_cross / matched IID   against Dirichlet concentration, K=10
  (c) t_memo / matched IID          per partition (K=10)
  (d) t_first_cross / matched IID   per partition

A setup whose line leaves the plot in (a) stopped memorising; one that stays in
(a) and leaves in (b) memorised and stopped generalising. Values are
Kaplan-Meier medians over runs censored at each run's budget, divided by the
matched IID cell's; bars span the per-run values over the same denominator.
Runs are exp3_curves' (paper_figures.fig4's) selection. C is faded (RESULTS 23).
"""
import importlib.util
import math
import os

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
DIRS = [0.01, 0.1, 0.5, 1.0, 10.0, 1000.0]
PARTS = ["operand", "label"]
PART_TICK = ["operand", "label"]
PARTS_REF = PARTS + ["dirichlet"]
PARTS_REF_TICK = PART_TICK + ["Dirichlet\n0.5"]
YLO, YHI = 0.3, 40.0


def cells(rows):
    """{setup: {"dir": {d: (runs, iid)}, "part": {p: (runs, iid)}}}"""
    out = {}
    for s in SETUPS:
        c = {"dir": {}, "part": {}}
        lad = ec.dirichlet_ladder(rows, s)
        if lad:
            _, by_d, base = lad
            c["dir"] = {d: (by_d[d], base) for d in DIRS if d in by_d}
        for K, _, parts, base in ec.partition_columns(rows, s):
            for p in PARTS:
                if p in parts and K == 10:
                    c["part"][p] = (parts[p], base)
            if "dirichlet" in parts and K == 10:
                c["part_ref"] = {**c.get("part_ref", {}), "dirichlet": (parts["dirichlet"], base)}
        c["part_ref"] = {**c["part"], **c.get("part_ref", {})}
        out[s] = c
    return out


def ratio(runs, base, key):
    """(KM ratio, lo, hi, fraction reaching the event) for key in memo / fc."""
    st, bst = pf.stats(runs), pf.stats(base)
    field = "t_memo" if key == "memo" else "t_first_cross"
    frac = sum(pf._finite(r[field]) for r in runs) / len(runs)
    b = bst[key]
    if not (pf._finite(st[key]) and pf._finite(b)):
        return math.inf, None, None, frac
    return st[key] / b, st[key + "_lo"] / b, st[key + "_hi"] / b, frac


def panel(ax, data, family, key, xs_of, dodge):
    for i, s in enumerate(SETUPS):
        cell = data[s][family]
        if not cell:
            continue
        alpha = pf.WITHHELD if s == "C" else 1.0
        col = pf.COL[s]
        pts = []
        for k, (runs, base) in cell.items():
            x = dodge(xs_of(k), i)
            r, lo, hi, frac = ratio(runs, base, key)
            if not pf._finite(r):
                pf.censored_x(ax, x, col, y=0.955, size=6, alpha=alpha)
                continue
            pts.append((x, r))
            ax.errorbar([x], [r], yerr=[[max(r - lo, 0)], [max(hi - r, 0)]], color=col,
                        capsize=2, elinewidth=0.8, capthick=0.8, ls="none", zorder=3,
                        alpha=alpha)
            pf.mark(ax, x, r, col, frac, size=4.5, alpha=alpha)
        if family == "dir" and len(pts) > 1:
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.2,
                    zorder=2, alpha=alpha)
    ax.axhline(1.0, color=pf.MUTED, ls="--", lw=0.9, zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(YLO, YHI)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.set_yticklabels(["0.5×", "1×", "2×", "5×", "10×", "20×"])
    ax.minorticks_off()


def main():
    rows = pf.load_rows()
    data = cells(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0),
                             gridspec_kw=dict(hspace=0.62, wspace=0.28))

    # Dirichlet rungs evenly spaced, IID-like on the left: on a log axis 1 and 0.5
    # sit on top of each other.
    order = DIRS[::-1]

    def dir_dodge(x, i):
        return x + (i - 2) * 0.1

    def dirichlet_axes(ax, key):
        panel(ax, data, "dir", key, order.index, dir_dodge)
        pf.cat_axis(ax, [f"{d:g}" for d in order])
        ax.grid(axis="x", visible=False)
        ax.set_xlabel(r"Dirichlet $\alpha_{\mathrm{dir}}$  (IID-like → concentrated labels)")

    for ax, key in ((axes[0, 0], "memo"), (axes[0, 1], "fc")):
        dirichlet_axes(ax, key)

    def part_dodge(x, i):
        return x + (i - 2) * 0.13

    for ax, key in ((axes[1, 0], "memo"), (axes[1, 1], "fc")):
        panel(ax, data, "part", key, PARTS.index, part_dodge)
        pf.cat_axis(ax, PART_TICK)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("partition (K = 10)")

    for ax in axes[:, 0]:
        ax.set_ylabel(pf.TMEMO + " / matched IID")
    for ax in axes[:, 1]:
        ax.set_ylabel(pf.TFC + " / matched IID")
    axes[0, 0].set_title("Time to Memorise — Label Skew")
    axes[0, 1].set_title("Time to Generalise — Label Skew")
    axes[1, 0].set_title("Time to Memorise — Partition Structure")
    axes[1, 1].set_title("Time to Generalise — Partition Structure")
    for ax, L in zip(axes.flat, "abcd"):
        pf.letter(ax, L)

    handles = legend_handles()
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07),
               fontsize=6.8)
    out = "figures/exp3"
    os.makedirs(out, exist_ok=True)
    save(fig, out, "exp3_two_clocks")

    # Panel (b) on its own: time to cross the bar against label skew.
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    dirichlet_axes(ax, "fc")
    ax.set_ylabel(pf.TFC + " / matched IID")
    # Finite values stay under 4x; the crosses sit on the top edge regardless.
    ax.set_ylim(0.4, 7)
    ax.set_yticks([0.5, 1, 2, 5])
    ax.set_yticklabels(["0.5×", "1×", "2×", "5×"])
    ax.set_title("Time to Generalise under Label Skew (K = 10)")
    fig.legend(handles=legend_handles(), loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.0), fontsize=6.6)
    fig.tight_layout()
    save(fig, out, "exp3_two_clocks_b")

    # Partitions on their own, one figure per clock, with Dirichlet 0.5 from the
    # same campaign as the unstructured reference beside the structured splits.
    for key, stem, title, ylab in (
            ("fc", "exp3_partitions_generalise", "Time to Generalise by Partition",
             pf.TFC + " / matched IID"),
            ("memo", "exp3_partitions_memorise", "Time to Memorise by Partition",
             pf.TMEMO + " / matched IID")):
        fig, ax = plt.subplots(figsize=(4.6, 3.5))
        panel(ax, data, "part_ref", key, PARTS_REF.index, part_dodge)
        pf.cat_axis(ax, PARTS_REF_TICK)
        ax.grid(axis="x", visible=False)
        ax.axvline(len(PARTS) - 0.5, color=pf.GRID, lw=0.8)
        ax.set_ylim(0.4, 16)
        ax.set_yticks([0.5, 1, 2, 5, 10])
        ax.set_yticklabels(["0.5×", "1×", "2×", "5×", "10×"])
        ax.set_xlabel("partition (K = 10)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        fig.legend(handles=legend_handles(), loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 0.0), fontsize=6.6)
        fig.tight_layout()
        save(fig, out, stem)

    for s in SETUPS:
        for fam in ("dir", "part"):
            for k, (runs, base) in data[s][fam].items():
                m = ratio(runs, base, "memo")
                f = ratio(runs, base, "fc")
                print(f"  {s} {fam} {k}: memo {m[0]:.2f} ({m[3]:.2f})  fc {f[0]:.2f} ({f[3]:.2f})")


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}/{name}.png/.pdf")


def legend_handles():
    handles = [Line2D([], [], color=pf.COL[s], marker="o", lw=1.2, ms=4.5,
                      alpha=pf.WITHHELD if s == "C" else 1.0,
                      label=pf.NAME[s] + (" (withheld)" if s == "C" else ""))
               for s in SETUPS]
    handles += [
        Line2D([], [], color=pf.INK2, marker="o", ls="none", ms=4.5, mfc=pf.INK2,
               label="all runs reached it"),
        Line2D([], [], color=pf.INK2, marker="o", ls="none", ms=4.5, mfc="white",
               mew=1.2, label="some runs did not"),
        Line2D([], [], color=pf.INK2, marker="x", ls="none", ms=6, mew=1.5,
               label="no run reached it"),
    ]
    return handles


if __name__ == "__main__":
    main()
