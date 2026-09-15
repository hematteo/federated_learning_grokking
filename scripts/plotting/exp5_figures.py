"""exp5, partial participation: figures in paper_figures' style.

    venv/bin/python scripts/plotting/exp5_figures.py         # -> figures/exp5/

The f ladder is `t5_participation`: FedAvg, K = 20, IID, E = 5, a fraction f of
clients sampled per round, f in {1, 0.5, 0.25}, rounds scaled by 1/f so every
cell does the same gradient work. Setup A also has `t4b_participation` at K = 50,
f in {0.2, 0.4, 0.6, 1}, drawn dashed. Selection is paper_figures._f_ladders'.

EVERYTHING IS IN COMMUNICATION ROUNDS. total_steps accumulates E * f per round,
so any step-denominated time halves with f by construction (RESULTS 21). A step
time converts to rounds as t / (E * f).

  exp5_two_clocks.png           (a) memorisation and (b) first-crossing rounds
                                relative to f = 1
  exp5_memorise_vs_f.png        (a) on its own
  exp5_generalise_vs_f.png      (b) on its own
  exp5_accuracy_vs_f.png        final test accuracy per run
  exp5_curves.png               test (top) and train (bottom) accuracy per run
                                against rounds, one column per ladder, f on a ramp

Ratios are Kaplan-Meier medians over runs, censored at each run's budget in
rounds, over the f = 1 cell's; bars span the per-run values. A cross on the top
edge: no run reached the event. C is faded in the summary panels (RESULTS 23),
not in the curves.
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

OUT = "figures/exp5"
# (key, setup, group, K, linestyle, label)
LADDERS = [
    ("A20", "A", "participation_setups", 20.0, "-", pf.NAME["A"]),
    ("B", "B", "participation_setups", 20.0, "-", pf.NAME["B"]),
    ("C", "C", "participation_setups", 20.0, "-", pf.NAME["C"]),
    ("D", "D", "participation_setups", 20.0, "-", pf.NAME["D"]),
    ("E", "E", "participation_setups", 20.0, "-", pf.NAME["E"]),
    ("A50", "A", "participation", 50.0, "--", "A: K = 50"),
]
FS = [0.2, 0.25, 0.4, 0.5, 0.6, 1.0]
# Ordered axis -> one hue; darker = fewer clients per round.
F_COL = dict(zip(sorted(FS, reverse=True),
                 ["#9cc2f0", "#6fa3e6", "#3f82d8", "#2463b4", "#154a8c", "#0a2f5c"]))


def ladders(rows):
    """{key: {f: runs}} with the f = 1 control from the matched banked K cell."""
    out = {}
    for key, s, group, K, _, _ in LADDERS:
        ps = pf.sel(rows, group=group, setup=s)
        if not ps:
            continue
        r0 = ps[0]
        ctrl = pf.best_control(pf.fed(
            rows, s, alpha=r0["alpha"], weight_decay=r0["weight_decay"], n_train=r0["n_train"],
            num_clients=K, group=lambda g: g not in ("participation_setups", "participation")))
        out[key] = {round(f, 2): rs for f, rs in sorted(pf.by_cell(ps + ctrl, "fraction_train").items())}
    return out


def in_rounds(runs, field):
    """(KM median, lo, hi, fraction reaching it) of a step time converted to rounds."""
    conv = [r[field] / (r["local_epochs"] * r["fraction_train"]) if pf._finite(r[field])
            else math.inf for r in runs]
    budget = [r["budget"] / (r["local_epochs"] * r["fraction_train"]) for r in runs]
    fin = [c for c in conv if math.isfinite(c)]
    return (pf.km(conv, budget), min(fin) if fin else math.inf, max(fin) if fin else math.inf,
            len(fin) / len(runs))


def fade(key):
    return pf.WITHHELD if key == "C" else 1.0


def colour(key):
    return pf.COL[dict((k, s) for k, s, *_ in LADDERS)[key]]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png/.pdf")


def ladder_handles(extra=True):
    h = [Line2D([], [], color=colour(k), ls=ls, marker="o", lw=1.2, ms=4.5, alpha=fade(k),
                label=lab + (" (withheld)" if k == "C" else "") + (", K = 20" if K == 20 and k != "A50" else ""))
         for k, s, g, K, ls, lab in LADDERS]
    if extra:
        h += [Line2D([], [], color=pf.INK2, marker="o", ls="none", ms=4.5, label="all runs reached it"),
              Line2D([], [], color=pf.INK2, marker="o", ls="none", ms=4.5, mfc="white", mew=1.2,
                     label="some runs did not"),
              Line2D([], [], color=pf.INK2, marker="x", ls="none", ms=6, mew=1.5,
                     label="no run reached it")]
    return h


def f_axis(ax):
    # 0.2 and 0.25 are too close to label both; the K = 20 ladder's rungs are
    # 0.25 / 0.5 / 1 and A's K = 50 ladder's 0.2 / 0.4 / 0.6 / 1.
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0.2", "0.4", "0.6", "0.8", "1"], fontsize=7)
    ax.set_xlim(0.12, 1.08)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("participation fraction f")


# ── two clocks, in rounds ────────────────────────────────────────────────────

def clock_panel(ax, lads, field, ylim=(0.4, 2.5)):
    for i, (key, *_rest) in enumerate(LADDERS):
        lad = lads.get(key)
        if not lad or 1.0 not in lad:
            continue
        ls = dict((k, l) for k, _, _, _, l, _ in LADDERS)[key]
        base = in_rounds(lad[1.0], field)[0]
        xs, ys, lo, hi, fr = [], [], [], [], []
        for f, runs in lad.items():
            m, a, b, frac = in_rounds(runs, field)
            x = f + (i - 2.5) * 0.006
            if not (math.isfinite(m) and math.isfinite(base)):
                pf.censored_x(ax, x, colour(key), y=0.955, size=6, alpha=fade(key))
                continue
            xs.append(x); ys.append(m / base); lo.append(a / base); hi.append(b / base); fr.append(frac)
        pf.series(ax, xs, ys, fr, colour(key), err=(lo, hi), alpha=fade(key), size=4.5, lw=1.2,
                  ls=ls)
    ax.axhline(1.0, color=pf.MUTED, ls="--", lw=0.9, zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ticks = [t for t in (0.5, 0.75, 1, 1.5, 2) if ylim[0] <= t <= ylim[1]]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}×" for t in ticks])
    ax.minorticks_off()
    f_axis(ax)


def two_clocks(lads):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.3))
    clock_panel(axes[0], lads, "t_memo")
    clock_panel(axes[1], lads, "t_first_cross")
    axes[0].set_ylabel(pf.TMEMO + " rounds / f = 1")
    axes[1].set_ylabel(pf.TFC + " rounds / f = 1")
    axes[0].set_title("Rounds to Memorise vs Participation")
    axes[1].set_title("Rounds to Generalise vs Participation")
    for ax, L in zip(axes, "ab"):
        pf.letter(ax, L)
    fig.legend(handles=ladder_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, fontsize=6.6)
    save(fig, "exp5_two_clocks")
    for field, name, title, ylab in (
            ("t_memo", "exp5_memorise_vs_f", "Rounds to Memorise vs Participation",
             pf.TMEMO + " rounds / f = 1"),
            ("t_first_cross", "exp5_generalise_vs_f", "Rounds to Generalise vs Participation",
             pf.TFC + " rounds / f = 1")):
        fig, ax = plt.subplots(figsize=(4.6, 3.5))
        clock_panel(ax, lads, field)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        fig.legend(handles=ladder_handles(), loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 0.0), fontsize=6.4)
        fig.tight_layout()
        save(fig, name)


# ── accuracy ─────────────────────────────────────────────────────────────────

def accuracy_vs_f(lads):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    rng = random.Random(0)
    for i, (key, s, g, K, ls, lab) in enumerate(LADDERS):
        lad = lads.get(key, {})
        xs, meds = [], []
        for f, runs in lad.items():
            x = f + (i - 2.5) * 0.008
            vals = [r["final_acc"] for r in runs if r["final_acc"] is not None]
            for v in vals:
                ax.plot(x + rng.uniform(-0.003, 0.003), v, "o", ms=2.6, color=colour(key),
                        alpha=0.5 * fade(key), mec="none", zorder=3)
            xs.append(x); meds.append(float(np.median(vals)))
        ax.plot(xs, meds, color=colour(key), ls=ls, lw=1.3, marker="o", ms=4.5,
                alpha=fade(key), zorder=4)
    for y, lab in ((95, "bar A, B"), (90, "bar E"), (85, "bar C, D")):
        ax.axhline(y, color=pf.MUTED, ls=":", lw=0.8, zorder=1)
        ax.text(1.01, y, lab, transform=ax.get_yaxis_transform(), fontsize=6,
                color=pf.MUTED, va="center", clip_on=False)
    f_axis(ax)
    ax.set_ylim(60, 102)
    ax.set_ylabel("final test accuracy (%)")
    ax.set_title("Final Test Accuracy vs Participation")
    fig.legend(handles=ladder_handles(extra=False), loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=6.4)
    fig.tight_layout()
    save(fig, "exp5_accuracy_vs_f")


# ── curves, in rounds ────────────────────────────────────────────────────────

def curves(lads):
    keys = [k for k, *_ in LADDERS if k in lads]
    fig, axes = plt.subplots(2, len(keys), figsize=(2.1 * len(keys), 4.6), sharey=True,
                             gridspec_kw=dict(hspace=0.15, wspace=0.12))
    present = set()
    for j, key in enumerate(keys):
        _, s, g, K, _, lab = next(L for L in LADDERS if L[0] == key)
        lad = lads[key]
        bar = None
        for f, runs in sorted(lad.items(), reverse=True):
            present.add(f)
            col = F_COL[f]
            for r in runs:
                h = pf.history(r["id"])
                if h is None:
                    continue
                pts = [(x, te, tr) for x, te, tr in zip(h["round"], h["test_acc"], h["train_acc"])
                       if x > 0]
                xs = [p[0] for p in pts]
                ls = ":" if ec.dropped_clients(r, h) else "-"
                axes[0, j].plot(xs, [p[1] for p in pts], color=col, lw=1.0, ls=ls, alpha=0.85)
                axes[1, j].plot(xs, [p[2] for p in pts], color=col, lw=1.0, ls=ls, alpha=0.85)
                bar = r["grok_threshold"]
                t = r["t_first_cross"]
                if pf._finite(t):
                    tr_ = t / (r["local_epochs"] * r["fraction_train"])
                    y = next(te for x, te, _ in pts if x >= tr_ - 1e-9)
                    axes[0, j].plot(tr_, y, "o", ms=4, color=col, mec="white", mew=0.7, zorder=5)
                else:
                    axes[0, j].plot(xs[-1], pts[-1][1], "x", ms=5.5, color=col, mew=1.3, zorder=5)
        for ax, y in ((axes[0, j], bar), (axes[1, j], 99.0)):
            ax.axhline(y, color=pf.MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
            ax.set_xscale("log")
            ax.set_ylim(-3, 103)
            ax.set_yticks([0, 50, 100])
            ax.tick_params(labelsize=6.5)
            ax.grid(axis="x", visible=False)
        axes[0, j].set_title(lab + (" (withheld)" if s == "C" else "") + f"\nK = {int(K)}",
                             fontsize=7)
        axes[1, j].set_xlabel("communication rounds", fontsize=7)
    axes[0, 0].set_ylabel("test accuracy (%)", fontsize=8)
    axes[1, 0].set_ylabel("train accuracy (%)", fontsize=8)
    handles = [Line2D([], [], color=F_COL[f], lw=1.8, label=f"f = {f:g}")
               for f in sorted(present, reverse=True)] + ec.MARK_HANDLES[:2]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(handles), fontsize=7)
    fig.suptitle("Training Curves per Run across Participation Fractions (x = rounds)",
                 fontsize=9.5, y=1.0)
    save(fig, "exp5_curves")


def main():
    rows = pf.load_rows()
    lads = ladders(rows)
    for key, lad in lads.items():
        print(f"  {key}: " + "; ".join(
            f"f={f:g}: {len(rs)} runs, memo {in_rounds(rs, 't_memo')[0]:.0f} r, "
            f"cross {in_rounds(rs, 't_first_cross')[0]:.0f} r, "
            f"{sum(pf._finite(r['t_first_cross']) for r in rs)}/{len(rs)} crossed"
            for f, rs in lad.items()))
    two_clocks(lads)
    accuracy_vs_f(lads)
    curves(lads)


if __name__ == "__main__":
    main()
