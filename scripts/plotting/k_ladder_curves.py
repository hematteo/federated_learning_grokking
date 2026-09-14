"""Accuracy curves for the IID client-count ladder, one panel per setup.

    venv/bin/python scripts/plotting/k_ladder_curves.py                 # -> figures/k_iid/
    venv/bin/python scripts/plotting/k_ladder_curves.py --out DIR --per-setup

Complements paper_figures.fig1/figA1 (the two clocks against K) and
exp2_slowdown_ratio (the ratio against K) with the trajectories those summaries
are read from: held-out accuracy against gradient steps for every run of each
setup's FedAvg K ladder, iid, E=5, at the setup's working point, with the
centralised control in gray. K is an ordered factor, so rungs take a single-hue
ramp (the documented blue steps) rather than categorical slots. Solid lines are
test accuracy, dashed lines (with --train) training accuracy; the end marker is
filled when the run held the bar, hollow when it crossed and fell back, a cross
when it never crossed. Selection, statistics and history loading are
paper_figures' own, so the runs drawn here are exactly the ones fig1 summarises.
"""
import argparse
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("pf", os.path.join(_HERE, "paper_figures.py"))
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)

RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
GRAY = "#898781"
SETUPS = ["A", "A'", "B", "C", "D", "E"]
LABEL = {"A": "quad-MLP, mod 97, GD", "A'": "quad-MLP, mod 97, AdamW", "B": "transformer, mod 113",
         "C": "transformer, S$_5$", "D": "quad-MLP, S$_5$", "E": "ReLU MLP, MNIST-1k"}


def ladder_runs(rows, s):
    """(series key, {K: runs}, centralised runs) for the setup's main iid K ladder."""
    rs = [r for r in pf.fed(rows, s) if r["group"] in pf.SRC1]
    ser = pf.by_cell(rs, "alpha", "weight_decay", "n_train")
    if not ser:
        return None, {}, []
    # the setup's working point: the most K rungs, then the standing transformer
    # decay (wd 0.1 for B and C, PROGRESS "Standing decisions"), then the harder alpha
    def rank(k):
        alpha, wd, _ = k
        return (1 if (s in ("B", "C") and wd == 0.1) else 0, len(pf.by_cell(ser[k], "num_clients")), -alpha)
    key = max(ser, key=rank)
    alpha, wd, ntr = key
    cells = {int(K): v for K, v in pf.by_cell(ser[key], "num_clients").items()}
    cent = pf.cent_runs(rows, s, alpha, wd)
    if s == "E":
        cent = [r for r in cent if r["n_train"] == ntr]
    return key, cells, cent


def draw(ax, s, key, cells, cent, train=False):
    alpha, wd, ntr = key
    Ks = sorted(cells)
    colors = {K: RAMP[round(i / max(1, len(Ks) - 1) * (len(RAMP) - 1))] for i, K in enumerate(Ks)}
    groups = [("centralised", GRAY, cent)] + [(f"K = {K}", colors[K], cells[K]) for K in Ks]
    bar = None
    for label, col, runs in groups:
        for r in runs:
            h = pf.history(r["id"])
            if h is None:
                continue
            steps = h.get("total_steps") or h.get("epoch")
            xs = [x for x in steps if x > 0]
            test = [y for x, y in zip(steps, h["test_acc"]) if x > 0]
            ax.plot(xs, test, color=col, lw=1.1, alpha=0.9, zorder=3)
            if train:
                tr = [y for x, y in zip(steps, h["train_acc"]) if x > 0]
                ax.plot(xs, tr, color=col, lw=0.8, ls="--", alpha=0.6, zorder=2)
            bar = r["grok_threshold"] or bar
            x_end, y_end = xs[-1], test[-1]
            if r["grokked"]:
                ax.plot(x_end, y_end, "o", ms=4, color=col, mec="white", mew=0.8, zorder=5)
            elif pf._finite(r["t_first_cross"]):
                ax.plot(x_end, y_end, "o", ms=4, mfc="white", mec=col, mew=1.2, zorder=5)
            else:
                ax.plot(x_end, y_end, "x", ms=5, color=col, mew=1.3, zorder=5)
    if bar:
        ax.axhline(bar, color=GRAY, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(0.99, bar / 100 - 0.03, f"bar {bar:g}", transform=ax.get_yaxis_transform(),
                ha="right", va="top", fontsize=7, color=GRAY)
    ax.set_xscale("log")
    ax.set_ylim(-2, 103)
    ax.set_yticks([0, 50, 100])
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ctx = f"n_train {int(ntr)}" if s == "E" else f"α {alpha:g}"
    n = sum(len(v) for v in cells.values()) + len(cent)
    ax.set_title(f"{s} · {LABEL[s]}\n{ctx} · wd {wd:g} · iid · E = 5 · {n} runs",
                 fontsize=8, loc="left", pad=6, linespacing=1.4)
    handles = [Line2D([], [], color=col, lw=1.6, label=label) for label, col, _ in groups]
    if train:
        handles += [Line2D([], [], color=GRAY, lw=1.2, label="test"),
                    Line2D([], [], color=GRAY, lw=1.0, ls="--", label="train")]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="upper left", ncol=2,
              handlelength=1.4, columnspacing=0.9, borderaxespad=0.2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/k_iid")
    ap.add_argument("--train", action="store_true", help="also draw training accuracy, dashed")
    ap.add_argument("--per-setup", action="store_true", help="one PNG per setup as well as the grid")
    ap.add_argument("--seed", type=int, default=None, help="draw only runs with this seed (per-setup figures)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict the per-setup figures to these setups")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = pf.load_rows()
    data = {s: ladder_runs(rows, s) for s in SETUPS}
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for ax, s in zip(axes.flat, SETUPS):
        key, cells, cent = data[s]
        if not cells:
            ax.set_axis_off(); continue
        draw(ax, s, key, cells, cent, train=args.train)
    for ax in axes[1]:
        ax.set_xlabel("gradient steps", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)" if args.train else "held-out accuracy (%)", fontsize=8)
    fig.suptitle("IID client-count ladder: held-out accuracy per run", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"k_ladder_curves.{ext}"), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {args.out}/k_ladder_curves.png/.pdf")
    if args.per_setup:
        for s in SETUPS:
            key, cells, cent = data[s]
            if not cells or (args.only and s not in args.only):
                continue
            if args.seed is not None:
                cells = {K: [r for r in v if int(r["seed"]) == args.seed] for K, v in cells.items()}
                cells = {K: v for K, v in cells.items() if v}
                cent = [r for r in cent if int(r["seed"]) == args.seed]
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            draw(ax, s, key, cells, cent, train=args.train)
            if args.seed is not None:
                ax.set_title(ax.get_title(loc="left").replace(" runs", f" runs · seed {args.seed}"), fontsize=8, loc="left", pad=6, linespacing=1.4)
            ax.set_xlabel("gradient steps", fontsize=8); ax.set_ylabel("accuracy (%)" if args.train else "held-out accuracy (%)", fontsize=8)
            fig.tight_layout()
            name = f"k_ladder_curves_{s.replace(chr(39), 'prime')}" + (f"_seed{args.seed}" if args.seed is not None else "") + ("_train" if args.train else "")
            fig.savefig(os.path.join(args.out, f"{name}.png"), dpi=180, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  wrote {args.out}/{name}.png  ({', '.join(f'K={K}: {len(v)}' for K, v in sorted(cells.items()))}; cent {len(cent)})")


if __name__ == "__main__":
    main()
