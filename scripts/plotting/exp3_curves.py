"""Training curves for exp3, data heterogeneity: the Dirichlet ladder and the partitions.

    venv/bin/python scripts/plotting/exp3_curves.py              # -> figures/exp3/
    venv/bin/python scripts/plotting/exp3_curves.py --out DIR

Two families of figure, each with test accuracy on the top row and train accuracy
below, against gradient steps:

  exp3a_dirichlet_curves.png     one column per setup (A-E): FedAvg, K=10, E=5,
  exp3a_dirichlet_curves_<S>.png Dirichlet label skew over dir_alpha in
                                 {0.01 ... 1000}, darker = more concentrated
                                 labels, with the matched IID run in gray.
  exp3b_partition_curves_<S>.png one column per K: operand / label
                                 (target on A-D, label_block on E) / Dirichlet 0.5
                                 against the matched IID baseline in gray.

Selection is paper_figures.fig4's own -- `dirichlet_setups` plus the 0.5 rung
from `partitions` (or `k_fixed_total` on A), and `partitions` against the IID
baseline `best_control` picks -- so these are the trajectories that figure
reduces to points and bars. A′ is out of the paper and is not drawn.

Markers: a dot at a run's first crossing of the bar, a cross at the end of a run
that never crossed. A DOTTED line is a run in which fewer clients reported than
were configured on at least one logged round (client failures Flower tolerated
silently); those runs are not the cell they are labelled as.
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

# Ordered axis -> one hue, light to dark (k_ladder_curves' ramp). Dark = skewed.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
GRAY = "#898781"
INK2, RULE = "#52514e", "#e1e0d9"
SETUPS = "ABCDE"
DIR_RUNGS = [1000.0, 10.0, 1.0, 0.5, 0.1, 0.01]          # IID-like -> concentrated
DIR_COL = dict(zip(DIR_RUNGS, RAMP))
PART_ORDER = ["operand", "label", "dirichlet"]
PART_LABEL = {"operand": "operand", "label": "label",
              "dirichlet": "Dirichlet 0.5"}
# `target` (label mod K) and `label_block` (contiguous label ranges) do the same
# job -- shard by output class -- and no setup ran both: target is A-D, and
# label_block exists only because target leaves clients empty on MNIST once K
# exceeds its 10 classes. They are drawn as one "label" condition, in target's colour.
PART_COL = {**pf.PART, "label": pf.PART["target"]}
IID_GROUPS = ("aggregation", "k_fixed_total", "setup_k_ladder")


# ── selection ────────────────────────────────────────────────────────────────

def iid_control(rows, s, K, like):
    return pf.best_control(pf.fed(
        rows, s, alpha=like["alpha"], weight_decay=like["weight_decay"],
        n_train=like["n_train"], num_clients=K, group=lambda g: g in IID_GROUPS))


def dirichlet_ladder(rows, s):
    ds = pf.sel(rows, group="dirichlet_setups", setup=s)
    if not ds:
        return None
    r0 = ds[0]
    if s == "A":
        mid = pf.sel(rows, group="k_fixed_total", setup="A", partition="dirichlet",
                     num_clients=10.0)
    else:
        mid = pf.sel(rows, group="partitions", setup=s, partition="dirichlet",
                     num_clients=10.0, alpha=r0["alpha"],
                     weight_decay=r0["weight_decay"], n_train=r0["n_train"])
    cells = pf.by_cell(ds + mid, "dirichlet_alpha")
    return r0, cells, iid_control(rows, s, 10.0, r0)


def partition_columns(rows, s):
    ps = pf.sel(rows, group="partitions", setup=s)
    cols = []
    for K in sorted({r["num_clients"] for r in ps}):
        here = [r for r in ps if r["num_clients"] == K]
        parts = pf.by_cell(here, "partition")
        label = parts.pop("target", []) + parts.pop("label_block", [])
        if label:
            parts["label"] = label
        # Coset (S5 only, K=5) is left out of the partition figures by choice.
        # Its K=5 column held nothing else, so it goes with it.
        parts.pop("coset", None)
        if not parts:
            continue
        cols.append((K, here[0], parts, iid_control(rows, s, K, here[0])))
    return cols


# ── drawing ──────────────────────────────────────────────────────────────────

def dropped_clients(r, h):
    """True if any logged round had fewer reporting clients than configured."""
    want = max(1, int(r["num_clients"] * (r["fraction_train"] or 1.0)))
    return any(n < want for n in h.get("n_participating", [])[1:])


def draw_groups(ax_test, ax_train, groups):
    """groups: [(color, runs, zorder, linewidth)], drawn in order. The IID baseline
    goes underneath and wider, so a heterogeneous run that tracks it stays visible."""
    bar, flagged = None, []
    for c, runs, z, lw in groups:
        for r in runs:
            h = pf.history(r["id"])
            if h is None:
                print(f"  no history for {r['id']}, skipped")
                continue
            steps = h.get("total_steps") or h.get("epoch")
            pts = [(x, te, tr) for x, te, tr in zip(steps, h["test_acc"], h["train_acc"])
                   if x > 0]
            xs = [p[0] for p in pts]
            ls = "-"
            if r["mode"] == "federated" and dropped_clients(r, h):
                ls = ":"
                flagged.append(r["id"])
            ax_test.plot(xs, [p[1] for p in pts], color=c, lw=lw, ls=ls, alpha=0.85, zorder=z)
            ax_train.plot(xs, [p[2] for p in pts], color=c, lw=lw, ls=ls, alpha=0.85, zorder=z)
            bar = r["grok_threshold"]
            t = r["t_first_cross"]
            if pf._finite(t):
                y = next(te for x, te, _ in pts if x >= t)
                ax_test.plot(t, y, "o", ms=4.5, color=c, mec="white", mew=0.8, zorder=z + 2)
            else:
                ax_test.plot(xs[-1], pts[-1][1], "x", ms=6, color=c, mew=1.4, zorder=z + 2)
    for ax, y, label in ((ax_test, bar, f"bar {bar:g}"), (ax_train, 99.0, "99%")):
        ax.axhline(y, color=GRAY, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(1.01, y, label, transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=6.5, color=GRAY, clip_on=False)
    for ax in (ax_test, ax_train):
        ax.set_xscale("log")
        ax.set_ylim(-2, 103)
        ax.set_yticks([0, 50, 100])
        ax.grid(False)
        ax.grid(axis="y", color=RULE, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=7, colors=INK2)
    return flagged


def context(r):
    data = f"n_train {int(r['n_train'])}" if r["setup"] == "E" else f"α {r['alpha']:g}"
    return f"{data} · wd {r['weight_decay']:g}"


def label_axes(axes):
    axes[0, 0].set_ylabel("test accuracy (%)", fontsize=8, color=INK2)
    axes[1, 0].set_ylabel("train accuracy (%)", fontsize=8, color=INK2)
    for ax in axes[1]:
        ax.set_xlabel("gradient steps", fontsize=8, color=INK2)


def finish(fig, handles, title, path, pdf=False):
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(path + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    if pdf:
        fig.savefig(path + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}.png")


MARK_HANDLES = [
    Line2D([], [], color=INK2, marker="o", ls="none", ms=4.5, label="first crossing"),
    Line2D([], [], color=INK2, marker="x", ls="none", ms=6, mew=1.4, label="never crossed"),
    Line2D([], [], color=INK2, lw=1.2, ls=":", label="clients dropped"),
]


# ── exp3a: Dirichlet ladder ──────────────────────────────────────────────────

def dirichlet_panel(ax_t, ax_b, s, ladder):
    r0, cells, base = ladder
    rungs = [d for d in DIR_RUNGS if d in cells]
    groups = [(GRAY, base, 2, 2.4)] + [(DIR_COL[d], cells[d], 3, 1.0) for d in rungs]
    flagged = draw_groups(ax_t, ax_b, groups)
    n = sum(len(cells[d]) for d in rungs)
    ax_t.set_title(f"{pf.NAME[s]}\nK = 10 · {context(r0)} · {n} runs + {len(base)} IID",
                   fontsize=8, loc="left", color=INK2)
    summary = ", ".join(
        f"{d:g}: {sum(pf._finite(r['t_first_cross']) for r in cells[d])}/{len(cells[d])}"
        for d in rungs)
    print(f"  {s}: crossed per dir_alpha {summary}; IID {len(base)} runs"
          + (f"; clients dropped in {flagged}" if flagged else ""))


def dirichlet_handles():
    return ([Line2D([], [], color=GRAY, lw=1.8, label="IID")]
            + [Line2D([], [], color=DIR_COL[d], lw=1.8, label=f"dir α = {d:g}")
               for d in DIR_RUNGS] + MARK_HANDLES)


def plot_dirichlet(rows, out):
    ladders = {s: dirichlet_ladder(rows, s) for s in SETUPS}
    ladders = {s: v for s, v in ladders.items() if v}
    title = ("exp3a — Dirichlet label skew: training curves per run\n"
             "FedAvg · K = 10 · E = 5 · darker = more concentrated labels")
    fig, axes = plt.subplots(2, len(ladders), figsize=(3.6 * len(ladders), 5.6),
                             sharex="col", squeeze=False)
    for j, (s, ladder) in enumerate(ladders.items()):
        dirichlet_panel(axes[0, j], axes[1, j], s, ladder)
    label_axes(axes)
    finish(fig, dirichlet_handles(), title, os.path.join(out, "exp3a_dirichlet_curves"),
           pdf=True)
    for s, ladder in ladders.items():
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex="col", squeeze=False)
        dirichlet_panel(axes[0, 0], axes[1, 0], s, ladder)
        label_axes(axes)
        finish(fig, dirichlet_handles(), title,
               os.path.join(out, f"exp3a_dirichlet_curves_{s}"))


# ── exp3b: partitions ────────────────────────────────────────────────────────

def plot_partitions(rows, out):
    for s in SETUPS:
        cols = partition_columns(rows, s)
        if not cols:
            continue
        fig, axes = plt.subplots(2, len(cols), figsize=(max(4.6 * len(cols), 7.0), 5.6),
                                 sharex="col", squeeze=False)
        present = set()
        for j, (K, r0, parts, base) in enumerate(cols):
            order = [p for p in PART_ORDER if p in parts]
            present |= set(order)
            groups = [(GRAY, base, 2, 2.4)] + [(PART_COL[p], parts[p], 3, 1.0) for p in order]
            flagged = draw_groups(axes[0, j], axes[1, j], groups)
            iid_note = f"{len(base)} IID" if base else "no matched IID"
            axes[0, j].set_title(f"K = {int(K)} · {context(r0)} · "
                                 f"{sum(len(parts[p]) for p in order)} runs + {iid_note}",
                                 fontsize=8, loc="left", color=INK2)
            summary = ", ".join(
                f"{p}: {sum(pf._finite(r['t_first_cross']) for r in parts[p])}/{len(parts[p])}"
                for p in order)
            print(f"  {s} K={int(K)}: {summary}; IID {len(base)}"
                  + (f"; clients dropped in {flagged}" if flagged else ""))
        label_axes(axes)
        handles = ([Line2D([], [], color=GRAY, lw=1.8, label="IID")]
                   + [Line2D([], [], color=PART_COL[p], lw=1.8, label=PART_LABEL[p])
                      for p in PART_ORDER if p in present] + MARK_HANDLES)
        finish(fig, handles,
               f"exp3b — partition structure, {pf.NAME[s]}: training curves per run\n"
               "FedAvg · E = 5 · each partition against the matched IID baseline (gray)",
               os.path.join(out, f"exp3b_partition_curves_{s}"), pdf=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/exp3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = pf.load_rows()
    print("exp3a (Dirichlet):")
    plot_dirichlet(rows, a.out)
    print("exp3b (partitions):")
    plot_partitions(rows, a.out)


if __name__ == "__main__":
    main()
