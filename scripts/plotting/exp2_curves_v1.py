"""Training curves behind exp2_slowdown_ratio_v1_<setup>.png.

    venv/bin/python scripts/plotting/exp2_curves_v1.py            # setup B -> figures/exp2/
    venv/bin/python scripts/plotting/exp2_curves_v1.py --setup D --out DIR

The ratio figure reduces each run to one number, t_first_cross. This draws the
trajectories it was read from: one column per data-axis value (one line on the
ratio plot), test accuracy on the top row and train accuracy below, every run of
the FedAvg K ladder coloured by K on a single-hue ramp, the centralised baseline
that line is divided by in gray. Runs are selected exactly as
exp2_slowdown_ratio.load() selects them -- by manifest run id, arm from the spec
-- so the curves are the runs the ratio plot summarises and no others. The dot on
each test curve is that run's first crossing; a cross marks a run that never
crossed within its budget.
"""
import argparse
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
from exp2_slowdown_ratio import MANIFESTS, SETUPS, _axis_value, _tfc   # noqa: E402
from fedgrok.manifest import load_manifest, run_id                     # noqa: E402

# Same ramp and neutral as k_ladder_curves.py, so the two curve figures agree.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
GRAY = "#898781"
INK2, RULE = "#52514e", "#e1e0d9"

# --clean fixes everything that otherwise varies with the data, so figures for
# different setups line up: one colour per K, one x-range, one canvas and margins.
CLEAN_KS = (2, 5, 10, 20, 50)
CLEAN_XLIM = (70, 3e5)
CLEAN_SIZE = (7.0, 5.6)
CLEAN_MARGINS = dict(left=0.115, right=0.985, top=0.985, bottom=0.185, hspace=0.12)


def select(setup, csv_path):
    """{axis value: {"cent": [rows], K: [rows]}} for the runs the ratio plot uses."""
    banked = {r["id"]: r for r in csv.DictReader(open(csv_path))}
    out, seen = {}, set()
    for m in MANIFESTS:
        for spec in load_manifest(m):
            rid = run_id(spec)
            if spec.get("setup") != setup or rid in seen or rid not in banked:
                continue
            arm = spec.get("arm")
            if arm not in ("cent_full", "fl"):
                continue                  # the reduced floor arm is not on the ratio plot
            seen.add(rid)
            row = banked[rid]
            key = "cent" if arm == "cent_full" else int(row["num_clients"])
            out.setdefault(_axis_value(row), {}).setdefault(key, []).append(row)
    # Keep only series the ratio plot draws: a baseline and at least one K.
    return {v: c for v, c in out.items() if "cent" in c and len(c) > 1}


def history(rid):
    paths = glob.glob(f"results/runs/{rid}/history_*.json")
    return json.load(open(paths[0])) if paths else None


def k_colors(ks):
    """One colour per K across the whole figure, so a rung keeps its colour in
    every column even when a column is missing the higher rungs."""
    ks = sorted(ks)
    return {k: RAMP[round(i / max(1, len(ks) - 1) * (len(RAMP) - 1))]
            for i, k in enumerate(ks)}


def draw(ax_test, ax_train, cells, col, clean=False):
    ks = sorted(k for k in cells if k != "cent")
    # Centralised last and on top: on setup A the federated curves sit exactly on
    # it, and drawn underneath it disappears.
    groups = [(k, col[k]) for k in ks] + [("cent", GRAY)]
    bar = None
    for key, c in groups:
        for r in cells[key]:
            h = history(r["id"])
            if h is None:
                print(f"  no history for {r['id']}, skipped")
                continue
            steps = h.get("total_steps") or h.get("epoch")
            pts = [(x, te, tr) for x, te, tr in zip(steps, h["test_acc"], h["train_acc"])
                   if x > 0]
            xs = [p[0] for p in pts]
            lw, z = (1.3, 4) if key == "cent" else (1.0, 3)
            ax_test.plot(xs, [p[1] for p in pts], color=c, lw=lw, alpha=0.85, zorder=z)
            ax_train.plot(xs, [p[2] for p in pts], color=c, lw=lw, alpha=0.85, zorder=z)
            bar = float(r["grok_threshold"])
            t = _tfc(r)
            if t != float("inf"):
                y = next(te for x, te, _ in pts if x >= t)
                ax_test.plot(t, y, "o", ms=4.5, color=c, mec="white", mew=0.8, zorder=5)
            elif not clean:
                ax_test.plot(xs[-1], pts[-1][1], "x", ms=6, color=c, mew=1.4, zorder=5)
    for ax, label in ((ax_test, f"bar {bar:g}"), (ax_train, "99% (t_memo)")):
        y = bar if ax is ax_test else 99.0
        ax.axhline(y, color=GRAY, lw=0.8, ls=(0, (4, 3)), zorder=1)
        # In the right margin, where no curve can run through it.
        if not clean:
            ax.text(1.01, y, label, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=7, color=GRAY, clip_on=False)
    for ax in (ax_test, ax_train):
        ax.set_xscale("log")
        if clean:
            ax.set_xlim(*CLEAN_XLIM)
        ax.set_ylim(-2, 103)
        ax.set_yticks([0, 50, 100])
        ax.grid(axis="y", color=RULE, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=11 if clean else 8, colors="black" if clean else INK2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--setup", default="B")
    ap.add_argument("--out", default="figures/exp2")
    ap.add_argument("--csv", default="results/data/runs_v2.csv")
    ap.add_argument("--drop", type=float, nargs="+", default=[],
                    help="data-axis values (alpha, or n_train on E) to leave out; "
                         "written to a separate file")
    ap.add_argument("--clean", action="store_true",
                    help="title reduced to setup, architecture and the data-axis "
                         "value; no panel header, no labels on the bar and t_memo "
                         "lines, and no crosses on runs that never crossed")
    ap.add_argument("--no-title", action="store_true", help="omit the figure title")
    ap.add_argument("--name", help="setup letter shown in the title (default: --setup)")
    a = ap.parse_args()
    # --clean is the paper version: larger, black text.
    fs, ink = (12, "black") if a.clean else (9, INK2)

    data = select(a.setup, a.csv)
    data = {v: c for v, c in data.items() if float(v) not in a.drop}
    if not data:
        sys.exit(f"setup {a.setup}: no banked series")
    vals = sorted(data, key=float)
    size = CLEAN_SIZE if a.clean else (max(5.2 * len(vals), 7.0), 5.6)
    fig, axes = plt.subplots(2, len(vals), figsize=size, sharex="col", squeeze=False)
    name = "n_train" if a.setup == "E" else "α"
    present = {k for c in data.values() for k in c if k != "cent"}
    col = k_colors(present | set(CLEAN_KS)) if a.clean else k_colors(present)
    col = {k: c for k, c in col.items() if k in present}
    for j, v in enumerate(vals):
        cells = data[v]
        draw(axes[0, j], axes[1, j], cells, col, clean=a.clean)
        r0 = cells["cent"][0]
        ks = sorted(k for k in cells if k != "cent")
        n = sum(len(x) for x in cells.values())
        if not a.clean:
            axes[0, j].set_title(f"{name} = {float(v):g} · wd {float(r0['weight_decay']):g} · "
                                 f"K = {', '.join(map(str, ks))} · {n} runs",
                                 fontsize=9, loc="left", color=INK2)
        axes[1, j].set_xlabel("gradient steps", fontsize=fs, color=ink)
        print(f"  {name}={float(v):g}: cent {len(cells['cent'])}, "
              + ", ".join(f"K={k}: {len(cells[k])}" for k in ks))
    axes[0, 0].set_ylabel("test accuracy (%)", fontsize=fs, color=ink)
    axes[1, 0].set_ylabel("train accuracy (%)", fontsize=fs, color=ink)
    label = dict(SETUPS)[a.setup]
    details = "E = 5 · FedAvg · IID"
    markers = " · dot = first crossing, × = never crossed"
    # Clean title: setup, architecture and -- with one column -- the data-axis value.
    short = [f"Setup {a.name or a.setup}", label.split(" · ")[0]]
    if len(vals) == 1:
        short.append(f"{name} = {float(vals[0]):g}")
    title = " · ".join(short) if a.clean else (
        f"Setup {a.setup} — {label}: training curves behind the slowdown ratio\n{details}{markers}")
    if not a.no_title:
        fig.suptitle(title, fontsize=10.5, x=0.01, ha="left")
    handles = [Line2D([], [], color=GRAY, lw=1.8, label="centralised")] + \
              [Line2D([], [], color=c, lw=1.8, label=f"K = {k}") for k, c in col.items()]
    if a.clean:
        fig.subplots_adjust(**CLEAN_MARGINS)
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=11,
                   frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.3,
                   handletextpad=0.5, columnspacing=0.9)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8.5,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.05, 1, 1.0 if a.no_title else 0.95))
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"exp2_curves_v1_{a.setup.replace(chr(39), 'prime')}")
    if a.drop:
        stem += "_without_" + "_".join(f"{v:g}" for v in a.drop)
    for ext in ("png", "pdf"):
        # No tight bbox under --clean: it would crop each canvas to its own legend.
        fig.savefig(f"{stem}.{ext}", dpi=200, facecolor="white",
                    bbox_inches=None if a.clean else "tight")
    plt.close(fig)
    print(f"  wrote {stem}.png/.pdf")


if __name__ == "__main__":
    main()
