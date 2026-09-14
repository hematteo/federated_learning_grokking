"""exp2 against the client count, drawn in the v1 style of results/figures/.

    venv/bin/python scripts/plotting/exp2_k_axis.py               # -> figures/exp2/
    venv/bin/python scripts/plotting/exp2_k_axis.py --out DIR

Two figures, each as one overview and one file per setup:

  exp2_tgrok_vs_K[_<setup>].png         grokking time vs K, centralized vs FL,
                                        one panel per data-axis value -- the form
                                        of v1's fig3_tgrok_vs_K.png
  exp2_slowdown_ratio_v1[_<setup>].png  FL / centralized vs K, one line per
                                        data-axis value -- v1's
                                        exp2_slowdown_ratio.png

Only the look is v1's. The rows, the statistic and the censoring come from
exp2_slowdown_ratio.load(), for the reasons its docstring gives: rows matched by
manifest run id, each alpha with its own centralized baseline, t_first_cross
rather than t_grok, medians with seed min-max bars, and censored cells drawn
rather than dropped. Setup A's alpha 0.35 / 0.5 series are v1's t_grok ladders
and are dashed and labelled as such.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from exp2_slowdown_ratio import SETUPS, load        # noqa: E402

# plot_exp2.py's rcParams, which produced the v1 figure set.
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
})

CENT, FL = "#2196F3", "#FF5722"              # fig3_tgrok_vs_K's two conditions
LINES = ["#2196F3", "#FF9800", "#009688"]    # plot_slowdown_ratio's alpha colours
CENSORED = "#d32f2f"
SUBTITLE = "E=5 · FedAvg · IID · median over 3 seeds, bars = seed range"


def _axis_label(setup, series):
    name = "n_train" if setup == "E" else "α"
    val = float(series["val"])
    return f"{name} = {val:g}" if setup == "E" else f"{name} = {val:.2f}"


def _k_axis(ax, ks):
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlim(min(ks) * 0.75, max(ks) * 1.3)
    ax.set_xlabel("K (clients)")


def _mark_censored(ax, cells, color=CENSORED, y=0.94):
    """Fully censored cells: a hollow marker pinned near the top, with n/n."""
    tr = ax.get_xaxis_transform()
    for c in cells:
        ax.plot([c["K"]], [y], marker="o", mfc="none", mec=color, mew=1.8,
                markersize=7, ls="none", transform=tr, clip_on=False, zorder=4)
        ax.annotate(f"{c['grokked']}/{c['n']}", (c["K"], y), xycoords=tr,
                    textcoords="offset points", xytext=(0, -11), ha="center",
                    va="top", fontsize=8, color=color)


def _mark_partial(ax, cells, scale=1.0):
    last = max(c["K"] for c in cells)
    for c in cells:
        if c["median"] is not None and c["grokked"] < c["n"]:
            # Label on the inside of the panel, so the last K does not run off it.
            right = c["K"] < last
            ax.annotate(f"{c['grokked']}/{c['n']}", (c["K"], c["median"] / scale),
                        textcoords="offset points", xytext=(7 if right else -7, 7),
                        ha="left" if right else "right", va="bottom", fontsize=8,
                        color=CENSORED)


# ── Figure 1: grokking time vs K ─────────────────────────────────────────────

def _tgrok_panel(ax, setup, s):
    ks = [c["K"] for c in s["cells"]]
    legacy = s.get("legacy", False)
    b = s["base"]
    ax.errorbar(ks, [b] * len(ks),
                yerr=[[b - s["base_lo"]] * len(ks), [s["base_hi"] - b] * len(ks)],
                marker="o", color=CENT, label=f"Centralized (full, n={s['base_n']})",
                capsize=3, linewidth=1.5, markersize=5)
    ok = [c for c in s["cells"] if c["median"] is not None]
    if ok:
        m = [c["median"] for c in ok]
        ax.errorbar([c["K"] for c in ok], m,
                    yerr=[[x - c["lo"] for x, c in zip(m, ok)],
                          [c["hi"] - x for x, c in zip(m, ok)]],
                    marker="s", color=FL, ls="--" if legacy else "-",
                    label="FL IID", capsize=3, linewidth=1.5, markersize=5)
    _mark_partial(ax, s["cells"])
    _mark_censored(ax, [c for c in s["cells"] if c["median"] is None])
    _k_axis(ax, ks)
    ax.set_title(_axis_label(setup, s) + ("  [v1, $T_{grok}$]" if legacy else ""))
    ax.grid(axis="y", alpha=0.3)


def _tgrok_row(axes, setup, series):
    for ax, s in zip(axes, series):
        _tgrok_panel(ax, setup, s)
    for ax in axes[len(series):]:
        ax.set_visible(False)
    top = max(max([s["base_hi"]] + [c["hi"] for c in s["cells"] if c["hi"]])
              for s in series)
    for ax in axes[:len(series)]:
        ax.set_ylim(0, top * 1.18)        # headroom for the censored markers
    axes[0].set_ylabel(r"$T_{first\ cross}$ (steps)")
    axes[len(series) - 1].legend(loc="upper left", fontsize=8)


def plot_tgrok_setup(setup, label, series, out_dir):
    fig, axes = plt.subplots(1, len(series), figsize=(4.5 * len(series), 4.5),
                             sharey=True, squeeze=False)
    _tgrok_row(list(axes[0]), setup, series)
    fig.suptitle(f"Exp 2: Grokking Time vs Number of Clients — Setup {setup} "
                 f"({label})\n{SUBTITLE}", fontsize=13)
    plt.tight_layout()
    return _save(fig, out_dir, f"exp2_tgrok_vs_K_{_slug(setup)}.png")


def plot_tgrok_all(data, out_dir):
    rows = [(s, l) for s, l in SETUPS if s in data]
    ncol = max(len(data[s]) for s, _ in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(4.5 * ncol, 3.9 * len(rows)),
                             squeeze=False)
    for i, (setup, label) in enumerate(rows):
        row = list(axes[i])
        for ax in row[1:]:
            ax.sharey(row[0])
        _tgrok_row(row, setup, data[setup])
        row[0].set_ylabel(f"Setup {setup}\n" + r"$T_{first\ cross}$ (steps)")
        row[0].annotate(label, xy=(0, 1.13), xycoords="axes fraction",
                        fontsize=9, color="#555555", style="italic")
    fig.suptitle(f"Exp 2: Grokking Time vs Number of Clients (per-setup scale)\n"
                 f"{SUBTITLE}", fontsize=14, y=1.01)
    plt.tight_layout()
    return _save(fig, out_dir, "exp2_tgrok_vs_K.png")


# ── Figure 2: slowdown ratio vs K ────────────────────────────────────────────

def _ratio_panel(ax, setup, label, series):
    all_ks, tops = set(), [1.0]
    for i, s in enumerate(series):
        col, b = LINES[i % len(LINES)], s["base"]
        legacy = s.get("legacy", False)
        all_ks |= {c["K"] for c in s["cells"]}
        ok = [c for c in s["cells"] if c["median"] is not None]
        if ok:
            r = [c["median"] / b for c in ok]
            tops += [c["hi"] / b for c in ok]
            ax.errorbar([c["K"] for c in ok], r,
                        yerr=[[x - c["lo"] / b for x, c in zip(r, ok)],
                              [c["hi"] / b - x for x, c in zip(r, ok)]],
                        marker="s" if legacy else "o", ls="--" if legacy else "-",
                        color=col, capsize=3, linewidth=1.8, markersize=6,
                        label=_axis_label(setup, s)
                        + ("  [v1, $T_{grok}$]" if legacy else ""))
        _mark_partial(ax, s["cells"], scale=b)
        # Offset each series' censored row so two cannot overprint.
        _mark_censored(ax, [c for c in s["cells"] if c["median"] is None],
                       color=col, y=0.94 - 0.09 * i)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.4, linewidth=1,
               label="No slowdown")
    _k_axis(ax, sorted(all_ks))
    ax.set_ylim(0, max(tops) * 1.2)
    ax.set_ylabel(r"$T^\mathrm{FL}_\mathrm{first\ cross}\;/\;"
                  r"T^\mathrm{cent}_\mathrm{first\ cross}$")
    ax.set_title(f"Setup {setup} — {label}", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)


def plot_ratio_setup(setup, label, series, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    _ratio_panel(ax, setup, label, series)
    ax.set_title(f"FL Slowdown Relative to Centralized — Setup {setup}\n{label}",
                 fontsize=14)
    ax.annotate(SUBTITLE, xy=(1, -0.16), xycoords="axes fraction", ha="right",
                fontsize=8, color="#555555")
    plt.tight_layout()
    return _save(fig, out_dir, f"exp2_slowdown_ratio_v1_{_slug(setup)}.png")


def plot_ratio_all(data, out_dir):
    rows = [(s, l) for s, l in SETUPS if s in data]
    ncol = 3
    nrow = -(-len(rows) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.9 * nrow),
                             squeeze=False)
    flat = [ax for r in axes for ax in r]
    for ax, (setup, label) in zip(flat, rows):
        _ratio_panel(ax, setup, label, data[setup])
    for ax in flat[len(rows):]:
        ax.set_visible(False)
    fig.suptitle(f"Exp 2: FL Slowdown Relative to Centralized\n{SUBTITLE}",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    return _save(fig, out_dir, "exp2_slowdown_ratio_v1.png")


def _slug(setup):
    return setup.replace("'", "prime")


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/exp2")
    ap.add_argument("--csv", default="results/data/runs_v2.csv")
    a = ap.parse_args()
    data = load(a.csv)
    print("  " + plot_tgrok_all(data, a.out))
    print("  " + plot_ratio_all(data, a.out))
    for setup, label in SETUPS:
        if setup not in data:
            print(f"  setup {setup}: no rows, skipped")
            continue
        print("  " + plot_tgrok_setup(setup, label, data[setup], a.out))
        print("  " + plot_ratio_setup(setup, label, data[setup], a.out))


if __name__ == "__main__":
    main()
