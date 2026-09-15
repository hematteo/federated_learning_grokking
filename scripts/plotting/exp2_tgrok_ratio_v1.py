"""exp2_slowdown_ratio_v1_<setup>.png, compact, without the footnote.

    venv/bin/python scripts/plotting/exp2_tgrok_ratio_v1.py              # setup B -> figures/exp2/
    venv/bin/python scripts/plotting/exp2_tgrok_ratio_v1.py --setup D --out DIR

Same runs, same numbers and the same v1 look as exp2_k_axis.plot_ratio_setup, on
a smaller canvas. The plotted statistic is still t_first_cross, from
exp2_slowdown_ratio.load(); only the y-axis label reads T_grok, for the paper's
notation. The caption must define T_grok as the first crossing of the bar.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_slowdown_ratio import SETUPS, load            # noqa: E402
from exp2_k_axis import _ratio_panel, _save, _slug      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--setup", default="B")
    ap.add_argument("--out", default="figures/exp2")
    ap.add_argument("--csv", default="results/data/runs_v2.csv")
    ap.add_argument("--name", help="setup letter shown in the title (default: --setup)")
    a = ap.parse_args()

    series = load(a.csv).get(a.setup)
    if not series:
        sys.exit(f"setup {a.setup}: no banked series")
    label = dict(SETUPS)[a.setup]

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    _ratio_panel(ax, a.setup, label, series)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.set_ylabel(r"$T^\mathrm{FL}_\mathrm{grok}\;/\;T^\mathrm{cent}_\mathrm{grok}$")
    ax.set_title(f"FL Slowdown Relative to Centralized\n"
                 f"Setup {a.name or a.setup} · {label}", fontsize=12)
    plt.tight_layout()
    print("  " + _save(fig, a.out, f"exp2_tgrok_ratio_v1_{_slug(a.setup)}.png"))


if __name__ == "__main__":
    main()
