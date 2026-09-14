"""exp2_slowdown_ratio_v1_<setup>.png with memorisation time in place of first crossing.

    venv/bin/python scripts/plotting/exp2_memo_ratio_v1.py              # setup B -> figures/exp2/
    venv/bin/python scripts/plotting/exp2_memo_ratio_v1.py --setup D --out DIR

Same runs, same arms and the same v1 look as exp2_k_axis.plot_ratio_setup; the
statistic is t_memo (first step train accuracy >= 99%) and each alpha is divided
by its own centralized t_memo. A cell whose runs never memorised is drawn as
censored, with the fraction that did.

The y axis is logarithmic: federated memorisation runs from ~3x to ~60x the
centralized value on B, which a linear axis flattens at the low K.

RESOLUTION. t_memo is read off the logged curve, so it is quantised to the
logging interval, and on B the centralized baseline is the coarse end: the
matched cent_full runs log every 200 epochs and record 200 on every seed, while
the same configuration logged every 50 records 150. The denominator is therefore
an upper bound and every ratio a lower bound by up to ~1.33x. The federated arm
evaluates every 20 rounds x E=5 = 100 steps.
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
from exp2_slowdown_ratio import MANIFESTS, SETUPS, _axis_value, _f   # noqa: E402
from exp2_k_axis import _ratio_panel, _save, _slug                  # noqa: E402
from fedgrok.manifest import load_manifest, run_id                  # noqa: E402

SUBTITLE = ("E=5 · FedAvg · IID · median over 3 seeds, bars = seed range · "
            "t_memo = first step train acc ≥ 99%")


def load_memo(setup, csv_path):
    """exp2_slowdown_ratio.load()'s series for one setup, on t_memo."""
    banked = {r["id"]: r for r in csv.DictReader(open(csv_path))}
    rows, seen = [], set()
    for m in MANIFESTS:
        for spec in load_manifest(m):
            rid = run_id(spec)
            if spec.get("setup") != setup or rid in seen or rid not in banked:
                continue
            seen.add(rid)
            rows.append({**banked[rid], "_arm": spec.get("arm")})

    series = []
    for val in sorted({_axis_value(r) for r in rows}, key=float):
        sub = [r for r in rows if _axis_value(r) == val]
        base = [x for x in (_f(r["t_memo"]) for r in sub if r["_arm"] == "cent_full")
                if np.isfinite(x)]
        if not base:
            continue
        cells = []
        for k in sorted({int(r["num_clients"]) for r in sub if r["_arm"] == "fl"}):
            v = [_f(r["t_memo"]) for r in sub
                 if r["_arm"] == "fl" and int(r["num_clients"]) == k]
            fin = [x for x in v if np.isfinite(x)]
            cells.append({"K": k, "n": len(v), "grokked": len(fin),
                          "median": float(np.median(fin)) if fin else None,
                          "lo": min(fin) if fin else None,
                          "hi": max(fin) if fin else None})
        if cells:
            series.append({"val": val, "base": float(np.median(base)),
                           "base_lo": min(base), "base_hi": max(base),
                           "base_n": len(base), "cells": cells})
    return series


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--setup", default="B")
    ap.add_argument("--out", default="figures/exp2")
    ap.add_argument("--csv", default="results/data/runs_v2.csv")
    a = ap.parse_args()

    series = load_memo(a.setup, a.csv)
    if not series:
        sys.exit(f"setup {a.setup}: no banked series")
    label = dict(SETUPS)[a.setup]

    fig, ax = plt.subplots(figsize=(7, 5))
    _ratio_panel(ax, a.setup, label, series)
    # Log scale, with room above the data for the censored row.
    ratios = [c[key] / s["base"] for s in series for c in s["cells"]
              for key in ("lo", "hi") if c[key]]
    ax.set_yscale("log")
    ax.set_ylim(min(ratios + [1.0]) * 0.6, max(ratios) * 4)
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}×"))
    ax.minorticks_off()
    ax.set_ylabel(r"$T^\mathrm{FL}_\mathrm{memo}\;/\;T^\mathrm{cent}_\mathrm{memo}$")
    for text, s in zip(ax.get_legend().get_texts()[1:], series):
        text.set_text(f"{text.get_text()}   (baseline {s['base']:,.0f})")
    ax.set_title(f"FL Memorisation Slowdown Relative to Centralized — Setup {a.setup}\n"
                 f"{label}", fontsize=14)
    ax.annotate(SUBTITLE, xy=(1, -0.16), xycoords="axes fraction", ha="right",
                fontsize=8, color="#555555")
    plt.tight_layout()
    path = _save(fig, a.out, f"exp2_memo_ratio_v1_{_slug(a.setup)}.png")
    print(f"  {path}")
    for s in series:
        print(f"  {'n_train' if a.setup == 'E' else 'α'}={float(s['val']):g} "
              f"baseline {s['base']:g}: " + ", ".join(
                  f"K={c['K']} {c['median'] / s['base']:.1f}× ({c['grokked']}/{c['n']})"
                  if c["median"] else f"K={c['K']} censored (0/{c['n']})"
                  for c in s["cells"]))


if __name__ == "__main__":
    main()
