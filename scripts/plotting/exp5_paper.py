"""exp5 (partial participation), paper version of the two-clocks figure.

    venv/bin/python scripts/plotting/exp5_paper.py        # -> figures/exp5/paper/

Runs are exp5_figures' ladders: FedAvg, K = 20, IID, E = 5, f in {1, 0.5, 0.25},
rounds scaled by 1/f. Setup A's K = 50 ladder (t4b_participation) is left out,
so every line is the same K.

  exp5_two_clocks.png   (a) t_memo and (b) t_grok (sustained: the bar holds for
                        the rest of the run), in COMMUNICATION ROUNDS, relative to
                        the same setup at f = 1. Styled as exp4_paper's two-clocks
                        figure: paper column order and lettering, one log scale.

Rounds, not steps: total_steps accumulates E * f per round, so a step-denominated
time halves with f by construction (RESULTS 21). Values are Kaplan-Meier medians
over runs censored at each run's budget in rounds; bars span the per-run values.
A series whose f = 1 median is censored has no denominator and is left out.
"""
import importlib.util
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e5 = _load("e5", "exp5_figures.py")
p4 = _load("p4", "exp4_paper.py")
pf = e5.pf

OUT = "figures/exp5/paper"
FS = [0.25, 0.5, 1.0]
KEY = {s: s for s in "BCDE"} | {"A": "A20"}   # repo setup -> exp5 ladder key


def panel(ax, lads, field, ylim=(0.4, 2.5)):
    """ylim=None fits the axis to the data, error bars included."""
    span = []
    for i, s in enumerate(p4.ORDER):
        lad, col = lads.get(KEY[s]), pf.COL[s]
        if not lad or 1.0 not in lad:
            continue
        base = e5.in_rounds(lad[1.0], field)[0]
        if not math.isfinite(base):
            print(f"  {p4.PAPER[s]} (repo {s}) omitted from {field}: f = 1 baseline censored")
            continue
        xs, ys, lo, hi, fr = [], [], [], [], []
        for f, runs in sorted(lad.items()):
            m, a, b, frac = e5.in_rounds(runs, field)
            x = f * 1.025 ** (i - 2)           # multiplicative dodge on the log f axis
            if not math.isfinite(m):
                pf.censored_x(ax, x, col, y=0.95, size=6)
                continue
            if field == "t_grok":
                frac = sum(bool(r["grokked"]) for r in runs) / len(runs)
            xs.append(x); ys.append(m / base); lo.append(a / base); hi.append(b / base)
            span += [a / base, b / base, m / base]
            fr.append(frac)
        pf.series(ax, xs, ys, fr, col, err=(lo, hi), size=4.8, lw=1.4)
    ax.axhline(1.0, color=pf.MUTED, lw=0.8, zorder=1)
    ax.set_yscale("log")
    if ylim is None:
        fin = [v for v in span if math.isfinite(v) and v > 0]
        ylim = (min(fin) / 1.06, max(fin) * 1.06)
        ticks = [t for t in (0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2)
                 if ylim[0] < t < ylim[1]]
    else:
        ticks = [t for t in (0.5, 0.75, 1, 1.5, 2) if ylim[0] < t < ylim[1]]
    ax.set_ylim(*ylim)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}×" for t in ticks])
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xscale("log")
    ax.set_xticks(FS)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(0.225, 1.11)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Participation fraction f")


def two_clocks(lads):
    with plt.rc_context(p4.RC):
        # Not sharey: (a) is zoomed to its own spread; (b) keeps the wider range
        # its crosses and the 0.58x point need.
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.36))
        panel(axes[0], lads, "t_memo", ylim=None)
        panel(axes[1], lads, "t_grok")
        axes[0].set_ylabel(r"$t_{\mathrm{memo}}\,/\,t_{\mathrm{memo}}^{f=1}$")
        axes[1].set_ylabel(r"$t_{\mathrm{grok}}\,/\,t_{\mathrm{grok}}^{f=1}$")
        axes[0].set_title("Rounds to Memorise vs Participation")
        axes[1].set_title("Rounds to Generalise vs Participation")
        for ax, L in zip(axes, "ab"):
            pf.letter(ax, L)
        p4.clock_legend(fig)
        os.makedirs(OUT, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT, f"exp5_two_clocks.{ext}"), bbox_inches="tight",
                        facecolor="white", dpi=300)
        plt.close(fig)
        print(f"  wrote {OUT}/exp5_two_clocks.png/.pdf")


def main():
    rows = pf.load_rows()
    lads = e5.ladders(rows)
    for s in p4.ORDER:
        lad = lads.get(KEY[s], {})
        print(f"  {p4.PAPER[s]} (repo {s}): " + "; ".join(
            f"f={f:g} memo {e5.in_rounds(rs, 't_memo')[0]:.0f} "
            f"grok {e5.in_rounds(rs, 't_grok')[0]:.0f} r, "
            f"{sum(bool(r['grokked']) for r in rs)}/{len(rs)} held" for f, rs in sorted(lad.items())))
    two_clocks(lads)


if __name__ == "__main__":
    main()
