"""exp4 (local epochs), paper versions.

    venv/bin/python scripts/plotting/exp4_paper.py        # -> figures/exp4/paper/

Same runs as exp4_figures (its `ladder`): FedAvg, K = 10, IID, rounds scaled by
5/E so every rung does the same gradient work; A also E = 1, 100, 200.

  exp4_phases.png   one panel per setup, absolute steps on a log axis: median time
                    to memorise (hollow) and to first crossing (filled) against E,
                    the delay between them shaded. A cell no run reached within
                    its budget is an up-triangle on its budget.
  exp4_scaling.png  (a) t_memo and (b) the delay, each relative to E = 5, on true
                    log-log axes with a line proportional to E -- the reference
                    for "memorisation cost grows linearly with local work".
  exp4_curves.png   test (top) and train (bottom) accuracy per run, E on a ramp.
  exp4_two_clocks.png
                    (a) t_memo and (b) t_grok (sustained: the bar holds for the
                    rest of the run) relative to the same setup at E = 5, both
                    panels on one log scale.
  exp4_two_clocks_cent.png
                    (a) t_memo and (b) t_first_cross relative to the CENTRALISED
                    run of the same configuration (same data, model, optimiser, lr,
                    decay, batch), not to E = 5. 1x is centralised; setup A's E = 1
                    rung should sit on it (FedAvg's full-batch GD identity).

Paper conventions, as in the cleaned exp3 figures: no titles, columns in the
paper's order (A, D, B, C, E from the repo) lettered by position, the first
crossing labelled t_grok, and nothing marked "withheld" (the caption says it).
Times are Kaplan-Meier medians over runs censored at each run's budget; bars
span the finite per-run values; delay is the median of per-run
t_first_cross - t_memo over runs where both are finite.
"""
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("e4", os.path.join(_HERE, "exp4_figures.py"))
e4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e4)
pf, ec = e4.pf, e4.ec

ORDER = "ADBCE"                       # repo setups, in the paper's column order
PAPER = dict(zip(ORDER, "ABCDE"))     # repo letter -> paper letter
OUT = "figures/exp4/paper"
TGROK = r"$t_{\mathrm{grok}}$"
TMEMO = r"$t_{\mathrm{memo}}$"
RC = {"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8,
      "ytick.labelsize": 8, "legend.fontsize": 8.5}


def name(s):
    return f"{PAPER[s]}: {pf.NAME[s].split(': ', 1)[1]}"


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), bbox_inches="tight",
                    facecolor="white", dpi=300)
    plt.close(fig)
    print(f"  wrote {OUT}/{stem}.png/.pdf")


def e_axis(ax, es, crowded=False):
    ax.set_xscale("log")
    shown = [e for e in es if e in (1, 5, 25, 100)] if crowded and len(es) > 5 else es
    ax.set_xticks(shown)
    ax.set_xticks([e for e in es if e not in shown], minor=True)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(min(es) / 1.5, max(es) * 1.5)
    ax.grid(axis="x", visible=False)


# ── 1. the two phases, absolute ──────────────────────────────────────────────

def phases(lads):
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(ORDER), figsize=(9.0, 2.7),
                                 gridspec_kw=dict(wspace=0.48))
        for ax, s in zip(axes, ORDER):
            lad, col = lads[s], pf.COL[s]
            es = sorted(lad)
            memo, fc = [], []
            for E in es:
                st = pf.stats(lad[E])
                memo.append((E, st["memo"], st["memo_lo"], st["memo_hi"], st["budget"]))
                fc.append((E, st["fc"], st["fc_lo"], st["fc_hi"], st["budget"]))
            # Delay band between the two medians wherever both exist.
            both = [(m[0], m[1], f[1]) for m, f in zip(memo, fc)
                    if pf._finite(m[1]) and pf._finite(f[1])]
            if len(both) > 1:
                ax.fill_between([b[0] for b in both], [b[1] for b in both],
                                [b[2] for b in both], color=col, alpha=0.12, lw=0, zorder=1)
            for pts, filled in ((memo, False), (fc, True)):
                fin = [p for p in pts if pf._finite(p[1])]
                ax.plot([p[0] for p in fin], [p[1] for p in fin], color=col, lw=1.3,
                        ls="-" if filled else "--", zorder=2)
                for E, y, lo, hi, _ in fin:
                    if pf._finite(lo) and pf._finite(hi):
                        ax.errorbar([E], [y], yerr=[[y - min(lo, y)], [max(hi, y) - y]],
                                    color=col, capsize=2, elinewidth=0.8, capthick=0.8,
                                    ls="none", zorder=3)
                    ax.plot([E], [y], "o", ms=4.8, mec=col, mew=1.2,
                            mfc=col if filled else "white", zorder=4)
                for E, y, _, _, budget in pts:
                    if not pf._finite(y):
                        ax.plot([E], [budget], "^", ms=5.5, mec=col, mew=1.2,
                                mfc=col if filled else "white", zorder=4)
            e_axis(ax, es, crowded=True)
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(pf._human))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            letter, desc = name(s).split(": ", 1)
            ax.set_title(f"{letter}\n{desc}")
            ax.set_xlabel("Local epochs E")
        axes[0].set_ylabel("Steps")
        for ax in axes:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo / 1.3, hi * 1.3)
        handles = [
            Line2D([], [], color=pf.INK2, ls="--", marker="o", mfc="white", ms=4.8,
                   label=f"Memorise ({TMEMO})"),
            Line2D([], [], color=pf.INK2, marker="o", ms=4.8, label=f"Generalise ({TGROK})"),
            Line2D([], [], color=pf.INK2, marker="^", ls="none", mfc="white", ms=5.5,
                   label="Not reached within budget (drawn at budget)"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))
        save(fig, "exp4_phases")


# ── 2. how each phase scales with E ──────────────────────────────────────────

# Config fields a centralised run must share with the federated ladder to count as
# its equivalent. Everything that changes what one gradient step computes.
MATCH = ["dataset", "model", "loss", "optimizer", "lr", "weight_decay", "momentum",
         "hidden_width", "n_layers", "d_mlp", "n_heads", "init_scale", "activation",
         "alpha", "n_train", "batch_size", "p"]


def cent_equivalent(rows, lad):
    """Centralised runs matching the ladder's configuration.

    Times are read off the logged curve, so they are quantised to the logging
    interval -- and the longest-budget control is usually the coarsest (setup D's
    250k-epoch arm logs every 500 and records t_memo = 500 where the same config
    logged every 25 records 250). So: the finest-logged matching cell among those
    in which every run crossed within budget, longest budget as the tie-break.
    """
    r0 = lad[5][0]
    cand = [r for r in rows
            if r["mode"] == "centralized" and r["arm"] != "cent_reduced"
            and r["group"] != "d_internals" and all(r.get(k) == r0.get(k) for k in MATCH)]
    cells = [c for c in pf.by_cell(cand, "epochs", "log_every").values()
             if all(pf._finite(r["t_first_cross"]) for r in c)]
    if not cells:
        return pf.best_control(cand)
    return min(cells, key=lambda c: (float(c[0]["log_every"]), -c[0]["budget"]))


def stats_ext(runs):
    """pf.stats plus the sustained grokking time t_grok (the bar holds for the rest
    of the run), KM-censored at each run's budget like the other clocks."""
    st = pf.stats(runs)
    st["grok"] = pf.km([r["t_grok"] for r in runs], [r["budget"] for r in runs])
    fin = [r["t_grok"] for r in runs if pf._finite(r["t_grok"])]
    st["grok_lo"], st["grok_hi"] = (min(fin), max(fin)) if fin else (float("inf"),) * 2
    return st


def scaling_panel(ax, lads, key, ylim=(0.3, 40), bases=None, guide=True):
    es_all = set()
    for i, s in enumerate(ORDER):
        lad, col = lads[s], pf.COL[s]
        if 5 not in lad:
            continue
        base = stats_ext(bases[s] if bases else lad[5])[key]
        if not pf._finite(base):
            # No finite denominator (e.g. E's E = 5 cell, where 2 of 3 runs dip
            # back under the bar, so its t_grok median is censored). A cross would
            # claim no run reached it at every E, which is false: leave it out.
            print(f"  {PAPER[s]} (repo {s}) omitted from {key}: E = 5 baseline censored")
            continue
        xs, ys, lo, hi, fr, cens = [], [], [], [], [], []
        for E, runs in sorted(lad.items()):
            st = stats_ext(runs)
            es_all.add(E)
            # Small multiplicative dodge so overlapping setups stay readable.
            x = E * 1.045 ** (i - 2)
            y = st[key]
            if key == "delay" and not (pf._finite(st["memo"]) and pf._finite(st["fc"])):
                # A delay needs both clocks; if either median is censored the cell
                # is censored here too, whatever the few runs that finished say.
                y = float("inf")
            if not (pf._finite(y) and pf._finite(base)):
                cens.append(x)
                continue
            if y / base <= ylim[0]:
                # Delay collapsed to (near) zero -- first crossing at about the same
                # step as memorisation: below the log axis, so a marker on the floor.
                ax.plot([x], [0.05], "v", ms=6, color=col, mec=col,
                        transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
                continue
            xs.append(x); ys.append(y / base)
            lo.append(st[key + "_lo"] / base); hi.append(st[key + "_hi"] / base)
            fr.append({"memo": sum(pf._finite(r["t_memo"]) for r in runs) / len(runs),
                       "grok": st["held"]}.get(key, st["crossed"]))
        pf.series(ax, xs, ys, fr, col, err=(lo, hi), size=4.8, lw=1.4)
        for x in cens:
            pf.censored_x(ax, x, col, y=0.95, size=6)
    es = sorted(es_all)
    e_axis(ax, es)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200)
             if ylim[0] < t < ylim[1]]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}×" for t in ticks])
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.axhline(1.0, color=pf.MUTED, ls="-", lw=0.8, zorder=1)
    ax.set_xlabel("Local epochs E")
    if not guide:
        return
    # Reference: cost proportional to E, anchored at E = 5.
    ref = [e for e in es if e >= 5]
    ax.plot(ref, [e / 5 for e in ref], color=pf.MUTED, ls=":", lw=1.2, zorder=1)
    ax.text(14, 14 / 5 * 1.25, r"$\propto E$", color=pf.MUTED, fontsize=8.5,
            ha="right", va="bottom")
    ax.set_xlabel("Local epochs E")


def scaling(lads):
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.28))
        scaling_panel(axes[0], lads, "memo")
        scaling_panel(axes[1], lads, "delay", ylim=(0.15, 40))
        axes[0].set_ylabel(r"$t_{\mathrm{memo}}\,/\,t^{E=5}_{\mathrm{memo}}$")
        axes[1].set_ylabel(r"Delay$\,/\,$Delay$^{E=5}$")
        for ax, L in zip(axes, "ab"):
            pf.letter(ax, L)
        setups = [Line2D([], [], color=pf.COL[s], marker="o", lw=1.4, ms=4.8, label=name(s))
                  for s in ORDER]
        blank = Line2D([], [], ls="none", label=" ")
        keys = [
            Line2D([], [], color=pf.INK2, marker="o", ls="none", ms=4.8, label="All runs"),
            Line2D([], [], color=pf.INK2, marker="o", ls="none", ms=4.8, mfc="white",
                   mew=1.2, label="Some runs"),
            Line2D([], [], color=pf.INK2, marker="x", ls="none", ms=6, mew=1.5,
                   label="No run within budget"),
            Line2D([], [], color=pf.INK2, marker="v", ls="none", ms=6,
                   label="Delay ≈ 0 (below axis)"),
            Line2D([], [], color=pf.MUTED, ls=":", lw=1.2, label=r"$\propto E$"),
        ]
        # Column-major fill: setups in the first two columns, the key in the last two.
        handles = setups + [blank] + keys + [blank]
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, -0.01), columnspacing=1.2)
        save(fig, "exp4_scaling")


# ── 3. curves ────────────────────────────────────────────────────────────────

def curves(lads):
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, len(ORDER), figsize=(9.0, 4.4), sharex="col",
                                 sharey=True, gridspec_kw=dict(hspace=0.12, wspace=0.12))
        present = set()
        for j, s in enumerate(ORDER):
            lad = lads[s]
            present |= set(lad)
            groups = [(e4.E_COL[E], lad[E], 3 + k, 1.0) for k, E in enumerate(sorted(lad))]
            ec.draw_groups(axes[0, j], axes[1, j], groups)
            for ax in axes[:, j]:
                for t in list(ax.texts):          # margin labels from draw_groups
                    t.remove()
                for ln in list(ax.lines):         # drop the never-crossed crosses
                    if ln.get_marker() == "x":
                        ln.remove()
                ax.xaxis.set_major_locator(mticker.LogLocator(numticks=10))
                ax.xaxis.set_minor_locator(mticker.NullLocator())
                ax.tick_params(labelsize=8, colors="black")
            letter, desc = name(s).split(": ", 1)
            axes[0, j].set_title(f"{letter}\n{desc}")
            axes[1, j].set_xlabel("Steps")
        axes[0, 0].set_ylabel("Test accuracy (%)")
        axes[1, 0].set_ylabel("Train accuracy (%)")
        handles = [Line2D([], [], color=e4.E_COL[E], lw=2, label=f"E = {E}")
                   for E in e4.ES if E in present]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                   bbox_to_anchor=(0.5, 0.0), handlelength=1.5, columnspacing=1.2)
        save(fig, "exp4_curves")


def clock_legend(fig, y=-0.01):
    """Setups in two columns, the marker key in a third."""
    setups = [Line2D([], [], color=pf.COL[s], marker="o", lw=1.4, ms=4.8, label=name(s))
              for s in ORDER]
    blank = Line2D([], [], ls="none", label=" ")
    keys = [
        Line2D([], [], color=pf.INK2, marker="o", ls="none", ms=4.8, label="All runs reached it"),
        Line2D([], [], color=pf.INK2, marker="o", ls="none", ms=4.8, mfc="white", mew=1.2,
               label="Some runs reached it"),
        # A cross is a censored MEDIAN: fewer than half the runs reached it. It can
        # still hide one run that did (D at E = 50 holds 1 of 3).
        Line2D([], [], color=pf.INK2, marker="x", ls="none", ms=6, mew=1.5,
               label="Median not reached in budget"),
    ]
    fig.legend(handles=setups + [blank] + keys, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, y), columnspacing=1.5, handlelength=1.8)


def two_clocks(lads):
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True,
                                 gridspec_kw=dict(wspace=0.36))
        scaling_panel(axes[0], lads, "memo", ylim=(0.3, 40), guide=False)
        scaling_panel(axes[1], lads, "grok", ylim=(0.3, 40), guide=False)
        axes[0].set_ylabel(r"$t_{\mathrm{memo}}\,/\,t_{\mathrm{memo}}^{E=5}$")
        axes[1].set_ylabel(r"$t_{\mathrm{grok}}\,/\,t_{\mathrm{grok}}^{E=5}$")
        axes[1].tick_params(labelleft=True)
        axes[0].set_title("Time to Memorise vs Local Epochs")
        axes[1].set_title("Time to Generalise vs Local Epochs")
        for ax, L in zip(axes, "ab"):
            pf.letter(ax, L)
        clock_legend(fig)
        save(fig, "exp4_two_clocks")


def two_clocks_cent(rows, lads):
    bases = {s: cent_equivalent(rows, lads[s]) for s in ORDER}
    for s in ORDER:
        st = pf.stats(bases[s])
        print(f"  centralised for {PAPER[s]} (repo {s}): {len(bases[s])} runs, group "
              f"{sorted({r['group'] for r in bases[s]})}, memo {st['memo']:g}, fc {st['fc']:g}")
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.28))
        scaling_panel(axes[0], lads, "memo", ylim=(0.3, 300), bases=bases, guide=False)
        scaling_panel(axes[1], lads, "fc", ylim=(0.3, 300), bases=bases, guide=False)
        axes[0].set_ylabel(r"$t^{\mathrm{FL}}_{\mathrm{memo}}\,/\,t^{\mathrm{cent}}_{\mathrm{memo}}$")
        axes[1].set_ylabel(r"$t^{\mathrm{FL}}_{\mathrm{grok}}\,/\,t^{\mathrm{cent}}_{\mathrm{grok}}$")
        for ax, L in zip(axes, "ab"):
            pf.letter(ax, L)
        setups = [Line2D([], [], color=pf.COL[s], marker="o", lw=1.4, ms=4.8, label=name(s))
                  for s in ORDER]
        blank = Line2D([], [], ls="none", label=" ")
        keys = [
            Line2D([], [], color=pf.INK2, marker="o", ls="none", ms=4.8, label="All runs"),
            Line2D([], [], color=pf.INK2, marker="o", ls="none", ms=4.8, mfc="white",
                   mew=1.2, label="Some runs"),
            Line2D([], [], color=pf.INK2, marker="x", ls="none", ms=6, mew=1.5,
                   label="No run within budget"),
        ]
        fig.legend(handles=setups + [blank] + keys, loc="upper center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, -0.01), columnspacing=1.2)
        save(fig, "exp4_two_clocks_cent")


def main():
    rows = pf.load_rows()
    lads = {s: e4.ladder(rows, s) for s in ORDER}
    for s in ORDER:
        print(f"  {PAPER[s]} (repo {s}): " + ", ".join(
            f"E={E} memo {pf.stats(rs)['memo']:.0f} fc {pf.stats(rs)['fc']:.0f} "
            f"delay {pf.stats(rs)['delay']:.0f}" for E, rs in lads[s].items()))
    phases(lads)
    scaling(lads)
    curves(lads)
    two_clocks(lads)
    two_clocks_cent(rows, lads)


if __name__ == "__main__":
    main()
