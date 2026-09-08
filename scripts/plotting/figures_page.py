"""The paper's figure set as one review page, with captions and the numbers behind them.

    python3 scripts/plotting/figures_page.py --out paper/figures/index.html

Embeds paper/figures/fig*.png as data URIs and reads figure_numbers.json, so the
page is self-contained and every number on it is the one the figure was drawn from.
"""

import argparse
import base64
import json
import os

FIG_DIR = "paper/figures"


def img(name):
    with open(os.path.join(FIG_DIR, name), "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def fmt(x, nd=0):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    return f"{x:,.{nd}f}"


def table(headers, rows, cls="data"):
    h = "".join(
        f"<th{' class=num' if i else ''}>{x}</th>" for i, x in enumerate(headers)
    )
    b = "".join(
        "<tr>"
        + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r))
        + "</tr>"
        for r in rows
    )
    return f'<div class="wrap"><table class="{cls}"><tr>{h}</tr>{b}</table></div>'


def build(numbers):
    n1, n2, n3, n4, n5, n6, n7 = (numbers.get(f"fig{i}", {}) for i in range(1, 8))

    # ── tables from the numbers file ──────────────────────────────────────
    def clock_rows(key):
        d = n1.get(key, {})
        return [
            [
                ("cent" if K == "1" else K),
                fmt(v["memo"]),
                fmt(v["delay"]),
                f"{int(round(v['held'] * v['n']))}/{v['n']}",
            ]
            for K, v in d.items()
        ]

    t1 = (
        table(["A, α=0.30 · K", "t_memo", "delay", "held"], clock_rows("A α=0.3"))
        + table(["A, α=0.25 · K", "t_memo", "delay", "held"], clock_rows("A α=0.25"))
        + table(
            ["B, α=0.30, wd=0.1 · K", "t_memo", "delay", "held"],
            clock_rows("B α=0.3, wd=0.1"),
        )
        + table(
            ["D, α=0.30, wd=1 · K", "t_memo", "delay", "held"],
            clock_rows("D α=0.3, wd=1"),
        )
    )
    rows = []
    for s in "ABCDE":
        d = n2.get(s, {})
        rows.append(
            [s]
            + [
                f"{fmt(d[E]['memo'])} · {fmt(d[E]['memo'] / int(E)) if d[E]['memo'] else '—'}"
                if E in d
                else "—"
                for E in ("1", "5", "10", "25", "50")
            ]
        )
    t2 = table(["setup", "E=1 (steps · rounds)", "E=5", "E=10", "E=25", "E=50"], rows)
    rows = []
    for tag, d in n3.items():
        rows.append(
            [tag]
            + [
                f"{fmt(d[f]['memo_r'])} / {fmt(d[f]['fc_r'])} / {fmt(d[f]['div'], 5)}"
                if f in d
                else "—"
                for f in ("1.0", "0.6", "0.5", "0.4", "0.25", "0.2")
            ]
        )
    t3 = table(["setup", "f=1", "0.6", "0.5", "0.4", "0.25", "0.2"], rows)
    rows = [
        [
            k,
            fmt(v["ratio"], 2) if v["ratio"] else "0/3",
            f"{int(round(v['held'] * 3))}/3",
            fmt(v["iid_fc"]),
        ]
        for k, v in n4.get("structure", {}).items()
    ]
    t4 = table(["cell · partition", "ratio to iid", "held", "iid t_first_cross"], rows)
    rows = []
    for m in ("fedavg", "fedadam", "fedyogi", "scaffold", "fedavgm", "fedprox"):
        rows.append(
            [m]
            + [
                f"{fmt(n5['algorithms'][f'{c} {m}']['fc'])} ({int(round(n5['algorithms'][f'{c} {m}']['held'] * 5))}/5)"
                if f"{c} {m}" in n5.get("algorithms", {})
                else "—"
                for c in ("H1", "H2", "H3")
            ]
        )
    t5 = table(["method", "H1", "H2", "H3"], rows)
    rows = [
        [
            k,
            fmt(v["t_memo"]),
            fmt(v["test@250k"], 1),
            fmt(v["test@2M"], 1),
            f"{v['train_loss@250k']:.3f} → {v['train_loss@2M']:.3f}",
            f"{v['wnorm@250k']:.1f} → {v['wnorm@2M']:.1f}",
            f"{v['drift@250k']:.2f} → {v['drift@2M']:.2f}",
        ]
        for k, v in n6.items()
    ]
    t6 = table(
        [
            "run",
            "t_memo",
            "test @250k",
            "test @2M",
            "train loss",
            "weight norm",
            "drift / round",
        ],
        rows,
    )
    ipr = n7.get("ipr", {})
    rounds = sorted({int(r) for arm in ipr.values() for r in arm})
    rows = [
        [arm] + [fmt(ipr[arm].get(str(r)), 4) for r in rounds]
        for arm in ("iid", "operand")
    ]
    t7 = table(["round"] + [f"{r:,}" for r in rounds], rows)

    figs = [
        (
            "fig1_two_clocks.png",
            "Figure 1 · Survival and the two clocks",
            "§4",
            "Federated, iid, E=5, FedAvg, full participation, each setup at its working point; a second series (dashed) where the corpus has one. Top: Kaplan–Meier median steps to memorise (train ≥ 99%) against client count, with the centralised run at <em>cent</em>. Bottom: median delay from memorisation to first crossing of the bar. Marker fill is the held fraction; a hollow marker near the top edge is a cell where no run reached the event in budget. The square on A is the operand split at K=97. Budgets are pooled where two campaigns share a cell (D at K ≤ 20; B at K=20 runs to 200k). C is drawn and withheld.",
            "aggregation · setup_k_ladder · k_collapse_budget · k_fixed_total · boundary; centralised references from the matching cent_full or decay-band cells.",
            t1,
        ),
        (
            "fig2_local_work.png",
            "Figure 2 · Local work, compute-matched",
            "§5.1",
            "K=10, iid, E ∈ {5, 10, 25, 50}, rounds ∝ 5/E so every rung does the same gradient work; E=5 is the banked K-ladder cell, and A carries E=1 as the FedAvg-identity check. (a) memorisation in steps, (b) the same numbers in aggregation rounds, (c) median delay. On the transformers memorisation scales with E in steps and is flat in rounds; on the quadratic MLPs it is flat in steps. C's delay is ≤ 0 from E=25 and is not drawn on the log axis; D's E=50 cell is the fixed point. E's partial fills are the sustain artefact (every run crosses).",
            "local_epochs, with E=5 controls from aggregation and setup_k_ladder.",
            t2,
        ),
        (
            "fig3_participation.png",
            "Figure 3 · Participation, in rounds",
            "§5.2",
            "K=20, iid, E=5, f ∈ {1, 0.5, 0.25}, rounds ∝ 1/f; the f=1 controls are the banked K=20 cells. Dashed: the anchor's K=50 ladder at α=0.25 (f ∈ {0.2, 0.4, 0.6}, f=1 from the boundary campaign), which is where the divergence rise was first measured. Rounds = steps ÷ (E·f). Memorisation is flat in rounds within 10% on every setup; A's first crossing is flat, B's shortens 1.8× at f=0.25; MNIST degrades at its degenerate K=20. Panel (d) is the mean client weight divergence per round over the rounds before first crossing (a whole-run mean is dominated by the post-grokking tail, whose length is set by the budget): on the anchor at K=50 it more than doubles as f falls from 1 to 0.2 and the first crossing, in rounds, does not move.",
            "participation_setups · participation, with f=1 controls from aggregation, k_collapse_budget, setup_k_ladder and boundary.",
            t3,
        ),
        (
            "fig4_heterogeneity_structure.png",
            "Figure 4 · Heterogeneity and partition structure",
            "§5.3",
            "(a) Fraction of runs that crossed the bar against Dirichlet concentration at K=10, E=5, with the 0.5 rung from the partition campaign; fill is the held fraction. (b) Median peak train accuracy: where a setup fails by not memorising, the failure is training, not generalisation. (c) The starvation control on the anchor at α=0.25: the same Dirichlet shard sizes with the labels randomised, run for run; every run whose smallest client held ≤ 2 samples is censored in both arms. (d) First-crossing time as a ratio to an exactly matched iid baseline; hollow markers at the top are 0/3 cells. Coherent sharding wins on A and C, loses on B and D; target is the worst partition everywhere it ran.",
            "dirichlet_setups · partitions · dirichlet_band · size_control; iid baselines from aggregation, k_fixed_total and setup_k_ladder.",
            t4,
        ),
        (
            "fig5_drift_mitigation.png",
            "Figure 5 · Drift and mitigation",
            "§6",
            "(a) Every FedAvg run on the anchor with a logged history: delay against mean client weight divergence per round before first crossing, coloured by which design axis moved; hollow triangles at the top are censored runs; SCAFFOLD and FedProx from the algorithm comparison are overlaid. Drift tracks the delay on every axis, and the two zero-cost axes (sampling, shard sizes) sit on the same band as the rest. <b>Caveat for §6:</b> on this pre-crossing window SCAFFOLD's divergence is not below FedAvg's (its clients disagree strongly in the few thousand steps before it groks and agree almost perfectly afterwards). The ledger's 188× and 51× reductions are whole-run means dominated by the post-grokking tail; re-derive them on a stated window before the paper quotes them. (b) The algorithm comparison at calibrated server learning rates on three hard anchor cells, five runs each: dots are runs, the bar is the KM median, hollow is censored at budget. (c) The H2 trajectories. FedProx at μ=0.01 is the one method slower than the baseline it is meant to improve.",
            "algorithms · server_lr_cal, and every anchor FedAvg group for the scatter.",
            t5,
        ),
        (
            "fig6_fixed_point.png",
            "Figure 6 · The fixed point",
            "§7",
            "Left: setup D at K=10, E=50, iid, three runs to 2,000,000 steps (the vertical line is the original 250k budget). Right: D at K=10, E=5, Dirichlet 1.0, 250,000 steps. Rows: test accuracy, train loss, total weight norm, mean client drift per round. Grey: centralised D at α=0.30, which groks at ~21,300. The training set is fit, the gradient and the decay are alive, and every quantity is stationary. The controls that decide whether this belongs to aggregation or to the per-round optimiser reset are still to run.",
            "e50_long · dirichlet_setups · central_anchor.",
            t6,
        ),
        (
            "fig7_mechanism.png",
            "Figure 7 · Mechanism",
            "§8",
            "(a) Per-neuron spectral IPR of the global first-operand block, K=97, α=0.25, five runs per arm with medians; the vertical line is the earliest first crossing anywhere in the cell. The arms separate thousands of rounds before any run generalises. (b) Within the iid arm alone, the two runs that later cross against the three that never do (n=5). (c) Setup D at α=0.40, centralised: the compositional circuit T read on its own keeps improving straight through the dip that the full model's test accuracy shows. IPR here is the per-round value logged in each run's history, which reproduces the checkpoint analysis exactly at the final round.",
            "boundary (K=97 histories) · d_internals.",
            t7,
        ),
    ]

    parts = []
    for fname, title, sec, caption, src, tbl in figs:
        parts.append(f"""<section>
  <h2>{title} <span class="sec">{sec}</span></h2>
  <figure><img src="{img(fname)}" alt="{title}"></figure>
  <p class="cap">{caption}</p>
  <p class="src"><b>Source groups</b> {src} · <b>File</b> <span class="mono">paper/figures/{fname}</span> and the PDF beside it.</p>
  <details><summary>The numbers this figure was drawn from</summary>{tbl}</details>
</section>""")
    body = "\n".join(parts)
    return f"""<title>Federated Grokking Paper Figures</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{color-scheme:light;--bg:#EEF0F2;--surface:#FCFCFB;--surface-2:#E6E9ED;--ink:#1B1F24;--ink-2:#4F565E;--ink-3:#7C848C;--rule:#D3D8DE;--accent:#2F4E9C;--accent-ink:#2F4E9C}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#15181C;--surface:#1C2026;--surface-2:#232830;--ink:#E4E7EB;--ink-2:#B4BAC2;--ink-3:#8A9199;--rule:#2E353D;--accent:#8FA8E8;--accent-ink:#A8BCF0}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#15181C;--surface:#1C2026;--surface-2:#232830;--ink:#E4E7EB;--ink-2:#B4BAC2;--ink-3:#8A9199;--rule:#2E353D;--accent:#8FA8E8;--accent-ink:#A8BCF0}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif}}
.page{{max-width:1100px;margin:0 auto;padding:40px 24px 80px}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}}
h1{{font-family:Spectral,Georgia,serif;font-weight:600;font-size:36px;line-height:1.1;margin:8px 0 10px;text-wrap:balance}}
h2{{font-family:Spectral,Georgia,serif;font-weight:600;font-size:22px;margin:0 0 10px;text-wrap:balance}}
h2 .sec{{font-family:"IBM Plex Mono",monospace;font-size:12px;font-weight:500;color:var(--ink-3);margin-left:8px}}
p{{max-width:78ch;margin:0 0 10px}}
.lede{{font-family:Spectral,Georgia,serif;font-size:18px;color:var(--ink-2);max-width:70ch}}
header{{border-bottom:1px solid var(--rule);padding-bottom:18px;margin-bottom:26px}}
section{{background:var(--surface);border:1px solid var(--rule);padding:18px 20px 14px;margin:0 0 22px}}
figure{{margin:0 0 12px;background:#fff;padding:6px;border:1px solid var(--rule)}}
figure img{{display:block;width:100%;height:auto}}
.cap{{max-width:none;color:var(--ink);font-size:14px}}
.src{{font-size:12.5px;color:var(--ink-2);max-width:none}}
.mono{{font-family:"IBM Plex Mono",monospace;font-size:.92em}}
details{{margin-top:6px}} summary{{cursor:pointer;font-size:13px;color:var(--accent-ink);font-weight:500}}
.wrap{{overflow-x:auto;margin:8px 0 12px}}
table{{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums;width:100%}}
th,td{{text-align:left;padding:5px 9px;border-bottom:1px solid var(--rule);vertical-align:top}}
th{{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink-2);white-space:nowrap}}
td.num,th.num{{text-align:right;font-family:"IBM Plex Mono",monospace;white-space:nowrap}}
pre{{background:var(--surface-2);padding:10px 12px;font-family:"IBM Plex Mono",monospace;font-size:12.5px;overflow-x:auto}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:12px 0 0}}
.card{{background:var(--surface);border:1px solid var(--rule);padding:12px 14px;font-size:13.5px}}
.card b{{display:block;margin-bottom:4px}}
a{{color:var(--accent-ink)}}
</style>
<div class="page">
<header>
  <div class="eyebrow">Federated grokking · figure set · regenerated from runs_v2.csv</div>
  <h1>Federated Grokking Paper Figures</h1>
  <p class="lede">The seven main figures of the rewritten paper, drawn from the 1,685-run corpus rather than the v1 logs, with the caption each one needs and the numbers behind it.</p>
  <p>Every figure is produced by one script from the run table and the per-run histories; nothing is typed in. Statistics: Kaplan–Meier medians over runs with right-censoring at each run's budget; delay = t_first_cross − t_memo per run; marker fill = held fraction. Setup C is drawn but its numbers are withheld until aggregation is made deterministic (RESULTS §23). Figure 7 uses the per-round IPR logged in the histories because the checkpoints are not on this machine; it reproduces the checkpoint analysis at the final round.</p>
  <pre>python3 scripts/plotting/paper_figures.py          # -> paper/figures/fig*.png, fig*.pdf, figure_numbers.json
python3 scripts/plotting/figures_page.py --out paper/figures/index.html</pre>
  <div class="grid">
    <div class="card"><b>Where the raw curves live</b>The <a href="https://claude.ai/code/artifact/c10d1879-8d79-46b0-a8fa-12bb823b8f4e">Grokking Run Atlas</a> draws every run in the corpus, one panel per cell, grouped by campaign with each campaign's design and what it answered. The earlier <a href="https://claude.ai/code/artifact/26cb008c-2fd0-4777-bb47-571c249342d6">Paper Axes Atlas</a> covers only the 2–4 September sweep.</div>
    <div class="card"><b>What this page is for</b>Reviewing the figures and captions together before they go into the LaTeX. The paper itself is the document; this page and the atlas are the evidence layer behind it. Section numbers are the rewritten paper's (<a href="https://claude.ai/code/artifact/01e35f35-07a4-47a5-82c7-0b0eb331819e">Writing Brief</a>).</div>
    <div class="card"><b>Not regenerated here</b>Appendix figures (α ladders, decay bands, capacity, server-LR calibration, the budget audit) and Fig. 1's K=97 inset, which the A panel now carries as the square marker. The fixed-point controls and the FedProx μ ladder have not run, so Figs. 5 and 6 carry those caveats in their captions.</div>
  </div>
</header>
{body}
</div>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(FIG_DIR, "index.html"))
    args = ap.parse_args()
    numbers = json.load(open(os.path.join(FIG_DIR, "figure_numbers.json")))
    html = build(numbers)
    with open(args.out, "w") as fh:
        fh.write(html)
    print(f"  wrote {args.out} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
