"""Every run in the corpus, in the Paper Axes Atlas page.

The Atlas published for the 2-4 September sweep covered its four manifests. This
builds the same page over the whole results table: one panel per cell (a config
minus its seed) with every run drawn, grouped by campaign, each campaign carrying
a one-paragraph description of its design and what it answered.

    python3 scripts/plotting/run_atlas.py --out paper/run_atlas.html
"""

import argparse
import collections
import csv
import glob
import json
import os
import re

TEMPLATE = os.path.join(os.path.dirname(__file__), "templates", "run_atlas.html")

# Narrative order: what each setup does alone, what federation does to it, the
# axes the paper reports, then the diagnostics. (group, label, description)
GROUPS = [
    (
        "central_anchor",
        "Centralized α ladders",
        "Gate A. Centralized α ladders for A–D (α 0.15–0.5, 5 runs) and MNIST's n_train ladder for E. Finds each setup's data cliff and its working point. RESULTS §11.",
    ),
    (
        "aprime_alpha",
        "A′ α ladder",
        "A's architecture and data under AdamW instead of GD. The optimiser moves the cliff below 0.20 and rescales the clock ~45×. §13.3.",
    ),
    (
        "c_alpha_w256",
        "C at width 256",
        "Setup C below α=0.5 at the width the capacity sweep selected, 100k epochs. C's usable working point is α=0.40; its cliff ≈0.30. §14.1.",
    ),
    ("c_alpha", "C at high α", "Setup C centralized at α ≥ 0.5 with 100k epochs."),
    (
        "c_capacity",
        "C capacity",
        "Setup C at width 128 vs 256: its Gate-A failure was censoring, not capacity. §11.",
    ),
    (
        "capacity",
        "Capacity",
        "Half / default / double width per setup at its working α. A′ fails at double width; B's width is not binding; D needs ≥256. §14.2.",
    ),
    (
        "d_alpha_fine",
        "D α ladder, fine",
        "Setup D centralized at 0.025 α resolution (tier X).",
    ),
    (
        "d_alpha_high",
        "D α ladder, high",
        "Setup D centralized above α=0.6 (tier X). The five α=1.0 rows have no held-out set and must be excluded from aggregates. §10.",
    ),
    (
        "d_alpha_cliff",
        "D α ladder, cliff",
        "Setup D centralized below its cliff: 0/20.",
    ),
    (
        "d_internals",
        "D internals",
        "Setup D centralized, 17 α rungs × 5 runs, with the compositional-circuit and irrep probes. Why α acts through the phase-1 plateau; the mid-training dip is masking, not decay. §16.3.",
    ),
    (
        "d_wd_ladder",
        "D weight-decay ladder",
        "Setup D centralized, lr·λ from 0 to 3e-3. wd=1.0 is the only band that groks every run. §17.3.",
    ),
    (
        "d_lr_control",
        "D learning-rate control",
        "Setup D at lr 2.5e-4 to 4e-3: lower is faster. §17.3.",
    ),
    (
        "d_gd_probe",
        "D under plain GD",
        "The quadratic MLP groks S₅ under GD at lr=50, wd=0 (3/3 at 22,600). The D′ setup exists. §13.1.",
    ),
    (
        "cd_decay_band",
        "C and D decay bands",
        "C's and D's weight-decay bands on the same lr·λ ladder as B. Neither inherited wd=1.0 moves, centrally. §13.2.",
    ),
    (
        "b_decay_band",
        "B decay band",
        "B at α=0.30, five decays. Every cell memorises at epoch 150; decay acts only on the delay, and t_first_cross ≈ 4,500/wd. §13.5.",
    ),
    (
        "b_wd01_alpha_ladder",
        "B α ladder at wd=0.1",
        "B's α ladder at the low decay adopted for federated transformer work on 2026-08-20.",
    ),
    (
        "b_wd_zero_alpha",
        "B at wd=0, α ladder",
        "Does data substitute for decay? At wd=0 B groks from α=0.60, with a ~29k-step delay intact.",
    ),
    (
        "b_wd_zero_a04_long",
        "B at wd=0, 1M steps",
        "B at wd=0, α=0.40, 1,000,000 steps: fp32 train loss underflows to zero and nothing moves. A genuine negative that is not the fixed point.",
    ),
    (
        "wd_zero",
        "wd=0 controls",
        "B, C and E centralized without decay: memorise by epoch 400, then sit at chance (E reaches 74–81%). §15.4.",
    ),
    (
        "wd_grid",
        "Anchor weight-decay grid",
        "The corrected sweep under GD at α=0.5. Decay is neutral-to-slowing and stops training above lr·λ=1e-3. §6.2.",
    ),
    (
        "mnist_wd_band",
        "MNIST decay band",
        "The Omnigrok band: decay accelerates grokking monotonically on MNIST. §6.3.",
    ),
    (
        "mnist_working_point",
        "MNIST working point",
        "Delay against shardability: every config with a long delay needs a large batch. §11.",
    ),
    (
        "poly_pilot",
        "Polynomial task gate",
        "x²+y² kept, x²+xy+y² excluded, at α=0.5. §7.",
    ),
    (
        "probe",
        "T1 probe",
        "The anchor at K=10, E ∈ {1, 5, 50}, iid vs operand. Local steps cost more than clients; the E=1 cells are budget-censored. §3.",
    ),
    (
        "k_fixed_total",
        "K breakdown, α=0.30",
        "The anchor's α=0.30 control: K ∈ {5,10,20,50} × {iid, operand, dirichlet(0.5)}, 5 runs, 10k rounds. Every cell 5/5; first sighting of the structure effect at K=50. §4.",
    ),
    (
        "boundary",
        "Boundary, α=0.25",
        "The anchor at α=0.25, E=5, 100k steps: K=20 and 50 iid, K=97 iid vs operand. v1's K=97 breakdown was a 50k clock; structure rescues K=97 (operand 5/5, iid 2/5). §5.",
    ),
    (
        "aggregation",
        "exp2 · aggregation vs fragmentation",
        "Does aggregation compensate for fragmenting the data? Per setup: cent_full (all data), cent_reduced (one client's 1/K shard) and fl (FedAvg, K ∈ {2,5,10,20,50}, E=5, iid). On the anchor 50 clients cost 17%; AdamW setups degrade with K and fail at K=50. §15.1.",
    ),
    (
        "aggregation_alpha2",
        "exp2 · second α",
        "A second, easier α per setup for the slowdown-ratio panels (B α=0.40, E n_train=4000). Partial.",
    ),
    (
        "setup_k_ladder",
        "Federated K ladder, B–E",
        "iid, E=5 K ladders on B (wd 0.1), C, D and E. D reproduces B's memorisation collapse at K=50 on a different architecture and task: the decay clock. §14.3.",
    ),
    (
        "k_collapse_wd",
        "K collapse · decay",
        "B at K ∈ {20,30,50}, wd 0.1 vs 1.0, 10k steps. Decay is the knob that moves the failure; these cells are clock-censored. §13.6.",
    ),
    (
        "k_collapse_budget",
        "K collapse · budget",
        "The same cells at budgets keyed to the measured centralized requirement. wd=0.1 groks 3/3 at K=20: there is no K≈30 collapse, only memorisation slowing with K. §13.7.",
    ),
    (
        "k50_hparam",
        "K=50 diagnosis · lr",
        "B at K=50: local step size. Lower lr is worse. §12.",
    ),
    ("k50_ladder", "K=50 diagnosis · ladder", "B at K=50: where training stops. §12."),
    (
        "b_k20_wd_control",
        "B K=20 decay control",
        "B at K=20, α=0.40, wd 1.0 vs 0.1, 10k steps: wd=1.0 blocks memorisation under federation. The cell behind the 2026-08-20 decay decision.",
    ),
    (
        "adam_restart",
        "Optimiser-state persistence",
        "B at K=10, E=5 and E=50, persist_local_opt_state False vs True. Persisting Adam state does not recover the ceiling. §15.3.",
    ),
    (
        "e1_identity",
        "A′ at E=1",
        "A′ at E=1 does not reproduce the ceiling: the FedAvg identity covers stateless optimisers only. §15.3.",
    ),
    (
        "partitions",
        "exp3b · partition structure",
        "iid vs operand vs target vs Dirichlet(0.5) per setup at two K, plus coset on S₅. Coherent sharding wins on A and C, loses on B and D; target is the worst partition everywhere. §17.2.",
    ),
    (
        "local_epochs",
        "Paper axis · local epochs",
        "K=10, iid, E ∈ {5,10,25,50} on every setup at matched compute (rounds ∝ 5/E), plus E=1 on A as the FedAvg-identity check. Memorisation scales with E on transformers, not on quadratic MLPs. §19.",
    ),
    (
        "participation_setups",
        "Paper axis · participation",
        "K=20, iid, E=5, f ∈ {0.5, 0.25} on every setup, rounds ∝ 1/f; the f=1 controls are the aggregation K=20 cells. Memorisation is flat in rounds on every setup; MNIST degrades at its degenerate K. §21.",
    ),
    (
        "participation",
        "exp4b · anchor",
        "The anchor at K=50, α=0.25, f ∈ {0.2,0.4,0.6}, rounds set to a common 100k steps. Flat in rounds; divergence up 1.8× at zero cost. §18.3.",
    ),
    (
        "dirichlet_setups",
        "Paper axis · heterogeneity",
        "Dirichlet concentration 0.01–1000 at K=10, E=5 on every setup (the 0.5 rung is in partitions). Shard starvation ruled out by construction on the algebraic setups. A and E immune; B and C stop memorising; D memorises and freezes. §20.",
    ),
    (
        "dirichlet_band",
        "exp3a · anchor",
        "The anchor at α=0.25, K=20/50, dir_α 0.01–1000. The 0.01 tail was client starvation. §18.1.",
    ),
    (
        "size_control",
        "Starvation control",
        "dirichlet_sizes: the Dirichlet shard sizes kept, the labels randomised. Every run whose smallest client held ≤2 samples dies in both arms. §18.2.",
    ),
    (
        "server_lr_cal",
        "Server-LR calibration",
        "FedAdam, FedYogi and FedAvgM on one heterogeneous anchor cell. Both adaptive methods cliff above server_lr=0.1; FedAvgM needs momentum. §14.5.",
    ),
    (
        "algorithms",
        "exp5 · aggregation rules",
        "Every method at its own calibrated server LR on three hard anchor cells (H1 α=0.25 K=10 E=25 iid; H2 same, Dirichlet 0.1; H3 α=0.30 E=50 Dirichlet 0.1), 5 runs. Adaptive server optimisers 10–20× faster than FedAvg; FedProx μ=0.01 worse than FedAvg. §17.1.",
    ),
    (
        "e50_long",
        "E=50 at 1M and 2M steps",
        "C and D at E=50 re-run at 8× budget. D's memorise-never-generalise fixed point; C's run-to-run nondeterminism. §22–23.",
    ),
    (
        "fl_probe",
        "FL probe",
        "First federated run of each new setup at K=10. Under-budgeted by construction; superseded.",
    ),
    ("probe_rerun", "FL probe re-run", "Probe cells re-run at 5× budget."),
    (
        "grok_confirm_fl",
        "FL confirmation",
        "One federated confirmation run per new setup.",
    ),
    (
        "mnist_fl",
        "Federated MNIST",
        "Federated MNIST probes for setup E's working point.",
    ),
]

CELL_KEYS = [
    "arm",
    "task",
    "p",
    "group_n",
    "alpha",
    "n_train",
    "batch_size",
    "hidden_width",
    "n_layers",
    "n_heads",
    "d_mlp",
    "init_scale",
    "num_clients",
    "local_epochs",
    "num_rounds",
    "fraction_train",
    "partition",
    "dirichlet_alpha",
    "strategy",
    "server_lr",
    "server_momentum",
    "proximal_mu",
    "lr",
    "weight_decay",
    "epochs",
    "persist_local_opt_state",
]
SHORT = {
    "num_clients": "K",
    "local_epochs": "E",
    "alpha": "α",
    "n_train": "n",
    "batch_size": "bs",
    "hidden_width": "d",
    "n_heads": "h",
    "d_mlp": "mlp",
    "weight_decay": "wd",
    "num_rounds": "R",
    "epochs": "ep",
    "dirichlet_alpha": "dir",
    "persist_local_opt_state": "persist",
    "fraction_train": "f",
    "server_lr": "slr",
    "server_momentum": "smom",
    "proximal_mu": "μ",
    "init_scale": "init",
    "n_layers": "L",
    "group_n": "S",
    "p": "p",
    "lr": "lr",
}
BARE = {"partition", "task", "strategy", "arm"}
RUNG_KEYS = [
    "num_clients",
    "local_epochs",
    "fraction_train",
    "dirichlet_alpha",
    "alpha",
    "n_train",
    "weight_decay",
    "server_lr",
    "proximal_mu",
    "hidden_width",
    "lr",
    "num_rounds",
    "epochs",
]


def infer_setup(row):
    if row.get("setup"):
        return row["setup"]
    key = (row.get("dataset"), row.get("model"))
    return {
        ("modular", "groknet"): "A",
        ("modular", "transformer"): "B",
        ("s5", "transformer"): "C",
        ("s5", "groknet"): "D",
        ("mnist", "mlp"): "E",
    }.get(key, "?")


def _num(x):
    """A JSON-safe number: blanks, inf and NaN all become null (JSON has no NaN)."""
    if x in ("", None) or x == "inf":
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    return v if v == v and abs(v) != float("inf") else None


def _label(key, varying):
    bits = []
    for i in varying:
        name, value = CELL_KEYS[i], key[i]
        if value == "":
            continue
        bits.append(value if name in BARE else f"{SHORT.get(name, name)}={value}")
    return "  ".join(bits) or "single cell"


def _rung(key, varying):
    for name in RUNG_KEYS:
        i = CELL_KEYS.index(name)
        if i in varying and key[i] not in ("",):
            try:
                return float(key[i])
            except ValueError:
                pass
    return 0.0


def _thin(xs, ys, n):
    if len(xs) <= n:
        return xs, ys
    idx = sorted(
        {0, len(xs) - 1} | {round(i * (len(xs) - 1) / (n - 1)) for i in range(n)}
    )
    return [xs[i] for i in idx], [ys[i] for i in idx]


def _history(run_id, hist_root, max_points):
    hits = glob.glob(os.path.join(hist_root, run_id, "history_*.json"))
    if not hits:
        return None
    with open(hits[0]) as fh:
        h = json.load(fh)
    steps = h.get("total_steps") or h.get("epoch") or []
    test, train = h.get("test_acc") or [], h.get("train_acc") or []
    n = min(len(steps), len(test), len(train))
    if n < 2:
        return None
    steps, test, train = steps[:n], test[:n], train[:n]
    s1, te = _thin(steps, test, max_points)
    _, tr = _thin(steps, train, max_points)
    return {
        "steps": [float(x) for x in s1],
        "test": [round(float(v), 1) if v == v else 0.0 for v in te],
        "train": [round(float(v), 1) if v == v else 0.0 for v in tr],
    }


def build(csv_path, hist_root, max_points):
    rows = list(csv.DictReader(open(csv_path)))
    by_group = collections.defaultdict(list)
    for r in rows:
        r["setup"] = infer_setup(r)
        by_group[r["group"]].append(r)

    order = [g for g, _, _ in GROUPS] + sorted(
        set(by_group) - {g for g, _, _ in GROUPS}
    )
    info = {g: (label, desc) for g, label, desc in GROUPS}
    manifests, runs, missing = [], [], 0
    for g in order:
        grs = by_group.get(g)
        if not grs:
            continue
        cells = collections.defaultdict(list)
        for r in grs:
            cells[tuple(r.get(k, "") or "" for k in CELL_KEYS)].append(r)
        varying_for = {}
        for setup in {r["setup"] for r in grs}:
            mine = [k for k, rs in cells.items() if rs[0]["setup"] == setup]
            varying_for[setup] = [
                i for i in range(len(CELL_KEYS)) if len({k[i] for k in mine if k[i] != ""}) > 1
            ]
        axis_names = sorted(
            {
                SHORT.get(CELL_KEYS[i], CELL_KEYS[i])
                for v in varying_for.values()
                for i in v
                if CELL_KEYS[i] not in ("num_rounds", "epochs")
            }
        )
        n_here = 0
        for key, rs in cells.items():
            varying = varying_for[rs[0]["setup"]]
            label, rung = _label(key, varying), _rung(key, varying)
            for r in rs:
                h = _history(r["id"], hist_root, max_points)
                if h is None:
                    missing += 1
                    continue
                n_here += 1
                runs.append(
                    {
                        "id": r["id"],
                        "man": g,
                        "setup": r["setup"],
                        "cell": label,
                        "rung": rung,
                        "seed": int(float(r.get("seed") or 0)),
                        "banked": False,
                        "held": r.get("grokked") in ("True", "true", "1"),
                        "t_memo": _num(r.get("t_memo")),
                        "t_fc": _num(r.get("t_first_cross")),
                        "peak": _num(r.get("peak_train_acc")) or 0.0,
                        "final": _num(r.get("final_acc")) or 0.0,
                        "thr": _num(r.get("grok_threshold")) or 95.0,
                        "budget": _num(r.get("steps_run")) or h["steps"][-1],
                        "wall": (_num(r.get("wall_s")) or 0.0) / 3600.0,
                        **h,
                    }
                )
        if n_here:
            label, desc = info.get(g, (g, ""))
            manifests.append(
                {
                    "key": g,
                    "label": label,
                    "axis": " · ".join(axis_names) or "—",
                    "desc": desc,
                    "n": n_here,
                }
            )
    return {"manifests": manifests, "runs": runs}, missing


def render(data, template_path):
    tpl = open(template_path).read()
    n_runs, n_cells = (
        len(data["runs"]),
        len({(r["man"], r["setup"], r["cell"]) for r in data["runs"]}),
    )
    n_groups = len(data["manifests"])
    header = f"""<header class="wrap">
  <div class="eyebrow">Federated grokking · every run in the corpus</div>
  <h1 style="margin-top:8px">Grokking Run Atlas</h1>
  <p>Test and train accuracy against gradient steps for all {n_runs:,} v2 runs, in {n_cells:,} cells across {n_groups} campaigns, every run drawn. Each campaign carries its design, what it answered and where the ledger discusses it. Click any panel for the large view with run ids. Companions: <a href="https://claude.ai/code/artifact/ba51c6aa-f43f-401b-b225-968924c762fe">Review Packet</a> · <a href="https://claude.ai/code/artifact/8caa5453-35dd-4bdb-87d7-30c3a830692f">Paper Axes Sweep</a> · <a href="https://claude.ai/code/artifact/01e35f35-07a4-47a5-82c7-0b0eb331819e">Writing Brief</a>.</p>
</header>"""
    tpl, n = re.subn(r'<header class="wrap">.*?</header>', header, tpl, flags=re.DOTALL)
    assert n == 1, "header not found"
    tpl = tpl.replace(
        "<title>Paper Axes Atlas</title>", "<title>Grokking Run Atlas</title>"
    )
    tpl = tpl.replace(
        '<option value="">all four</option>', '<option value="">all campaigns</option>'
    )
    tpl = tpl.replace(
        '<label class="chk"><input type="checkbox" id="f-banked" checked> include banked controls</label>',
        '<label class="chk" hidden><input type="checkbox" id="f-banked" checked> include banked controls</label>',
    )
    tpl = tpl.replace(
        "downsampled to ≤200 points", f"downsampled to ≤{MAX_POINTS} points"
    )
    tpl = tpl.replace(
        "</style>",
        ".group .desc{color:var(--ink-2);font-size:.9rem;max-width:84ch;margin:8px 0 0}\n</style>",
        1,
    )
    old = "const MANLABEL=Object.fromEntries(D.manifests.map(m=>[m.key,m.label]));"
    assert old in tpl, "MANLABEL line not found"
    tpl = tpl.replace(
        old,
        old
        + "const MANDESC=Object.fromEntries(D.manifests.map(m=>[m.key,m.desc||'']));",
    )
    old = "h('h2',{html:`${MANLABEL[c.man]} <small>${c.man}</small>`},gwrap);"
    assert old in tpl, "group heading line not found"
    tpl = tpl.replace(
        old, old + "if(MANDESC[c.man])h('p',{class:'desc',text:MANDESC[c.man]},gwrap);"
    )
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return tpl.replace("__DATA__", payload)


MAX_POINTS = 160


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/data/runs_v2.csv")
    ap.add_argument("--hist-root", default="results/runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-points", type=int, default=MAX_POINTS)
    args = ap.parse_args()
    data, missing = build(args.csv, args.hist_root, args.max_points)
    html = render(data, TEMPLATE)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(html)
    for m in data["manifests"]:
        print(f"  {m['key']:22s} {m['n']:4d} runs   axis: {m['axis']}")
    print(
        f"\n  {len(data['runs'])} runs drawn, {missing} without a history -> {args.out} "
        f"({len(html) / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
