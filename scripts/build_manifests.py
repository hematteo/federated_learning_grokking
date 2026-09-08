"""Generate the versioned run manifests under manifests/.

Each function builds one manifest (a JSONL of run specs) for a tier of the
plan. Regenerating is deterministic — run ids are content hashes — so
re-running never changes ids and the launcher's resume stays valid.

    python scripts/build_manifests.py            # write all
    python scripts/build_manifests.py t0_wd_grid # write one

Run ids are content hashes of the config, so a cell that appears in two
manifests (e.g. an E-spine point that is also a probe point) gets the SAME id
and the launcher executes it once — the second manifest's copy is skipped by
the normal resume check. Overlap between tiers is therefore free, not waste.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fedgrok.manifest import (
    TAG_KEYS, expand_grid, load_manifest, run_id, write_manifest,
)

MANIFEST_DIR = os.path.join(os.path.dirname(__file__), "..", "manifests")
SEEDS5 = [42, 123, 456, 789, 1011]
SEEDS3 = [42, 123, 456]

# Every FL cell runs a FIXED number of communication rounds, so E is a clean
# *communication* axis and rounds — the expensive resource in FL — are matched
# across cells.
#
# NOTE (deliberate choice): we do NOT set num_rounds = budget // E to equalise
# gradient steps across E. That compute-matching is done at ANALYSIS time
# instead, using the `total_steps` axis the training loop already logs
# (alongside `sequential_steps` and `n_participating`) — see the Phase 0.6 step
# accounting. Keeping it out of the run config means:
#   - num_rounds stays an explicit, readable field in the manifest rather than
#     an implicit function of another field;
#   - a single run supports BOTH readings (per-round and per-step) rather than
#     baking one in;
#   - the consequence to keep in mind is that total gradient work scales with E
#     (steps = rounds x E), so raw wall-clock and total compute are NOT matched
#     across the E-spine, and low-E cells see proportionally fewer steps. When
#     comparing cells at equal compute, slice on `total_steps`, not on round.
#
# E RANGE (set by the t1_probe results, 2026-07-28): with rounds fixed, the
# extremes of the E axis are unusable in opposite directions, so the spine is
# restricted to E in {5, 10, 25, 50}:
#   - E = 1, 2 are UNDER-BUDGETED. At 10k rounds E=1 gets 10k gradient steps,
#     but this cell needs ~12.9k to grok (measured), so it censors for lack of
#     budget rather than for any federated reason — an uninformative cell that
#     invites exactly the wrong conclusion. (The E=1 identity is anyway proven
#     exactly in tests/test_fedavg_identity.py, so the spine does not need it.)
#   - E = 100, 250 are TOO EXPENSIVE. E=250 x 10k rounds is 2.5M gradient steps
#     per run (~1.5-2.5h); the probe's E=250 cells were cancelled for this.
# E in {5, 10, 25, 50} is a 10x span, all cells affordable and all informative.
FL_ROUNDS = 10_000
E_SPINE = [5, 10, 25, 50]
FL_EVAL_EVERY = 20          # ~500 curve points per run

# The one-shot (R=1) cell is the sole exception: with a single round there is no
# round axis, so its local-step count is stated outright.
ONE_SHOT_LOCAL_STEPS = 50_000


def t0_wd_grid():
    """Tier 0: loss x optimizer x weight-decay grid on setup A (mod-97 MLP).

    Fixes the coupled-WD defect and answers the lazy->rich / numerical-stability
    critiques. Weight-decay values are chosen so lr*wd (the comparable quantity)
    is log-spaced; see fedgrok.core.utils.check_decay_stability.

      GD    (lr=50)   : wd in {0, 2e-7, 2e-6, 2e-5, 2e-4} -> lr*wd {0,1e-5..1e-2}
      AdamW (lr=1e-3) : wd in {0, 0.01, 0.1, 1.0}         -> lr*wd {0,1e-5..1e-3}

    Centralized, 5 seeds. p=97, quadratic MLP, MSE (loss is fixed here; the CE
    arm needs the Phase 2/3 loss registry and is generated separately later).
    """
    specs = []
    specs += expand_grid(
        {"mode": "centralized", "task": "addition", "p": 97, "alpha": 0.5,
         "hidden_width": 256, "activation": "quadratic", "optimizer": "gd",
         "lr": 50.0, "epochs": 50000, "log_every": 100},
        {"weight_decay": [0.0, 2e-7, 2e-6, 2e-5, 2e-4], "seed": SEEDS5},
        tags={"tier": "T0", "group": "wd_grid", "experiment": "wd"},
    )
    specs += expand_grid(
        {"mode": "centralized", "task": "addition", "p": 97, "alpha": 0.5,
         "hidden_width": 256, "activation": "quadratic", "optimizer": "adamw",
         "lr": 1e-3, "epochs": 50000, "log_every": 100},
        {"weight_decay": [0.0, 0.01, 0.1, 1.0], "seed": SEEDS5},
        tags={"tier": "T0", "group": "wd_grid", "experiment": "wd"},
    )
    return specs


def t0_poly_pilot():
    """Tier 0 pilot: do x2+y2 and x2+xy+y2 grok centrally before FL compute?

    Doshi et al. (2406.03495) separate learnable from non-learnable modular
    polynomials for this architecture; a task that does not grok centrally can
    tell us nothing about FL, so gate the operation set on this.

    RESULT (run 2026-07-28, 3 seeds, GD lr=50, alpha=0.5, p=97):
        x2_plus_y2  3/3 grok, KM median 7000 [6900, 7000]  -> KEEP
        x2_y2_xy    0/3 grok (censored at 50k)             -> EXCLUDE
    The operation set in t1_replication therefore uses x2_plus_y2 and not
    x2_y2_xy, which is exactly what this gate is for.
    """
    return expand_grid(
        {"mode": "centralized", "p": 97, "alpha": 0.5, "hidden_width": 256,
         "activation": "quadratic", "optimizer": "gd", "lr": 50.0,
         "epochs": 50000, "log_every": 100},
        {"task": ["x2_plus_y2", "x2_y2_xy"], "seed": SEEDS3},
        tags={"tier": "T0", "group": "poly_pilot", "experiment": "pilot"},
    )


def t0_mnist_wd_band():
    """Locate the Omnigrok Goldilocks weight-decay band for MNIST-1k.

    The pipeline is verified (it reaches ~91% test), but a clean grok — fast
    memorisation then *delayed* generalisation — only appears in a narrow decay
    band: too weak and test never catches up within budget; too strong and there
    is no delay at all. This sweep over lr*wd (with large init) finds the band.
    Centralized, MSE, 3-layer MLP width 200, minibatch, 3 seeds.
    """
    return expand_grid(
        {"mode": "centralized", "dataset": "mnist", "model": "mlp",
         "hidden_width": 200, "n_layers": 3, "init_scale": 9.0,
         "loss": "mse", "optimizer": "adamw", "lr": 1e-3, "batch_size": 200,
         "n_train": 1000, "n_test": 5000, "epochs": 20000, "log_every": 100},
        # lr*wd in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}
        {"weight_decay": [0.01, 0.03, 0.1, 0.3, 1.0], "seed": SEEDS3},
        tags={"tier": "T0", "group": "mnist_wd_band", "experiment": "mnist"},
    )


def t3_server_lr_calibration():
    """Fair server-LR tuning for the server-optimiser strategies.

    The original exp5 tuned FedAdam's server LR but not the others', so its
    ~10x "speedup" was partly a tuning artifact. This sweeps server_lr for every
    tunable strategy on one representative heterogeneous cell, 3 seeds, so each
    method can be fixed at its own best LR before the main comparison. FedAvg /
    FedProx / SCAFFOLD have no server-LR knob (implicitly 1.0) and are the fixed
    references. FedAvgM additionally needs its momentum, swept lightly here.
    """
    cell = {"mode": "federated", "task": "addition", "p": 97, "alpha": 0.3,
            "hidden_width": 256, "num_clients": 10, "local_epochs": 5,
            "partition": "dirichlet", "dirichlet_alpha": 0.1,
            "num_rounds": FL_ROUNDS, "lr": 50.0, "eval_every": FL_EVAL_EVERY}
    specs = []
    for strat in ("fedadam", "fedyogi"):
        specs += expand_grid(
            {**cell, "strategy": strat},
            {"server_lr": [0.01, 0.1, 0.3, 1.0], "seed": SEEDS3},
            tags={"tier": "T3", "group": "server_lr_cal", "experiment": "cal",
                  "algorithm": strat},
        )
    specs += expand_grid(
        {**cell, "strategy": "fedavgm"},
        {"server_lr": [0.1, 0.3, 1.0], "server_momentum": [0.0, 0.9],
         "seed": SEEDS3},
        tags={"tier": "T3", "group": "server_lr_cal", "experiment": "cal",
              "algorithm": "fedavgm"},
    )
    return specs


# Setup A (the anchor): Gromov quadratic MLP, mod-97, MSE, full-batch GD.
SETUP_A = {"mode": "federated", "task": "addition", "p": 97, "alpha": 0.3,
           "model": "groknet", "hidden_width": 256, "loss": "mse",
           "optimizer": "gd", "lr": 50.0,
           "num_rounds": FL_ROUNDS, "eval_every": FL_EVAL_EVERY}


def t1_probe():
    """The early breakdown check — COMPLETED 2026-07-28, kept as a record.

    E values are left as they were actually run ({1,5,50,250}) so this manifest
    documents the executed sweep. Do NOT relaunch it: the E=250 cells were
    cancelled for cost and the E=1 cells proved under-budgeted, which is exactly
    why the going-forward spine is E_SPINE = {5,10,25,50} (see the E RANGE note
    at the top of this module). The T2 e-spine supersedes this.

    RESULT (18/24 completed, K=10, alpha=0.3, p=97, 3 seeds, KM median T_grok):
        E=1   iid/operand  0/3 grok  — censored by BUDGET, not by federation
                                       (10k steps available, ~12.9k needed).
                                       iid and operand were identical seed-for-
                                       seed: the E=1 FedAvg identity in the wild.
        E=5   iid 12900 / operand 12700   3/3
        E=50  iid 23000 / operand 17000   3/3
    Verdict: DELAY, not breakdown, at K=10 — grokking slows ~1.8x from E=5 to
    E=50 on the compute-matched step axis, but never fails. The breakdown search
    moves to higher K and stronger heterogeneity (the T2 K-sweep).
    """
    return expand_grid(
        {**SETUP_A, "num_clients": 10},
        {"local_epochs": [1, 5, 50, 250], "partition": ["iid", "operand"],
         "seed": SEEDS3},
        tags={"tier": "T1", "group": "probe", "experiment": "probe"},
    )


def t1_replication():
    """Tier 1: does the effect replicate across setups, tasks and moduli?

    Four blocks (MNIST is centralized-only — federated MNIST has no operand
    structure, so make_federated_datasets rejects it by design):
      - transformer @ mod-113 (Nanda config, CE)
      - S5 composition on both architectures, incl. the coset partition
      - prime ladder p in {53, 97, 113, 151} (finite-size scaling)
      - operation set (addition/subtraction/division/x2+y2) at p=97
    """
    specs = []

    # Transformer, Nanda config (CE + AdamW), mod-113.
    specs += expand_grid(
        {"mode": "federated", "task": "addition", "p": 113, "alpha": 0.3,
         "model": "transformer", "hidden_width": 128, "loss": "ce",
         "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1.0,
         "num_clients": 10,
         "num_rounds": FL_ROUNDS, "eval_every": FL_EVAL_EVERY},
        {"local_epochs": [5, 25],
         "partition": ["iid", "operand", "dirichlet"], "seed": SEEDS3},
        tags={"tier": "T1", "group": "transformer", "experiment": "replication"},
    )

    # S5 composition. The coset partition needs exactly 5 clients (5 cosets of
    # S4), so it is a separate block from the K=10 iid/dirichlet cells.
    s5_base = {"mode": "federated", "dataset": "s5", "group_n": 5, "alpha": 0.5,
               "loss": "ce", "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1.0,
               "num_rounds": FL_ROUNDS, "eval_every": FL_EVAL_EVERY}
    for model, width in (("groknet", 256), ("transformer", 128)):
        specs += expand_grid(
            {**s5_base, "model": model, "hidden_width": width, "num_clients": 10},
            {"local_epochs": [5, 25], "partition": ["iid", "dirichlet"],
             "seed": SEEDS3},
            tags={"tier": "T1", "group": "s5", "experiment": "replication"},
        )
        specs += expand_grid(
            {**s5_base, "model": model, "hidden_width": width,
             "num_clients": 5, "partition": "coset", "coset_subgroup": "s_nm1"},
            {"local_epochs": [5, 25], "seed": SEEDS3},
            tags={"tier": "T1", "group": "s5_coset", "experiment": "replication"},
        )

    # Prime ladder (finite-size scaling of the breakdown boundary).
    specs += expand_grid(
        {**SETUP_A, "num_clients": 10},
        {"p": [53, 97, 113, 151], "local_epochs": [5, 50],
         "partition": ["iid", "operand"], "seed": SEEDS3},
        tags={"tier": "T1", "group": "prime_ladder", "experiment": "replication"},
    )

    # Operation set at p=97. Multiplication is deliberately excluded (it is
    # cyclic Z_{p-1} in disguise — see DEGENERATE_TASKS).
    specs += expand_grid(
        {**SETUP_A, "num_clients": 10},
        {"task": ["addition", "subtraction", "division", "x2_plus_y2"],
         "local_epochs": [5, 50], "partition": ["iid", "operand"],
         "seed": SEEDS3},
        tags={"tier": "T1", "group": "operations", "experiment": "replication"},
    )
    return specs


def t2_phase_diagram():
    """Tier 2: the phase diagram on setup A — the paper's central grid.

    E-spine (the load-bearing axis), K-sweep in both disentangled forms,
    participation (the one FL knob with no centralized analogue), and the
    one-shot E->inf endpoint.
    """
    specs = []

    # E-spine: E from the exact centralized identity (E=1) out to near-independent.
    specs += expand_grid(
        {**SETUP_A, "num_clients": 10},
        {"local_epochs": E_SPINE,
         "partition": ["iid", "dirichlet", "operand"], "seed": SEEDS5},
        tags={"tier": "T2", "group": "e_spine", "experiment": "phase"},
    )

    # K-sweep, fixed TOTAL data (per-client shards shrink as K grows).
    specs += expand_grid(
        {**SETUP_A},
        {"num_clients": [5, 10, 20, 50], "local_epochs": [5, 50],
         "partition": ["iid", "dirichlet", "operand"], "seed": SEEDS5},
        tags={"tier": "T2", "group": "k_fixed_total", "experiment": "phase"},
    )

    # K-sweep, fixed PER-CLIENT data (alpha scales with K, so shard size is
    # constant). This disentangles "more clients" from "less data each" — the
    # confound flagged in the review.
    for k, alpha in ((5, 0.15), (10, 0.30), (20, 0.60), (50, 0.90)):
        specs += expand_grid(
            {**SETUP_A, "num_clients": k, "alpha": alpha},
            {"local_epochs": [5, 50], "partition": ["iid", "dirichlet", "operand"],
             "seed": SEEDS5},
            tags={"tier": "T2", "group": "k_fixed_per_client", "experiment": "phase"},
        )

    # Partial participation — no centralized analogue.
    specs += expand_grid(
        {**SETUP_A, "num_clients": 20},
        {"fraction_train": [0.2, 0.4, 0.6, 0.8, 1.0], "local_epochs": [5, 50],
         "partition": ["iid", "operand"], "seed": SEEDS5},
        tags={"tier": "T2", "group": "participation", "experiment": "phase"},
    )

    # One-shot FL: E -> infinity, R = 1. Each client trains to convergence
    # independently, then a single merge. The far endpoint of the E-spine, and
    # where frequency collision is most likely. This is the ONE cell where a
    # fixed round count makes no sense (R is 1 by definition), so E carries the
    # whole budget explicitly.
    specs += expand_grid(
        {**SETUP_A, "num_clients": 10, "num_rounds": 1,
         "local_epochs": ONE_SHOT_LOCAL_STEPS, "eval_every": 1},
        {"partition": ["iid", "dirichlet", "operand"], "seed": SEEDS5},
        tags={"tier": "T2", "group": "one_shot", "experiment": "phase"},
    )
    return specs


def t3_algorithm_comparison():
    """Tier 3: the FL algorithm comparison on the hard cells.

    Server-LR-tunable methods should be fixed at their calibrated LR (see
    t3_server_lr_calibration) before this is treated as final; the values here
    are placeholders that keep the grid runnable end to end.
    """
    hard_cells = [
        {"label": "H1", "alpha": 0.25, "num_clients": 10, "local_epochs": 25,
         "partition": "iid"},
        {"label": "H2", "alpha": 0.25, "num_clients": 10, "local_epochs": 25,
         "partition": "dirichlet", "dirichlet_alpha": 0.1},
        {"label": "H3", "alpha": 0.30, "num_clients": 10, "local_epochs": 50,
         "partition": "dirichlet", "dirichlet_alpha": 0.1},
    ]
    algos = [
        ("fedavg", {}),
        ("fedprox", {"proximal_mu": 0.01}),
        ("scaffold", {}),
        ("fedavgm", {"server_lr": 1.0, "server_momentum": 0.9}),
        ("fedadam", {"server_lr": 0.1}),
        ("fedyogi", {"server_lr": 0.1}),
    ]
    specs = []
    for cell in hard_cells:
        label = cell.pop("label")
        for strategy, kw in algos:
            specs += expand_grid(
                {**SETUP_A, **cell, "strategy": strategy, **kw},
                {"seed": SEEDS5},
                tags={"tier": "T3", "group": "algorithms", "experiment": "algo",
                      "setting": label, "algorithm": strategy},
            )
        cell["label"] = label
    return specs


def t2_k_breakdown():
    """The t1_probe follow-up: push K up, where breakdown is most likely.

    The probe found DELAY but no breakdown at K=10. Fragmenting the data across
    more clients is the axis most likely to actually break the circuit, so this
    is the K-sweep alone at E=5 (the cheap, known-good E), 3 partitions x 5
    seeds. It is a strict subset of t2_phase_diagram's k_fixed_total group, so
    ids match and running it means those cells are already done when the full
    T2 sweep is launched.
    """
    return expand_grid(
        {**SETUP_A, "local_epochs": 5},
        {"num_clients": [5, 10, 20, 50],
         "partition": ["iid", "dirichlet", "operand"], "seed": SEEDS5},
        tags={"tier": "T2", "group": "k_fixed_total", "experiment": "phase"},
    )


def t2_boundary():
    """The campaign that settles whether federation ever BREAKS grokking.

    Everything measured so far is delay, not breakdown: the alpha=0.3 plane
    groks 60/60 across K in {5..50}, and the probe found delay at K=10. The one
    standing breakdown claim comes from v1 exp2 at alpha=0.25, and it does not
    survive inspection.

    WHY THE OLD RESULT IS NOT EVIDENCE. exp2 ran `t_max=50000`. Its alpha=0.25,
    E=5, iid ladder (3 seeds, T_grok):

        K=2   23285 25315 27190      K=20  27215 29720 33025
        K=5   23625 25845 27880      K=50  40600 43975 CENSORED
        K=10  24450 26780 28790      K=97  CENSORED x3

    K=50 groks at 41-44k against a 50k budget. K=97's 0/3 is exactly what
    continued monotone delay looks like when it runs past the budget — the same
    trap as the E=1 probe cells, in the one place where censoring is supposed to
    BE the signal. Nothing there separates "federation broke the circuit" from
    "K=97 needs 60-90k steps".

    THE BUDGET. 20k rounds x E=5 = 100k gradient steps:
      - 2.3x the largest observed K=50 grok time (44k), so a censored K=97 sits
        well clear of its neighbour rather than 14% above it;
      - 1.33x the largest T_grok ever recorded at alpha<=0.25 (75425).
    Uniform across cells so every run shares one censoring time — a survival
    comparison across K is meaningless otherwise.

    THE CELLS.
      K=97 iid      the question.
      K=50 iid      proves the budget sufficed. If K=97 AND K=50 both censor,
                    the budget is still short and the sweep says nothing.
      K=20 iid      reproduces v1's 27-33k, confirming the v2 harness matches
                    exp2 before anything is concluded from a difference.
      K=97 operand  the rescue test. t2_k_breakdown found operand significantly
                    FASTER than iid at K=50 (13700 [13000,14000] vs 15200
                    [14600,16000]), and dirichlet tracking iid exactly — so it
                    is structure, not heterogeneity. Whether that advantage
                    survives to K=97 is a sharper question than the breakdown.
      Nearly free in wall-clock: the ten K=97 runs occupy slots the 1.2h K=20
      runs were never going to fill.

    5 seeds, not 1: grokking is stochastic (v1's K=50 cell was 2/3). Not 20
    either — seeds only buy CI separation where the grok fraction is partial,
    and which cell that is, is what this wave finds out. Deepening is wave 2.

    CHECKPOINTS ON. `checkpoint_every` is a config field and so feeds the run-id
    hash: enabling it later re-runs everything. The per-client W1 snapshots are
    what the frequency-consensus mechanism analysis needs, and this is the wave
    we least want to repeat. ~9.6 MB per checkpoint at K=97, 20 checkpoints per
    run, ~2.5 GB total.

    Verified before writing: all three partitions shard cleanly at alpha=0.25
    up to K=97 (smallest shard 7 samples, dirichlet; no empty shards).
    Cost, from the fitted model (9.8 + 1.291*K + 0.418*E) min at 10k rounds:
    K=20 1.2h, K=50 2.5h, K=97 4.5h per run -> ~7.5h wall-clock on 12 slots.
    """
    boundary = {**SETUP_A, "alpha": 0.25, "local_epochs": 5,
                "num_rounds": 20_000,
                "checkpoint_every": 1_000,
                "checkpoint_client_weights": True}
    specs = expand_grid(
        {**boundary, "partition": "iid"},
        {"num_clients": [20, 50, 97], "seed": SEEDS5},
        tags={"tier": "T2", "group": "boundary", "experiment": "boundary"},
    )
    specs += expand_grid(
        {**boundary, "partition": "operand", "num_clients": 97},
        {"seed": SEEDS5},
        tags={"tier": "T2", "group": "boundary", "experiment": "boundary"},
    )
    return specs


# ── The four new setups ──────────────────────────────────────────────────────
# Every federated result so far is SETUP_A. These are the setups built during the
# v2 rewrite and verified to grok centrally, but never run federated. Each is
# defined once here and referenced by every s5_* builder, so a setup's identity
# lives in exactly one place.
#
# `setup` is a TAG, not a config field, so adding it costs no run ids.

# ── TRANSFORMER WEIGHT DECAY: 1.0 HERE IS HISTORY, NOT THE CURRENT CHOICE ────
# As of 2026-08-20 new transformer work (B and C) uses **weight_decay = 0.1**.
# A new builder must override it explicitly:  {**SETUP_B, "weight_decay": 0.1}.
#
# The constants below stay at 1.0 deliberately. weight_decay is a Config field,
# so it is inside the content hash: changing it here changes every B and C spec
# id, main()'s `unchanged` check then fails, and write_manifest rewrites
# t2_aggregation and the rest -- orphaning all 305 banked wd=1.0 transformer
# runs from the ids `scripts/plotting/exp2_slowdown_ratio.py` selects on.
#
# Budget accordingly: on C's centralized decay band at alpha=0.5, wd=0.1 first
# crosses at 70,200 against 19,900 at wd=1.0 -- **3.5x slower** -- and below
# wd=0.03 C does not grok at all. Sizing a wd=0.1 cell from a banked wd=1.0
# number is how a cell gets censored.
#
# Why the change is nonetheless right for federated work: at K=20 on B, wd=1.0
# stops MEMORISATION -- two of three seeds sat at ~41% train through 10,000
# steps with t_memo infinite, while at wd=0.1 all three memorised cleanly
# (t_memo 3,600-4,200) and were merely waiting to generalise. High decay drives
# generalisation on these setups but blocks memorisation once federation has
# slowed it.
SETUP_B = {"dataset": "modular", "task": "addition", "p": 113,
           "model": "transformer", "hidden_width": 128, "n_heads": 4,
           "d_mlp": 512, "loss": "ce", "optimizer": "adamw", "lr": 1e-3,
           "weight_decay": 1.0}

SETUP_C = {"dataset": "s5", "group_n": 5, "model": "transformer",
           "hidden_width": 128, "n_heads": 4, "d_mlp": 512, "loss": "ce",
           "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1.0}

SETUP_D = {"dataset": "s5", "group_n": 5, "model": "groknet",
           "hidden_width": 256, "loss": "ce", "optimizer": "adamw",
           "lr": 1e-3, "weight_decay": 1.0}

# MNIST: lr*wd = 1e-4 is the measured best band (t0_mnist_wd_band -- highest test
# accuracy, clean 3300-epoch delay). alpha is IGNORED for MNIST; its data-fraction
# axis is n_train.
SETUP_E = {"dataset": "mnist", "model": "mlp", "hidden_width": 200,
           "n_layers": 3, "init_scale": 9.0, "loss": "mse", "optimizer": "adamw",
           "lr": 1e-3, "weight_decay": 0.1, "batch_size": 200}

NEW_SETUPS = {"B": SETUP_B, "C": SETUP_C, "D": SETUP_D, "E": SETUP_E}


# ── The optimiser as a control variable ──────────────────────────────────────
#
# The campaign is a 2x2 -- architecture (quad-MLP / transformer) x task (modular
# / S_5) -- and a factorial only reads if everything OFF the axis is held fixed.
# It is not. A is GD+MSE (Gromov's config, inherited from v1); B, C and D are
# AdamW+CE. So:
#
#     B vs C   task, on the transformer      AdamW+CE both      CLEAN
#     C vs D   architecture, on S_5          AdamW+CE both      CLEAN
#     A vs B   architecture, on modular      GD+MSE vs AdamW+CE confounded
#     A vs D   task, on the quad-MLP         GD+MSE vs AdamW+CE confounded
#
# B, C and D are internally consistent; the odd one out is A, and A cannot move
# -- it is Gromov's published config, the anchor to 870 v1 runs, and every banked
# federated result in the project. Changing it orphans all of that.
#
# So the fix is the missing cell, not a change to an existing one.

# A' -- A's architecture and task under B/C/D's optimiser. Closes the factorial.
#
# MSE and not CE, deliberately: A vs A' is then a SINGLE-variable optimiser
# contrast, which is worth having on its own terms because RESULTS.md 6.4 already
# claims the optimiser (AdamW's decoupled decay vs GD's coupled decay) is what
# flips weight decay's sign -- and that claim currently rests on comparing two
# different datasets. The cost is that A' vs B still differs in loss as well as
# architecture, which is stated rather than hidden.
#
# weight_decay is left at the measured band rather than inherited: t0_wd_grid's
# AdamW arm covers exactly this cell (quad-MLP, mod-97, AdamW, MSE) and lr*wd
# 1e-4 is the strongest setting that still groks under GD. It needs confirming at
# a working alpha, which is what x_aprime_alpha does -- the banked arm is all
# alpha=0.5, where this setup reaches 100/100 by epoch 200 and has no delay at
# all to measure.
SETUP_A_PRIME = {"dataset": "modular", "task": "addition", "p": 97,
                 "model": "groknet", "hidden_width": 256,
                 "activation": "quadratic", "loss": "mse",
                 "optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.1}

# C and D at a decay chosen FOR them rather than inherited from Nanda's mod-113
# transformer. wd is left unset here on purpose: x_cd_decay_band measures it, and
# these constants are filled in once that lands. Until then they are unused.
#
# NOTE these are new constants rather than edits to SETUP_C / SETUP_D. Run ids
# are content hashes and `s5_central_anchor` references those constants directly,
# so editing them in place would change 60 banked ids and `write_manifest` would
# (correctly) refuse to rewrite the anchor manifest. Keeping both means the
# wd=1.0 ladders stay valid and become a deliberate decay comparison instead of
# discarded work.


def s5_central_anchor():
    """STAGE 1: locate each new setup's own data cliff, centrally. Gate A.

    This is the cheapest and highest-value stage in the campaign, and it is a
    prerequisite rather than a formality, for two reasons.

    FIRST, THERE IS NO ANCHOR. No manifest has ever produced a centralized
    transformer or S5 run: the published grok times (transformer 6200, S5 ~14000)
    came from ad-hoc runs with no result JSON, and `results/data/runs/` contains
    zero of either. Federated delay is a RATIO, so without a pipeline-produced
    centralized T_grok there is no denominator.

    SECOND, alpha=0.25/0.3 IS SETUP-A'S BOUNDARY, NOT EVERYONE'S. Setup A's cliff
    is at alpha_c ~= 0.198 and its FL cells sit just above it. Where the cliff is
    for a transformer, for S5, or for MNIST is unknown. Placing FL cells before
    measuring it is how this project has twice produced a censored cell that meant
    nothing -- the E=1 probe cells and v1's K=97 breakdown claim, both budget
    artifacts (see the E RANGE note above and t2_boundary's docstring).

    Budgets are generous on purpose: a censored cell here must mean "past the
    cliff", not "past the clock".
    """
    specs = []

    # Setups B/C/D: the alpha ladder. Spans setup A's known boundary region so
    # the cliffs are directly comparable.
    for label, setup, epochs in (("B", SETUP_B, 30_000),
                                 ("C", SETUP_C, 40_000),
                                 ("D", SETUP_D, 40_000)):
        specs += expand_grid(
            {"mode": "centralized", **setup, "epochs": epochs, "log_every": 100},
            {"alpha": [0.15, 0.20, 0.25, 0.30, 0.40, 0.50], "seed": SEEDS5},
            tags={"tier": "S1", "group": "central_anchor",
                  "experiment": "anchor", "setup": label},
        )

    # Setup A on the same ladder, for a like-for-like reference measured by the
    # same harness rather than quoted from v1.
    specs += expand_grid(
        {"mode": "centralized", "task": "addition", "p": 97, "model": "groknet",
         "hidden_width": 256, "loss": "mse", "optimizer": "gd", "lr": 50.0,
         "epochs": 30_000, "log_every": 100},
        {"alpha": [0.15, 0.20, 0.25, 0.30, 0.40, 0.50], "seed": SEEDS5},
        tags={"tier": "S1", "group": "central_anchor", "experiment": "anchor",
              "setup": "A"},
    )

    # Setup E: alpha does not apply, so the data axis is n_train. The batch-size
    # block is NOT optional -- see s5_mnist_working_point below.
    specs += expand_grid(
        {"mode": "centralized", **SETUP_E, "n_test": 5000, "epochs": 20_000,
         "log_every": 100},
        {"n_train": [500, 1000, 2000, 4000], "seed": SEEDS5},
        tags={"tier": "S1", "group": "central_anchor", "experiment": "anchor",
              "setup": "E"},
    )
    return specs


def s5_mnist_working_point():
    """STAGE 1b: find an (n_train, batch_size) for MNIST that supports a K-sweep.

    At the Omnigrok grok point (n_train=1000) federated MNIST is degenerate. Every
    shard is n_train/K samples, so at batch_size=200:

        n_train  K    per client   batches per local epoch
           1000  10          100   1   <- batch_size is inert
           1000  50           20   1
           4000  10          400   2
           4000   5          800   4

    With one batch per local epoch, `local_epochs` stops meaning what it means on
    every other setup and the E axis is not comparable. So MNIST needs a working
    point that still groks centrally AND leaves >= 2 batches per local epoch at
    the campaign's largest K. That is what this measures.
    """
    return expand_grid(
        {"mode": "centralized", **{k: v for k, v in SETUP_E.items()
                                   if k != "batch_size"},
         "n_test": 5000, "epochs": 20_000, "log_every": 100},
        {"n_train": [1000, 2000, 4000], "batch_size": [25, 50, 100, 200],
         "seed": SEEDS3},
        tags={"tier": "S1", "group": "mnist_working_point",
              "experiment": "anchor", "setup": "E"},
    )


def s5_fl_probe():
    """STAGE 2: does FL run at all on each new setup, and what does it cost? Gate B.

    The cost model (9.8 + 1.291*K + 0.418*E) min/10k rounds is fitted on GrokNet
    alone. Wall-clock here is ~99% Flower/Ray orchestration, and orchestration is
    weight-shipping, so what matters is payload per client per round:

        GrokNet + modular   74,496 params    291 KB
        Transformer        225,792 params    882 KB   <- 3x
        GrokNet + S5        92,160 params    360 KB
        MLP + MNIST        199,210 params    778 KB

    Compute is ~6x for the transformer but is only ~1% of wall-clock, so the
    payload is the term to worry about. Rather than extrapolate, measure: this
    stage refits the coefficients per setup before anything expensive is planned.

    Also the first FL run of each new setup at a realistic K -- worth knowing
    before committing to a 200-run stage.
    """
    specs = []
    for label, setup in NEW_SETUPS.items():
        cell = {"mode": "federated", **setup, "num_rounds": 2_000,
                "local_epochs": 5, "eval_every": FL_EVAL_EVERY}
        if label == "E":
            # MNIST: no alpha, and operand/coset do not apply.
            cell.update({"n_train": 4000, "n_test": 5000})
            partitions = ["iid", "label_block"]
        elif label in ("C", "D"):
            cell["alpha"] = 0.5
            partitions = ["iid", "operand"]
        else:
            cell["alpha"] = 0.3
            partitions = ["iid", "operand"]
        specs += expand_grid(
            cell,
            {"num_clients": [10, 50], "partition": partitions, "seed": SEEDS3},
            tags={"tier": "S2", "group": "fl_probe", "experiment": "probe",
                  "setup": label},
        )

    # The AdamW confound, quantified. Rebuilding the optimizer every round is a
    # genuine no-op for GD at momentum=0 -- which is why setup A's E axis is clean
    # -- but under AdamW it makes every round E bias-corrected COLD-START Adam
    # steps. All four new setups use AdamW, so without this the new E results and
    # setup A's are not measuring the same quantity. 12 runs to find out.
    specs += expand_grid(
        {"mode": "federated", **SETUP_B, "alpha": 0.3, "num_clients": 10,
         "partition": "iid", "num_rounds": 2_000, "eval_every": FL_EVAL_EVERY},
        {"persist_local_opt_state": [False, True], "local_epochs": [5, 50],
         "seed": SEEDS3},
        tags={"tier": "S2", "group": "adam_restart", "experiment": "probe",
              "setup": "B"},
    )
    return specs


def x_d_alpha_fine():
    """TANGENT (not part of the staged campaign): a fine alpha ladder on setup D.

    Interstitial points between the Gate A ladder's rungs for the quadratic MLP
    on S5, which currently reads:

        a=0.5  7,200 | 0.4  12,600 | 0.3  21,300 | 0.25  36,200 | 0.2  0/5

    Adding 0.325/0.35/0.375, 0.425/0.45/0.475 and 0.525/0.55 takes that from 6
    rungs to 14, which is what a power-law fit of T_grok ~ (alpha - alpha_c)^-gamma
    needs to be worth quoting -- setup A's exponent (gamma ~ 0.99, alpha_c ~ 0.198,
    R^2 = 0.978) was fitted on a comparable density.

    Identical to the s5_central_anchor block for setup D in every other respect:
    same SETUP_D, same 40,000-epoch budget, same 5 seeds. Every new alpha is at or
    above 0.325, where the ladder already shows T_grok <= 21,300, so 40k leaves
    at least 2x headroom and nothing here should censor on the clock.

    Tagged tier "X" so it stays out of the Gate A / Stage 3 analysis by default.
    """
    return expand_grid(
        {"mode": "centralized", **SETUP_D, "epochs": 40_000, "log_every": 100},
        {"alpha": [0.325, 0.35, 0.375, 0.425, 0.45, 0.475, 0.525, 0.55],
         "seed": SEEDS5},
        tags={"tier": "X", "group": "d_alpha_fine", "experiment": "tangent",
              "setup": "D"},
    )


def x_d_alpha_high():
    """TANGENT: setup D's ladder extended up to alpha = 1.00 in 0.025 steps.

    Same cell as x_d_alpha_fine and the Gate A setup-D block in every respect --
    same SETUP_D, same 40,000-epoch budget, same 5 seeds -- so all three compose
    into one ladder. At these alphas T_grok is well under 5,000, so the budget is
    ample.

    alpha = 1.00 IS NO LONGER EMITTED, and the note that used to sit here was
    wrong in the one way that mattered. It said the empty-test-set cells "record
    t_grok = inf", which sounds harmless. They recorded `t_grok = 0,
    grokked = True`: compute_accuracy over zero samples returns NaN, `nan <
    threshold` is False, so the sustained-crossing scan found no point below the
    bar and reported it as held from the first sample. Five banked rows in this
    group say setup D groks instantly at alpha = 1.0, and nothing measured them.

    `split_indices` now rejects any alpha that empties either side of the split
    and `compute_t_grok` treats NaN as below the bar, so neither half can recur.
    The ladder stops at 0.975, which already leaves only 360 test samples. The
    five banked JSONs stay on disk as the record; RESULTS.md carries the
    exclusion note.
    """
    return expand_grid(
        {"mode": "centralized", **SETUP_D, "epochs": 40_000, "log_every": 100},
        {"alpha": [0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775,
                   0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975],
         "seed": SEEDS5},
        tags={"tier": "X", "group": "d_alpha_high", "experiment": "tangent",
              "setup": "D"},
    )


def x_d_alpha_cliff():
    """TANGENT: is setup D's cliff real, or is it the 40,000-epoch clock?

    The 14-rung ladder (x_d_alpha_fine + the Gate A block) is fitted far better
    by an exponential, T_grok ~ 10^(-2.78 alpha), than by the power law setup A
    follows -- R^2 0.983 on two parameters against 0.971 on three, and the power
    law's alpha_c runs to the bottom of its search range.

    Extrapolating that exponential to the two censored rungs:

        alpha = 0.20  ->  ~45,000 epochs
        alpha = 0.15  ->  ~62,000 epochs

    Both are ABOVE the 40,000-epoch budget they were run at. So their 0/5 is
    exactly what a smooth exponential looks like when it runs past the clock, and
    the apparent cliff between 0.20 and 0.25 may not exist at all. Reading it as
    an abrupt failure would repeat the mistake that cost v1 its headline claim.

    150,000 epochs -- 2.4x the exponential's prediction at the hardest rung -- and
    two intermediate rungs to see whether the curve simply continues.
    """
    return expand_grid(
        {"mode": "centralized", **SETUP_D, "epochs": 150_000, "log_every": 200},
        {"alpha": [0.15, 0.175, 0.2, 0.225], "seed": SEEDS5},
        tags={"tier": "X", "group": "d_alpha_cliff", "experiment": "tangent",
              "setup": "D"},
    )


def x_d_internals():
    """TANGENT: what is happening inside setup D across the alpha range 0.2-0.6?

    THE QUESTION. Read at fixed step counts rather than threshold crossings, the
    32-rung alpha ladder is two phases, not a family of different phenomena:

        alpha   0.30  0.375  0.40  0.45  0.50  0.55  0.60
        @1800    3.1   19.1  32.6  57.1  77.4  88.2  94.0    <- phase-1 plateau
        @3600    2.4   20.2  32.6  55.9  73.4  84.1  91.5    <- the dip
        T@95   27067  19400 17200 13960 11380  8920  6700    <- phase-2

    Phase 1 finishes by ~1,800 steps at EVERY alpha and lands on a plateau
    P(alpha) that is a sigmoid with midpoint ~0.47. The dip bottoms at ~3,600
    steps at every alpha. Phase 2's rate is smooth, T@95 ~ exp(-n/3190) over
    alpha in [0.3, 0.6] to 4.7% residual. So alpha's dramatic effect is almost
    entirely P(alpha); the apparent cliff is where P crosses the 85% grok bar and
    it moves when the bar moves (T@25 collapses at 0.40, T@50 at 0.45, T@85 at
    0.55). None of that says WHY.

    WHAT THIS RUN ADDS. The accuracy histories cannot answer it, because nothing
    logged separates a memorising circuit from a compositional one. Setup D's
    quadratic activation makes that separation exact rather than fitted -- see
    metrics/quadratic_circuits.py: logit = A[c,a] + 2T[c,a,b] + B[c,b], where A
    and B cannot compose by construction, so T IS the compositional circuit. And
    metrics/irreps.py gives S_5's exact analogue of the modular study's Fourier
    spectrum via the isotypic decomposition. Both are new, both log per eval, and
    NO banked run has them -- nor any saved weights, so this cannot be done
    post-hoc on the 160 ladder runs.

    A single pilot at alpha=0.55 already shows the measures move with the dip:
    the compositional circuit's participation ratio bottoms at 54 units at step
    1,800 (the plateau) and rises to 75 at step 3,600 (the trough), exactly
    anti-correlated with test accuracy. That is one seed at one alpha, which is
    what this manifest is for.

    BUDGET. Two blocks, because the low rungs are slow and the high ones are not:
    alpha <= 0.275 gets 150,000 epochs (alpha=0.25's T@85 is 35,140 and 0.225 is
    censored at 40k), the rest 40,000. log_every=25 throughout -- the dip is
    ~1,800 steps wide and the ladder's 100-step resolution smears its edges.
    checkpoint_every=500 so the post-hoc questions the time series cannot answer
    (CKA to the final representation, cumulative unit ablation) stay open without
    a re-run; at 369 KB a checkpoint that is ~4 GB, against 32 TB free.
    """
    slow = expand_grid(
        {"mode": "centralized", **SETUP_D, "epochs": 150_000,
         "log_every": 25, "checkpoint_every": 500},
        {"alpha": [0.2, 0.225, 0.25, 0.275], "seed": SEEDS5},
        tags={"tier": "X", "group": "d_internals", "experiment": "tangent",
              "setup": "D"},
    )
    fast = expand_grid(
        {"mode": "centralized", **SETUP_D, "epochs": 40_000,
         "log_every": 25, "checkpoint_every": 500},
        {"alpha": [0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5,
                   0.525, 0.55, 0.575, 0.6],
         "seed": SEEDS5},
        tags={"tier": "X", "group": "d_internals", "experiment": "tangent",
              "setup": "D"},
    )
    return slow + fast


def x_d_wd_ladder():
    """TANGENT: is setup D's fixed-epoch dip a weight-decay transient?

    THE OBSERVATION. Read at fixed step counts rather than at threshold
    crossings, every rung of the 32-point alpha ladder has the same shape: a
    fast rise that plateaus by ~1,800 steps, a dip bottoming at ~3,600, then a
    slow climb. The dip sits at the SAME step for every alpha, while T_grok over
    that ladder spans 300 to 35,000 steps -- two orders of magnitude. Whatever
    sets the dip's clock is therefore not the data.

    Test acc at fixed steps (5-seed means), showing plateau height P(alpha):

        alpha  0.30  0.375  0.40  0.45  0.50  0.55  0.60
        @1800   3.1   19.1  32.6  57.1  77.4  88.2  94.0
        @3600   2.4   20.2  32.6  55.9  73.4  84.1  91.5

    P(alpha) is a sigmoid with midpoint ~0.47, and it -- not any transition in
    the dynamics -- is what makes the ladder's curves look like different
    phenomena. The apparent alpha cliff moves with the grok threshold (T@25
    collapses at alpha~0.40, T@50 at ~0.45, T@70 at ~0.50, T@85 at ~0.55),
    i.e. it is just where P(alpha) crosses the chosen bar.

    THE HYPOTHESIS. The one clock in this setup that ignores alpha is decoupled
    weight decay: SETUP_D runs lr=1e-3, wd=1.0, so weights shrink by (1 - lr*wd)
    per step and the decay timescale is tau = 1/(lr*wd) = 1,000 steps. The
    observed plateau is at ~1.8 tau and the trough at ~3.6 tau.

    THE TEST. Move tau and see whether the dip moves with it.

        wd    lr*wd    tau      predicted trough (3.6 tau)
        0.0   0        none     no dip at all
        0.1   1e-4     10,000   ~36,000
        0.3   3e-4      3,333   ~12,000
        1.0   1e-3      1,000    ~3,600   (the banked ladder -- the control)
        3.0   3e-3        333    ~1,200

    Proportional movement confirms decay. No movement falsifies it cheaply and
    sends the search to the task instead. All five sit under the lr*wd <= 0.1
    band that check_decay_stability enforces.

    Three alphas spanning the sigmoid -- 0.30 (below the midpoint, dip masked by
    growth), 0.45 (mid-rise, deepest dip), 0.55 (near ceiling) -- so the same
    runs also show whether P(alpha) itself depends on decay, and whether phase
    2's rate constant (T@95 ~ exp(-n/3190) over alpha in [0.3, 0.6]) rescales.

    log_every=25 rather than the ladder's 100: at wd=3.0 the whole event is
    predicted to be over by step ~1,200, which 100-step resolution would smear.
    """
    return expand_grid(
        {"mode": "centralized", **SETUP_D, "epochs": 40_000, "log_every": 25},
        {"weight_decay": [0.0, 0.1, 0.3, 1.0, 3.0],
         "alpha": [0.30, 0.45, 0.55],
         "seed": SEEDS3},
        tags={"tier": "X", "group": "d_wd_ladder", "experiment": "tangent",
              "setup": "D"},
    )


def x_d_lr_control():
    """TANGENT: the confound control for x_d_wd_ladder -- decay clock or step size?

    x_d_wd_ladder varies wd at fixed lr, so it varies lr*wd. But lr also sets
    the step size, and a dip that moves with 1/lr would look identical to one
    that moves with 1/(lr*wd) in that ladder alone. The two are separated by
    holding lr*wd FIXED and varying lr:

        lr       wd     lr*wd    tau        step size vs control
        2.5e-4   4.0    1e-3     1,000      0.25x
        1.0e-3   1.0    1e-3     1,000      1x        (control)
        4.0e-3   0.25   1e-3     1,000      4x

    Products are exactly 1e-3 in all three, so tau is identical by construction.
    Predictions:
      dip at the same STEP in all three  -> the decay timescale sets the clock
      dip epoch scaling as 1/lr          -> it is the step size, and the wd
                                            ladder's result is coincidental

    Two alphas (0.45, 0.55) rather than three: this arm only has to localise the
    dip, which the ladder shows is deepest and cleanest above the P(alpha)
    midpoint.

    60,000 epochs rather than 40,000. At lr=2.5e-4 every timescale that is
    gradient-driven stretches 4x, and alpha=0.45's T@95 of ~13,960 at the control
    lr would land near 55,000. The dip is early either way, but the extra budget
    keeps phase 2 measurable in the slow arm so the same runs report whether the
    rate constant follows lr or lr*wd.

    lr=4e-3 is 4x the published grokking band and may be unstable on a quadratic
    activation. That is a real risk and an informative outcome rather than a
    failure -- divergence is visible in the loss curve, and the arm is 6 runs.
    """
    specs = []
    for lr, weight_decay in [(2.5e-4, 4.0), (1.0e-3, 1.0), (4.0e-3, 0.25)]:
        specs += expand_grid(
            {"mode": "centralized", **SETUP_D, "epochs": 60_000,
             "log_every": 25, "lr": lr, "weight_decay": weight_decay},
            {"alpha": [0.45, 0.55], "seed": SEEDS3},
            tags={"tier": "X", "group": "d_lr_control", "experiment": "tangent",
                  "setup": "D"},
        )
    return specs


# ── exp2's floor arm ─────────────────────────────────────────────────────────

# Fields that only mean something federated. A centralized spec carrying them
# would raise in build_config, since Config has no such attributes.
_FED_ONLY = {"num_clients", "num_rounds", "local_epochs", "fraction_train",
             "partition", "dirichlet_alpha", "proximal_mu", "strategy",
             "server_lr", "server_momentum", "tau", "feddyn_alpha",
             "eval_every", "track_client_drift", "persist_local_opt_state",
             "checkpoint_client_weights", "coset_subgroup"}


def reduced_arm(fed_specs, budget_multiple=2.0):
    """v1 exp2's centralized-REDUCED condition: one model on one client's shard.

    exp2 ran three arms per (alpha, K) -- centralized-full as the ceiling, this
    as the floor, and FL as the test -- and its headline was that FL groks in 23
    of 30 cells where the floor groks in 1. Read carefully that gap is mostly
    "FL sees K times more data than the floor", not evidence about aggregation,
    which is why the going-forward framing measures FL against the CEILING on the
    compute-matched step axis. The floor is kept anyway, because it is the arm
    that says whether a client could have done this alone, and that is the
    question a federated result is actually answering.

    HOW THE DATA IS REDUCED IS DATASET-DEPENDENT, which is why this is not a
    one-liner. `alpha` is the data-fraction axis for the grid datasets, but MNIST
    IGNORES alpha entirely -- its axis is n_train -- so reducing MNIST by scaling
    alpha would produce a spec identical to the full arm and a floor that
    silently equals the ceiling.

    BUDGET. The floor gets `budget_multiple` times the FL arm's total gradient
    steps (rounds x E), not the same. A floor arm is only informative if its
    failure is attributable to having less data, and matching the budget exactly
    leaves "it ran out of time" open -- which is how v1 read K=97 as a breakdown
    when two of five seeds simply needed more than the 50k they were given. The
    floor is cheap (one model, 1/K of the data), so headroom is nearly free.

    Ids are content hashes, so a floor cell shared by two FL cells -- e.g. the
    same (alpha, K) at different E -- resolves to one run and executes once.
    """
    out, seen = [], set()
    for spec in fed_specs:
        if spec.get("mode") != "federated":
            raise ValueError("reduced_arm expects federated specs")
        k = spec["num_clients"]
        reduced = {key: value for key, value in spec.items()
                   if key not in _FED_ONLY and key not in TAG_KEYS}
        reduced["mode"] = "centralized"

        if spec.get("dataset") == "mnist":
            n_train = spec.get("n_train", 1000)
            reduced["n_train"] = max(1, n_train // k)
            if reduced["n_train"] < spec.get("batch_size", 0):
                # One partial batch per epoch: the optimiser's effective batch
                # changes with K, so the floor would differ from the ceiling by
                # more than data volume. Shrink the batch with the shard.
                reduced["batch_size"] = reduced["n_train"]
        else:
            reduced["alpha"] = spec["alpha"] / k

        steps = spec.get("num_rounds", 10_000) * spec.get("local_epochs", 5)
        reduced["epochs"] = int(steps * budget_multiple)
        reduced["log_every"] = max(10, reduced["epochs"] // 500)

        reduced["id"] = run_id(reduced)
        if reduced["id"] in seen:
            continue
        seen.add(reduced["id"])
        # `arm` rides along as a tag so the three conditions are separable in the
        # results table without reconstructing which alpha was a reduction.
        out.append({**reduced, "arm": "cent_reduced", "reduced_from_k": k})
    return out


def p1_d_gd_probe():
    """PHASE 1: does the quadratic MLP grok S_5 under GD + MSE at all?

    Setup D is Gromov's architecture -- the SAME model as setup A -- but running
    AdamW + CE where A runs GD + MSE. Nothing recorded says why; it appears to
    have inherited the optimiser from the S_5 side (setup C) rather than from the
    architecture side. The consequence is that A vs D, which is meant to isolate
    the TASK holding architecture fixed, moves task, optimiser and loss together.

    And it has never been checked: of 370 banked S_5 runs, ZERO use GD.

    alpha=0.5 is D's easiest working point (T_grok 7,200 under AdamW), so if GD
    works anywhere it works here. lr is swept because Gromov's 50 was tuned for a
    194-dim input and 97 classes; S_5 is 240-dim with 120, and a quadratic
    activation at too large a step diverges rather than degrading. Divergence in
    the low-lr arm would be informative, not a wasted cell.

    50,000 epochs is ~7x A's alpha=0.5 requirement, so nothing here censors on
    the clock.

    > DECISION RULE. If any lr reaches 5/5, D is redefined as GD + MSE: A vs D
    > becomes a clean single-variable task comparison, D inherits A's immunity to
    > the K>=30 collapse (no decay clock at wd=0), and setup C keeps AdamW with
    > C vs D then differing in architecture AND optimiser -- stated, not hidden.
    > If none groks, "S_5 requires adaptivity on this architecture" is itself a
    > result, it retroactively justifies D's config, and the confound is recorded
    > as a limitation instead of a mistake.
    """
    return expand_grid(
        {"mode": "centralized", "dataset": "s5", "group_n": 5,
         "model": "groknet", "hidden_width": 256, "activation": "quadratic",
         "loss": "mse", "optimizer": "gd", "weight_decay": 0.0,
         "epochs": 50_000, "log_every": 100},
        {"lr": [5.0, 10.0, 50.0], "seed": SEEDS3},
        tags={"tier": "P1", "group": "d_gd_probe", "experiment": "optimiser",
              "setup": "D"},
    )


def p1_cd_decay_band():
    """PHASE 1: measure C's and D's weight decay instead of inheriting Nanda's.

    wd=1.0 is Nanda's published value for a 1-layer transformer on mod-113. B has
    a reason to carry it -- B IS that replication. C and D carry it because they
    were written next to B, and for D nothing about the architecture, the task or
    the class count was consulted.

    That matters twice over. It is the likeliest cause of the K>=30 collapse: at
    K=50 on setup B, wd=1.0 gives ~3.6% train accuracy and wd=0.1 gives 70.2% --
    the only knob tried that moves the failure at all, and lr moves it the WRONG
    way. And setup E, the one AdamW setup whose decay WAS measured for its own
    setup (t0_mnist_wd_band), is the one that does not collapse.

    The sweep is on lr*wd, log-spaced, exactly as t0_wd_grid and
    t0_mnist_wd_band are -- weights shrink by (1 - lr*wd) per step, so that
    product is the comparable quantity and 1/(lr*wd) is the decay timescale:

        wd     0.01    0.03    0.1     0.3     1.0
        lr*wd  1e-5    3e-5    1e-4    3e-4    1e-3  (the inherited value)

    C runs at hidden_width 256, not the 128 of its Gate A ladder: the capacity
    sweep halved its T_grok at 256 (21,600 vs 51,200 at 12/12 each), so 256 is
    the capacity C will actually use and the band should be measured there.

    Budgets are ~4.5x each setup's measured T_grok at the inherited decay
    (D 21,300 at alpha=0.30; C 21,600 at alpha=0.50, width 256).

    > DECISION RULE. Take the band with the highest fraction-grokked, breaking
    > ties on the shortest T_grok. If the best band is NOT 1e-3, C and D are
    > redefined at it and their alpha ladders must be re-measured there -- the
    > ladder is what sets the working point and every downstream budget, so it
    > does not survive a change of decay. That re-ladder is the real cost of this
    > sweep and is Phase 2, not an afterthought.
    """
    specs = []
    # D: quad-MLP on S_5, at its working alpha.
    specs += expand_grid(
        {"mode": "centralized", **SETUP_D, "alpha": 0.30,
         "epochs": 100_000, "log_every": 50},
        {"weight_decay": [0.01, 0.03, 0.1, 0.3, 1.0], "seed": SEEDS3},
        tags={"tier": "P1", "group": "cd_decay_band", "experiment": "decay",
              "setup": "D"},
    )
    # C: transformer on S_5, at the capacity the capacity sweep selected.
    specs += expand_grid(
        {"mode": "centralized", **SETUP_C, "hidden_width": 256, "alpha": 0.50,
         "epochs": 100_000, "log_every": 50},
        {"weight_decay": [0.01, 0.03, 0.1, 0.3, 1.0], "seed": SEEDS3},
        tags={"tier": "P1", "group": "cd_decay_band", "experiment": "decay",
              "setup": "C"},
    )
    return specs


def t3b_partitions():
    """PART 2: exp3b -- does HOW you shard matter more than how far you shard?

    main's exp3b compared iid / operand / target at K=10. This keeps those three
    and adds two that main could not run:

      dirichlet  unstructured non-IID. THE CONTROL, and the reason this is not
                 just main's experiment again. t2_k_breakdown found operand
                 significantly FASTER than iid at K=50 while dirichlet tracked
                 iid exactly at every K -- so the effect is structure, not
                 heterogeneity. Without a dirichlet arm the two are confounded.
      coset      S_5's algebraically coherent split (S_4 -> 5 cosets). On S_5 the
                 operand partition shards by first-operand ELEMENT, which is NOT
                 coherent the way a mod-p operand shard is; coset is. C and D
                 only. K must equal the coset count exactly, so it is its own
                 block at K=5.

    THE CLAIM UNDER TEST. "Coherent shards beat random ones, and the gap grows
    with K" is the project's strongest result and rests on setup A alone. It is
    also the one claim insensitive to where the budget was set. This asks whether
    it is a property of grokking or of one architecture.

    K IS CHOSEN PER SETUP FROM exp2, not shared. exp2 measured where each setup
    still functions federated, and there is no point asking about partitions
    where the setup cannot grok under ANY partition:

        A   K in {10, 50}   FL tracks the ceiling to 1.17x at K=50
        C   K in {10, 50}   3/3 at every K
        A'  K in {10, 20}   already 14.6x at K=20
        B   K in {10, 20}   0/3 at K=50 -- the decay clock, not the partition
        D   K in {10, 20}   0/3 at K=50, same reason
        E   K in {10, 20}   shards degenerate past 20 at batch=100

    Budgets are exp2's, which are the measured ones.

    > DECISION RULE. Per setup, compare operand (and coset on S_5) against iid at
    > matched K, with dirichlet as the control. If structured partitions are
    > faster wherever the setup groks at all, the claim generalises beyond the
    > anchor. If the ordering flips on any setup -- and s5_fl_probe already hinted
    > D's operand cell is WORSE, which would invert the headline -- that setup's
    > shard geometry is the thing to explain, and the coset arm is what
    > distinguishes "structured" from "algebraically coherent".
    """
    BLOCKS = [
        ("A",  {k: v for k, v in SETUP_A.items()
                if k not in ("mode", "num_rounds", "eval_every")},
         {"alpha": 0.30}, 10_000, [10, 50]),
        ("A'", SETUP_A_PRIME, {"alpha": 0.20}, 20_000, [10, 20]),
        ("B",  SETUP_B, {"alpha": 0.30}, 20_000, [10, 20]),
        ("C",  SETUP_C, {"alpha": 0.40, "hidden_width": 256}, 40_000, [10, 50]),
        ("D",  SETUP_D, {"alpha": 0.30}, 50_000, [10, 20]),
    ]
    specs = []
    for label, base, wp, rounds, Ks in BLOCKS:
        tags = {"tier": "T3b", "group": "partitions", "experiment": "exp3b",
                "setup": label}
        common = {"mode": "federated", **base, **wp, "local_epochs": 5,
                  "strategy": "fedavg", "fraction_train": 1.0,
                  "num_rounds": rounds, "eval_every": FL_EVAL_EVERY,
                  "checkpoint_every": max(1, rounds // 10),
                  "checkpoint_client_weights": True}
        specs += expand_grid(
            common,
            {"partition": ["iid", "operand", "target", "dirichlet"],
             "num_clients": Ks, "seed": SEEDS3},
            tags=tags,
        )
        # coset: S_5 only, and K must BE the coset count (S_4 -> 5)
        if base.get("dataset") == "s5":
            specs += expand_grid(
                {**common, "num_clients": 5, "partition": "coset",
                 "coset_subgroup": "s_nm1"},
                {"seed": SEEDS3}, tags=tags,
            )
    # E: MNIST has no operand or coset structure; label_block is its analogue
    specs += expand_grid(
        {"mode": "federated", **{k: v for k, v in SETUP_E.items()
                                 if k != "batch_size"},
         "n_train": 2000, "n_test": 5000, "batch_size": 100, "local_epochs": 5,
         "strategy": "fedavg", "fraction_train": 1.0, "num_rounds": 8_000,
         "eval_every": FL_EVAL_EVERY, "checkpoint_every": 800,
         "checkpoint_client_weights": True},
        {"partition": ["iid", "dirichlet", "label_block"],
         "num_clients": [10, 20], "seed": SEEDS3},
        tags={"tier": "T3b", "group": "partitions", "experiment": "exp3b",
              "setup": "E"},
    )
    return specs


def x_controls():
    """Two controls for claims already being made. 12 runs, ~3 slot-hours.

    ---- 1. A' AT E=1: is exp2's A' column measuring anything real? ----

    exp2 reads A' at K=2 as 13.3x its centralized ceiling (53,200 against 4,000),
    consistent across all three seeds, with memorisation identical at 200. K=2 is
    where federation should be nearly a no-op, so either that is a real and very
    large E-effect or the harness is wrong -- and if the harness is wrong, A's
    1.00x agreement at the same K would be the coincidence, not this.

    E=1 settles it, because at E=1 FedAvg with n_k/n weighting is an EXACT
    algebraic identity with centralized GD -- proven in tests/test_fedavg_identity
    and observed in the wild (the T1 probe's iid and operand runs returned
    identical test accuracies seed-for-seed). So this cell MUST reproduce the
    ceiling.

    > DECISION RULE. If E=1/K=2 matches A's centralized ladder, the pipeline is
    > sound and A's 13x is a genuine E-effect -- most likely AdamW's optimiser
    > restart (persist_local_opt_state=False rebuilds Adam every round, so each
    > round is E cold-start bias-corrected steps) amplified by A' sitting AT its
    > cliff, margin +0.01 against +0.10 for every other setup. Both are testable
    > afterwards and neither invalidates exp2. If E=1 does NOT match, exp2's A'
    > column is not interpretable and the fault may not stop at A'.

    Budget 30,000 rounds. At E=1 one round IS one step, so rounds = steps and
    the Flower round-trip is paid 30,000 times -- this is the most
    orchestration-heavy cell per unit of learning in the campaign, which is
    exactly why E=1 was trimmed from the going-forward E_SPINE. A' groks
    centrally at 3,000 / 4,000 / 8,600 across its three seeds, so 30,000 is ~3.5x
    the slowest and the control cannot plausibly censor. A first draft of this
    manifest asked for 300,000 and would have cost 22 slot-hours to answer a
    question that needs three.

    ---- 2. wd=0 ON B, C AND E: is the inherited decay load-bearing? ----

    Setup D has the with/without control and it is decisive: it MEMORISES at
    epoch ~250 in every band from wd=0 to wd=1.0, and generalises only at 1.0.
    At wd=0 it sits at 1.2% test -- chance on S_5 is 0.83% -- with 100% train,
    indefinitely. A clean dose-response: 1.2 -> 1.3 -> 1.7 -> 52 -> 73 -> 96%.

    B, C and E have NO run at wd=0. Their decay bands started at 0.01. So "these
    setups need weight decay" is measured for D and inherited from Nanda and
    Omnigrok for the rest -- and this project has now been wrong about an
    inherited value three times (C's cliff, D's optimiser, B's band), each time
    in a way that survived into RESULTS before being caught.

    Budgets are each setup's own measured centralized requirement with headroom,
    so a failure here means the decay and not the clock.

    > DECISION RULE. If a setup groks at wd=0, its decay is not load-bearing and
    > every claim resting on the decay clock (14.3) needs re-examining for it. If
    > it memorises and sits at chance like D, the mechanism generalises and the
    > inherited values are vindicated on measurement rather than on citation.
    """
    specs = []
    # 1. the FedAvg identity control
    specs += expand_grid(
        {"mode": "federated", **SETUP_A_PRIME, "alpha": 0.20,
         "local_epochs": 1, "num_clients": 2, "partition": "iid",
         "strategy": "fedavg", "fraction_train": 1.0,
         "num_rounds": 30_000, "eval_every": 50},
        {"seed": SEEDS3},
        tags={"tier": "X", "group": "e1_identity", "experiment": "control",
              "setup": "A'"},
    )
    # 2. wd=0 on the three setups that have never been asked
    for label, base, wp, epochs in (
            ("B", SETUP_B, {"alpha": 0.30}, 100_000),
            ("C", SETUP_C, {"alpha": 0.40, "hidden_width": 256}, 200_000),
            ("E", {k: v for k, v in SETUP_E.items() if k != "batch_size"},
             {"n_train": 2000, "n_test": 5000, "batch_size": 100}, 40_000)):
        specs += expand_grid(
            {"mode": "centralized", **base, **wp, "weight_decay": 0.0,
             "epochs": epochs, "log_every": max(10, epochs // 500)},
            {"seed": SEEDS3},
            tags={"tier": "X", "group": "wd_zero", "experiment": "control",
                  "setup": label},
        )
    return specs


def t2_aggregation():
    """PART 1: exp2 -- does aggregation compensate for fragmenting the data?

    v1's exp2 ran three conditions per (alpha, K) and this is the same three:

        ceiling  centralized on the full training set
        floor    one model on ONE client's shard  (reduced_arm)
        FL       K clients, FedAvg, iid, full participation

    No v2 manifest has ever had the floor. reduced_arm has been written, tested
    and plumbed through TAG_KEYS/PREFERRED_COLUMNS for months with nothing
    calling it, so exp2 has been a two-arm subset.

    WHAT THE FLOOR IS AND IS NOT FOR. v1's headline was "FL groks in 23 of 30
    cells where the floor groks in 1". Most of that gap is just FL seeing K times
    more data than one shard -- it is not evidence about aggregation. So the
    headline comparison here is FL against the CEILING on the compute-matched
    step axis. The floor answers the narrower question a federated result is
    practically answering: could a client have done this alone? It is also the
    cheapest arm by a wide margin (centralized, 1/K of the data).

    DROPPED FROM v1's GRID: the alpha sweep. exp1 covers alpha centrally,
    t2_k_breakdown showed the alpha=0.3 plane uniformly safe, and K's cost is now
    known to be set by t_memo(K) + delay rather than by proximity to the cliff.
    Each setup sits at ITS OWN working point instead.

    BUDGETS ARE MEASURED, NOT ASSUMED. Every one exceeds the slowest banked
    first-crossing for that setup at the largest K it runs (RESULTS 14.3):

        setup  working pt        slowest measured   budget    headroom
        A      alpha 0.30            15,200          50,000     3.3x
        A'     alpha 0.20             5,500 (cent)  100,000    18x
        B      alpha 0.30, wd 1.0     7,500 (K=20)  100,000    13x
        C      alpha 0.40, w 256     56,900 (K=50)  200,000     3.5x
        D      alpha 0.30            95,600 (K=20)  250,000     2.6x
        E      n_train 2000          11,900 (K=20)   40,000     3.4x

    D sets the ceiling on cost: its K=20 cell used 95,600 of a 100,000 budget in
    the ladder -- 1.0x headroom, i.e. right at the edge -- and K=50 never
    memorised at all. 250,000 is the honest number for it.

    B RUNS AT ITS PUBLISHED wd=1.0, not the wd=0.1 that reopens its K axis. B's
    value is that it IS the Nanda replication, and 13.5 confirmed wd=1.0 is also
    its fastest band centrally. Its K=50 cell is expected to fail to memorise --
    that is the decay clock (14.3), it is understood, and it is a property of the
    setup rather than a defect in this manifest. The aggregation question at high
    K for B is a wd=0.1 follow-up, deliberately not folded in here.

    K=2 is main's lowest rung and is kept deliberately. It is the cheapest cell
    in the sweep, and it anchors the low end of the K curve closest to the
    centralized arm -- which is where cent_full and fl should agree if the
    compute-matched step axis is right. Disagreement at K=2 is a harness fault
    rather than a federated effect, so the rung doubles as a control.

    E stops at K=20: at n_train=2000 the shards are 1000/400/200/100 at
    K=2/5/10/20 against batch=100, so K=50 has no viable local epoch.

    > DECISION RULE. Per setup, compare FL against the CEILING on total_steps. If
    > FL tracks the ceiling as K grows, aggregation compensates for fragmentation
    > and the delay law is the whole story. If FL degrades toward the FLOOR, the
    > averaging is not recovering what fragmentation costs, and the K at which
    > that starts is the number the paper reports. The floor is the reference for
    > "could one client have done this alone", not for the aggregation claim.
    """
    # (label, base, working-point overrides, num_rounds, ceiling epochs, Ks)
    BLOCKS = [
        ("A",  {**{k: v for k, v in SETUP_A.items()
                   if k not in ("mode", "num_rounds", "eval_every")}},
         {"alpha": 0.30}, 10_000, 50_000, [2, 5, 10, 20, 50]),
        ("A'", SETUP_A_PRIME, {"alpha": 0.20}, 20_000, 100_000, [2, 5, 10, 20, 50]),
        ("B",  SETUP_B, {"alpha": 0.30}, 20_000, 100_000, [2, 5, 10, 20, 50]),
        ("C",  SETUP_C, {"alpha": 0.40, "hidden_width": 256},
         40_000, 200_000, [2, 5, 10, 20, 50]),
        ("D",  SETUP_D, {"alpha": 0.30}, 50_000, 250_000, [2, 5, 10, 20, 50]),
        ("E",  {k: v for k, v in SETUP_E.items() if k != "batch_size"},
         {"n_train": 2000, "n_test": 5000, "batch_size": 100},
         8_000, 40_000, [2, 5, 10, 20]),
    ]
    specs = []
    for label, base, wp, rounds, cent_epochs, Ks in BLOCKS:
        tags = {"tier": "T2", "group": "aggregation", "experiment": "exp2",
                "setup": label}
        fl = expand_grid(
            {"mode": "federated", **base, **wp, "local_epochs": 5,
             "partition": "iid", "strategy": "fedavg", "fraction_train": 1.0,
             "num_rounds": rounds, "eval_every": FL_EVAL_EVERY,
             "checkpoint_every": max(1, rounds // 10),
             "checkpoint_client_weights": True},
            {"num_clients": Ks, "seed": SEEDS3},
            tags={**tags, "arm": "fl"},
        )
        specs += fl
        # (b) the floor -- one model on one client's shard, 2x the FL arm's steps
        specs += [{**r, **tags, "arm": r["arm"], "reduced_from_k": r["reduced_from_k"]}
                  for r in reduced_arm(fl)]
        # (a) the ceiling -- centralized on the full training set. Dedupes against
        # Gate A wherever the working point already has banked runs.
        specs += expand_grid(
            {"mode": "centralized", **{k: v for k, v in base.items()
                                       if k not in _FED_ONLY},
             **wp, "epochs": cent_epochs, "log_every": max(10, cent_epochs // 500)},
            {"seed": SEEDS3},
            tags={**tags, "arm": "cent_full"},
        )
    return specs


def t2_aggregation_alpha2():
    """exp2's SECOND alpha per setup, so the slowdown-ratio figure gets two lines.

    main's exp2_slowdown_ratio.png carried four lines per panel, one per alpha in
    {0.25, 0.30, 0.35, 0.50}. v2's version (paper/exp2_slowdown_ratio_*.png)
    carries one, because t2_aggregation deliberately dropped the alpha sweep and
    put each setup at its own working point instead. This adds one more rung so
    every panel shows how the FL/centralized ratio moves with alpha.

    THE SECOND RUNG IS EASIER, ONE STEP ON EACH SETUP'S OWN LADDER. Harder is not
    available: A' is 0/5 at alpha=0.175 (13.3) and C is 1/3 at 0.30 and 0/3 at
    0.25 (14.1), so on two of six setups a harder rung would be a censored line.
    Stepping one rung rather than jumping to 0.50 keeps a usable baseline --
    measured centralized t_first_cross at the new rung:

        A   0.30 -> 0.40    8,800     B   0.30 -> 0.40    1,400
        A'  0.20 -> 0.30      300     C   0.40 -> 0.50   15,200
        D   0.30 -> 0.40   12,600     E   n_train 2000 -> 4000

    A' IS THE ONE TO READ CAREFULLY. At alpha=0.30 it crosses at 300 steps, so
    E=5 leaves federation 60 rounds to act -- 13.3's "there is no delay left to
    disrupt", and A' already reads 13.3x at K=2 from optimiser restart alone
    (15.3). Its second line is that fixed cost over a short baseline, not an
    alpha effect. There is no better rung: every easier alpha is faster still
    (0.25 -> 500, 0.40 -> 200) and there is no harder one.

    BUDGETS. Federated num_rounds is each setup's existing exp2 value, unchanged:
    they were sized for the HARDER working alpha, so they are generous here, and
    holding C's at 40,000 is what lets its cells dedup. Centralized epochs are
    exp2's too, except A' (5,000, matching aprime_alpha) and C (100,000, matching
    c_capacity) where a banked budget with ample headroom buys the arm for free.

    DEDUP, verified by exact config comparison rather than assumed: C's
    K in {5,10,20,50} at alpha=0.50 already exist in setup_k_ladder (R=40,000,
    E=5, w256, wd=1.0, eval_every=20, checkpoint_every=4,000, client weights on,
    seeds 42/123/456), and A' and C's centralized arms exist in aprime_alpha and
    c_capacity. 18 of 105 specs are already banked.

    ARMS. fl + cent_full only. The ratio is FL / cent_full and main's figure never
    used the floor, so reduced_arm is not built here.

    > DECISION RULE. Per setup, plot FL/cent_full against K at both alphas.
    > If the easier alpha's curve sits BELOW the working alpha's, the slowdown
    > grows as data gets scarce -- main's reading, and the delay law's prediction.
    > If the two curves coincide, the ratio is set by K alone and alpha only moves
    > the baseline. If the easier curve sits ABOVE, the ratio is dominated by a
    > fixed per-round cost divided by a shrinking baseline, which is what A' is
    > expected to show and what would make that panel uninterpretable.
    """
    # (label, base, working-point overrides, num_rounds, ceiling epochs, Ks)
    BLOCKS = [
        ("A",  {**{k: v for k, v in SETUP_A.items()
                   if k not in ("mode", "num_rounds", "eval_every")}},
         {"alpha": 0.40}, 10_000, 50_000, [2, 5, 10, 20, 50]),
        ("A'", SETUP_A_PRIME, {"alpha": 0.30}, 20_000, 5_000, [2, 5, 10, 20, 50]),
        ("B",  SETUP_B, {"alpha": 0.40}, 20_000, 100_000, [2, 5, 10, 20, 50]),
        ("C",  SETUP_C, {"alpha": 0.50, "hidden_width": 256},
         40_000, 100_000, [2, 5, 10, 20, 50]),
        ("D",  SETUP_D, {"alpha": 0.40}, 50_000, 250_000, [2, 5, 10, 20, 50]),
        ("E",  {k: v for k, v in SETUP_E.items() if k != "batch_size"},
         {"n_train": 4000, "n_test": 5000, "batch_size": 100},
         8_000, 40_000, [2, 5, 10, 20]),
    ]
    specs = []
    for label, base, wp, rounds, cent_epochs, Ks in BLOCKS:
        tags = {"tier": "T2", "group": "aggregation_alpha2", "experiment": "exp2",
                "setup": label}
        # `strategy` and `fraction_train` are OMITTED, not set to their defaults.
        # run_id hashes the raw spec, so a key present with its default value
        # hashes differently from the same key absent -- and t1_setup_k_ladder,
        # which already holds C's K in {5,10,20,50} at this alpha, omits both.
        # Writing them out would re-run 12 banked C cells for ~50 slot-hours and
        # produce bit-identical results. FedConfig defaults are strategy="fedavg"
        # and fraction_train=1.0, so the resolved config is unchanged.
        specs += expand_grid(
            {"mode": "federated", **base, **wp, "local_epochs": 5,
             "partition": "iid",
             "num_rounds": rounds, "eval_every": FL_EVAL_EVERY,
             "checkpoint_every": max(1, rounds // 10),
             "checkpoint_client_weights": True},
            {"num_clients": Ks, "seed": SEEDS3},
            tags={**tags, "arm": "fl"},
        )
        specs += expand_grid(
            {"mode": "centralized", **{k: v for k, v in base.items()
                                       if k not in _FED_ONLY},
             **wp, "epochs": cent_epochs, "log_every": max(10, cent_epochs // 500)},
            {"seed": SEEDS3},
            tags={**tags, "arm": "cent_full"},
        )
    return specs


def p0_c_alpha_width256():
    """PART 0.2: C's alpha ladder at the capacity it will actually use.

    THE ERROR THIS CORRECTS. "Setup C's working alpha is >= 0.5" has been carried
    since Gate A and used to argue C cannot be matched to the other setups. It
    rests on one cell: alpha=0.30, 0/5, at hidden_width 128 with 40,000 epochs.

    But at that SAME width 128, alpha=0.50 has a KM median of 39,800 (12/12 given
    100,000 epochs, s5_setup_c_capacity). So the ladder was cut off essentially
    AT THE MEDIAN FOR AN EASIER ALPHA. A harder alpha censoring under that budget
    is the expected outcome, not evidence of a cliff. It is the same
    budget-manufactured boundary this project has now produced seven times, and
    it went into RESULTS as a setup property.

    C has NEVER been run below alpha=0.5 at width 256 -- every width-256 run in
    the corpus is alpha=0.50. And 256 is the capacity C will use: the capacity
    sweep halved its T_grok there (22,500 against 39,800 at 12/12 each).

    BUDGET. 100,000 epochs, not the 40,000 of the original ladder. C's
    first-crossing median at width 256 / alpha=0.50 is ~16,100, so 100k is ~6x
    the easiest rung's requirement and leaves room for the low rungs to be slow
    rather than censored. Re-using 40,000 here would reproduce the exact defect
    this manifest exists to correct.

    log_every=50 because C is the unstable setup -- it dips back below the bar
    after crossing it, up to 28 times at wd=0.3 -- so t_grok on C is partly a
    measure of the logging rate (RESULTS 13.4). Read `t_first_cross` here.

    > DECISION RULE. If C reaches 5/5 (or 3/3) at alpha=0.30 with T_grok below a
    > third of budget, C joins the campaign at the SAME working point as B and D
    > and the "C cannot be matched" caveat is withdrawn. If its cliff really does
    > sit above 0.4 even at width 256, C stays an existence proof at its own
    > alpha and the caveat stands -- but then it is a measured caveat rather than
    > an inherited one.
    """
    return expand_grid(
        {"mode": "centralized", **SETUP_C, "hidden_width": 256,
         "epochs": 100_000, "log_every": 50},
        {"alpha": [0.25, 0.30, 0.40, 0.50, 0.60], "seed": SEEDS3},
        tags={"tier": "P0", "group": "c_alpha_w256", "experiment": "boundary",
              "setup": "C"},
    )


def p0_capacity():
    """PART 0.3: exp0's analogue -- is each setup's width sufficient?

    v1's exp0 asked one question before anything else ran: does the chosen width
    actually support grokking across the alpha range, or is a null result just an
    underparameterised model? Only setup C has ever been asked it here
    (s5_setup_c_capacity), and the answer moved a real number -- width 256 halved
    C's T_grok against 128, which is why C runs at 256 now.

    Every other setup's width is inherited: A and D at 256 from Gromov, B at 128
    from Nanda, E at 200 from Omnigrok. Those are published values and probably
    fine, but "probably fine" is what was said about C's decay and about D's
    optimiser, and both turned out to be worth measuring.

    Half / default / double per setup, at its own working alpha, centralized:

        A'  128 / 256 / 512      alpha 0.20   (its only alpha with a real delay)
        B    64 / 128 / 256      alpha 0.30
        D   128 / 256 / 512      alpha 0.30
        E   100 / 200 / 400      n_train 2000, batch 100

    Budgets are ~5x each setup's measured centralized T_grok, so a narrow model
    failing means capacity and not the clock. B carries the widest margin because
    its seed variance is bimodal (4,400 to 20,400) and a halved width will be
    slower still.

    > DECISION RULE. If the default width is within noise of double, capacity is
    > not binding and the campaign proceeds at the inherited value. If double is
    > materially faster -- as it was for C -- that setup moves to double before
    > any federated cell is spent on it, because every federated budget is a
    > multiple of the centralized requirement and would otherwise be set from a
    > handicapped baseline.
    """
    specs = []
    specs += expand_grid(
        {"mode": "centralized", **SETUP_A_PRIME, "alpha": 0.20,
         "epochs": 50_000, "log_every": 50},
        {"hidden_width": [128, 256, 512], "seed": SEEDS3},
        tags={"tier": "P0", "group": "capacity", "experiment": "width",
              "setup": "A'"},
    )
    specs += expand_grid(
        {"mode": "centralized", **SETUP_B, "alpha": 0.30,
         "epochs": 100_000, "log_every": 50},
        {"hidden_width": [64, 128, 256], "seed": SEEDS3},
        tags={"tier": "P0", "group": "capacity", "experiment": "width",
              "setup": "B"},
    )
    specs += expand_grid(
        {"mode": "centralized", **SETUP_D, "alpha": 0.30,
         "epochs": 100_000, "log_every": 50},
        {"hidden_width": [128, 256, 512], "seed": SEEDS3},
        tags={"tier": "P0", "group": "capacity", "experiment": "width",
              "setup": "D"},
    )
    specs += expand_grid(
        {"mode": "centralized", **{k: v for k, v in SETUP_E.items()
                                   if k != "hidden_width"},
         "n_train": 2000, "n_test": 5000, "batch_size": 100,
         "epochs": 20_000, "log_every": 25},
        {"hidden_width": [100, 200, 400], "seed": SEEDS3},
        tags={"tier": "P0", "group": "capacity", "experiment": "width",
              "setup": "E"},
    )
    return specs


def t1_setup_k_ladder():
    """TIER 1: the K ladder on every setup, measuring BOTH timescales.

    THE LIMITATION THIS EXISTS TO CLOSE. Every federated result in the project
    uses one setup -- the quadratic MLP on modular addition. RESULTS 10 states
    that as the headline caveat, and nothing measured so far tests whether the
    delay law or the partition-structure effect is a property of grokking or a
    property of that one architecture.

    WHY A LADDER AND NOT A REPLICATION. The obvious version of this manifest
    re-runs the anchor's cells on the new setups and compares T_grok. That would
    have been the wrong experiment, because the two setups measured so far
    decompose in OPPOSITE ways, and T_grok alone hides it:

        setup A (GD, wd=0), alpha=0.30, iid, E=5
            t_memo   3,700  3,700  3,700  3,700     K = 5, 10, 20, 50
            delay    9,500  9,700 10,000 11,500     <- delay carries the K effect

        setup B (AdamW, wd=0.1), alpha=0.30, iid, E=5
            t_memo   --     --     3,600  53,300    K = 20, 50
            delay    --     --    ~90,000    ?      <- memorisation carries it

    A's memorisation is flat in K and its delay grows; B's memorisation explodes
    and its delay may not move at all. Same axis, same statistic, opposite
    mechanism -- and a table of T_grok values shows one number for both. The
    plausible reason is the decay clock: A runs wd=0 and has none, so shard size
    does not fight memorisation; B's decoupled decay is applied per local step
    and is independent of shard size, so smaller shards lose the race. That
    predicts D (AdamW) resembles B and that anything at wd=0 resembles A.

    So the measurement is t_memo AND delay per K per setup, not T_grok per K.

    DESIGN. K in {5, 10, 20, 50} is the anchor's own ladder (t2_k_breakdown), so
    every cell here has a same-K, same-E, same-partition counterpart on A already
    banked. iid only: partition structure is a separate axis and mixing it in
    would leave the K effect and the structure effect confounded in a first look.

    Each setup sits at ITS OWN working point (RESULTS 11 and 13), never a shared
    alpha -- setup C's cliff is above 0.5 while everyone else's is near 0.20, and
    a shared alpha would put C past its cliff and call it a federated effect.

    BUDGETS are set as t_memo(K) + delay with headroom ABOVE the estimate rather
    than at it, per the rule the K-collapse investigation produced. They are not
    multiples of the centralized T_grok: that is the error that manufactured six
    boundaries in this project, most recently the wd=0.1 arm censored at 10,000
    steps against a 45,050 requirement.

    B carries only its missing rungs. Its K = 20, 30, 50 cells at wd=0.1 are
    already banked at long budget (p1_k_collapse_budget); K = 5 and 10 complete
    the memo(K) curve at its low end, which is where a blow-up would first be
    distinguishable from a constant.

    E (MNIST) stops at K=20, and its last rung is degenerate ON PURPOSE.
    (n_train=2000, batch=100) is the working point with both a real delay and
    more than one batch per local epoch, but that holds only to K=10: the shards
    are 400 / 200 / 100 at K = 5 / 10 / 20, so K=20 is exactly one full-batch
    step per local epoch and `local_epochs` stops meaning what it means on every
    other setup. validate_manifest flags it, correctly. It is kept as the control
    that shows what degeneracy looks like -- the same convention s5_mnist_fl
    uses -- and the ladder cannot extend past it, because shrinking batch_size
    with K would confound the K axis with a change in the effective batch, which
    is the very thing MNIST's delay depends on.

    > DECISION RULE. Read t_memo(K) and delay(K) per setup, not T_grok(K).
    > If the AdamW setups (B, C, D) all show memorisation carrying the K effect
    > while the wd=0 setups (A) show the delay carrying it, the decay clock is
    > the mechanism, per-setup budgets follow from which term dominates, and the
    > campaign's high-K cells are budgeted from the dominant term. If the
    > decomposition does not sort by decay, the two setups measured so far are
    > coincidence and the K axis needs a per-setup budget measured empirically
    > before any comparison across setups is meaningful.
    """
    specs = []

    # D -- quad-MLP on S_5, working alpha 0.30 (T_grok 21,300 centralized).
    # 20,000 rounds = 100,000 steps, ~4.7x centralized. D is AdamW at wd=1.0, so
    # if it follows B its memorisation is what runs out first at high K; the
    # headroom is there to see that rather than censor it.
    specs += expand_grid(
        {"mode": "federated", **SETUP_D, "alpha": 0.30, "local_epochs": 5,
         "partition": "iid", "num_rounds": 20_000, "eval_every": FL_EVAL_EVERY,
         "checkpoint_every": 2_000, "checkpoint_client_weights": True},
        {"num_clients": [5, 10, 20, 50], "seed": SEEDS3},
        tags={"tier": "T1", "group": "setup_k_ladder", "experiment": "replication",
              "setup": "D"},
    )

    # C -- transformer on S_5. alpha=0.50 and width 256: its cliff sits far above
    # the others' and the capacity sweep halved its T_grok at 256. 40,000 rounds
    # = 200,000 steps, double everyone else, because C needed >=100k epochs
    # CENTRALLY and is the setup whose Gate A verdict was itself censoring.
    specs += expand_grid(
        {"mode": "federated", **SETUP_C, "hidden_width": 256, "alpha": 0.50,
         "local_epochs": 5, "partition": "iid", "num_rounds": 40_000,
         "eval_every": FL_EVAL_EVERY,
         "checkpoint_every": 4_000, "checkpoint_client_weights": True},
        {"num_clients": [5, 10, 20, 50], "seed": SEEDS3},
        tags={"tier": "T1", "group": "setup_k_ladder", "experiment": "replication",
              "setup": "C"},
    )

    # B -- only the rungs the K-collapse work did not already buy, at the decay
    # that trains at every K (wd=0.1). K=20/30/50 are banked at 100k-200k steps.
    specs += expand_grid(
        {"mode": "federated", **SETUP_B, "weight_decay": 0.1, "alpha": 0.30,
         "local_epochs": 5, "partition": "iid", "num_rounds": 20_000,
         "eval_every": FL_EVAL_EVERY,
         "checkpoint_every": 2_000, "checkpoint_client_weights": True},
        {"num_clients": [5, 10], "seed": SEEDS3},
        tags={"tier": "T1", "group": "setup_k_ladder", "experiment": "replication",
              "setup": "B"},
    )

    # E -- MNIST. batch_size FIXED across the ladder: letting it shrink with K to
    # keep shards viable would confound the K axis with a change in the
    # optimiser's effective batch, and the delay depends on exactly that.
    specs += expand_grid(
        {"mode": "federated", **{k: v for k, v in SETUP_E.items()
                                 if k != "batch_size"},
         "n_train": 2000, "n_test": 5000, "batch_size": 100, "local_epochs": 5,
         "partition": "iid", "num_rounds": 4_000, "eval_every": FL_EVAL_EVERY,
         "checkpoint_every": 400, "checkpoint_client_weights": True},
        {"num_clients": [5, 10, 20], "seed": SEEDS3},
        tags={"tier": "T1", "group": "setup_k_ladder", "experiment": "replication",
              "setup": "E"},
    )
    return specs


def p1_b_decay_band():
    """PHASE 1: B's decay band -- the control the K-collapse diagnosis is missing.

    THE GAP. p1_cd_decay_band measured C and D. It did not measure B, and B is
    the setup carrying the K>=30 collapse diagnosis: p1_k_collapse_wd runs B at
    wd=0.1 against a wd=1.0 control, and EVERY centralized B run in the corpus
    is wd=1.0. So the arm that is supposed to reopen the K axis is being read
    against a reference that does not exist.

    WHY THAT MATTERS RIGHT NOW. At K=20 and K=30 the wd=0.1 cells memorise
    cleanly -- 100% train by epoch 3,500-3,900, 3/3 -- and then sit at 0.2-0.5%
    TEST for the rest of the run. Against chance (1/113 = 0.88%) that is not
    "generalising slowly", it is not generalising at all. Read next to the
    wd=1.0 control, where one K=20 seed groks at 7,500, it looks like a
    federated effect and it would be the headline of the whole campaign.

    But the runs got 2,000 rounds x E=5 = 10,000 steps, and the only centralized
    number available to size that against is B at wd=1.0 (T_grok 6,600 at
    alpha=0.30, worst seed 20,400). Lower decay means a LONGER delay in this
    regime -- t0_mnist_wd_band is the direct precedent, where lr*lambda=1e-5
    never generalised inside 20,000 epochs while 1e-4 took 3,800 -- so 10,000
    steps at wd=0.1 is plausibly under-budget before federation is invoked at
    all. This project has manufactured a boundary out of a budget four times
    (v1's headline claim, the E=1 probe cells, the first FL probe, setup C's
    Gate A verdict). This is the check that stops the fifth.

    THE SWEEP. Deliberately identical in shape to p1_cd_decay_band -- same
    5-point log-spaced lr*lambda ladder, same 3 seeds -- so the three AdamW
    setups' bands are directly comparable:

        wd     0.01    0.03    0.1     0.3     1.0
        lr*wd  1e-5    3e-5    1e-4    3e-4    1e-3  (the inherited value)

    alpha=0.30 is B's Gate A working point. 50,000 epochs is ~7.5x B's WORST
    observed seed rather than a multiple of its median, because B's seed
    variance is intrinsic and bimodal (4,400 / 6,100 / 6,600 / 19,500 / 20,400)
    -- sizing off the median would censor the slow cluster and hand back exactly
    the ambiguity this sweep exists to remove.

    > DECISION RULE. Read B's centralized T_grok at whichever band the federated
    > arm uses. If it is comfortably under 10,000 steps, those federated cells
    > were given enough time and "memorises at K=20/30, never generalises" is a
    > FEDERATED breakdown of grokking with memorisation intact -- the mechanism
    > this project has been looking for, with the per-client checkpoints as its
    > evidence base. If it is above 10,000, the cells were censored by the clock
    > and say nothing yet; re-run them at 5x the measured requirement before
    > claiming anything. Either way this gates the K axis for B/C/D/E, and with
    > it exp2/exp3/exp4, whose primary axis is K.
    """
    return expand_grid(
        {"mode": "centralized", **SETUP_B, "alpha": 0.30,
         "epochs": 50_000, "log_every": 50},
        {"weight_decay": [0.01, 0.03, 0.1, 0.3, 1.0], "seed": SEEDS3},
        tags={"tier": "P1", "group": "b_decay_band", "experiment": "decay",
              "setup": "B"},
    )


def p1_k_collapse_budget():
    """PHASE 1: the K-collapse arm again, at a budget keyed to a MEASURED number.

    WHAT p1_b_decay_band CHANGED. That sweep supplied the centralized reference
    the collapse diagnosis never had. Setup B, alpha=0.30, 3 seeds, memorising at
    epoch 150 in every cell:

        lr*wd    1e-5    3e-5    1e-4      3e-4     1e-3 (inherited)
        T_grok   never   never   45,050    20,000   6,100

    Every cell memorises at epoch 150, so decay does nothing to memorisation here
    and the whole effect is on the generalisation timescale. wd=1.0 is optimal,
    which vindicates the inherited value a third time after C and D.

    So weight decay ACCELERATES grokking on B, monotonically -- the AdamW pattern
    RESULTS 6.4 predicts, now measured on a third setup. And the federated wd=0.1
    cells were given 2,000 rounds x E=5 = 10,000 steps against a centralized
    requirement of 45,050 (KM median, 3/3). They were censored by the clock. "Memorises at
    K=20/30 and never generalises" is not evidence of anything federated yet;
    it is the fifth budget-manufactured boundary in this project, caught before
    it was claimed rather than after.

    THE BUDGET. Not the "5x measured requirement" rule of thumb: 5 x 45,050 =
    225,000 steps, which at K=50 on a transformer is 15-20 h/run and not
    affordable at 3 seeds. 100,000 steps is 2.2x the centralized requirement and
    matches the t2_boundary precedent (100k for a setup needing ~47k at K=50).

    WHY K=20 GETS DOUBLE. There is one honest worry about 100,000 being enough.
    Federated memorisation on this setup is ~23x slower than centralized (t_memo
    150 -> 3,500 at K=20), and if the generalisation timescale stretched by
    anything like that factor no affordable budget would reach it. The anchor
    says otherwise -- federation costs only +9% to +21% on T_grok at K=20-50 --
    but that is a different setup, and betting the answer on it is the mistake
    this manifest exists to stop repeating. So the CHEAPEST cell carries the
    DEEPEST budget: K=20 gets 200,000 steps (4.4x). If wd=0.1 is merely slow
    federated, K=20 is where that shows up, and it costs the least to find out.

    THE wd=1.0 CONTROL, at 1 seed. Its documented failure is that the model never
    memorises (peak ~= final, 3-5% train), which no budget fixes -- but that was
    read off 10,000-step runs too, and asserting "more budget will not help" is
    the mirror image of the error above. Two runs settle it.

    > DECISION RULE. If wd=0.1 groks at K=20 within 200,000 steps, the collapse
    > is an inherited-hyperparameter defect: the K axis reopens for B/C/D/E at a
    > corrected decay, exp2/exp3/exp4 can be specified on their primary axis, and
    > D' is not needed. Read K=30 and K=50 for how the requirement scales with K
    > -- that scaling IS the federated cost, and it is the number the campaign
    > budgets come from. If wd=0.1 memorises and does not generalise at K=20 with
    > 4.4x the centralized requirement, that is a federated breakdown of grokking
    > with memorisation intact, and it is the mechanism the project has been
    > looking for rather than a nuisance.
    """
    base = {**SETUP_B, "mode": "federated", "alpha": 0.30, "local_epochs": 5,
            "partition": "iid", "eval_every": FL_EVAL_EVERY,
            # per-client weights ON: if this turns out to BE the breakdown, the
            # evidence base has to already exist. checkpoint_* are Config fields,
            # so adding them afterwards changes the ids and re-runs the sweep.
            "checkpoint_every": 2_000, "checkpoint_client_weights": True}
    specs = []
    # The decisive cell: cheapest K, deepest budget (200,000 steps).
    specs += expand_grid(
        {**base, "weight_decay": 0.1, "num_clients": 20, "num_rounds": 40_000},
        {"seed": SEEDS3},
        tags={"tier": "P1", "group": "k_collapse_budget",
              "experiment": "diagnosis", "setup": "B"},
    )
    # How the requirement scales with K (100,000 steps).
    specs += expand_grid(
        {**base, "weight_decay": 0.1, "num_rounds": 20_000},
        {"num_clients": [30, 50], "seed": SEEDS3},
        tags={"tier": "P1", "group": "k_collapse_budget",
              "experiment": "diagnosis", "setup": "B"},
    )
    # The control: is "never memorises" really budget-independent?
    specs += expand_grid(
        {**base, "weight_decay": 1.0, "num_rounds": 20_000},
        {"num_clients": [30, 50], "seed": [42]},
        tags={"tier": "P1", "group": "k_collapse_budget",
              "experiment": "diagnosis", "setup": "B"},
    )
    return specs


def p1_aprime_alpha():
    """PHASE 1: setup A' -- A's architecture and task under AdamW. The alpha ladder.

    A' is the cell that closes the 2x2 (see SETUP_A_PRIME). It needs the same
    thing every other setup needed before it could be run federated: its own
    cliff and its own working point.

    THE GRID SITS LOW, AND THE RESOLUTION SCALES, because a pilot showed A' is a
    different animal from A at the same alpha. At alpha=0.30, one seed:

        A  (GD)     memorise ~2,000    grok 13,100    delay ~11,000
        A' (AdamW)  memorise    100    grok    300    delay      200

    ~40x faster with a ~55x smaller delay -- the phenomenon is barely present at
    the alpha where A shows it most cleanly. Gridding A' on A's rungs would spend
    most of the ladder in a regime with nothing to measure, and at log_every=50 a
    200-epoch delay is four points wide. So the rungs move down and each block
    gets a resolution matched to its own expected T_grok.

    THE BUDGET IS SPLIT for the opposite reason at the other end. T_grok diverges
    as (alpha - alpha_c)^-gamma, so A jumps 25,300 -> 805,000 across a single rung
    near alpha_c ~ 0.198. A flat budget is therefore guaranteed to censor the low
    rungs and report a cliff that is really the clock -- the mistake that has now
    cost this project its v1 headline claim, the E=1 probe cells, the first FL
    probe, and setup C's Gate A verdict. The low block gets 300,000 epochs.

    alpha goes down to 0.125, below A's cliff: if AdamW is this much more
    data-efficient, A' may have a LOWER alpha_c, and a ladder that stops at A's
    cliff could not see it. If 0.125 censors at 300k that is a bound, not a
    boundary, and gets reported as one.

    > DECISION RULE. A' is ready when it has a fitted T(alpha) and a working alpha
    > whose T_grok leaves >=3x headroom in an affordable FL budget AND whose delay
    > is large enough to measure a federated effect against -- a setup with a
    > 200-epoch delay cannot show a 10% federated slowdown. Its cliff is then
    > compared to A's alpha_c ~ 0.198: if they coincide, the data threshold is a
    > property of the TASK and the optimiser only sets the clock, which is the
    > cleanest possible statement of what alpha does. If they differ, the
    > threshold is optimiser-dependent and every cross-setup alpha comparison in
    > the study needs that caveat.
    """
    # Three blocks, each with log_every ~ 1/30th of its expected T_grok so the
    # memorise->generalise gap is resolved rather than smeared.
    blocks = [
        # alpha,                        epochs,  log_every
        ([0.30, 0.40, 0.50],             5_000,   10),
        ([0.225, 0.25],                 50_000,   25),
        ([0.125, 0.15, 0.175, 0.20],   300_000,  100),
    ]
    specs = []
    for alphas, epochs, log_every in blocks:
        specs += expand_grid(
            {"mode": "centralized", **SETUP_A_PRIME, "epochs": epochs,
             "log_every": log_every},
            {"alpha": alphas, "seed": SEEDS5},
            tags={"tier": "P1", "group": "aprime_alpha", "experiment": "ladder",
                  "setup": "A'"},
        )
    return specs


def p1_k_collapse_wd():
    """PHASE 1: is the K>=30 collapse a decay artifact? The federated test.

    THE OBSERVATION. Setup B, one seed per cell, default hyperparameters -- peak
    train accuracy against client count:

        K        10     20     30     40     50
        peak    100.0   98.2   42.8    5.9    5.0

    A smooth degradation, not a cliff, and it is a TRAINING failure: these models
    never memorise, so no budget fixes them. Setup A under plain GD at wd=0 groks
    5/5 at K=50 and is the only setup with no decay clock at all.

    WHY DECAY. Ruled out already: weight-norm collapse, client drift, and local
    step size (lower lr is WORSE -- 4.6% train at lr=1e-4). The one knob that
    moves it is weight decay, at one seed: at K=50, wd=1.0 gives ~3.6% train and
    wd=0.1 gives 70.2%. The mechanism is plausible -- decoupled decay is applied
    per local step and is independent of shard size, while the learning signal
    from a 77-sample shard is not, so the balance between them shifts with K even
    though tau = 1/(lr*wd) does not.

    WHY THIS AND NOT THE CENTRALIZED BAND SWEEP. p1_cd_decay_band measures decay
    with ONE client. It cannot show that a better band restores training at K=50,
    because the quantity that breaks is the ratio between decay and a per-shard
    gradient, and that ratio has no centralized analogue. This has to be measured
    federated.

    Setup B carries it: it is the setup with the existing K ladder, so the new
    cells drop straight onto measured points. The wd=1.0 arm re-uses banked ids
    where the seeds coincide -- run ids are content hashes, so the launcher skips
    them and the control arm is nearly free.

    > DECISION RULE. If wd=0.1 restores training at K=50 (t_memo finite, 3/3),
    > this is an inherited-hyperparameter defect: C and D adopt their measured
    > bands, the K axis reopens to K=97, and the campaign proceeds as designed.
    > If it does not, the collapse is a genuine federated breakdown mechanism on
    > adaptive optimisers -- which is a headline rather than a nuisance, and the
    > per-client checkpoints become the evidence base for explaining it.
    """
    return expand_grid(
        {**SETUP_B, "mode": "federated", "alpha": 0.30, "local_epochs": 5,
         "partition": "iid", "num_rounds": 2_000, "eval_every": FL_EVAL_EVERY},
        {"num_clients": [20, 30, 50], "weight_decay": [0.1, 1.0],
         "seed": SEEDS3},
        tags={"tier": "P1", "group": "k_collapse_wd", "experiment": "diagnosis",
              "setup": "B"},
    )


def s5_setup_c_capacity():
    """GATE A follow-up: does setup C fail because of alpha, or because of capacity?

    C (transformer on S5) never groks reliably: 4/5 at alpha=0.5, 3/5 at 0.4, 0/5
    below. Two explanations, and they lead to opposite decisions.

    (a) THE CLIFF IS SIMPLY HIGHER. S5 is a harder function than modular addition
        -- 120 classes, non-abelian -- so it may just need more data. Tested by
        extending the ladder to alpha 0.6/0.7.

    (b) THE MODEL IS TOO SMALL. d_model=128 against 120 output classes is thin,
        and the suspicious part is that the *quadratic MLP* at width 256 handles
        the same task cleanly (setup D: 5/5 down to alpha=0.25). A transformer
        losing to a 2-layer MLP on the canonical grokking benchmark is more
        likely a capacity or tuning artifact than a fact about transformers.
        Tested by sweeping n_heads / d_mlp / hidden_width -- newly possible, since
        those only became Config fields in this session; a manifest setting them
        used to raise in build_config.

    100k epochs throughout: C's slowest observed grok is 33,100, so a censored
    cell here means "past the cliff", not "past the clock".

    Decision rule: C stays in the campaign if some configuration reaches 5/5 at a
    workable alpha with T_grok below a third of budget. Otherwise it is dropped
    and the campaign runs A/B/D/E -- which still separates architecture (B vs A)
    from task (D vs A); C is the interpolation cell, not a load-bearing one.
    """
    specs = expand_grid(
        {"mode": "centralized", **SETUP_C, "epochs": 100_000, "log_every": 200},
        {"alpha": [0.6, 0.7], "seed": SEEDS5},
        tags={"tier": "S1b", "group": "c_alpha", "experiment": "gate_a", "setup": "C"},
    )
    for width in (128, 256):
        specs += expand_grid(
            {"mode": "centralized", **SETUP_C, "hidden_width": width,
             "alpha": 0.5, "epochs": 100_000, "log_every": 200},
            {"n_heads": [4, 8], "d_mlp": [512, 1024], "seed": SEEDS3},
            tags={"tier": "S1b", "group": "c_capacity", "experiment": "gate_a",
                  "setup": "C"},
        )
    return specs


def s5_mnist_fl():
    """GATE A follow-up: federated MNIST at a config that actually groks.

    The first probe ran MNIST at (n_train=4000, batch=200) -- which the
    working-point sweep then showed has NO DELAY at all centrally: T_grok 500
    against memorisation at 500. So the probe measured federation on a setup that
    was not grokking to begin with, and its censoring says nothing about
    federation.

    The working-point sweep's finding is that delay and shardability oppose each
    other. A large memorise->generalise gap needs a large batch; a large batch on
    a shard of n_train/K samples degenerates to a single full-batch step, at which
    point `local_epochs` stops meaning what it means on every other setup. The
    two cells that have both a real delay and >= 2 batches per local epoch:

        (2000, 50)   delay 200,  8/4/2 batches at K = 5/10/20
        (2000, 100)  delay 500,  4/2/1 batches at K = 5/10/20

    Both are swept. batch_size is held FIXED across the K sweep within each arm --
    varying it with K to keep shards viable would confound the K axis with a
    change in the optimiser's effective batch, which is the very thing the delay
    depends on. The (100, K=20) cell is therefore degenerate by construction and
    is kept deliberately, as the control that shows what degeneracy looks like.

    4,000 rounds x E=5 = 20,000 steps, ~25x the centralized requirement.
    """
    return expand_grid(
        {"mode": "federated", **{k: v for k, v in SETUP_E.items() if k != "batch_size"},
         "n_train": 2000, "n_test": 5000, "num_rounds": 4_000, "local_epochs": 5,
         "eval_every": FL_EVAL_EVERY},
        {"batch_size": [50, 100], "num_clients": [5, 10, 20],
         "partition": ["iid", "label_block"], "seed": SEEDS3},
        tags={"tier": "S2b", "group": "mnist_fl", "experiment": "gate_a", "setup": "E"},
    )


def s5_probe_rerun():
    """GATE A follow-up: the probe cells that genuinely ran out of budget.

    Deliberately narrow. Of the probe's censored cells, only these three had
    MEMORISED and were waiting to generalise -- the rest sit at 1-5% train
    accuracy, which no amount of extra budget fixes (see s5_k50_diagnosis).

        D quad/S5  K=50 iid      99% train, 77-82% test against an 85 bar
        D quad/S5  K=10 operand  95% train, 13% test
        B tfmr/mod K=10 operand  50-96% train, 8-91% test -- one seed nearly made it

    10,000 rounds x E=5 = 50,000 steps, 5x the original probe and >2x each
    setup's centralized T_grok, so a cell that still censors here is reporting
    federation rather than the clock.
    """
    specs = expand_grid(
        {"mode": "federated", **SETUP_D, "alpha": 0.5, "num_clients": 50,
         "partition": "iid", "num_rounds": 10_000, "local_epochs": 5,
         "eval_every": FL_EVAL_EVERY},
        {"seed": SEEDS3},
        tags={"tier": "S2b", "group": "probe_rerun", "experiment": "gate_a", "setup": "D"},
    )
    specs += expand_grid(
        {"mode": "federated", **SETUP_D, "alpha": 0.5, "num_clients": 10,
         "partition": "operand", "num_rounds": 10_000, "local_epochs": 5,
         "eval_every": FL_EVAL_EVERY},
        {"seed": SEEDS3},
        tags={"tier": "S2b", "group": "probe_rerun", "experiment": "gate_a", "setup": "D"},
    )
    specs += expand_grid(
        {"mode": "federated", **SETUP_B, "alpha": 0.3, "num_clients": 10,
         "partition": "operand", "num_rounds": 10_000, "local_epochs": 5,
         "eval_every": FL_EVAL_EVERY},
        {"seed": SEEDS3},
        tags={"tier": "S2b", "group": "probe_rerun", "experiment": "gate_a", "setup": "B"},
    )
    return specs


def s5_k50_diagnosis():
    """GATE A follow-up: why does every AdamW setup fail to TRAIN at K=50?

    B, C and D all sit at 1-5% train accuracy at K=50 -- not memorised-but-not-
    generalising, but barely learning -- while setup A under plain GD groks 5/5 at
    the same K. So it is not client count by itself, and it is not budget: a model
    at 4% train after 10,000 steps, whose centralized twin reaches 100% train by
    epoch 200, is not short of steps.

    Two mechanisms already ruled out from the logged history:
      * weight-norm collapse (AdamW's decay is data-independent, so it survives
        averaging intact while gradient terms partially cancel) -- norms are flat
        or GROWING; D K=50 operand reaches a norm comparable to the grokking K=10
        run while sitting at 2% train;
      * client drift -- elevated but not decisive, and the failing D K=50 operand
        drifts LESS than the working D K=50 iid.

    So this sweep asks the two cheap remaining questions:

      LOCAL STEP SIZE. lr and wd are tuned for full-batch centralized training. A
      K=50 client holds ~76 samples and takes 5 full-batch steps on them; the same
      lr may simply be far too large at that shard size, so 50 clients overfit in
      opposite directions and the average is noise.

      WHERE IT BREAKS. K=10 works and K=50 does not. A ladder between them says
      whether this is a sharp transition (interesting) or a smooth degradation
      (a tuning gradient).

    Decision rule: if a lower local lr recovers training, this is a
    hyperparameter mismatch and Stage 3 must tune per-K or move adaptivity
    server-side (FedAdam/FedYogi are already implemented). If nothing recovers it,
    it is a candidate breakdown mechanism and becomes a headline rather than a
    nuisance -- with the per-client signature checkpoints as the evidence base.
    """
    cell = {"mode": "federated", **SETUP_B, "alpha": 0.3, "partition": "iid",
            "num_rounds": 2_000, "local_epochs": 5, "eval_every": FL_EVAL_EVERY}
    specs = expand_grid(
        {**cell, "num_clients": 50},
        {"lr": [1e-4, 3e-4, 1e-3], "weight_decay": [0.1, 1.0], "seed": [42]},
        tags={"tier": "S2b", "group": "k50_hparam", "experiment": "gate_a", "setup": "B"},
    )
    specs += expand_grid(
        cell,
        {"num_clients": [20, 30, 40], "seed": [42]},
        tags={"tier": "S2b", "group": "k50_ladder", "experiment": "gate_a", "setup": "B"},
    )
    return specs


def estimate_minutes(spec):
    """Rough wall-clock for one spec, used only to order a manifest.

    `launch_sweep.py` fills slots strictly FIFO in manifest order, so a long run
    that happens to sit last starts last and tails the whole sweep behind it.
    Emitting longest-first fills the short runs in behind the long ones instead;
    on t2_boundary that was worth ~18% (7.2h -> 5.9h).

    These constants are deliberately crude -- ordering only needs the ranking to
    be right, not the magnitude. Measured on an L4, 2026-07-29. Stage 2 replaces
    them with a per-setup fit; until then a transformer is charged ~3x a GrokNet
    because wall-clock here is ~99% weight-shipping and it ships ~3x the payload.
    """
    # The two modes are bottlenecked by DIFFERENT things, so they need different
    # model multipliers. Federated wall-clock is ~99% Flower/Ray weight-shipping,
    # which scales with parameter count: the transformer's 226k params against
    # GrokNet's 74k is the ~3x. Centralized wall-clock is per-step compute, where
    # the transformer's attention and MLP make it ~9x GrokNet -- measured 8.4
    # ms/step at p=113 against 1.3 ms/step. Using one multiplier for both
    # under-costs the centralized transformer threefold.
    if spec.get("mode") == "federated":
        payload_cost = {"transformer": 3.0, "mlp": 2.7}.get(spec.get("model"), 1.0)
        K = spec.get("num_clients", 5)
        E = spec.get("local_epochs", 5)
        rounds = spec.get("num_rounds", 10_000)
        return (9.8 + 1.291 * K + 0.418 * E) * rounds / 10_000 * payload_cost
    model_cost = {"transformer": 9.0, "mlp": 1.0}.get(spec.get("model"), 1.4)

    # Centralized cost tracks GRADIENT STEPS, not epochs. Under minibatching an
    # epoch is n_train/batch_size steps, so a batch-size sweep spans an order of
    # magnitude at a fixed epoch budget: MNIST at (n_train 4000, batch 25) is
    # 3.2M steps against 100k at (1000, 200) -- 32x. Costing by epochs alone
    # scored all of those equal and put the most expensive cell two-thirds of the
    # way down the file, which is precisely the tail-blocking this ordering
    # exists to avoid.
    epochs = spec.get("epochs", 10_000)
    batch = spec.get("batch_size", 0)
    n_train = spec.get("n_train", 1000)
    steps = epochs * max(1, -(-n_train // batch)) if batch else epochs
    return steps * 0.0009 / 60 * model_cost      # ~0.9 ms/step on an L4


def _longest_first(specs):
    return sorted(specs, key=estimate_minutes, reverse=True)


def t4b_participation():
    """PORT of main's exp4b: does PARTIAL PARTICIPATION delay grokking?

    v2 has never run it. All 601 banked federated runs are fraction_train=1.0,
    so full participation is an untested assumption across the entire study and
    RESULTS 10 carries it as a limitation. v1's own 72 runs cannot fill the gap:
    their times sit on the pre-Phase-0.6 step axis, which over-counted work
    under fraction_train < 1 by ~2.5x. Their OUTCOMES stand (72/72 grokked) but
    they were measured at alpha=0.30, K=10 -- the plane RESULTS 4 has since
    shown is uniformly safe, 60/60. This is therefore not a reproduction; it is
    the same axis asked where federated effects actually appear.

    WHERE. alpha=0.25, K=50, E=5, iid -- t2_boundary's cell, which groks 5/5 at
    a KM median of 46,800. A failure here is attributable to participation
    rather than to the working point, which is the whole reason to pay for K=50.

    BUDGET, and why it is not a fixed round count. total_steps accumulates
    `E * samples_this_round / n_train_total` per round (Phase 0.6), so a cell at
    fraction f does f x the gradient work per round. Holding rounds fixed would
    starve the low-f cells by exactly the factor under test -- the mechanism
    behind seven manufactured boundaries in this project. Rounds are therefore
    set to reach a COMMON 100,000 total_steps, which is t2_boundary's budget and
    gave that cell 2.1x headroom over its measured requirement:

        f=1.0 ->  20,000 rounds   <- the control. Emitted WITHOUT a
                                     fraction_train key so its content hash
                                     matches t2_boundary's and the three seeds
                                     dedup instead of re-running.
        f=0.6 ->  33,400 rounds
        f=0.4 ->  50,000 rounds
        f=0.2 -> 100,000 rounds

    > DECISION RULE. Read on total_steps against the f=1.0 control.
    > (a) Cells track the control -> participation is free once compute is
    >     matched. v1's null reproduced where it counts, and the axis closes.
    > (b) Delay grows as f falls at matched compute -> participation costs
    >     something beyond compute. Then read client_weight_divergence: 17.4
    >     predicts it rises as f falls, because sampling fewer clients per round
    >     raises the variance of the aggregate. That would be the first test of
    >     the drift mechanism on an axis that moves disagreement WITHOUT moving
    >     K or E, which is exactly what 17.4 says is missing.
    > (c) Low-f cells censor at 100,000 total_steps -> re-budget before
    >     concluding anything. The requirement is a lower bound, not a
    >     prediction.
    """
    base = {k: v for k, v in SETUP_A.items()
            if k not in ("mode", "num_rounds", "eval_every")}
    common = {"mode": "federated", **base, "alpha": 0.25, "num_clients": 50,
              "local_epochs": 5, "partition": "iid", "eval_every": FL_EVAL_EVERY,
              "checkpoint_client_weights": True}
    tags = {"tier": "T4b", "group": "participation", "experiment": "exp4b",
            "setup": "A"}
    specs = []
    # The control: no fraction_train key, so the id matches t2_boundary's.
    specs += expand_grid({**common, "num_rounds": 20_000, "checkpoint_every": 1000},
                         {"seed": SEEDS3}, tags=tags)
    for frac, rounds in ((0.6, 33_400), (0.4, 50_000), (0.2, 100_000)):
        specs += expand_grid(
            {**common, "fraction_train": frac, "num_rounds": rounds,
             "checkpoint_every": rounds // 20},
            {"seed": SEEDS3}, tags=tags,
        )
    return specs


def t3a_dirichlet_band():
    """PORT of main's exp3a: does the AMOUNT of heterogeneity matter?

    RESULTS 4's "dirichlet tracks iid exactly at every K" is load-bearing -- it
    is the entire basis for "structure, not heterogeneity" (5.4), which 17.2
    narrowed but did not withdraw. It rests on TWO concentration values, 0.1 and
    0.5, both on the heterogeneous side, and both measured on the alpha=0.3
    plane that is uniformly safe. v1 swept five orders of magnitude and got
    100/100 grokked, but at alpha>=0.3, K=10 -- away from any boundary, so it
    could not have found an effect had there been one.

    WHERE. alpha=0.25, E=5, 20,000 rounds. t2_boundary's iid arms at K=20
    (5/5, 29,800) and K=50 (5/5, 46,800) are the controls and are ALREADY
    BANKED, so this manifest buys only the ladder.

    WHY TWO CLIENT COUNTS, AND WHY THE LADDER IS SHORTER AT K=50. The Dirichlet
    partitioner shards over target classes, and mod-97 at alpha=0.25 supplies
    ~2,350 training samples across 97 classes -- about 24 per class. A
    near-uniform draw therefore hands each of 50 clients a vanishing share of
    each class and leaves some client empty, which the partitioner correctly
    refuses (data/partition.py). Measured before writing this: at K=50,
    dirichlet_alpha 10 and 1000 raise on 2-3 seeds of 3, while 0.01/0.1/1.0 are
    feasible; at K=20 the whole ladder is feasible on every seed. So:

        K=20  dirichlet_alpha in {0.01, 0.1, 1.0, 10.0, 1000.0}   full ladder
        K=50  dirichlet_alpha in {0.01, 0.1, 1.0}                 feasible part

    That constraint is a property of the instrument, not of the phenomenon, and
    it is the reason the ladder is not simply run at the most fragmented cell.

    > DECISION RULE. Compare each rung to the banked iid arm at its own K, on
    > t_first_cross.
    > (a) All rungs track iid -> the claim hardens across five orders of
    >     magnitude AT the boundary, and "structure, not heterogeneity" becomes
    >     established rather than assumed.
    > (b) Low rungs degrade while high rungs track -> the claim needs
    >     qualifying to MODERATE heterogeneity, and both 5.4 and 17.2 need that
    >     qualifier written in.
    > (c) dirichlet_alpha=1000 at K=20 MUST match the iid arm to within seed
    >     noise. At that concentration the draw is numerically uniform, so a
    >     discrepancy there is a harness bug and not a finding -- this rung is
    >     the partitioner's own control, not a data point.
    """
    base = {k: v for k, v in SETUP_A.items()
            if k not in ("mode", "num_rounds", "eval_every")}
    common = {"mode": "federated", **base, "alpha": 0.25, "local_epochs": 5,
              "partition": "dirichlet", "num_rounds": 20_000,
              "eval_every": FL_EVAL_EVERY, "checkpoint_every": 1000,
              "checkpoint_client_weights": True}
    tags = {"tier": "T3a", "group": "dirichlet_band", "experiment": "exp3a",
            "setup": "A"}
    specs = []
    specs += expand_grid({**common, "num_clients": 20},
                         {"dirichlet_alpha": [0.01, 0.1, 1.0, 10.0, 1000.0],
                          "seed": SEEDS3}, tags=tags)
    specs += expand_grid({**common, "num_clients": 50},
                         {"dirichlet_alpha": [0.01, 0.1, 1.0], "seed": SEEDS3},
                         tags=tags)
    return specs


def t3a_size_control():
    """THE CONTROL for exp3a's breakdown: heterogeneity, or starved clients?

    t3a_dirichlet_band found the first breakdown candidate in this project that
    does not dissolve on inspection. At dirichlet_alpha=0.01 the models memorise
    (97-100% train) and then sit at 1-5% test with a ZERO slope, at 2-3x the
    budget the matched iid arm needed: 0/3 at K=50, 1/3 censored at K=20. Not a
    clock running out -- five of six trajectories are flat.

    But `dirichlet` moves two things at once, and at low concentration it moves
    them together. Measured on the actual shards:

        K=50   median shard   classes/client   min shard   outcome
        a=0.1       43              19            18-21     3/3 grok
        a=0.01    41-51              5             1-8      0/3

    The MEDIAN shard is unchanged, so this is not a bulk data-quantity effect.
    What collapses is label diversity (19 -> 5) and the size TAIL (one client
    got a single sample). So "extreme label heterogeneity stops grokking" and
    "starving a few clients stops grokking" are confounded, and the 0/3 cannot
    tell them apart.

    `dirichlet_sizes` (data/partition.py) holds the geometry and removes the
    label structure: it draws the Dirichlet partition through the same rng to
    get its sizes -- byte-identical per seed, asserted in tests -- then
    re-assigns every index uniformly at random into shards of exactly those
    sizes. Same shard sizes, IID labels.

    Cells mirror the band exactly: alpha=0.25, E=5, 20,000 rounds, K in {20,50},
    dirichlet_alpha in {0.01, 0.1}, 3 seeds. 0.1 is included because it is the
    graded rung -- it costs ~15% under `dirichlet`, and whether that 15% is
    labels or geometry is the same question asked where the answer is not
    binary.

    > DECISION RULE. Compare against the SAME (K, dirichlet_alpha) cell of
    > t3a_dirichlet_band, on t_first_cross.
    > (a) Control groks where `dirichlet` failed -> the breakdown is LABEL
    >     heterogeneity. That is a genuine federated boundary with a mechanism,
    >     and it is what this project has been looking for since v1.
    > (b) Control fails too -> the breakdown is shard SIZE. The finding narrows
    >     to "a client with ~1 sample poisons the aggregate", which is worth
    >     stating but is not about heterogeneity at all.
    > (c) Control matches the iid arm outright -> the size profile costs
    >     nothing on its own and the entire dirichlet ladder is a label effect.
    """
    base = {k: v for k, v in SETUP_A.items()
            if k not in ("mode", "num_rounds", "eval_every")}
    common = {"mode": "federated", **base, "alpha": 0.25, "local_epochs": 5,
              "partition": "dirichlet_sizes", "num_rounds": 20_000,
              "eval_every": FL_EVAL_EVERY, "checkpoint_every": 1000,
              "checkpoint_client_weights": True}
    tags = {"tier": "T3a", "group": "size_control", "experiment": "exp3a",
            "setup": "A"}
    specs = []
    for K in (20, 50):
        specs += expand_grid({**common, "num_clients": K},
                             {"dirichlet_alpha": [0.01, 0.1], "seed": SEEDS3},
                             tags=tags)
    return specs


def x_b_wd_zero_fl():
    """Setup B at wd=0, federated -- does the memorisation collapse need DECAY?

    THE CONTROL THAT IS MISSING. RESULTS 14.3's decay clock says memorisation
    blows up with K on setups with decoupled decay and stays flat on those
    without, because AdamW's decay is applied per local step and is independent
    of shard size while the learning signal from a 1/K shard is not. The
    evidence for it is B/C/D/E (AdamW, wd>0) degrading against A (GD, wd=0).

    But A differs from B in the OPTIMISER as well as the decay, so "AdamW" and
    "decoupled decay" are confounded in exactly the comparison the mechanism
    rests on. B at wd=0 is AdamW with no decay clock at all, and it separates
    them on a single setup with everything else held fixed.

    > DECISION RULE. Read t_memo(K) against the banked wd ladder at the same
    > alpha, K and budget.
    >   If t_memo at wd=0 stays FLAT in K (~ the centralized 200) while wd=0.1
    >   and wd=1.0 climb, the decay clock is about decay and the mechanism
    >   stands as written.
    >   If t_memo at wd=0 climbs with K anyway, the collapse is a property of
    >   AdamW under fragmentation and 14.3's mechanism is wrong -- the paper
    >   would then say "adaptive optimiser", not "decay clock".
    > Generalisation is expected to fail either way: 15.4 has B at wd=0 sitting
    > at 2.5-5.9% test for 100,000 centralized epochs against a 0.88% chance
    > level, so decay is what CAUSES grokking here. A federated null on
    > t_first_cross is therefore the centralized result reproduced, not a
    > federated effect, and t_memo is the reading that matters.

    alpha=0.30, NOT 0.40. It is the plane with a complete wd ladder already
    banked at both K (below), and the centralized wd=0 arm in `x_controls` is at
    0.30 too, so the ceiling comes free and matched.

    BUDGET 20,000 rounds = 100,000 steps, and it is set from the arm it has to
    out-live rather than from this arm's expectation. Banked t_first_cross at
    wd=0.1 reaches 77,200 (K=10) and 98,200 (K=20); anything shorter turns an
    honest null into the eighth censored boundary. It also matches the wd=1.0
    aggregation arm's 20,000 rounds and the centralized arm's 100,000 epochs, so
    all four wd levels are read at one budget.

    The comparison set this completes -- setup B, alpha=0.30, iid, E=5, 3 seeds,
    median t_memo / t_first_cross:

        wd    centralized        K=10                    K=20
        1.0   150 / 45,050       4,800-8,000 / 3/3       6,100-12,300 / 3/3
        0.1   150 / 4,350        1,300-1,400 / 3/3       3,500-4,100 / 3/3
        0.0   200 / never        THIS                    THIS

    NOT a new setup and not a working-point change: B's published config stays
    wd=1.0 (13.5), and new transformer work runs at 0.1 (the 2026-08-20 standing
    decision). This is a mechanism control, tagged tier X like the other ones.
    """
    base = {**SETUP_B, "weight_decay": 0.0, "alpha": 0.30}
    tags = {"tier": "X", "group": "b_wd_zero", "experiment": "control",
            "setup": "B"}

    # The ceiling. Already banked in x_controls -- reproduced here so the
    # manifest declares the whole comparison; identical spec, identical content
    # hash, so the launcher skips it and it costs nothing.
    specs = expand_grid(
        {"mode": "centralized", **base, "epochs": 100_000, "log_every": 200},
        {"seed": SEEDS3},
        tags=tags,
    )

    # The federated arm. K=10 and K=20 are where the wd ladder above is complete.
    specs += expand_grid(
        {"mode": "federated", **base, "local_epochs": 5, "partition": "iid",
         "strategy": "fedavg", "fraction_train": 1.0,
         "num_rounds": 20_000, "eval_every": FL_EVAL_EVERY},
        {"num_clients": [10, 20], "seed": SEEDS3},
        tags=tags,
    )
    return specs


def x_b_wd_zero_alpha():
    """Is there ANY alpha at which setup B groks with no weight decay at all?

    WHY BUDGET CANNOT ANSWER IT. B's centralized decay band at alpha=0.30 fits
    t_first_cross ~ 4,500 / wd almost exactly -- 4,350 / 18,100 / 45,050
    measured at wd 1.0 / 0.3 / 0.1 against 4,500 / 15,000 / 45,000 predicted.
    Two things follow. The wd=0.03 and wd=0.01 cells of `p1_b_decay_band` were
    given 50,000 steps against a predicted 150,000 and 450,000, so they are
    CENSORED rather than decided and 13.5's "sharp threshold below wd=0.1" is a
    clock artifact. And the law has no finite value at wd=0: raising the budget
    is not a route, which is why the wd=0 control sat at 2.5-5.9% test after
    100,000 centralized epochs and was still creeping at +0.001 to +0.058 points
    per 1,000 steps.

    WHY ALPHA IS THE KNOB THAT CAN. B's low-decay alpha ladder at wd=0.1
    (60,000 epochs, 3 seeds, 3/3 everywhere, t_memo=150 throughout):

        alpha        0.3      0.4      0.5     0.6     0.7
        t_fc      45,050   15,500    7,050   3,150   1,550

    alpha 0.3 -> 0.7 buys 29x; dropping wd 1.0 -> 0.1 costs 10x. So alpha more
    than compensates for a decade of decay, and by alpha=0.7 the whole delay is
    ~1,400 steps over t_memo. That is the regime where the memorising and the
    generalising solution have nearly converged -- 70% of the grid makes the
    held-out pairs close to interpolation -- and it is the plausible place for
    generalisation to happen with nothing driving it. No one has measured it:
    every wd=0 run in the project is at alpha=0.30 (B) or 0.40 (C).

    > DECISION RULE. Report the LOWEST alpha that groks 3/3, not the highest.
    > 13.3 records that A' has no usable federated working point above
    > alpha=0.20 because "there is no delay left for federation to disrupt", and
    > the same trap applies here: a federated K effect needs a delay to act on,
    > so the useful rung is the marginal one. If NO rung groks, alpha is not the
    > knob and the next candidate is init_scale < 1 (Omnigrok's norm-travel
    > story, decay's mechanism without decay) -- which is NOT currently wired
    > into the transformer, only into `models/mlp.py`.

    BUDGET 100,000 epochs, 1.67x the wd=0.1 ladder it is read against. Compare
    on t_first_cross, which 14.4 establishes as the statistic that survives a
    budget difference; t_grok does not. log_every=50 matches the wd=0.1 ladder
    so the two are read at one logging rate as well.

    alpha=0.30 is not a rung: it is banked (group `wd_zero`, 100,000 epochs,
    0/3, final test 2.5-5.9%) and is the ladder's negative anchor.
    """
    return expand_grid(
        {"mode": "centralized", **SETUP_B, "weight_decay": 0.0,
         "epochs": 100_000, "log_every": 50},
        {"alpha": [0.4, 0.5, 0.6, 0.7, 0.8], "seed": SEEDS3},
        tags={"tier": "X", "group": "b_wd_zero_alpha", "experiment": "control",
              "setup": "B"},
    )


def x_b_wd_zero_a04_long():
    """The low-alpha anchor for the wd=0 budget curve. One rung, ten times the budget.

    `x_b_wd_zero_alpha` established that setup B groks with NO weight decay from
    alpha=0.60, and that the rungs below it are censored rather than decided --
    every cell at alpha>=0.5 was still climbing at the 100,000-epoch ceiling.
    That leaves the curve anchored at only two usable points, because alpha=0.80
    has no delay left to measure:

        alpha            0.6        0.7      0.8
        delay        95,150     28,850      450   (t_first_cross - t_memo)

    Two points cannot say HOW the requirement diverges as alpha falls, and that
    divergence is the alpha-direction counterpart to the decay-direction law
    t_first_cross ~ 4,500/wd that B's decay band already fits. Together they are
    a two-parameter statement -- data and decay are exchangeable, at a rate this
    project can measure -- which is worth more than either alone.

    ALPHA=0.40 AND NOT 0.30. Both were considered at long budgets. At alpha=0.30
    all three banked seeds extrapolate to 1.65M / 4.4M / 88M steps and none is
    accelerating, so 1,000,000 buys three more censored cells. At alpha=0.40 one
    seed is unambiguously mid-transition -- seed 123's quarterly slope runs
    +0.096 -> +0.297 -> +0.365 accuracy points per 1,000 steps -- and
    extrapolates to ~283,000.

    > EXPECTED: 1/3 or 2/3, NOT 3/3, and that is the honest prior rather than a
    > disappointment. Seed 123 (~283,000) is safe; seed 42 (~834,000, slopes
    > noisy at +0.155 -> +0.058 -> +0.111) is a coin flip; seed 456 is
    > DECELERATING (+0.116 -> +0.176 -> +0.020, ~4.1M) and will not make it.
    > Read the result as a point on the budget-vs-alpha curve, not as "alpha=0.40
    > groks" -- the fraction is the wrong headline for a cell chosen because it
    > straddles the transition.

    NOT A FEDERATED WORKING POINT, and not intended as one. A centralized
    requirement near 300,000 steps puts the matching federated arm at ~86
    slot-hours across K=10 and K=20 on measured rates, before federation's own
    added delay. The federated wd=0 arm belongs at alpha=0.60-0.70. This manifest
    buys a curve, not a working point.

    log_every=50 matches `x_b_wd_zero_alpha` exactly. 14.4 is the reason: a
    t_first_cross is quantised by the logging rate, so a rung read at a different
    rate is not on the same curve as the others. It costs 20,000 evaluations
    rather than 2,000, and the ~8.8 h/run estimate already includes them.

    Cost: 3 runs, ~8.8 h each, ~26.5 slot-hours, ~9 h wall at --per-gpu 4.
    These are new run ids -- `epochs` is inside the content hash, so the banked
    100,000-epoch cells cannot be extended and are re-run from scratch.
    """
    return expand_grid(
        {"mode": "centralized", **SETUP_B, "weight_decay": 0.0, "alpha": 0.40,
         "epochs": 1_000_000, "log_every": 50},
        {"seed": SEEDS3},
        tags={"tier": "X", "group": "b_wd_zero_a04_long", "experiment": "control",
              "setup": "B"},
    )


# ── The paper's three open FL axes ───────────────────────────────────────────
#
# The campaign measured K exhaustively and partition structure exhaustively. The
# other three FL design variables -- local epochs E, participation fraction f,
# and the AMOUNT of heterogeneity -- are measured on the ANCHOR ONLY, so every
# statement the paper makes about them is a statement about one architecture on
# one task. These three builders close that, each one extending the manifest
# whose figure it joins:
#
#   t5_local_epochs        extends t1_setup_k_ladder  (same base, same decays)
#   t5_participation       extends t1_setup_k_ladder  (same base, same decays)
#   t3a_dirichlet_setups   extends t3b_partitions     (same base, same decays)
#
# THE DECAY THIS IMPLIES, stated because it looks like a violation of the
# 2026-08-20 standing decision and is not. B and C carry the decay of the
# manifest they extend -- B at wd=0.1 in the two K-ladder builders, B and C at
# wd=1.0 in the Dirichlet ladder -- because each of these axes is read against a
# BANKED control at that decay. Switching a ladder to a different decay from its
# own control does not produce a cleaner number; it produces a comparison
# between two cells that differ in two things. "New work uses wd=0.1" governs a
# new working point, not the completion of a banked figure whose other lines are
# already drawn -- and PROGRESS's own rule for mixed figures (label which decay
# each line carries) is what applies here.


def _e_scaled(common, rounds_at_e5, E, ckpt_at_e5=0):
    """One E-ladder cell, compute-matched to the E=5 cell it is scaled from.

    ROUNDS. total_steps accumulates E per round at full participation, so
    rounds = rounds_at_e5 * 5/E holds total gradient work FIXED across the
    ladder. That is the opposite convention to the E-spine at the top of this
    file, which fixes rounds and lets compute scale with E, and the choice is
    deliberate: the anchor already has the fixed-rounds reading banked
    (RESULTS 17.4's E table, 10,000 rounds at E in {1,5,50}), so the
    communication-matched view exists. What does not exist anywhere is the
    compute-matched one, and it is the view that separates "local work delays
    grokking" from "this cell simply ran longer".

    EVAL CADENCE IS MATCHED IN STEPS, NOT ROUNDS -- eval_every scales by the
    same 5/E. A fixed eval_every would sample the curve every E*eval_every
    steps, so the E=50 cells would be read at 10x the granularity of E=5 and the
    E axis would be confounded with the measurement resolution. This project has
    lost two results to exactly that (13.4 and 14.4: t_grok measuring the
    logging rate), and it is cheap to avoid here. Same scaling for
    checkpoint_every, so the mech-interp channel samples the same step grid on
    every rung.
    """
    scale = 5.0 / E
    spec = {**common, "local_epochs": E,
            "num_rounds": max(1, round(rounds_at_e5 * scale)),
            "eval_every": max(1, round(FL_EVAL_EVERY * scale))}
    if ckpt_at_e5:
        spec["checkpoint_every"] = max(1, round(ckpt_at_e5 * scale))
    return spec


# Each block is the setup's own K=10 cell from the manifest named beside it --
# copied field-for-field so that the E=5 rung is a content-hash match and
# dedups instead of re-running.
# (label, common, control_rounds_at_e5, rung_rounds_at_e5, ckpt_at_e5).
#
# CONTROL AND RUNG BUDGETS DIFFER ON B, and only on B. The banked K-ladder
# controls were audited against their own t_first_cross before this manifest
# was finalised, and two failed the headroom rule:
#
#   B  K=10  t_fc 55,200-77,200 at a 100,000-step budget -> 1.3x. The control
#      itself is 3/3 and MEASURED, so it stays banked -- but rungs at the same
#      budget would censor under any E-induced slowdown comparable to the
#      anchor's (delay roughly doubles by E=50, RESULTS 17.4). New rungs get
#      200,000 steps. Reading rung-vs-control across budgets is exactly what
#      the standing decision "compare t_first_cross, not t_grok, when budgets
#      differ" (14.4) licenses.
#   D  K=10  the K-ladder cell is 2/3 -- seed 42 CENSORED at 100,000 steps
#      (t_fc inf). A censored reference is not a control, so D's whole ladder
#      rebases on t3b_partitions' iid K=10 cell instead: 250,000 steps, 3/3 at
#      t_fc ~78,900, banked, and its t3b-style common (explicit strategy /
#      fraction_train, checkpoint_every 5,000) hash-matches it -- verified
#      against the run store at authoring time. Zero re-runs, and every D rung
#      is compute-matched at 250,000 steps.
#
# A (3.8x), C (>20x) and E (3.6x) pass the audit and keep a single budget.
def _k10_blocks():
    return [
        # A -- t2_k_breakdown's cell. No checkpoint fields on that manifest, so
        # none here either; A's per-client weights come from t3b/t4b instead.
        ("A", {**{k: v for k, v in SETUP_A.items()
                  if k not in ("num_rounds", "eval_every")},
               "num_clients": 10, "partition": "iid"},
         FL_ROUNDS, FL_ROUNDS, 0),
        # B -- t1_setup_k_ladder's K=10 cell; rungs at double budget, above.
        ("B", {"mode": "federated", **SETUP_B, "weight_decay": 0.1,
               "alpha": 0.30, "partition": "iid", "num_clients": 10,
               "checkpoint_client_weights": True}, 20_000, 40_000, 2_000),
        ("C", {"mode": "federated", **SETUP_C, "hidden_width": 256,
               "alpha": 0.50, "partition": "iid", "num_clients": 10,
               "checkpoint_client_weights": True}, 40_000, 40_000, 4_000),
        # D -- t3b_partitions' iid K=10 cell, NOT the K-ladder's (see above).
        ("D", {"mode": "federated", **SETUP_D, "alpha": 0.30,
               "local_epochs": 5, "strategy": "fedavg", "fraction_train": 1.0,
               "partition": "iid", "num_clients": 10,
               "checkpoint_client_weights": True}, 50_000, 50_000, 5_000),
        ("E", {"mode": "federated", **{k: v for k, v in SETUP_E.items()
                                       if k != "batch_size"},
               "n_train": 2000, "n_test": 5000, "batch_size": 100,
               "partition": "iid", "num_clients": 10,
               "checkpoint_client_weights": True}, 4_000, 4_000, 400),
    ]


def t5_local_epochs():
    """PAPER AXIS 1: local epochs, on every setup, at matched compute.

    THE GAP. 562 of the 673 banked federated runs are at E=5, and every cell
    that is not sits on the anchor (E in {1, 25, 50}) or is B's 6-run
    persist-state probe. So the paper's local-work axis is one architecture on
    one task, and RESULTS 17.4's E column -- drift/round 0.018 -> 0.167 -> 1.92
    across E = 1, 5, 50 -- is the whole of the evidence that local work is what
    generates client disagreement.

    WHY THIS IS THE AXIS THE DRIFT STORY NEEDS. 18.4 concluded that systematic
    disagreement delays grokking while sampling noise does not. E is the one
    knob that raises disagreement by making clients travel FURTHER from a shared
    starting point without changing the data any client holds -- the opposite
    manipulation to exp3a (which changes what the data IS) and to exp4b (which
    changes who is sampled). If the delay tracks E at matched compute, "local
    work generates the conflict" is measured rather than assumed on all five
    setups; if it does not, the drift-per-round table is a correlate of E and
    not a mechanism.

    DESIGN. K=10, iid, E in {5, 10, 25, 50}, 3 seeds, each setup at its own
    working point. E=5 is the banked K-ladder cell and dedups. Rounds and both
    sampling cadences scale by 5/E -- see _e_scaled for why compute-matching and
    not round-matching, and why the eval grid has to move with it.

    A CARRIES ONE EXTRA RUNG, E=1, and only A can. Under plain GD at
    momentum=0, FedAvg at E=1 is an exact algebraic identity with centralized
    training (tests/test_fedavg_identity.py), so A's E=1 cell is the axis's
    zero-drift endpoint with a known answer -- it must reproduce the centralized
    curve, and if it does not the harness is wrong rather than the finding
    interesting. On B/C/D/E it would measure nothing of the kind: 15.3
    established that with persist_local_opt_state=False (the default here, and
    standard FedAvg semantics) every AdamW round at E=1 is a single cold-start
    bias-corrected Adam step, which is closer to signSGD than to Adam. That cell
    is uninterpretable as a federated result, so it is not bought. Its cost on A
    is 5x the control's rounds and it is the single most expensive rung here.

    THE OPTIMISER-RESTART CONFOUND IS BOUNDED, NOT REMOVED. Rebuilding the local
    optimiser every round is what FedAvg does, so the E axis on an AdamW setup
    carries restart frequency alongside local drift -- fewer, longer rounds also
    means fewer restarts. s5_fl_probe already bounds it on B: 12 banked runs,
    E=5 and E=50, persist False against True, and persisting Adam state does not
    recover the ceiling (3/3 against 2/3, overlapping times). So the confound is
    measured and small on the one setup that has the data, and no new
    persist=True arm is bought here.

    > DECISION RULE. Read delay = t_first_cross - t_memo against E on the
    > total_steps axis, per setup, with mean_client_drift and
    > client_weight_divergence beside it.
    > (a) Delay grows with E at matched compute, on the AdamW setups as well as
    >     on A -> local work costs generalisation time, the drift-per-round
    >     table generalises, and 18.4's "systematic disagreement" gains its
    >     third and cleanest manipulation.
    > (b) Delay flat in E at matched compute -> E buys communication savings for
    >     free at these K, the anchor's E table was measuring the extra compute
    >     that fixed rounds handed the high-E cells, and RESULTS 17.4's E column
    >     needs withdrawing as evidence for the mechanism.
    > (c) The two AdamW failure modes are separable and must be read apart:
    >     t_memo climbing with E is the decay clock (14.3) reaching the
    >     memorisation phase, not a drift result.
    """
    specs = []
    for label, common, ctrl_r5, rung_r5, ckpt5 in _k10_blocks():
        tags = {"tier": "T5", "group": "local_epochs", "experiment": "exp_e",
                "setup": label}
        rungs = [1, 5, 10, 25, 50] if label == "A" else [5, 10, 25, 50]
        for E in rungs:
            r5 = ctrl_r5 if E == 5 else rung_r5
            c5 = ckpt5 if E == 5 else (ckpt5 and round(ckpt5 * rung_r5 / ctrl_r5))
            specs += expand_grid(_e_scaled(common, r5, E, c5),
                                 {"seed": SEEDS3}, tags=tags)
    return specs


def t5_participation():
    """PAPER AXIS 2: partial participation, on the four setups that lack it.

    THE GAP. 664 of 673 banked federated runs are at f=1.0. The nine that are
    not are t4b_participation, on the anchor, at K=50 -- and 18.3's reading of
    them is one of the load-bearing results in the drift story: sampling 10 of
    50 clients reaches the bar in the SAME number of rounds while doing a fifth
    of the gradient work, so on the compute-matched axis partial participation
    is free. 18.4 then makes that cell carry the whole "sampling noise averages
    out, systematic conflict does not" distinction. One setup, one K.

    WHY IT MATTERS THAT IT IS ONE SETUP. The claim is mechanistic -- it says
    what KIND of disagreement costs time -- and the anchor is the setup with no
    weight decay, no adaptive optimiser and a memorisation time flat in K. Every
    other setup fails at high K through memorisation rather than through delay
    (14.3), and subsampling clients is exactly a manipulation of how much
    gradient signal reaches the aggregate per round. Whether f is free on a
    setup whose memorisation is already starved is not something the anchor can
    answer.

    DESIGN. K=20, iid, E=5, f in {1.0, 0.5, 0.25} -> 20, 10 and 5 clients per
    round, 3 seeds, each setup at its K-ladder working point. K=20 rather than
    the anchor's K=50: on B, C and D, K=50 is where memorisation itself
    collapses (14.3, 17.2), and a participation null measured inside a training
    failure says nothing about participation.

    ROUNDS SCALE AS 1/f, exactly as t4b did, and for the same reason: a round at
    fraction f does f times the gradient work, so a fixed round count would
    starve the low-f cells by precisely the factor under test. Every cell here
    reaches the same total_steps as its own f=1.0 control. eval_every and
    checkpoint_every scale by 1/f as well, so all three cells are sampled on the
    same total_steps grid rather than the same round grid.

    THE f=1.0 CONTROL IS EMITTED WITHOUT A fraction_train KEY so its content
    hash matches the banked K-ladder cell and it dedups. Emitting it as
    fraction_train=1.0 would be the same experiment with a different id, which
    is the hazard manifest.orphaned_ids exists for.

    A IS IN, AT K=20, DESPITE t4b -- because t4b is at K=50 and this figure is
    at K=20, and a panel whose anchor line sits at a different K from every
    other line is not the anchor as a reference, it is a second variable. A's
    control is t2_k_breakdown's K=20 iid cell (banked, 5 seeds), which carries
    no strategy/fraction_train/checkpoint keys, so A's block is built from
    SETUP_A directly rather than from the shared common dict -- adding those
    keys would change the control's hash and re-run banked work. The f<1 arms
    do carry per-client checkpoints (new runs regardless), matching t4b's own
    convention. If A's K=20 axis is flat like its K=50 axis, 18.3 gains a
    second K for free; if it is not, that is itself a finding about where the
    marginal client starts to matter.

    E'S K=20 IS THE DEGENERATE RUNG (one full-batch step per local epoch --
    the K-ladder's own flagged control). That degeneracy is CONSTANT across
    the f arms, and the reading is within-setup against E's own f=1.0 control,
    so it cancels; but E's panel is not comparable to the others on the E
    semantics and must not be read as if it were.

    COST WARNING. C's block is roughly half the manifest's slot-hours on its
    own (40,000 rounds at f=1.0 becomes 160,000 at f=0.25, on the slowest setup
    in the study). Specs are emitted longest-first, so truncating the sweep
    after C's f=0.25 cells still leaves every other setup complete.

    > DECISION RULE. Compare t_first_cross in ROUNDS and in total_steps against
    > the f=1.0 control, per setup, with client_weight_divergence beside it.
    > (a) Flat in rounds, as on the anchor -> 18.4's distinction holds across
    >     architectures and the paper can state it as a property of grokking
    >     under federation rather than of the anchor.
    > (b) Delay grows as f falls -> the marginal client carries signal on this
    >     setup, and "sampling noise averages out" needs the qualifier that it
    >     does so only where memorisation is not already the bottleneck. Read
    >     t_memo first: if t_memo is what moves, this is the decay clock at a
    >     smaller effective client population, not a participation result.
    """
    # BUDGETS WERE AUDITED against banked t_first_cross before finalising, and
    # three of the four original K-ladder-derived controls failed:
    #
    #   B  the K=20 wd=0.1 cell crosses at 66,100-98,200 -- measured at
    #      p1_k_collapse_budget's 200,000-step budget. The 100,000 steps first
    #      drafted here would censor two of three seeds ON THE CONTROL. Base is
    #      40,000 rounds, and with checkpoint_every=2,000 the control
    #      hash-matches p1_k_collapse_budget's banked cell (verified).
    #   D  the K-ladder K=20 cell is 1/3 GROKKED (t_fc inf, inf, 95,600 at
    #      100,000 steps) -- a censored reference is not a control. Rebased on
    #      t3b_partitions' iid K=20 cell: 250,000 steps, banked, t_fc 94,300.
    #   E  1.7x headroom against t3b's 3.6x at the same K. Rebased on t3b's
    #      8,000-round cell, also banked.
    #
    # C passes (t_fc 5,600-9,100 against 200,000 steps) and keeps the K-ladder
    # cell. Consequence of the rebasing: D's and E's commons are t3b-style and
    # carry explicit strategy/fraction_train keys (the f arms override
    # fraction_train); B's and C's carry neither, like their source manifests.
    # The asymmetry is hash-dictated -- normalising it would orphan the banked
    # controls, which is the exact waste this comment exists to prevent.
    BLOCKS = [
        ("B", {"mode": "federated", **SETUP_B, "weight_decay": 0.1,
               "alpha": 0.30, "partition": "iid", "num_clients": 20,
               "checkpoint_client_weights": True}, 40_000, 2_000),
        ("C", {"mode": "federated", **SETUP_C, "hidden_width": 256,
               "alpha": 0.50, "partition": "iid", "num_clients": 20,
               "checkpoint_client_weights": True}, 40_000, 4_000),
        ("D", {"mode": "federated", **SETUP_D, "alpha": 0.30,
               "strategy": "fedavg", "fraction_train": 1.0,
               "partition": "iid", "num_clients": 20,
               "checkpoint_client_weights": True}, 50_000, 5_000),
        ("E", {"mode": "federated", **{k: v for k, v in SETUP_E.items()
                                       if k != "batch_size"},
               "n_train": 2000, "n_test": 5000, "batch_size": 100,
               "strategy": "fedavg", "fraction_train": 1.0,
               "partition": "iid", "num_clients": 20,
               "checkpoint_client_weights": True}, 8_000, 800),
    ]
    specs = []
    # A -- built bare so the control hashes to t2_k_breakdown's banked cell.
    a_tags = {"tier": "T5", "group": "participation_setups",
              "experiment": "exp4b", "setup": "A"}
    a_base = {**SETUP_A, "local_epochs": 5, "num_clients": 20,
              "partition": "iid"}
    specs += expand_grid(a_base, {"seed": SEEDS3}, tags=a_tags)
    for frac in (0.5, 0.25):
        specs += expand_grid(
            {**a_base, "fraction_train": frac,
             "num_rounds": round(FL_ROUNDS / frac),
             "eval_every": round(FL_EVAL_EVERY / frac),
             "checkpoint_every": round(FL_ROUNDS / frac) // 10,
             "checkpoint_client_weights": True},
            {"seed": SEEDS3}, tags=a_tags,
        )
    for label, common, rounds, ckpt in BLOCKS:
        tags = {"tier": "T5", "group": "participation_setups",
                "experiment": "exp4b", "setup": label}
        base = {**common, "local_epochs": 5, "eval_every": FL_EVAL_EVERY}
        # The control: no fraction_train key, so the id matches the banked
        # K-ladder cell.
        specs += expand_grid({**base, "num_rounds": rounds,
                              "checkpoint_every": ckpt},
                             {"seed": SEEDS3}, tags=tags)
        for frac in (0.5, 0.25):
            specs += expand_grid(
                {**base, "fraction_train": frac,
                 "num_rounds": round(rounds / frac),
                 "eval_every": round(FL_EVAL_EVERY / frac),
                 "checkpoint_every": round(ckpt / frac)},
                {"seed": SEEDS3}, tags=tags,
            )
    return specs


def t3a_dirichlet_setups():
    """PAPER AXIS 3: the Dirichlet ladder on B, C, D and E.

    Written against plans/exp3a-dirichlet-ladder-across-setups.md, which carries
    the feasibility measurements this builder relies on. Two departures from that
    plan, both deliberate: A' is dropped (it is out of the paper), and the rung
    list for E follows the plan's own feasibility table rather than the algebraic
    setups' full ladder.

    THE GAP. RESULTS 18.1 found the first breakdown in this project that does not
    dissolve on inspection -- concentrating the Dirichlet draw costs 2.01x -- and
    18.2 then showed the tail of that ladder is client STARVATION rather than
    heterogeneity. Both readings are setup A. Meanwhile 17.2 withdrew the
    partition-structure headline as a general claim, on evidence that the
    ordering of structured partitions differs per setup. The unstructured axis
    now has to be asked the same question, or "heterogeneity costs time" inherits
    exactly the generality that "coherent shards help" just lost.

    A IS IN, AT K=10, DESPITE THE PLAN SAYING IT NEEDS NOTHING. The plan's
    "A is banked" is t3a_dirichlet_band -- K in {20, 50} at alpha=0.25, whose
    low rungs are exactly the cells 18.2 showed were starvation, not
    heterogeneity. This figure's panels are all K=10 at working points, and at
    K=10 A's min shard at dir_alpha=0.01 is ~116 (plan's own table, verified
    below at generation time), clear of the starvation regime. So A's K=10
    ladder is the one version of A's ladder that is BOTH matched to the other
    panels and clean of the confound -- and its 0.5 rung dedups against
    t3b_partitions like everyone else's. The banked K=20/50 ladder stays as
    the 18.1/18.2 reading; this is not a re-run of it.

    K=10, AND THAT IS THE WHOLE REASON THE LADDER IS FEASIBLE. The Dirichlet
    partitioner shards over target classes, so a concentrated draw on a dataset
    with few samples per class empties someone's shard and the partitioner
    correctly refuses. The plan measured the minimum shard directly at each
    setup's working point: >= 116 samples at K=10 on every algebraic setup
    against the <= 2 that killed cells in 18.2. So the size confound that
    required t3a_size_control beside A's low rungs is not in play here, and no
    size-control arm is bought.

    E IS THE EXCEPTION, ON BOTH COUNTS. MNIST has 10 classes against 97-120, so
    dirichlet_alpha=0.01 at K=10 empties a shard and raises; its ladder starts at
    0.1. And the size confound the algebraic setups escape is present here --
    partitions built for these exact specs, min shard across the three seeds:

        dir_a  0.1 -> 5, 25, 56      1.0 -> 101    10 -> 172    1000 -> 193
        the banked 0.5 rung -> 62, 83

    against batch_size=100. So on E the 0.1 rung (and the banked 0.5 rung it is
    read against) puts at least one client below a single batch, which is 18.2's
    starvation regime rather than a heterogeneity reading. Kept, because dropping
    it would leave E with no concentrated end at all, but E's low rungs are NOT
    comparable to the algebraic setups' and the panel must say so. Separating the
    two would need t3a_size_control's dirichlet_sizes arm rebuilt for MNIST --
    12 runs, not written, and only worth buying if E's low rung is the cell that
    ends up carrying a claim.

    THE 0.5 RUNG IS FREE, AND ONLY IF WRITTEN BARE. FedConfig.dirichlet_alpha
    defaults to 0.5 and t3b_partitions' banked dirichlet cells OMIT the field, so
    a spec that names it explicitly is a different content hash and re-runs
    banked work. Every block below inherits t3b's own common dict and adds the
    rungs as an axis, with the 0.5 rung emitted as a separate bare spec. Verify
    at generation time: the block should report 3 banked cells per setup.

    > DECISION RULE. Per setup, plot t_first_cross against dirichlet_alpha with
    > the banked iid cell at the same K as the horizontal reference.
    > (a) The ladder is flat until the concentrated end on every setup -> 18.1's
    >     threshold reading generalises and the paper states heterogeneity as a
    >     threshold effect rather than a gradient.
    > (b) The threshold sits at a different concentration per setup -> read it
    >     against each setup's class count and shard geometry before calling it
    >     an architecture effect; the plan's feasibility table is the first place
    >     to look, since classes-per-client is what differs most across setups.
    > (c) Any cell that fails to memorise is not a heterogeneity result. Check
    >     peak_train_acc first: on the AdamW setups the decay clock (14.3) is the
    >     competing explanation for every failure, and it is the one that has
    >     been right so far.
    """
    LADDER = [0.01, 0.1, 1.0, 10.0, 1000.0]
    specs = []
    for label, base, wp, rounds, _Ks in [
        ("A", {k: v for k, v in SETUP_A.items()
               if k not in ("mode", "num_rounds", "eval_every")},
         {"alpha": 0.30}, 10_000, None),
        ("B", SETUP_B, {"alpha": 0.30}, 20_000, None),
        ("C", SETUP_C, {"alpha": 0.40, "hidden_width": 256}, 40_000, None),
        ("D", SETUP_D, {"alpha": 0.30}, 50_000, None),
    ]:
        tags = {"tier": "T3a", "group": "dirichlet_setups", "experiment": "exp3a",
                "setup": label}
        common = {"mode": "federated", **base, **wp, "local_epochs": 5,
                  "strategy": "fedavg", "fraction_train": 1.0,
                  "num_clients": 10, "partition": "dirichlet",
                  "num_rounds": rounds, "eval_every": FL_EVAL_EVERY,
                  "checkpoint_every": max(1, rounds // 10),
                  "checkpoint_client_weights": True}
        # The 0.5 rung, BARE -- this is t3b_partitions' banked cell.
        specs += expand_grid(common, {"seed": SEEDS3}, tags=tags)
        specs += expand_grid(common, {"dirichlet_alpha": LADDER,
                                      "seed": SEEDS3}, tags=tags)
    # E -- MNIST, and 0.01 is infeasible at K=10 (10 classes, empty shard).
    common = {"mode": "federated", **{k: v for k, v in SETUP_E.items()
                                      if k != "batch_size"},
              "n_train": 2000, "n_test": 5000, "batch_size": 100,
              "local_epochs": 5, "strategy": "fedavg", "fraction_train": 1.0,
              "num_clients": 10, "partition": "dirichlet", "num_rounds": 8_000,
              "eval_every": FL_EVAL_EVERY, "checkpoint_every": 800,
              "checkpoint_client_weights": True}
    tags = {"tier": "T3a", "group": "dirichlet_setups", "experiment": "exp3a",
            "setup": "E"}
    specs += expand_grid(common, {"seed": SEEDS3}, tags=tags)
    specs += expand_grid(common, {"dirichlet_alpha": LADDER[1:],
                                  "seed": SEEDS3}, tags=tags)
    return specs

def x_e50_long():
    """TIER X: the two E=50 cells t5_local_epochs could not decide, at 4-10x budget.

    THE TWO CELLS, and they fail in OPPOSITE phases. Both are on S_5, which is
    the split that matters: A, B and E are 3/3 at E=50 and only the S_5 setups
    break there.

      D (quad-MLP, S_5, wd=1.0) at 250,000 steps: 0/3, and it is NOT a training
        failure -- peak train accuracy is 100.0 on all three seeds, t_memo
        2,500-3,500, and test then stalls at 80.4 / 80.8 / 81.7%. Final-quarter
        slopes are -0.0019, +0.0035 and +0.0146 points per 1,000 steps, so one
        seed is DECLINING and the other two extrapolate to 0.9M and 4.2M further
        steps. Memorisation intact, generalisation absent: the exact signature
        p1_k_collapse_budget's decision rule named as "a federated breakdown of
        grokking with memorisation intact ... the mechanism the project has been
        looking for".

      C (transformer, S_5, wd=1.0) at 200,000 steps: 1/3, and it IS a training
        failure -- the two failing seeds peak at 86.6% and 86.9% TRAIN, below
        the bar, so they never memorise. C's t_memo scales with E (3,200 at E=5
        to 32,500 at E=50), so the budget that was ample at E=5 is not at E=50.
        This is the same shape as 17.2's "C at K=50 is a training failure, not a
        partition result", reached along the E axis instead.

    WHY THIS IS NOT JUST A LONGER RE-RUN. Eight boundaries in this project have
    dissolved into censoring on re-measurement, and the ONE that did not
    (RUNS_TODO 3: B at wd=0 stops rather than slows) was settled by exactly this
    move -- a single arm at ~10x budget, which found a hard plateau and the
    mechanism behind it. D's cell is the current best candidate for a second
    genuine negative, and it cannot be claimed at 250,000 steps.

    BUDGETS are set ABOVE the extrapolations rather than at them, per the rule
    that produced every re-measurement in this file:

      D -> 40,000 rounds = 2,000,000 steps (8x). Covers the median seed's 0.9M
           extrapolation with >2x margin. The slowest seed's 4.2M is NOT covered
           on purpose: if D is still at ~81% after 2M steps with a flat slope,
           the plateau is the finding and a 4M-step run buys nothing but cost.
      C -> 20,000 rounds = 1,000,000 steps (5x). The question here is only
           whether the two seeds MEMORISE; the grokked seed crossed at 31,400
           with t_fc ~= t_memo (C's delay has collapsed to zero by E=50), so any
           seed that memorises groks almost immediately.

    EVAL CADENCE is coarsened to every 10 rounds (500 steps) rather than the
    ladder's every 2 rounds (100 steps). At 2,000,000 steps the ladder's cadence
    would log 20,000 points per run for a question -- "does it ever cross?" --
    that 500-step resolution answers. t_memo is ~3,000 on D, so 500 steps still
    resolves it to ~15%.

    > DECISION RULE.
    > D: (a) crosses the bar within 2M steps -> the E=50 cell was censored, the
    >    delay simply grows steeply with E, and D's E axis is a delay story like
    >    A's. (b) still ~81% with a flat slope -> a real breakdown with
    >    memorisation intact, the first on the E axis, and it belongs in the
    >    paper as a stated positive result rather than a censored cell.
    >    Read final_acc and the final-quarter slope, NOT t_grok.
    > C: (a) the two seeds memorise and grok -> E=50 was under-budgeted and C's
    >    ladder is complete as a t_memo story. (b) they still peak below the bar
    >    at 1M steps -> C cannot train at E=50, which is a statement about the
    >    setup's optimisation and not about grokking; it then needs saying that
    >    C's E ladder stops at E=25.
    """
    specs = []
    # D -- identical to t5_local_epochs' E=50 cell except the budget and cadence.
    specs += expand_grid(
        {"mode": "federated", **SETUP_D, "alpha": 0.30, "local_epochs": 50,
         "strategy": "fedavg", "fraction_train": 1.0, "partition": "iid",
         "num_clients": 10, "num_rounds": 40_000, "eval_every": 10,
         "checkpoint_every": 4_000, "checkpoint_client_weights": True},
        {"seed": SEEDS3},
        tags={"tier": "X", "group": "e50_long", "experiment": "exp_e", "setup": "D"},
    )
    # C -- same, at its own working point (width 256, alpha 0.50).
    specs += expand_grid(
        {"mode": "federated", **SETUP_C, "hidden_width": 256, "alpha": 0.50,
         "local_epochs": 50, "partition": "iid", "num_clients": 10,
         "num_rounds": 20_000, "eval_every": 10, "checkpoint_every": 2_000,
         "checkpoint_client_weights": True},
        {"seed": SEEDS3},
        tags={"tier": "X", "group": "e50_long", "experiment": "exp_e", "setup": "C"},
    )
    return specs

BUILDERS = {
    "x_e50_long": x_e50_long,
    "t5_local_epochs": t5_local_epochs,
    "t5_participation": t5_participation,
    "t3a_dirichlet_setups": t3a_dirichlet_setups,
    "x_b_wd_zero_a04_long": x_b_wd_zero_a04_long,
    "x_b_wd_zero_alpha": x_b_wd_zero_alpha,
    "x_b_wd_zero_fl": x_b_wd_zero_fl,
    "p1_d_gd_probe": p1_d_gd_probe,
    "p1_cd_decay_band": p1_cd_decay_band,
    "p1_b_decay_band": p1_b_decay_band,
    "t3b_partitions": t3b_partitions,
    "x_controls": x_controls,
    "t2_aggregation": t2_aggregation,
    "t2_aggregation_alpha2": t2_aggregation_alpha2,
    "p0_c_alpha_width256": p0_c_alpha_width256,
    "p0_capacity": p0_capacity,
    "t1_setup_k_ladder": t1_setup_k_ladder,
    "p1_aprime_alpha": p1_aprime_alpha,
    "p1_k_collapse_wd": p1_k_collapse_wd,
    "p1_k_collapse_budget": p1_k_collapse_budget,
    "x_d_alpha_fine": x_d_alpha_fine,
    "x_d_alpha_cliff": x_d_alpha_cliff,
    "x_d_alpha_high": x_d_alpha_high,
    "x_d_internals": x_d_internals,
    "x_d_wd_ladder": x_d_wd_ladder,
    "x_d_lr_control": x_d_lr_control,
    "s5_setup_c_capacity": s5_setup_c_capacity,
    "s5_mnist_fl": s5_mnist_fl,
    "s5_probe_rerun": s5_probe_rerun,
    "s5_k50_diagnosis": s5_k50_diagnosis,
    "t4b_participation": t4b_participation,
    "t3a_dirichlet_band": t3a_dirichlet_band,
    "t3a_size_control": t3a_size_control,
    "s5_central_anchor": s5_central_anchor,
    "s5_mnist_working_point": s5_mnist_working_point,
    "s5_fl_probe": s5_fl_probe,
    "t0_wd_grid": t0_wd_grid,
    "t0_poly_pilot": t0_poly_pilot,
    "t0_mnist_wd_band": t0_mnist_wd_band,
    "t1_probe": t1_probe,
    "t1_replication": t1_replication,
    "t2_phase_diagram": t2_phase_diagram,
    "t2_k_breakdown": t2_k_breakdown,
    "t2_boundary": t2_boundary,
    "t3_server_lr_calibration": t3_server_lr_calibration,
    "t3_algorithm_comparison": t3_algorithm_comparison,
}


# Manifests allowed to DROP run ids they previously claimed. write_manifest
# refuses that by default, because an orphaned id means the launcher stops
# recognising a banked run as done and re-runs completed work. That guard is
# right for every ordinary edit; it has to be overridden when the dropped cells
# are ones that must never run again -- which is what its own error message says
# force is for. One entry, with the reason:
FORCE_REWRITE = {
    "x_d_alpha_high": (
        "drops the five alpha=1.0 cells. alpha is the TRAINING fraction, so 1.0 "
        "leaves no test set; compute_accuracy over zero samples returns NaN, and "
        "NaN passed the old sustained-crossing scan, banking them as "
        "grokked=True at t_grok=0. split_indices now rejects the spec outright, "
        "so leaving them in would only queue five guaranteed failures. Their "
        "result JSONs stay on disk as the record."
    ),
}


def main():
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    which = sys.argv[1:] or list(BUILDERS)
    for name in which:
        if name not in BUILDERS:
            raise SystemExit(f"Unknown manifest {name}; options: {list(BUILDERS)}")
        specs = BUILDERS[name]()
        # Only the new staged manifests are reordered. The completed ones are
        # left byte-for-byte as they ran, so their record stays exactly as
        # executed -- and write_manifest would refuse to orphan their ids anyway.
        # t2_aggregation_alpha2 is named explicitly rather than by a "t2_"
        # prefix: the other t2_ manifests have already run and their files are
        # the record of the order they ran in. Simulated on the outstanding
        # cells, FIFO in build order finishes in 16.7 h against 14.2 h
        # longest-first, because setup D's 6.2 h K=50 runs sit in the fifth
        # block and would start last.
        if (name.startswith(("s5_", "x_", "p1_", "t5_"))
                or name in ("t2_aggregation_alpha2", "t3a_dirichlet_setups")):
            specs = _longest_first(specs)
        path = os.path.join(MANIFEST_DIR, name + ".jsonl")

        # Idempotence: if the file already claims exactly this set of run ids,
        # leave it alone. Builders have been edited over the campaign's life, so
        # regenerating can emit the same cells in a different ORDER -- which
        # rewrites a completed sweep's record for no reason and shows up as a
        # spurious diff (t2_boundary, 20 lines reordered, ids untouched). Order
        # only ever mattered for scheduling, and a manifest that has already run
        # has nothing left to schedule. A genuine change still writes, because
        # any edit to a spec changes its content hash and therefore the id set.
        unchanged = (os.path.exists(path)
                     and {s["id"] for s in load_manifest(path)}
                     == {s.get("id") or run_id(s) for s in specs})
        if not unchanged:
            write_manifest(specs, path, force=name in FORCE_REWRITE)
        est = sum(estimate_minutes(s) for s in specs)
        print(f"{name}: {len(specs)} runs -> {os.path.relpath(path)} "
              f"(~{est / 60:.1f} slot-hours){'  [unchanged]' if unchanged else ''}")


if __name__ == "__main__":
    main()
