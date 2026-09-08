# RUNS_TODO

What still needs to be run, decided from scratch.

This supersedes every "Next" / "Outstanding" / "Stale" list that used to live in
`PROGRESS.md`, `RESULTS.md` and `plans/`. Nothing carries over by default — an
item appears here only after it has been looked at explicitly and kept.

Ground truth for what is already banked: `results/data/runs_v2.csv` and
`results/data/runs/*.json`.

---

## To run

### 1. Setup B at wd=0, federated — is the memorisation collapse about DECAY or about ADAMW?

> **SUPERSEDED IN ONE FIELD by entry 2 (2026-08-23): move α from 0.30 to 0.70.**
> At α=0.30, wd=0 cannot grok at any budget — B's decay band fits
> `t_first_cross ≈ 4,500/wd`, which diverges as wd→0, and the banked centralized
> control confirms it (2.5–5.9% test after 100,000 epochs). Entry 2 measured the
> α ladder: wd=0 groks **3/3 at α=0.70** with a **~28,850-step delay** intact.
> At α=0.70 the arm therefore answers the `t_memo(K)` question below *and*
> carries a `delay(K)` reading, instead of being a guaranteed null.
>
> The cost of moving: the banked wd=1.0 and wd=0.1 federated cells at K=10/20
> are at α=0.30, so the matched wd ladder has to be re-run at α=0.70 (~12 runs,
> cheap — everything is fast at that α) or the comparison is unmatched. The
> manifest `manifests/x_b_wd_zero_fl.jsonl` below is still at α=0.30 and has NOT
> been regenerated. Decide the α before launching it.

**Manifest** `manifests/x_b_wd_zero_fl.jsonl` — 9 specs, **6 to run** (the 3
centralized cells hash to runs already banked in `x_controls` and are skipped).

| | |
|---|---|
| setup | B (Nanda transformer, mod-113, CE+AdamW, lr 1e-3, width 128) |
| **weight_decay** | **0.0** |
| alpha | 0.30 |
| arms | centralized (banked) · federated **K=10** · federated **K=20** |
| E / partition / strategy | 5 · iid · FedAvg, full participation |
| seeds | 42, 123, 456 |
| budget | 20,000 rounds = **100,000 steps** (centralized arm: 100,000 epochs) |
| cost | **~8.6 slot-hours** — ~71 min/run at K=10, ~102 min/run at K=20, from the median `wall_s` of the matched banked cells. (`build_manifests` prints 19.4 h; its fitted model over-costs non-anchor setups ~2.6×.) |

**Why.** RESULTS §14.3's decay clock says memorisation blows up with K where
decay is decoupled and stays flat where it is absent — AdamW's decay is applied
per local step and does not scale with shard size, the learning signal from a
1/K shard does. The evidence is B/C/D/E (AdamW, wd>0) degrading against A (GD,
wd=0). **But A differs from B in the optimiser as well as the decay**, so
"AdamW" and "decoupled decay" are confounded in the one comparison the mechanism
rests on. B at wd=0 is AdamW with no decay clock, everything else fixed.

**Decision rule.** Read `t_memo(K)`, not `t_grok`.

- `t_memo` **flat in K** (≈ the centralized 200) while the wd=0.1 and wd=1.0
  ladders climb → the decay clock is about decay; §14.3 stands as written.
- `t_memo` **climbs with K anyway** → the collapse is a property of AdamW under
  fragmentation, and the paper has to say "adaptive optimiser", not "decay
  clock".

**These runs will NOT grok in budget, and that is the design, not a censoring
risk.** The banked centralized wd=0 curves end at 3.21 / 5.86 / 2.49% test
(chance 0.88, bar 95) with final-quarter slopes of +0.021 / +0.058 / +0.001
points per 1,000 steps. Extrapolated, the fastest seed needs **~1.55 M further
steps** to reach the bar and the slowest ~90 M. Centralized is the upper bound
for the federated arms on this setup, so a federated null at 100,000 steps
reproduces §15.4's centralized result rather than reporting a federated effect.
`t_memo` is the reading, and it cannot be censored here: at wd=0 it is bounded
above by the wd=1.0 value of 12,300 steps at K=20 (less decay memorises faster),
which 100,000 steps clears 8×.

**Do not trim the budget to the `t_memo` requirement.** 8,000 rounds would read
`t_memo` at ~3.5 slot-hours, but `num_rounds` is inside the content hash, so a
longer budget is a different run id — a short guess is re-run from scratch, not
resumed. It would also foreclose the one hypothesis under which these runs
return something other than a null: setup C's delay *collapses* with K and at
K=50 its first crossing precedes memorisation, so if averaging acts as an
implicit regulariser, B at wd=0 could generalise federated where it does not
centrally.

**Budget is set from the arm it must out-live, not from this arm's expectation.**
Banked `t_first_cross` at wd=0.1 reaches 77,200 (K=10) and 98,200 (K=20).
Anything shorter converts an honest null into the ninth censored boundary in
this project. 100,000 steps also matches the wd=1.0 aggregation arm and the
centralized wd=0 arm, so all three decay levels are read at one budget.

**What it completes** — setup B, α=0.30, iid, E=5, 3 seeds, banked `t_memo` range
and grokked fraction:

| wd | centralized `t_memo` / `t_fc` | K=10 | K=20 |
|---|---|---|---|
| 1.0 | 150 / 45,050 | 4,800–8,000 · 3/3 | 6,100–12,300 · 3/3 |
| 0.1 | 150 / 4,350 | 1,300–1,400 · 3/3 | 3,500–4,100 · 3/3 |
| **0.0** | **200 / never** | **← this** | **← this** |

```bash
setsid nohup venv/bin/python -u scripts/launch_sweep.py \
    manifests/x_b_wd_zero_fl.jsonl --gpus 0 --per-gpu 4 \
    > logs/sweeps/b_wd_zero_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

Not a working-point change: B's published config stays wd=1.0, and new
transformer work stays at the 2026-08-20 wd=0.1 decision. This is a tier-X
mechanism control.

---

### 2. Setup B at wd=0 — a centralized α ladder. **DONE**

`manifests/x_b_wd_zero_alpha.jsonl` — **15/15, 0 failures, 289 min wall**,
2026-08-23. B centralized, wd=0.0, 100,000 epochs, `log_every=50`, 3 seeds.

**ANSWER: yes, α makes B grok with no weight decay. The working point is α=0.70.**

| α | grokked | median `t_first_cross` | delay over `t_memo` | vs wd=0.1 | final test per seed |
|---|---|---|---|---|---|
| 0.4 | 0/3 | — | — | — | 13.8 · 28.3 · 14.1% |
| 0.5 | 0/3 | — | — | — | 43.1 · 69.6 · 74.6% |
| 0.6 | 1/3 | 95,300 | 95,150 | 30× | 95.5 · 84.5 · 93.9% |
| **0.7** | **3/3** | **29,000** | **28,850** | 19× | 99.0 · 96.5 · 97.4% |
| 0.8 | 3/3 | 600 | 450 | — | 98.0 · 99.3 · 98.2% |

`t_memo` = **150 at every α and every seed**, so the whole α effect is on the
delay and none of it is a training difference. Per-seed `t_first_cross` at
α=0.7 is 3,900 / 29,000 / 31,200 and at α=0.8 is 450 / 600 / 1,650.

**Why 0.70 and not 0.80.** The decision rule was the *lowest* α that groks 3/3,
because a federated K effect needs a delay to act on. α=0.8 collapses the delay
to **450 steps** — §13.3's A′ trap, "no delay left for federation to disrupt".
α=0.7 keeps ~28,850, which is 19× what wd=0.1 has at the same α precisely
*because* nothing is driving the transition.

**The sub-threshold rungs are censored, not decided.** Final-quarter slopes and
the steps still needed to reach the bar: α=0.5 seed 123 at +0.604 pts/1k (~42k
short), α=0.6 seed 123 at +0.223 (~47k short), α=0.6 seed 456 at +0.126 (~8.5k
short). Every cell at α≥0.5 was still climbing at the ceiling, so α=0.6 would
likely be 3/3 at ~150,000 epochs. The α boundary is a smooth transition the
budget cuts across, not a cliff — **do not quote α=0.7 as "the threshold"**,
quote it as the lowest rung that groks 3/3 *within 100,000 epochs*.

**Read `t_first_cross`, never `t_grok`, on this ladder.** α=0.8 seed 42 crosses
at 600 and records `t_grok` = 91,200; α=0.7 seed 42 crosses at 3,900 and records
65,300. Post-crossing dips dominate `t_grok` here exactly as §14.4 describes.

**Side finding, still unacted.** B's decay band fits `t_first_cross ≈ 4,500/wd`
at α=0.30, which puts `p1_b_decay_band`'s wd=0.03 and wd=0.01 cells at ~150,000
and ~450,000 steps required against the **50,000** they were given. They are
censored, not decided, and RESULTS §13.5's "sharp threshold below wd=0.1" is a
clock artifact.

---

### 3. Setup B at wd=0, α=0.40 to 1,000,000 steps — the low-α anchor. **DONE**

`manifests/x_b_wd_zero_a04_long.jsonl` — **3/3 completed, 0 failures, 386.7 min
wall** (6.44 h/run, against a 8.8 h estimate). 2026-08-24.

**RESULT: 0/3, and the pre-registered prediction was WRONG.** The prediction on
record was 1/3–2/3 — seed 123 "safe" at ~283,000 from a linear extrapolation of
its accelerating slope. What happened instead:

| seed | test @100k | @300k | @500k | @1M | plateau |
|---|---|---|---|---|---|
| 42 | 13.8% | 39.8% | 70.1% | **70.2%** | flat from ~500k |
| 123 | 28.3% | 56.1% | 63.4% | **63.7%** | flat from ~400k |
| 456 | 14.1% | 36.3% | 60.7% | **70.2%** | flat, one late step at ~850k |

**Why: the gradient dies, and with wd=0 nothing else moves the weights.** Train
accuracy is 100% from step 150 and stays there, so `train_loss` decays toward
zero and reaches **exactly 0.0 in float32** at step **233,600 / 212,450 /
257,450**. After that there is no gradient, and at wd=0 there is no decay term
either, so the model is at a fixed point:

| seed | weight-norm range after 500k | test range after 500k |
|---|---|---|
| 42 | **0.0000** | 70.1–70.2% |
| 123 | 0.0007 | 63.3–63.7% |
| 456 | 2.2198 | 53.9–70.3% |

Two seeds froze completely. Seed 456 kept a little residual drift — the loss
oscillates between 0 and the ~2.33e-11 float32 floor and Adam normalises by
`sqrt(v)`, so denormal gradients can still produce steps — and it gained ~10
points late. That is the exception that shows the rule.

> **This is the first negative result in this project that is NOT a clock
> running out.** Eight apparent boundaries have dissolved into censoring on
> re-measurement. This one does not: at wd=0 the run does not slow down, it
> *stops*, and no budget reaches the bar. The lesson for the method is that
> extrapolating a wd=0 curve from a pre-saturation window overestimates badly —
> the process has a hard stop the early window cannot see. Every slope-based
> forecast in this file inherits that caveat.

**What it means for the science.** Three things fall out of one mechanism:

- **It explains `t_first_cross ≈ 4,500/wd`.** Once the training loss dies, decay
  is the only remaining force, so the rate of post-memorisation travel is
  proportional to wd and the requirement diverges as wd→0. The law and this
  plateau are the same fact seen twice.
- **It reframes what α buys.** Grokking without decay is a **race between
  generalisation and gradient death**. α sets who wins: at α=0.4 the model
  reaches 60–70% before the loss underflows around ~230,000 steps and is then
  frozen there; at α≥0.6 it crosses the bar first.
- **The plateau is a measurable quantity** — the generalisation available "for
  free" from the memorising solution at a given α. ~70 / 64 / 70% at α=0.40.

**Caveat to state if this is published.** The hard stop is partly a *precision*
fact: CE on a perfectly memorised training set decays toward zero and float32
makes it exactly zero. In float64, or with label smoothing or a loss floor, the
gradient would persist and the plateau might not be exact. The mechanism —
nothing drives the weights once the loss dies and decay is off — is real either
way, but "no budget suffices" is a claim about this numerical setup.

**Consequence for entry 2's follow-up, and it survives.** The α=0.6 rung's two
uncrossed seeds need ~8,700 and ~47,000 more steps (to ~109k and ~147k total),
while gradient death arrives around 200,000–260,000. The curve wins that race,
so **α=0.6 → 150,000 epochs is still a good bet for 3/3** — and now for a
mechanistic reason rather than an extrapolated slope. At 100k every ladder run
still had `train_loss` ~1e-8, three orders above the floor, with weight norms
drifting 2–4.8 units per 25k steps, so none of them had stalled.

---

---

## The paper's three open FL axes

Added 2026-09-01 from the paper's `\section{Notes}`. The contributions the paper
claims are (1) grokking under FL, (2) FL-specific dynamics via FL-specific
hyperparameters, (3) drift-mitigation techniques against grokking, (4) mech
interp of why fragmentation delays it. Contribution 2 is the one the banked data
under-supports: **K** is measured on every setup and partition **structure** is
measured on every setup, but **local epochs, participation fraction and the
amount of heterogeneity are measured on the anchor alone** — 562 of 673 banked
federated runs sit at E=5, 664 at f=1.0, and every Dirichlet ladder rung outside
`t3b_partitions`' single 0.5 cell is setup A.

Three manifests, written and generated, none launched. **150 runs, ~267
slot-hours, ~67 h wall** at `--gpus 0 --per-gpu 4` (every cell is K ≤ 20, inside
the range where four concurrent runs cost nothing per-run).

Each extends the manifest whose figure it joins and inherits that manifest's
working points and decay — `t5_*` extend `t1_setup_k_ladder` (B at wd=0.1),
`t3a_dirichlet_setups` extends `t3b_partitions` (B and C at wd=1.0). That is not
a violation of the 2026-08-20 wd=0.1 standing decision: each new cell is read
against a banked control at its own decay, and switching a ladder off its own
control's decay buys a comparison that differs in two things. Every builder
docstring says this in place.

**Budget audit, 2026-09-01 — three controls failed and were rebased.** Every
control cell was checked against its banked `t_first_cross` before finalising,
per the headroom rule, and the first drafts repeated the project's signature
mistake three times:

| block | first draft | banked t_fc | fix |
|---|---|---|---|
| B participation (K=20, wd=0.1) | 100,000 steps | 66,100–98,200 *at 200k* | base → 40,000 rounds; control now hash-matches `p1_k_collapse_budget`'s banked cell |
| D participation (K=20) | K-ladder cell as control | control is **1/3 grokked** (inf · inf · 95,600) | rebased on `t3b_partitions`' iid K=20 cell — 250,000 steps, banked |
| D E-ladder (K=10) | K-ladder cell as control | control is **2/3** (seed 42 censored) | rebased on t3b's iid K=10 cell — 250,000 steps, 3/3 at 78,900, banked |
| B E-ladder (K=10) | 100,000 steps/rung | control 55,200–77,200 (1.3×) | control stays banked; **new rungs at 200,000 steps**, read on `t_first_cross` per §14.4 |
| E participation (K=20) | 20,000 steps (1.7×) | 11,100–11,900 | rebased on t3b's 8,000-round cell (3.6×), banked |

All rebased controls were verified to hash-match banked runs before
regeneration, so the fixes cost only the f/E arms' extra length (~42 slot-h) —
and every manifest's control column now dedups 3/3 on every setup. A and C
passed the audit unchanged (3.7× and >20×).

### 4. `t5_local_epochs` — the E axis on all five setups, at matched compute. **DONE**

> **48/48 run, 0 failures, 65 slot-h, 2026-09-02.** Readings in RESULTS §19; the
> D E=50 and C E=50 cells were re-run at 8×/5× budget (`x_e50_long`, entry 8).

`manifests/t5_local_epochs.jsonl` — 63 specs, **48 to run**, ~32 slot-h.
K=10, iid, E ∈ {5, 10, 25, 50}, 3 seeds, each setup at its K-ladder working
point. E=5 is the banked K-ladder cell and dedups (15 of the 63).

**Rounds scale as 5/E, so total gradient work is FIXED across the ladder** — the
opposite convention to the E-spine at the top of `build_manifests.py`, and
deliberately so: the anchor already has the fixed-rounds reading banked (§17.4's
E table), so the communication-matched view exists and the compute-matched one
does not. `eval_every` and `checkpoint_every` scale by the same factor, so every
rung is sampled on the same *step* grid; a fixed `eval_every` would read the E=50
cells at 10× the granularity of E=5 and confound the axis with its own
measurement resolution (§13.4, §14.4).

**A carries one extra rung, E=1, and only A can.** Under GD at momentum=0 FedAvg
at E=1 is an exact identity with centralized training, so it is the axis's
zero-drift endpoint with a known answer. On the AdamW setups §15.3 already
established the same cell is a cold-start-Adam artifact and measures nothing
federated. It is the most expensive rung here (5× the control's rounds, ~7 h of
setup A's ~7.2).

> **Decision rule.** Read `delay = t_first_cross − t_memo` against E on the
> `total_steps` axis, with `mean_client_drift` and `client_weight_divergence`
> beside it. Delay grows with E at matched compute → local work costs
> generalisation time and §18.4's "systematic disagreement" gains its cleanest
> manipulation. Delay flat → the anchor's E table was measuring the extra compute
> that fixed rounds handed the high-E cells, and §17.4's E column has to be
> withdrawn as evidence. `t_memo` climbing with E is the decay clock, not a drift
> result, and must be read apart.

The optimiser-restart confound is bounded rather than removed: `s5_fl_probe`'s
12 banked persist=True runs on B show persisting Adam state does not recover the
ceiling, so no new persist arm is bought. `persist_local_opt_state` stays **False**
everywhere — standard FedAvg semantics, and a no-op under pure GD.

### 5. `t5_participation` — partial participation on B, C, D, E. **DONE**

> **30/30 run, 0 failures, 134 slot-h, 2026-09-04.** Readings in RESULTS §21. Read
> on the ROUNDS axis — `total_steps` accumulates E·f per round, so step-denominated
> quantities halve with f by construction.

`manifests/t5_participation.jsonl` — 45 specs, **30 to run**, ~120 slot-h.
K=20, iid, E=5, f ∈ {1.0, 0.5, 0.25} → 20 / 10 / 5 clients per round, 3 seeds,
**all five setups**. A is in at K=20 despite `t4b_participation` being banked —
t4b is at K=50, and a panel whose anchor line sits at a different K from every
other line carries a second variable. A's control is `t2_k_breakdown`'s banked
K=20 iid cell (built bare — no strategy/fraction_train/checkpoint keys — so it
hashes to it and dedups); if A's K=20 axis is flat like its K=50 one, §18.3
gains a second K for free. E's K=20 is the degenerate rung (one full-batch step
per local epoch); constant across the f arms so it cancels within-setup, but
E's panel is not comparable to the others on the E semantics.

**K=20 and not the anchor's K=50** — on B, C and D, K=50 is where *memorisation*
collapses (§14.3, §17.2), and a participation null measured inside a training
failure says nothing about participation.

**Rounds scale as 1/f**, exactly as `t4b` did: a round at fraction f does f× the
gradient work, so a fixed round count would starve the low-f cells by precisely
the factor under test. `eval_every` and `checkpoint_every` scale with it. The
f=1.0 control is emitted **without a `fraction_train` key** so its hash matches
the banked K-ladder cell — 9 of the 12 controls dedup; B's K=20 cell is the one
that does not exist yet and costs 3 runs.

> **Decision rule.** Compare `t_first_cross` in rounds *and* in `total_steps`
> against the f=1.0 control, per setup, with `client_weight_divergence` beside
> it. Flat in rounds, as on the anchor → §18.4's "sampling noise averages out"
> holds across architectures and the paper can state it as a property of grokking
> under federation. Delay grows as f falls → read `t_memo` first: if that is what
> moves, this is the decay clock at a smaller effective client population, not a
> participation result.

**Cost warning: C's block is ~39 of the 80 slot-hours** (40,000 rounds at f=1.0
becomes 160,000 at f=0.25, on the slowest setup in the study). Specs are emitted
longest-first, so truncating after C's f=0.25 cells still leaves every other
setup complete.

### 6. `t3a_dirichlet_setups` — the heterogeneity ladder on B, C, D, E. **DONE**

> **72/72 run, 0 failures, 137 slot-h, 2026-09-03.** Readings in RESULTS §20. The
> anchor's ladder is flat: §18.1's 2.01× was starvation, as §18.2 said.

`manifests/t3a_dirichlet_setups.jsonl` — 87 specs, **72 to run**, ~114 slot-h.
K=10, E=5, `dirichlet_alpha` ∈ {0.01, 0.1, 1.0, 10, 1000}, 3 seeds, **all five
setups** at their `t3b_partitions` working points and budgets. Written against
`plans/exp3a-dirichlet-ladder-across-setups.md`, with three departures: **A′ is
dropped** (out of the paper), E's rung list follows the plan's own feasibility
table, and **A is in** — the plan's "A needs nothing" points at
`t3a_dirichlet_band`, which is K ∈ {20, 50} at α=0.25 and whose low rungs are
the starvation-contaminated cells of §18.2. At K=10 A's min shard at
dir_α=0.01 is 116–184 (verified per seed against these exact specs), so this is
the one version of A's ladder that is both matched-K to the other panels and
clean of the confound. The banked ladder stays as the §18.1/§18.2 reading.

**The 0.5 rung is free, and only if written bare.** `FedConfig.dirichlet_alpha`
defaults to 0.5 and `t3b_partitions`' banked cells omit the field, so naming it
explicitly re-runs banked work. The builder emits it as a bare spec: 15 of the 87
dedup, 3 per setup, exactly as the plan predicted.

**Partitions pre-flighted for these exact specs** (built, not assumed). Min shard
at K=10, dir_α=0.01: B 298, C 386, D 194 — an order of magnitude clear of the ≤2
that made §18.2's tail a starvation artifact, so no `dirichlet_sizes` control is
bought for the algebraic setups.

**E is the exception on both counts.** 10 classes against 97–120, so 0.01 empties
a shard and raises (its ladder starts at 0.1) — and its min shards are 5 / 25 / 56
at dir_α=0.1 and 62 / 83 at the banked 0.5 rung, against `batch_size`=100. E's low
rungs therefore sit **below one batch per client** and are in the starvation
regime, not a heterogeneity reading. Kept so E has a concentrated end at all, but
its panel must say so, and separating the two would need `dirichlet_sizes`
rebuilt for MNIST (12 runs, not written).

> **Decision rule.** Per setup, plot `t_first_cross` against `dirichlet_alpha`
> with the banked iid cell at the same K as the reference. Flat until the
> concentrated end on every setup → §18.1's threshold reading generalises. A
> different threshold per setup → read it against class count and shard geometry
> before calling it an architecture effect. **Any cell that fails to memorise is
> not a heterogeneity result** — check `peak_train_acc` first; on the AdamW setups
> the decay clock is the competing explanation and it has been right every time.

**CANCELLED 2026-09-02 — launched on CaMLSys by mistake, stopped after 15 min
with no results banked.** The intended venue was the CS department's shared L4
box (`dev-gpu-acs`), which is *not* permitted for experiments — interactive
development and <2-minute tests only, per Malcolm's email — so it cannot host
this either. Venue is an open decision: University HPC (CSD3) or vast.ai. What
follows is the CaMLSys record, kept because the job script and its findings
transfer to whichever Slurm venue is chosen.

Venue facts checked 2026-09-02: `dev-gpu-acs` (alias `cam-gpu-acs`, key
`id_cambridge`) is already accessible — host `gxp-l4-0`, 8× L4 24 GB, 32 cores,
503 GB — but the department page it points at, the local compute guide and
Malcolm's email all say the same thing: a few minutes at most, no training, one
GPU (it pins `CUDA_VISIBLE_DEVICES` and CPU affinity per login). CSD3 (alias
`csd3`) refuses key-only login — it needs password + TOTP interactively — so a
first login has to be done by hand; SL3 gives 3,000 GPU-h/quarter, 12 h/job, 32
GPUs, and this campaign is ~67 GPU-h at four runs per GPU.

What the 15 minutes established: the full 595-test suite passes on a Linux A40
node with torch 2.10.0+cu128 and flwr 1.27.0; and **16 Ray heads started at the
same instant produced Ray worker-registration failures on 2 of 16 runs within
60 s** (`Failed to register worker to Raylet ... End of file`). The 3080 box
never ran more than 4 concurrently. Whatever venue is next, stagger the slot
starts in `launch_sweep.py` (a few seconds apart) or cap at ~8 concurrent
before trusting a 16-wide sweep.

Jobs **55062 → 55063 → 55064**, chained with `afterany` because the `cls_master` QoS
allows one running job and the `ampere` partition caps at 12 h; each job
re-invokes the same idempotent script and `launch_sweep` resumes from the result
JSONs. Script: `scripts/slurm/camlsys_paper_axes.sh` — 4× A40, 44 CPUs, 200 GB,
`--per-gpu 4` (16 concurrent runs), sweeps in the order local_epochs →
dirichlet → participation. It builds a fresh venv on `/dev/shm` from
`pyproject.toml`, **runs the full pytest suite on the node as the gate** (the
non-FL half passes locally; `flwr` is only installed there), and only then
launches. Ray temp and object store are on `/dev/shm` too — `/nfs-share` is at
~180 of ~200 GB quota and mauao's `/tmp` was 99% full. **`flwr` is pinned to
1.27.\*** in the job: `pyproject` only lower-bounds it, the first submission
(55059, cancelled after 4 min with no results) resolved 1.36, and the banked
controls are all 1.27 — Flower owns the client-sampling RNG, so an unpinned
resolve would have been a version confound under the participation arms.

```bash
ssh taranaki "squeue -u mh2274"                                             # chain state
ssh taranaki "tail -40 /nfs-share/mh2274/federated_learning_grokking/logs/slurm-55062.out"
ssh taranaki "ls /nfs-share/mh2274/federated_learning_grokking/results/data/runs | wc -l"   # 1529 + done
# pull results when a job ends (JSONs + histories + checkpoints):
rsync -az taranaki:/nfs-share/mh2274/federated_learning_grokking/results/ results/
```

A run in flight when a job hits 12 h is re-run from scratch by the next job in
the chain, so **no single run may exceed 12 h**. The longest cells by estimate
are D's f=0.25 (~10.6 h on the 3080's per-round timing) and B's f=0.25 (~8.9 h);
check per-round timing in job 55059's logs before the participation sweep is
reached, and if A40+Slurm is slower than the 3080, hold those six cells back.

Original single-box launch, kept for reference:

```bash
for m in t5_local_epochs t3a_dirichlet_setups t5_participation; do
  setsid nohup venv/bin/python -u scripts/launch_sweep.py \
      manifests/$m.jsonl --gpus 0 --per-gpu 4 \
      > logs/sweeps/${m}_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null
done &
```

### 7. Contribution 3 is BLOCKED on SCAFFOLD, and the blocker is in the code

The paper's third contribution — drift-mitigation techniques against grokking —
currently rests on `t3_algorithm_comparison` (90 runs, §17.1), which is **setup A
only**, and on the SCAFFOLD-vs-FedProx contrast that §17.4 makes the mechanism
argument out of. It cannot be ported as written:

**`_build_strategy` raises on `strategy="scaffold"` whenever `optimizer="adamw"`**
(`training/federated.py:485`), and deliberately — SCAFFOLD's Option-II control
variate `c_i⁺ = c_i − c + (x − y_i)/(η·K)` inverts `x − y_i = η·Σg`, which holds
for SGD and not under Adam's per-coordinate preconditioning. Setup A is the only
GD setup in the study, so **SCAFFOLD is unavailable on B, C, D and E**, and it is
the only drift-correction method in §17.1 that works.

Three ways forward, unresolved:

1. **Implement Option I** — accumulate the mean local gradient during the local
   steps and use it as `c_i⁺` directly, which is unbiased under any optimiser.
   ~30 lines in `training/scaffold.py` plus the accumulation hook in the client
   loop, and it makes the whole drift axis portable. The Option-I/Option-II
   distinction is Karimireddy's own, so this is not an invention.
2. **Restrict contribution 3 to the anchor** and state it as a limitation, as
   §17.2 did for the partition claim.
3. **Port only the AdamW-safe methods** — FedProx's μ ladder, FedAvgM, FedAdam,
   FedYogi. Cheap in code, but the server-side adaptive methods need a per-setup
   `server_lr` calibration first (`t3_server_lr_calibration` was 42 runs on A
   alone), and §17.4's mechanism argument needs SCAFFOLD specifically: FedProx is
   the method that *fails*.

Also still unwritten, and flagged twice in RESULTS (§17.4, §18.4) as the one
control that would settle the mechanism: **FedAvg damped to FedProx's effective
step size**, ~15 runs on the anchor, separating "corrected direction" from
"suppressed magnitude".


### 8. `x_e50_long` — C and D at E=50, 5× and 8× budget. **DONE**

`manifests/x_e50_long.jsonl` — **6/6, 0 failures, 65 slot-h**, 2026-09-04.

**D: a real breakdown, decision rule (b).** 0/3 at 2,000,000 steps as at 250,000;
train 100% from step ~3,000, test 80.5 / 82.3 / 82.6%; weight norm 101.5 → 101.6,
drift/round 8.0 → 8.0, train loss 0.065 → 0.061 (min 0.055 — never the fp32
underflow of entry 3). A stationary equilibrium, not a slow run. RESULTS §22.

**C: verdict withheld.** 2/3 memorise and grok at 1M steps, but the same
config+seed at 200k had reached only 67% train at the step where the 1M run
reports `t_memo`. Trajectories are not reproducible on C — RESULTS §23 — so
"under-budgeted" and "unstable" cannot be separated with n=3. Not to be quoted.

## Reproducibility — the harness is not run-to-run deterministic

Found by the six duplicate cells above. Same config and seed, two runs: D differs by
0.00 on every seed; C by 12.7 / 0.0 / 7.7 points of peak train accuracy, against a
between-seed spread of 13.4. The client loop is seeded (and full-batch on C, so it
draws nothing), init is seeded, the probes draw nothing. What remains is
floating-point ordering: Flower's `aggregate()` sums client weights in ARRIVAL order
and nothing sorts them; no deterministic-algorithm flag is set. Ten float32 layers
summed in 200 orders give 200 results ~1.5 ulp apart, injected every round.

**To do, in this order.** (1) `results.sort(key=lambda r: r[0].cid)` before
aggregation — free. (2) `torch.use_deterministic_algorithms(True)` +
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. (3) A regression test that two runs of a small
config produce identical histories; none exists — `test_fedavg_identity` proves the
algebra, not run-to-run. (4) Either bring C's three key cells to n=10 (~21 runs,
~70 slot-h) or state C qualitatively. (5) Change "seeds" to "runs" in the paper's
statistics language.

## Decided against

**α=0.30 and α=0.40 to 250,000 / 200,000 steps** (10.7 slot-h). Every one of the
six cells still censors at those budgets: α=0.30 needs 1.65M–88M steps and
α=0.40's best seed needs ~283,000 against the 200,000 proposed. Cost without a
result.

**α=0.30 to 1,000,000 steps** (21.6 slot-h). 0/3 by extrapolation, and no seed
is accelerating. Would only be worth buying as a deliberate negative result at
B's own campaign working point — "wd=0 is dead at α=0.30 even at 1M" as a stated
limitation. Not currently needed.
