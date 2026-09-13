# exp5 across setups — the algorithm comparison the paper's contribution 3 needs

Written 2026-09-09. **Update 2026-09-11: B's calibration block is read and
Phase 2's builder is written** (`t6_algo_comparison`, B filled in, 63 runs to
run). Headline: the adaptive optimum is setup-dependent — server_lr 0.03 on B
against the anchor's 0.1, with 0.1 destroying memorisation on B — so the
calibration this plan exists to buy was necessary, not hygiene. FedProx loses
at every mu on B, confirming §17.1 rather than overturning it. Numbers and the
wd=0.1/wd=1.0 decay fork are in RUNS_TODO entry 0.

Status: **Phase 0 done; Phases 1 and 4 launched 2026-09-09
on cam-gpu-acs** via `scripts/run_algo_chain.sh` (RUNS_TODO entry 0). Phase 2
is written once Phase 1's rungs are selected; Phase 3 once Phase 2 names the
winners. Changes from the first draft: SCAFFOLD Option I is in the calibration
manifest for B–E (12 runs) so Phase 2 is not its first run off the anchor; B's
calibration budget is the audited 40,000-round rung budget, not the
20,000-round control; the H2 mechanism block also carries FedProx at μ=0.001,
v1's grokking value; damped FedAvg at 0.1 has 3 seeds, not 5.

## Why

Contribution 3 of the paper (drift-mitigation techniques against grokking) rests on
`t3_algorithm_comparison` (90 runs) and `t3_server_lr_calibration` (42 runs).
Both are setup A only. As of `runs_v2.csv` at 1,685 rows:

| method | runs | setups | hyperparameters covered | comparison cells | checkpoints |
|---|---|---|---|---|---|
| FedAvg | 712 | A, A′, B, C, D, E | K, E, f, partition, dir_α, wd | hundreds | 653 runs |
| FedAdam | 27 | A | server_lr ∈ {0.01, 0.1, 0.3, 1.0} at E=5; τ=1e-3 fixed | H1, H2, H3 | none |
| FedYogi | 27 | A | as FedAdam | H1–H3 | none |
| FedAvgM | 33 | A | server_lr ∈ {0.1, 0.3, 1.0} × momentum ∈ {0, 0.9} | H1–H3 | none |
| FedProx | 15 | A | **μ = 0.01 only** (v1 swept μ; v2 did not) | H1–H3 | none |
| SCAFFOLD | 15 | A | no knobs; raises under AdamW | H1–H3 | none |

Four defects, each of which a reviewer can find from the tables:

1. **FedProx was compared at the μ that fails.** v1's exp5 ran μ ∈ {0.001, 0.01,
   0.1, 1.0}; μ=0.001 grokked 3/3 on H1 and H3 and μ≥0.01 never grokked anywhere.
   v2 fixed μ=0.01 with no calibration arm, so §17.1's "FedProx loses to the
   baseline" and §17.4's direction-vs-magnitude argument rest on an uncalibrated
   method.
2. **Calibration and comparison run at different E.** Server LRs were chosen at
   E=5 on one cell and used at E=25 (H1, H2) and E=50 (H3). RESULTS §10 flags
   the transfer as unchecked.
3. **Every comparison cell is one FedAvg wins 5/5.** The comparison measures
   speed on cells where FedAvg groks (45,500 / 61,000 / 31,000). No alternative
   method has been run on any cell where FedAvg fails — B at dir_α=0.1 (0/3,
   does not memorise), D at E=50 (0/3, the §22 equilibrium), D's structured
   partitions (0/3), E's `label_block` (0/3), C's `target` (0/3), or any K=50
   AdamW cell. Whether mitigation *rescues* a breakdown is the result the paper
   does not have.
4. **No algorithm run carries weights** (`checkpoint_every=0` on all 132), so the
   spectral-IPR mechanism of §16.2 cannot be read on SCAFFOLD or FedProx, and the
   damped-FedAvg control that §17.4 and §18.4 both name as the settling test was
   never written.

## Phase 0 — code, before any run

1. **Determinism.** `results.sort(key=lambda r: r[0].cid)` before aggregation;
   `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
   a regression test that two runs of a small config produce identical
   histories. This is `RUNS_TODO`'s reproducibility entry and it gates everything
   here: a 5-seed comparison on C without it is noise (§23).
2. **SCAFFOLD Option I.** Accumulate the mean local gradient during the local
   steps and use it as `c_i⁺` directly (Karimireddy et al.'s Option I). Unbiased
   under any local optimiser, so the AdamW guard in `_build_strategy` becomes a
   choice of estimator rather than a refusal. ~30 lines in
   `training/scaffold.py` plus an accumulation hook in `GrokClient.fit()`.
   Test: under GD at E=1, Option I and Option II agree to fp32 tolerance
   (`x − y_i = η·Σg` holds exactly there). Keep Option II as the default on GD
   so the 15 banked anchor SCAFFOLD runs stay reproducible.
3. Nothing else needs code. `proximal_mu`, `server_lr`, `server_momentum`, `lr`,
   `checkpoint_every` and `checkpoint_client_weights` are all in the run-id
   hash already, so every arm below is a manifest builder.

## Phase 1 — calibrate every method on every setup

One cell per setup: its banked K=10, E=5, iid working point from
`t1_setup_k_ladder` / `t2_k_breakdown` (the same controls `t5_local_epochs`
dedups against), on the control's own budget so the FedAvg baseline is free.

| setup | working point | rounds | banked FedAvg `t_first_cross` | wall/run |
|---|---|---|---|---|
| A | α=0.30, wd=0 | 10,000 | 12,900 (3/3) | ~0.4 h |
| B | α=0.30, wd=0.1 | 20,000 | 55,900 (3/3) | ~0.6 h |
| C | α=0.40, wd=1.0 | 40,000 | 7,800 (3/3, withheld §23) | ~2.9 h |
| D | α=0.50, wd=1.0 | 50,000 | 78,900 (3/3) | ~0.9 h |
| E | n_train=1000, wd=0.1 | 8,000 | 5,200 (3/3) | ~0.25 h |

Ladders, 3 seeds each:

| method | knob | rungs | configs |
|---|---|---|---|
| FedAdam | server_lr | 0.01, 0.03, 0.1, 0.3 | 4 |
| FedYogi | server_lr | 0.01, 0.03, 0.1, 0.3 | 4 |
| FedAvgM | (server_lr, momentum) | {0.3, 1.0} × {0.5, 0.9} | 4 |
| FedProx | μ | 1e-4, 1e-3, 1e-2, 1e-1 | 4 |

16 configs × 3 seeds × 5 setups = **240 runs**. The anchor's FedAdam/FedYogi/
FedAvgM rungs at 0.1 and 1.0 partly dedup against `t3_server_lr_calibration`
(different cell: that was dir_α=0.1; this is iid, so they do not hash-match —
accept the 12-run overlap for a matched panel). Cost from the wall column:
A 20 · B 30 · C 140 · D 45 · E 12 ≈ **250 slot-h, 110 without C**.

The anchor's adaptive cliff sat between 0.1 and 0.3; 0.03 is added so the
optimum is bracketed on both sides on setups whose loss scale differs (CE on
B–E against MSE on A).

> **Selection rule, fixed before launch.** Per (setup, method): the rung with
> the lowest median `t_first_cross` among rungs that grok 3/3; ties within the
> seed spread go to the smaller step size (smaller server_lr, smaller μ). A
> method with no 3/3 rung on a setup is reported as failing calibration there
> and still enters Phase 2 at its best partial rung, labelled. Read
> `t_memo` beside `t_first_cross`: an adaptive server optimiser that moves
> memorisation is the decay clock (§14.3) wearing a new label.

## Phase 2 — the fair comparison, two cells per setup

Every method at its Phase-1 point, **5 seeds**, `checkpoint_every` on and
`checkpoint_client_weights=True` (this is what §16 could not read on the
algorithm arms). Two cells per setup:

**Cell W** — the working point above. Speed comparison; FedAvg is banked at
3 seeds, so 2 more seeds per setup.

**Cell R** — a banked cell where FedAvg fails. Rescue comparison; this is the
new result.

| setup | cell R | banked FedAvg | budget | why this cell |
|---|---|---|---|---|
| A | H2: α=0.25, K=10, E=25, dir_α=0.1 | 5/5 at 61,000 | 10,000 rounds | the anchor's comparison cell, kept so the panel has a line the paper already discusses |
| B | K=10, E=5, dir_α=0.1 | **0/3**, peak train 31% | as `t3a_dirichlet_setups` | fails to memorise (§20) |
| C | K=10, E=5, `target` | **0/3** | as `t3b_partitions` | incoherent structure kills it (§17.2) |
| D | K=10, E=50, iid | **0/3** at 250k and 2M steps | 250,000 steps | the memorise-never-generalise equilibrium (§22) |
| E | K=10, E=5, `label_block` | **0/3** | as `t3b_partitions` | the only MNIST breakdown |

Methods: FedAvg, FedProx, FedAvgM, FedAdam, FedYogi, SCAFFOLD (Option II on A,
Option I on B–E). 6 × 5 seeds × 10 cells = 300, minus banked FedAvg (3 seeds
on 9 cells, 5 on H2) and the banked anchor H2 arms ≈ **220 runs**. Cost, by
the cells' banked per-run walls: ≈ **270 slot-h, 190 without C**. D's E=50 cell
is the long one (~2 h/run); it is also the one where a rescue would matter most.

> **Decision rules.**
> - Cell W: rank methods on median `t_first_cross`; report the ratio to
>   FedAvg with the bootstrap CI. The anchor's 10–20× adaptive advantage either
>   generalises or is an MSE/GD anchor property.
> - Cell R: the order parameter is `frac` grokked, not time. A method that
>   takes a 0/3 cell to ≥3/5 has rescued it. **Check `peak_train_acc` first**:
>   on B and C the failure is memorisation, so a rescue must move `t_memo`
>   from ∞; on D memorisation is intact and a rescue must break the
>   equilibrium (weight norm and drift/round must leave 101.5 / 8.0).
> - SCAFFOLD vs FedProx at calibrated μ is the §17.4 argument re-run fairly.
>   If FedProx at μ=1e-3 groks where μ=0.01 did not, §17.4's "suppressed
>   magnitude" reading is withdrawn and the damped-FedAvg control (Phase 4)
>   becomes the only remaining discriminator.

## Phase 3 — axes for the winners

Best adaptive method (from Phase 2 W) and SCAFFOLD Option I, 3 seeds:

- **K ladder** K ∈ {5, 20, 50} at E=5 iid on **B and D**, the two setups where
  FedAvg's memorisation collapses with K (§14.3). K=10 is Phase 2. Budgets from
  `t1_setup_k_ladder`'s banked cells, headroom above the banked `t_memo(K)`.
  Question: does a server-side adaptive step or drift correction move the
  decay clock, or only the delay? 2 methods × 3 K × 2 setups × 3 = 36 runs.
- **E ladder** E ∈ {25, 50} at K=10 iid on **A and B**, compute-matched as in
  `t5_local_epochs`. This is the calibration-transfer check §10 asks for: if
  the E=5 server_lr is off its cliff at E=50, the H1–H3 numbers need a note.
  2 methods × 2 E × 2 setups × 3 = 24 runs, plus FedAvg controls banked.
- **Participation** f ∈ {0.5, 0.25} at K=20 on B for the best adaptive method
  only: the server-side optimiser sees a noisier pseudo-gradient at low f.
  12 runs.

≈ **72 runs, ~110 slot-h.**

## Phase 4 — mechanism

- **Damped FedAvg.** The anchor's H2 cell, FedAvg with local `lr` scaled by
  {0.5, 0.2, 0.1}, 5 seeds, checkpoints on. 15 runs, ~10 slot-h. Separates
  "FedProx suppresses the learning signal" from "FedProx is a smaller step".
  If damped FedAvg at the step size FedProx's proximal term implies also fails
  to grok, FedProx's failure is a step-size story and §17.4 is rewritten; if it
  groks, the proximal term is doing something the step size does not.
- **Spectral IPR per method**, read from Phase 2's checkpoints with
  `analyze_mechanism.py`'s global-model path: does SCAFFOLD build Fourier
  structure earlier than FedAvg, as operand sharding does (§16.2)? No new runs.
- **Client divergence per method** is already logged on every run; the Phase 2
  table gets `client_weight_divergence` beside `t_first_cross`, per setup.

## Cost and order

| phase | runs | slot-h | slot-h without C | needs |
|---|---|---|---|---|
| 0 code | — | — | — | determinism, SCAFFOLD Option I, tests |
| 1 calibration | 240 | ~250 | ~110 | Phase 0 |
| 2 comparison | ~220 | ~270 | ~190 | Phase 1 points |
| 3 axes | ~72 | ~110 | ~110 | Phase 2 winners |
| 4 mechanism | 15 | ~10 | ~10 | nothing (can run first) |
| **total** | **~550** | **~640** | **~420** | |

~80 h wall at 8 slots; ~55 h without C. Phases 1 and 2 are the minimum the
paper's contribution 3 needs. Phase 4's damped-FedAvg control is independent
and cheap, so launch it alongside Phase 1.

**C is run last and is droppable.** Its numbers are withheld under §23 until
Phase 0 lands, it is the most expensive setup per run, and its target cell is
a training failure. If Phase 0's determinism fix does not make C reproducible
at n=5, state C qualitatively and spend its slot-hours on 5 seeds in Phase 3.

Costs use banked per-round walls on cam-gpu-acs (A 150, B 204, C 260, D 167,
E 201 ms/round at K=10, E=5, four runs per card). Every cost estimate in this
project has carried ~2× uncertainty; budget the venue for that.

**Venue.** Per RUNS_TODO entry 6, `dev-gpu-acs` is not permitted for
experiments. CSD3 (SL3: 3,000 GPU-h/quarter, 12 h/job) fits; no single cell
here exceeds ~3 h. Stagger slot starts or cap at ~8 concurrent Ray heads.

## Not in this plan, and why

- **FedDyn.** Not implemented; needs per-client state like SCAFFOLD. Add only
  if SCAFFOLD Option I rescues a cell and a second drift-correcting method is
  wanted as replication.
- **FedNova, MOON, FedDecorr, FedBN, robust aggregators.** Out of scope per
  RESULTS §10.
- **A′.** Out of the paper; uninterpretable on partitions (§17.2).
- **τ, β₁, β₂ for the adaptive methods.** Flower defaults; one server_lr
  ladder per setup is the calibration the field does. Note it as a limitation.
