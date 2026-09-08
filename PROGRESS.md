# v2-multisetup — progress and resume notes

Working branch for the multi-setup rewrite. Read in this order:

| | |
|---|---|
| **`RUNS_TODO.md`** | what still needs running, decided from scratch. **Supersedes every "Next" / "Outstanding" / "Stale" list that used to live in this file.** |
| **`RESULTS.md`** | every measured number, with the run data behind it |
| **this file** | what is built, how to run it, and the reasoning that is expensive to re-derive |
| `plans/` | the campaign plans; index in `plans/README.md` |

**Ground truth is `results/data/runs_v2.csv`, not the prose.** These notes lag the
banked data; the CSV and the result JSONs do not. Regenerate with
`collect_runs.py`, and see §Resume for the exact "what is still missing" recipe.

The build changelog that used to occupy the middle of this file — nine
`Done — Phase N` sections — is in `git log`, which is where a changelog belongs.
Per-sweep result tables that duplicated `RESULTS.md` have gone the same way; the
section numbers in the table above are the citable ones.

## Current state — 2026-09-07

**1,685 runs banked · ~1,366 machine-hours · 0 failed runs all campaign.**

The paper's three FL axes ran 2–4 Sep on `cam-gpu-acs` (gxp-l4-0, 2× L4, 8
slots): `t5_local_epochs` 48, `t3a_dirichlet_setups` 72, `t5_participation` 30,
plus `x_e50_long` 6 — **156/156, 0 failures, 401 slot-h**. Readings in RESULTS
§19–§23; the run-by-run status and the reproducibility to-do are `RUNS_TODO.md`
entries 4–8. Headlines: D reaches a stationary federated equilibrium that
memorises and never generalises (§22, 8× budget); memorisation cost scales with
local work on transformers and not on MLPs (§19); the anchor is immune to
unstructured heterogeneity once starvation is controlled, contradicting §18.1
(§20); partial participation is free in rounds everywhere but MNIST (§21); and
**runs are not reproducible run-to-run** (§23) — zero effect on D, outcome-flipping
on C. Nothing from this sweep is committed yet.

**Hardware note.** The 8× L4 box is available again and `cl-alloc-gpu` assigns
each user one card by `UID % 8` (mine is GPU 1) plus that card's NUMA-node CPUs.
That is a convention, not a reservation; GPUs 1–2 were used with the group's two
other active users at ~2.5 cores between them. Per-round cost is near-linear in
E, not the mild scaling `build_manifests.estimate_minutes` assumes.

## Current state — 2026-08-24

**1,529 runs banked · ~965 machine-hours · 0 failed runs all campaign.**

The all-5 campaign is finished and its readings are in `RESULTS.md` §14–§18.
Live work is setup B's weight-decay investigation (`RUNS_TODO.md` entries 1–3):
the wd=0 α ladder and the million-step α=0.40 anchor are **done**, and the
federated wd=0 arm is written but not launched and needs its α decided first.

`t2_aggregation_alpha2` has 60 specs outstanding. The other 439 outstanding
specs belong to `t1_replication`, `t2_phase_diagram` and `t1_probe`, which
predate Gate A and have been "rewrite or delete" for weeks — they are a decision,
not a backlog, and `RUNS_TODO.md` is where that decision gets made.

### Standing decisions

**Transformer weight decay is 0.1 (2026-08-20).** Setups **B** and **C** use
`weight_decay = 0.1` for all new work, replacing the 1.0 inherited from Nanda. A
new manifest builder must override it explicitly (`{**SETUP_B, "weight_decay": 0.1}`);
`SETUP_B`/`SETUP_C` in `scripts/build_manifests.py` stay at 1.0 on purpose,
because `weight_decay` is inside the run-id hash and editing the constant would
rewrite `t2_aggregation` and orphan all 305 banked wd=1.0 transformer runs.
Budget ~3.5× the banked wd=1.0 numbers. Evidence in `RESULTS.md` §"short
version" item 9.

**Compare `t_first_cross`, not `t_grok`, across cells** whenever budgets or
logging rates differ (§14.4). `t_grok` requires the bar to hold for the rest of
the run, so a longer budget can report a 22× worse time on an identical
trajectory prefix.

**Budget every federated cell as `t_memo(K) + delay`,** never as a multiple of
the centralized `T_grok`. Which term carries the K dependence is setup-dependent:
on the anchor (GD, wd=0) memorisation is flat in K and the delay grows; on the
AdamW setups memorisation explodes. Measure it rather than assuming — eight
boundaries in this project were manufactured by getting this wrong.

## State

| | |
|---|---|
| Branch | `v2-multisetup` (branched from `main` @ `41c3fa8`; `main` has nothing this lacks) |
| Frozen reference | tag `v1-single-setup` — the state that produced the 32 figures in `results/figures/` |
| Tests | **592 collected** (`venv/bin/python -m pytest tests/ -q`, ~9 min incl. FL integration; the non-FL half is ~45 s) |
| Runs banked | **1,529** result JSONs, and `results/data/runs_v2.csv` matches at 1,529 (1,016 grokked, 513 censored). Plus 870 v1 runs in `runs.csv`. ~965 machine-hours |
| Setups | A quad-MLP/mod-97 · A′ quad-MLP/mod-97/AdamW (**measured**, §13.3) · B transformer/mod-113 · C transformer/S₅ · D quad-MLP/S₅ · E MLP/MNIST-1k. **D′ dropped** — the gate that would have required it opened (§13.7) |
| FL algorithms | FedAvg, FedProx, FedAvgM, FedYogi, FedAdam (native) + SCAFFOLD (adapted, **raises under AdamW by design**) |
| Statistics | censored survival (KM median + fraction-grokked + bootstrap CI); `scripts/summarize_runs.py` |
| deps | torch 2.10 + torchvision 0.25 (pinned pair); flwr 1.27 |
| Env | `venv/` (py3.10, torch 2.10, flwr 1.27); package installed via `pip install -e . --no-deps` |
| Hardware | **1× RTX 3080 Laptop (8 GB), 16 cores, 30 GB RAM** since 2026-08-17 — was 8× L4 (23 GB, 64 cores, shared). Every `--gpus 0,1,...` in this file and in `scripts/run_*.sh` is a historical record, not a runnable command. Use `--gpus 0 --per-gpu 4` (measured, below) |

## RESOLVED: there was never a K≈30 collapse

**The blocker is cleared and the K axis is open** (RESULTS.md §13.7). The whole
thing was a censored measurement, three times over.

The table that defined the blocker — setup B, peak train 100 / 98.2 / 42.8 / 5.9 /
5.0 at K = 10 / 20 / 30 / 40 / 50 — was read off runs given **10,000 steps**. At
K=30, memorisation alone takes **12,900**. Given 100,000 steps the same
configuration memorises and groks at 13,200. "42.8%" is where that run had got to
when the clock stopped.

**What federation actually does**, once `t_memo` is recorded separately from
`t_grok` (setup B, iid, E=5, α=0.30, wd=0.1):

| | t_memo | T_grok | delay |
|---|---|---|---|
| centralized | 150 | 45,050 (3/3) | 44,900 |
| K=20 | 3,600 | 93,700 (**3/3**) | 90,100 |
| K=30 | 5,900 | 94,300 (1/3) | 88,400 |
| K=50 | 53,300 | censored (0/3) | — |

Two separable effects, and only one depends on K:

1. **Memorisation slows steeply with K** — 24×, 39×, then 355× the centralized
   value at K = 20, 30, 50.
2. **The delay is large and roughly doubles under federation** — ~90,000 against
   the centralized 44,900, established at K=20 across 3 seeds.

**RESOLVED by the K ladder (§14.3): the delay DOES grow with K.** The low rungs
close the curve — 44,900 centralized → 50,400 (K=5) → 54,600 (K=10) → 90,100
(K=20) → 88,400 (K=30). It roughly doubles by K=20, then plateaus. So both terms
grow; what the decay clock controls is whether *memorisation* also blows up.

`T_grok(K) ≈ t_memo(K) + delay` therefore accounts for the censored cells: K=30
needs ~95,900 against the 100,000 it got (hence 1/3, at 94,300), and K=50 needs
≳143,000 and got 100,000. **143,000 stays a lower bound** — budgeting exactly to
an estimate is what recreates the censoring it came from.

**Consequences for the campaign:**

- **D′ is not needed.** The decision rule's first branch fired, so the A-vs-D
  optimiser confound is recorded as a stated limitation rather than fixed with a
  seventh setup. `p1_dprime_alpha` is not written and should not be.
- **Budgets come from `t_memo(K) + delay`**, never from a multiple of the
  centralized number. That is the rule every per-setup FL manifest must follow.
- **B's K ladder is bounded by cost, not capability.** K=97 extrapolates well past
  200,000 steps, so either B stops at K=50 or the campaign accepts one expensive
  high-K cell per setup.
- **No separate confirmation run.** Run ids are content hashes, so a campaign cell
  at K=50 with headroom above 143,000 *is* the test.
- **The decay clock is confirmed, not hypothesised** (§14.3) — D reproduces B's
  collapse on a different architecture and task. Expect any AdamW setup to lose
  memorisation at high K, and any wd=0 setup not to.

The only cell that still looks like a genuine wall is **K=50 at the inherited
wd=1.0**, which fails to memorise even at 100,000 steps — though at peak train
37.8% against the 5.0% recorded at 10,000, so it too is slow rather than stuck.

**Ruled out along the way:** weight-norm collapse, client drift, and local step
size (lower lr is *worse* — 4.6% train at lr=1e-4).

Careful with the number: lr·λ=1e-3 is *also* where the anchor's WD sweep found
decay outrunning learning, but that sweep was **GD** (coupled decay) and
RESULTS.md §6.4 makes the point that the optimiser is exactly what discriminates.
The numerical coincidence is not the argument; the federated wd evidence is.

## The optimiser is a control variable, and it is not currently controlled

The campaign is a 2×2 (architecture × task). A is GD+MSE (Gromov, inherited from
v1); B, C, D are AdamW+CE. So:

| comparison | isolates | verdict |
|---|---|---|
| B vs C | task, on the transformer | **clean** |
| C vs D | architecture, on S₅ | **clean** |
| A vs B | architecture, on modular | confounded (optimiser + loss) |
| A vs D | task, on the quad-MLP | confounded (optimiser + loss) |

B/C/D are internally consistent; A is the odd one out and **cannot move** — it is
Gromov's config, the anchor to 870 v1 runs, and every banked federated result. So
the fix is the missing cell, **A′** (`SETUP_A_PRIME`): A's architecture and task
under AdamW, MSE not CE so that A vs A′ is a single-variable optimiser contrast.

Provenance of the AdamW choices, since only two are inherited:

- **B** — Nanda's published config verbatim (lr=1e-3, wd=1.0, CE). Its value is
  fidelity; do not tune it or it stops being a replication.
- **E** — Omnigrok's family, but wd=0.1 was **measured here** (`t0_mnist_wd_band`).
  The only AdamW setup whose decay was chosen for the setup it runs in — and the
  only one that does not collapse at K≥20.
- **C** — inherited from B, and since **measured** (`p1_cd_decay_band`): wd=1.0
  is optimal for C too, so the inheritance was lucky rather than justified.
- **D** — **inherited from nothing.** Gromov's architecture running Nanda's
  optimiser. Now measured twice over: wd=1.0 is the only band D groks in at all
  (`p1_cd_decay_band`), and the quad-MLP *does* grok S₅ under GD+MSE
  (`p1_d_gd_probe`). The confound is therefore *closable* via D′ — but the K gate
  opened without needing it (§13.7), so **A vs D stays confounded and that is
  recorded as a stated limitation** rather than fixed with a seventh setup.

## Phase 1 — RUN. The pilots that define the setups.

All four executed; full readings and tables in **RESULTS.md §13**. Each manifest's
docstring carries the decision rule it was written against — read it before the
results, not after.

| manifest | runs | verdict |
|---|---|---|
| `p1_d_gd_probe` | 9/9 | **The quad-MLP does grok S₅ under GD+MSE** — lr=50 → 3/3 at 22,600 (α=0.5); lr∈{5,10} → 0/3. D′ was the gated response; the gate opened without it, so **D′ is dropped** and A vs D stays a stated limitation |
| `p1_b_decay_band` | 15/15 | **B's decay does not move either** — wd=1.0 is optimal, and every cell memorises at epoch 150, so decay acts purely on the generalisation timescale. Supplies the control the K-collapse arm was missing |
| `p1_cd_decay_band` | 30/30 | **Neither C's nor D's decay moves.** D groks only at wd=1.0 (0/3 everywhere else); C ties 3/3 at wd∈{0.1,0.3,1.0} and wd=1.0 wins on both statistics. New: **C is unstable** — it dips back below the bar after crossing it, up to 28 times at wd=0.3 |
| `p1_aprime_alpha` | 45/45 | A′'s cliff sits between α=0.175 and 0.20, **below A's**, and it groks **~45× faster at every shared α**. Its only usable federated working point is **α=0.20** — above that there is no delay left for federation to disrupt |
| `p1_k_collapse_budget` | 11/11 | **THE GATE, and it opened.** wd=0.1 groks **3/3 at K=20** given 200,000 steps. `T_grok ≈ t_memo(K) + 90,000` predicts every previously-censored cell; §12's headline table was itself measured at 10,000 steps and is superseded |
| `p1_k_collapse_wd` | 18/18 | **Graded recovery.** wd=0.1 restores memorisation fully at K=20/30, partially at K=50 (~78% peak). Its zero-generalisation cells turned out to be **censored at 10,000 steps against a measured 45,050 requirement** — `p1_k_collapse_budget` re-runs them |

**Phase 2's hidden cost did not materialise.** The plan reserved ~60 runs to
re-measure C's and D's α ladders in case their decay moved. Neither moved, so both
Gate A ladders stand as written. That is Phase 1's largest saving.

**One gap Phase 1 missed, now closed:** `p1_cd_decay_band` measured C and D but
not **B** — the setup carrying the K-collapse diagnosis. `p1_b_decay_band` supplied
the missing centralized reference, and it immediately paid for itself by showing
the federated arm had been reading a censored cell as a result.

## How to run a sweep now

```bash
venv/bin/python scripts/build_manifests.py               # regenerate manifests/
venv/bin/python scripts/launch_sweep.py manifests/t0_wd_grid.jsonl --dry-run
venv/bin/python scripts/launch_sweep.py manifests/t0_wd_grid.jsonl --gpus 0 --per-gpu 4
venv/bin/python scripts/collect_runs.py                  # per-run JSONs -> results/data/runs_v2.csv
```
Resume is automatic (skips runs whose result JSON exists). One run = one subprocess with `CUDA_VISIBLE_DEVICES` pinned.

## Deferred follow-ups (lower priority, not blocking)

- **Native robust aggregators** — FedMedian / FedTrimmedAvg / Krum / Bulyan are all in stock Flower
  (zero custom code); add to `_build_strategy` if the robustness axis is wanted.
- **The `metrics/fourier.py` split** (fourier/spectral/basic/nonabelian) — cosmetic.
- **The deferred DP / compression / Byzantine follow-up paper** — out of scope for v2 (see plan).
- **`paper/` holds six figures and no manuscript** — the per-setup
  `exp2_slowdown_ratio_*.png` panels. There is no LaTeX source in the tree.

**Mechanistic metrics are wired into per-epoch history.**
`metrics/probes.py::mechanistic_probe(cfg)` dispatches per (dataset, model) —
coset attribution + the exact quadratic-circuit and irrep decompositions on S₅, `embed_ipr`
on the modular transformer, `ipr` on the anchor — and **both** training loops call it inside
the eval block. `client_signature(model, cfg)` likewise generalises per-client checkpointing
to every architecture (`W1_operand` / `W_E` / `layer0`).

**Equivalence harness** (reuse for every training-loop change): `traj.py` in the scratchpad
runs one config and dumps history JSON; diff against `ref_run1.json` (archived c7c997f
trajectory). Worst |Δ| < 1e-5 = fp32 noise = OK. It has now caught two real bugs (the 1.1
RNG-order bug and the stale `ipr` reference in 2b).

## Concurrency on one card — MEASURED, 2026-08-19

Client `num_gpus=1/K`, `num_cpus=1` reservations are unchanged. The old note here
worried about 12 runs × large K on 64 cores; the box is now 16 cores and one 8 GB
card, so the question is settled differently:

**`--gpus 0 --per-gpu 4` is the working default at K ≤ 20.** Measured on the live
alpha2 E block: K=20 ran at **235 ms/round with four runs sharing the card**,
against **244 ms/round banked single-slot on an L4**. Per-run wall is unchanged and
throughput is ~4×, because the workload is orchestration-bound rather than
GPU-bound — the same fact that makes cost scale with client count.

Above K=20, cap the client population instead of lowering `--per-gpu`:
`FEDGROK_GPU_CLIENT_CAP=8` (see the README's VRAM section). Ray queues rather than
deadlocks when clients outnumber cores, so an uncapped K=50 run is slow, not
broken — 50 client processes on 16 cores contend instead of computing.

## Correctness fixes, 2026-08-31

Four defects found by reading the code rather than the results. All four are
fixed with tests; the suite is **592 collected**.

- **NaN passed the grok bar.** `compute_t_grok` scanned for the last point
  *below* the threshold, and `nan < threshold` is False — so an all-NaN test
  curve reported the bar as held from the first sample: `grokked=True,
  t_grok=0`. `alpha=1.0` produces exactly that (no held-out set → division by
  zero in `compute_accuracy`), and five setup-D rows are banked that way.
  Now: `split_indices` rejects an α that empties either side of the split,
  `_below()` counts NaN as below the bar, and `x_d_alpha_high` stops at 0.975.
- **The result-JSON writer emitted invalid JSON.** `_write_json_atomic`
  serialised `inf` as `"inf"` but left NaN as a bare `NaN` token, which is a
  Python extension no other parser accepts — and `final_ipr` is NaN on every
  non-modular run, so 1,038 of the 1,529 banked files carry one. It is invisible
  from inside the project, because everything that reads them is Python. The
  writer now handles all three non-finite values and sets `allow_nan=False`, so
  **new** runs are portable.
  **The 1,038 banked files are deliberately left as they are.** The exposure is
  narrow — the JSONs are committed, so it only bites a non-Python consumer of a
  clone, and nothing in the repo is one. `scripts/repair_result_json.py` will
  rewrite them (idempotent, `--dry-run` first) if the data is ever going
  somewhere that needs it. `runs_v2.csv` is byte-identical either way, so no
  analysis moves whichever is chosen.
- **Client minibatch order was unseeded.** `torch.manual_seed(cfg.seed)` runs in
  the driver; clients are Ray actors with their own unseeded generators. All 94
  banked setup-E federated runs are affected, and asymmetrically — centralized
  MNIST *is* seeded. Now a private CPU generator seeded per
  (run, client, round). Deliberately created only on the minibatch branch, so
  the full-batch path stays bit-identical to every banked run (tested).
- **`strategy="feddyn"` silently ran FedAvg.** It was in FedConfig's Literal with
  no branch in `_build_strategy`, so it fell through to the default return and
  banked a row labelled `feddyn`. Typos did the same. `_build_strategy` now
  raises on anything unrecognised, and `feddyn` is out of the Literal
  (`feddyn_alpha` stays — it is in the schema of every banked row).

Lower severity, same pass — all guards against states no current
configuration can reach, added because this project's expensive mistakes have
all been silent-wrong rather than loud-wrong: `_optimizer_cache` keyed on
lr/wd/optimizer/momentum as well as `_client_key`; `_load_ndarrays_into` and
SCAFFOLD's `apply_correction` raise on length/shape mismatch instead of letting
`zip` truncate silently.

**Not changed, on purpose.** `bootstrap_ci` takes its lower bound as the α/2
quantile of the FINITE resamples, which is not the α/2 quantile of the full
distribution once some resamples are inf. Rescaling by 1/(1-frac_inf) is closer,
but measured across all 204 real cells it moves 3 of them by 1.1–2.1% — and the
rescaled version is not exact either, since numpy indexes quantiles at q*(M-1).
The existing estimator errs toward a WIDER interval, which is the safe direction.
Left alone rather than perturbing a published statistic for a cosmetic gain.

A second pass over the modules the first did not reach found two more traps, both
latent — no banked run is affected by either:

- **`Config.apply_adamw_defaults` was a safety net detached from the path it
  guarded.** It was v1 argparse machinery: the CLI set `_lr_set` when the user
  passed `--lr`, and the method filled AdamW defaults for the rest. Nothing on
  the manifest path ever called it, and it *could not* be wired in — a spec has
  no "was it set" flag, so calling it after `build_config` would have overwritten
  every deliberately chosen lr with 1e-4. Consequence: an `optimizer="adamw"`
  spec omitting `lr` inherited **GD's 50.0** and nothing rejected it —
  `check_decay_stability` returns early at `weight_decay=0`, so the run would
  proceed, diverge, and bank NaN accuracies as an ordinary censored result. The
  method and its three flags are deleted; `build_config` now requires an explicit
  `lr` under AdamW. A test asserts all 1,290 banked AdamW specs already satisfy
  it.
- **The centralized loop never called `model.eval()`.** The federated
  `evaluate_fn` does. No model here has dropout, batch norm, or (deliberately, in
  GrokFormer) LayerNorm, so the two loops agree today — but the asymmetry was
  undocumented, and it is the kind that surfaces only once a model gains a
  stochastic layer and *only the centralized numbers move*. The one remaining
  difference is noted in place: centralized `train_acc` still comes from the
  training forward pass, federated recomputes it under eval().

Also: `backfill_runs.py` and `plotting/alpha_ladder.py` had the same
position-dependent `max()` over NaN that `extract_grokking_results` did;
`launch_sweep.py` exited 0 after a sweep in which every run failed, which for a
detached `nohup` sweep is the only status signal there is.

**No published number moves.** Every fix either guards a state no banked run is
in, or corrects a computation whose inputs no banked run has. `runs_v2.csv` is
byte-identical; regenerating it with `collect_runs.py` confirms that.

## Findings worth not re-deriving

- **Weight decay was a values bug, not a mechanism bug.** Both SGD (coupled) and AdamW
  (decoupled) shrink weights by `(1 - lr*wd)` per step, so `lr*wd` is the comparable
  quantity and `1/(lr*wd)` the decay timescale. exp5 ran lr=50 × wd∈{0.01,0.1,1.0} →
  `lr*wd` ∈ {0.5, 5, 50}. All nine cells returned `T_grok=inf` for numerical reasons.
  Guard in `core/utils.check_decay_stability` rejects divergence (`lr*wd >= 2`) *and*
  the subtler destructive case (`lr*wd > 0.1`; 0.5 never diverges but halves every
  weight every step). Replacement sweep λ ∈ {2e-7, 2e-6, 2e-5, 2e-4} at lr=50.
- **History filenames are not unique run keys.** The `_adam_tau*_slr*` / `_wd*` suffixes
  landed mid-campaign (`3796754`, 2026-03-23), so earlier exp5 arms wrote *identical*
  paths — FedAvg, FedAdam-0.1, FedAdam-0.01 and FedAvg+WD-* collided per (setting, seed).
  61 paths were written by >1 run. Any surviving JSON holds only the last run written.
  `harvest_logs.py` therefore takes algorithm identity from the **log filename**.
- **Wall-clock is ~99% orchestration.** Measured on an L4: 50k full-batch GD steps at
  p=97 take **23 s** (0.458 ms/step; p=151 is 1.077 ms/step). The README budgeted ~26 min
  for the same run. Flower/Ray costs ~156 ms/round. Phase 1 targets ~40 ms/round.
- **The E-sweep is the expensive part**, because rounds = total_steps / E. E=1 means
  50,000 rounds. Those cells dominate the manifest's cost.
- **Grid holes:** 6 empty `exp3a` α=0.20 logs, and `exp3a_a0.30_dir10.0` crashed at
  round 6491 on a Ray actor failure. Tracked in `results/data/runs_skipped.csv`.
- **exp4b partial-participation T_grok values sit on the OLD (inflated) step axis.** The
  0.6 fix changes the x-axis under `fraction_train < 1.0`, so those rows in `runs.csv` are
  ~2.5× too high on `total_steps`. They will be corrected when exp4b is re-run.
- **Known benign warning:** `test_drift_metrics_in_history` emits `invalid value in
  subtract` — a pre-existing test-hygiene issue (default lr=50 diverges on p=7, weights
  overflow to inf, drift-std hits inf−inf). Library bug, not; test still passes. Out of
  Phase 0 scope.

## Resume

```bash
cd <repo root>
git checkout v2-multisetup
venv/bin/python -m pytest tests/ -q          # ~530 passed, ~10.5 min
nvidia-smi                                   # one card now; check it is free first
```

**First, find the real state.** These notes lag the banked data; the CSV and the
result JSONs do not:

```bash
venv/bin/python scripts/collect_runs.py                  # per-run JSONs -> runs_v2.csv
venv/bin/python scripts/summarize_runs.py results/data/runs_v2.csv --group group,setup,alpha
# what each manifest still owes -- run ids are content hashes, so this is exact
venv/bin/python - <<'EOF'
import sys, os, glob; sys.path.insert(0, 'src')
from fedgrok.manifest import load_manifest, run_id
banked = {os.path.basename(p)[:-5] for p in glob.glob('results/data/runs/*.json')}
for m in sorted(glob.glob('manifests/*.jsonl')):
    ids = [run_id(s) for s in load_manifest(m)]
    print(f"{os.path.basename(m):<32} {len(ids):>4} specs, "
          f"{sum(i not in banked for i in ids):>4} missing")
EOF
```

**The setup/check phase is done**, and the first campaign manifest is written:
`t1_setup_k_ladder` (39 runs, ~88 slot-h, not launched). The rules that came out
of the setup phase:

> **1. Budget every federated cell as `t_memo(K) + delay`, never as a multiple of
> the centralized T_grok.** A centralized-anchored budget under-provisions exactly
> the high-K cells the campaign cares about. Six boundaries in this project have
> been manufactured by getting this wrong.
>
> **2. Which of the two terms carries the K dependence is SETUP-DEPENDENT.** On
> the anchor (GD, wd=0) memorisation is flat in K and the delay grows; on setup B
> (AdamW) memorisation explodes. A table of T_grok shows one number for both, so
> budget from whichever term dominates for that setup — and measure it rather
> than assuming, which is what `t1_setup_k_ladder` is for.

Budget B's K=50 campaign cells with headroom **above** 143,000 (that figure is a
lower bound, not a prediction) and they double as the test of the additive model —
no separate confirmation run is needed, since run ids are content hashes.

**Still to write before exp2 can run:**

- Per-setup FL manifests, budgeted as `t_memo(K) + delay` per the rule above —
  NOT as a multiple of the centralized T_grok, which is what under-provisioned
  every high-K cell in the K-collapse diagnosis. Working points per setup are in
  RESULTS §11 and §13.
- `checkpoint_every` + `checkpoint_client_weights` on the cells feeding the
  circuit analysis — **no B/C/D/E run has per-client weights**, so exp7 has no
  data on any new setup. These are config fields, so they change run ids: setting
  them after a sweep re-runs it. The 20 anchor runs that do have them are
  findable by query (`checkpoint_client_weights == True`).
- The three-arm wiring for exp2: `reduced_arm()` builds the floor condition
  (dataset-aware — α/K for grid datasets, `n_train`/K for MNIST) and is tested,
  but **no manifest calls it yet**.
- Per-setup capacity (exp0) checks: done for C only.
