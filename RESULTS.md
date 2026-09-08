# Results — federated grokking

Everything measured, as of 2026-09-07. Branch `v2-multisetup`.

Companion documents: `PROGRESS.md` (what is built and what remains) and `plans/`
(index in `plans/README.md`) — the multi-setup campaign and the boundary campaign
are both under `plans/closed/`. They moved into the repo on 2026-08-17; earlier
revisions of this file cited them at `~/.claude/plans/`, which no longer resolves.

**Data behind every number here:**
`results/data/runs_v2.csv` (1,685 v2 runs) and `results/data/runs.csv` (870 v1 runs,
recovered from logs). Both are committed. Regenerate any table with:

```bash
venv/bin/python scripts/summarize_runs.py results/data/runs_v2.csv --group <cols>
```

All T_grok figures are Kaplan–Meier medians with bootstrap 95% CIs over seeds,
right-censoring runs that did not grok within budget. `frac` is the fraction of
seeds that grokked — the order parameter, and the honest headline when a cell is
partially censored.

---

## The short version

1. **Federation does not break grokking; it delays it** — modestly. Ten times the
   clients costs ~16% more training time at α=0.3.
2. **v1's one breakdown claim was a budget artifact.** At α=0.25, K=97, two of five
   seeds grok at 95.6k/98.5k steps. The original experiment stopped at 50k.
3. **Partition structure is the variable that matters.** At K=97 a structured split
   groks 5/5 while a random split manages 2/5. Unstructured non-IID (Dirichlet)
   behaves exactly like random. *How* you partition matters more than *how far* you
   fragment.
4. **All four setups grok**, so the phenomenon is not an artifact of one architecture.
5. **Weight decay's effect depends on the optimizer, not the loss** — AdamW's
   decoupled decay accelerates grokking, plain GD's coupled decay does not.

6. **Every setup has its own cliff, and A, B and D nearly coincide** at α≈0.20
   (§11) — but that is not because the threshold belongs to the task family.
   **The optimiser moves it**: A′ groks 5/5 at α=0.20 where A, same architecture
   and same data under GD, manages 0/5 (§13.3).
7. **There is no K≈30 collapse** — it was a censored measurement (§13.7).
   Federation costs `t_memo(K) + delay` and **both terms grow with K**. What the
   **decay clock** controls is whether *memorisation* also blows up: flat under GD
   at wd=0, explosive under AdamW. Confirmed on a second setup — D (quad-MLP, S₅)
   reproduces B's collapse exactly, sharing neither its architecture nor its task
   (§14.3).

8. **The optimiser is worth ~45× on the anchor's own task** — the identical
   architecture on the identical data groks in 500 steps under AdamW where it
   takes 25,300 under Gromov's GD, at every α they share (§13.3).
9. **All three inherited weight decays survived being checked.** C's, D's and B's
   were adopted from Nanda's mod-113 transformer with justification only for B,
   and measuring each for its own setup returns the same value every time
   (§13.2, §13.5). On B, decay leaves memorisation untouched at epoch 150 and
   acts purely on the generalisation timescale.

   **Superseded for new transformer work, 2026-08-20: B and C now run at
   `weight_decay = 0.1`, not 1.0.** The check above was centralized, and
   centrally 1.0 is right — on C's band at α=0.5, wd=1.0 crosses at 19,900
   against 70,200 at 0.1, and below 0.03 C does not grok at all. **Under
   federation the conclusion inverts.** At K=20 on B (α=0.40, 10,000 steps),
   wd=1.0 stops *memorisation*: two of three seeds sat at ~41% peak train with
   `t_memo` infinite, while the identical cell at wd=0.1 memorised completely
   on all three (`t_memo` 3,600–4,200) and was merely waiting to generalise.
   High decay drives generalisation on these setups, but once federation has
   slowed memorisation enough, it starts blocking that too — the decay clock
   (§14.3) reaching the phase it usually leaves alone.

   Two consequences. **Budgets need ~3.5× the banked wd=1.0 numbers**, which is
   the ratio C's band measures. And the **305 banked wd=1.0 transformer runs are
   not comparable** to new ones, so any figure mixing them must label which
   decay each line carries.

10. **Seven budget-manufactured boundaries so far.** Every headline failure this
    project has reported has, on re-measurement, been a clock running out — the
    latest being setup C's supposed α≥0.5 cliff, which was a ladder cut off at the
    median for an easier α (§14.1). `t_memo` is recorded alongside `t_grok`
    precisely because one number cannot tell "never learned" from "learned and not
    yet generalised".
11. **`t_grok` is not comparable across budgets.** Because it requires the bar to
    hold for the rest of the run, *extending* a run can make its measured grokking
    time 22× worse on an identical trajectory prefix (§14.4). Compare
    `t_first_cross` whenever budgets or logging rates differ.

12. **The partition result now has a mechanism, and it runs the right way round**
    (§16.2). Coherent shards build Fourier structure *before* they generalise:
    at K=97 the operand and iid arms separate completely on per-neuron spectral
    IPR by round 6,000, against an earliest first crossing anywhere of round
    11,560. Within the iid arm alone the same measure orders the two seeds that
    eventually grok above the three that never do, ~9,000 rounds early.

13. **The per-client checkpoints cannot test that mechanism on the arm that
    carries it** (§16.1). The saved signature is `W1[:, :p]` and `operand`
    shards on the first operand, so at K=97 each client trains exactly one
    column of the matrix being read — 98.8% of its deviation energy, against
    10% under iid. Instrument and treatment are the same variable.

14. **Setup D's mid-training dip is masking, not decay** (§16.3). The
    non-compositional circuit is at chance (0.1–0.4%) at every α; the
    compositional circuit alone *beats the full model* by up to 20 points, and
    keeps improving straight through the dip while its share of the logit falls.
    The dip is the marginal terms growing over a circuit that never degrades.

15. **Adaptive server optimisers are 10–20× faster than FedAvg on the hard
    cells** (§17.1), with every method at its own calibrated LR — so v1's
    FedAdam claim survives being made fair, and the advantage is larger than v1
    reported. FedYogi ties or beats FedAdam once both are tuned. **FedProx at
    μ=0.01 loses to the baseline it is meant to improve**: censored on two cells
    of three.

16. **The partition headline narrows to two setups** (§17.2). Against exactly
    matched iid baselines, coherent sharding wins on A and on C — 1.9× on C, the
    strongest instance in the study — and fails on B and D. C and D differ only
    in architecture, so *whether* coherence helps is a property of the setup.
    Incoherent structure (`target`) is the worst partition everywhere it ran.

17. **Client drift tracks delay on every design axis and is not the cause**
    (§17.4). K, E and partition all move drift and delay together, and SCAFFOLD
    confirms it by intervention — 188× less client divergence, 12× faster. But
    FedProx cuts divergence 51× and never groks, so drift magnitude is neither
    necessary nor sufficient. What separates them is whether the *direction* is
    corrected or the magnitude merely suppressed.

18. **Heterogeneity has a threshold, and below it the instrument breaks first**
    (§18.1–18.2). The Dirichlet knob is flat from 1000 down to 1.0, costs ~15%
    at 0.1, and at 0.01 stops measuring heterogeneity: failure there tracks the
    **smallest shard**, seed for seed, and survives randomising the labels. A
    client holding one sample poisons the aggregate and no budget fixes it. The
    eighth apparent boundary in this project to dissolve — and the first that
    dissolved into something other than a budget artifact.

19. **Partial participation is free per communication round** (§18.3). Sampling
    10 of 50 clients reaches the bar in the same number of rounds while doing a
    fifth of the gradient work. Which makes it the test §17.4 wanted: f raises
    client divergence 1.8× without touching K, E or the data, and costs nothing
    — so **systematic disagreement delays grokking and sampling noise does
    not** (§18.4).

20. **Federation can drive a grokking model into a stable equilibrium that
    memorises perfectly and never generalises** (§22). Setup D at E=50: 0/3 at
    250,000 steps *and* 0/3 at 2,000,000; train 100% from step ~3,000, test
    80–83%, weight norm 101.5 → 101.6 and drift/round 8.0 → 8.0 across 1.75M
    steps, train loss never below 0.055. Not a clock running out — a fixed point
    with the gradient alive. The same signature appears on D's Dirichlet axis.
    The first breakdown in the project to *survive* a longer budget.

21. **On transformers, memorisation cost scales linearly with local work; on
    full-batch quadratic MLPs it does not — and decay is not why** (§19). At
    matched compute, B's `t_memo` goes 1,300 → 12,700 over E = 5 → 50 (9.8× on
    10×) and C's 3,200 → 32,500; A's and D's are flat. D carries ten times B's
    weight decay. §14.3's decay clock does not explain the E axis.

22. **The anchor is immune to unstructured heterogeneity** (§20). K=10, dir_α over
    five orders of magnitude, 18/18 grokked, `t_memo` 3,600–3,900, `t_first_cross`
    12,900 → 14,100 (1.09×). **§18.1's 2.01× is withdrawn as a heterogeneity
    result**: it was measured at K=20/50 where §18.2 had already shown the low
    rungs were shard starvation; remove the starvation and the effect is gone.

23. **Heterogeneity attacks a different phase on each architecture** (§20). It
    stops the transformers *memorising* (B: peak train 20–31% below dir_α=0.5, a
    step; C: graded) and lets D memorise while stopping it *generalising* (0/3 at
    dir_α = 0.5 and 1.0 with train at 100%, test 70–82%), at a threshold one to
    two orders stricter. E, like A, is unaffected. "Heterogeneity costs time"
    is not a claim that can be stated once for the study.

24. **Partial participation is free per round on every setup except MNIST**
    (§21). Memorisation rounds are flat in f everywhere (A: 740/760/800). B's
    delay shortens 1.8× in rounds; E degrades, 3/3 → 0/3 and 94% → 72% final —
    measured at its degenerate K=20 rung, so within-configuration only. Read on
    the rounds axis: `total_steps` accumulates E·f per round and every
    step-denominated number halves with f by construction.

25. **Runs are not reproducible run-to-run** (§23). Same config and seed, two
    runs: D differs by 0.00 on every seed; C by 12.7 / 0.0 / 7.7 points of peak
    train accuracy against a between-seed spread of 13.4. Float summation order
    in Flower's arrival-ordered `aggregate()`, amplified by dynamics near the
    threshold. **Every quantitative claim on C is withheld** until aggregation is
    made deterministic or C is run at n≈10. "Seeds" should read "runs".

Open: whether K=97 IID *fails* or is merely *slow* — both successes landed within
5% of the budget ceiling, so that cell is not yet resolved.

---

## 23. Reproducibility — the harness is not run-to-run deterministic

The long-budget re-run of the E=50 cells (§22) duplicated six configurations at
a second `num_rounds`, which is the first time this project has had two runs of
the same config *and* seed. `num_rounds` touches nothing but logging and the
server's round count, and there is no learning-rate schedule, so the first 4,000
rounds should be identical. Peak train accuracy over the overlapping 200,000
steps:

| setup | seed | run 1 (200k budget) | run 2 (1M/2M budget) | run-to-run \|Δ\| |
|---|---|---|---|---|
| D | 42 / 123 / 456 | 100.0 / 100.0 / 100.0 | 100.0 / 100.0 / 100.0 | **0.00 / 0.00 / 0.00** |
| C | 42 / 123 / 456 | 86.6 / 100.0 / 86.9 | 99.3 / 100.0 / 79.2 | **12.72 / 0.00 / 7.65** |

C's between-seed spread on the same statistic is 13.40. **On C, rerunning the
seed is as different as changing it.** Seed 456 reaches 86% train in one run and
26% in the other at 75,000 steps.

**What is ruled out**, in code: the client training loop draws no random numbers
on C (`batch_size=None`, the full-batch path the loop comments call
"RNG-identical"); where minibatching is used it is seeded per
`(seed, partition_id, server_round)`; initialisation is `torch.manual_seed`ed in
the driver; `probes.py`, `nonabelian.py` and `grokking_metrics.py` contain no
`rand`, `shuffle`, `permutation` or `Generator`. Eval cadence therefore cannot
touch the trajectory — which also clears §19's design, where `eval_every` scales
with E.

**What remains** is floating-point ordering, twice over. Flower's `aggregate()`
computes `reduce(np.add, layer_updates) / total`, a sequential float32 sum in the
order results arrive from Ray, and nothing in `training/federated.py` sorts them
(the only `sorted()` is over probe names). Summing ten GrokFormer-sized float32
layers in 201 orders gives 200 distinct results, max |Δ| 1.8e-7 (~1.5 ulp). And no
`torch.use_deterministic_algorithms`, `cudnn.deterministic` or
`CUBLAS_WORKSPACE_CONFIG` appears anywhere in `src/`. The perturbation is injected
every round; whether it matters is set by the local dynamics. D is contractive —
it falls into the same 80% attractor from any nearby start — and C sits at its
threshold, where the same noise decides whether a seed memorises at all.

PROGRESS.md records the equivalence harness seeing `|Δ| < 1e-5` and calling it
"fp32 noise = OK". The measurement was right; the conclusion is not safe for a
setup near a critical point.

> **Consequences.** (1) Each run is still a draw from the trajectory
> distribution, so "k of n grokked", the KM medians and the bootstrap intervals
> estimate what they claim to; what breaks is any claim whose effect is
> comparable to the noise. (2) On C that is every quantitative claim in §19–§21.
> On D the plateau reproduces exactly. Everything in §19–§22 stated as a result
> has an effect many times the noise. (3) The statistics language should say
> *runs*, not *seeds*: for C the seed carries no information. (4) §11's "B's
> seed variance is intrinsic and bimodal" may be this effect wearing a seed
> label; it needs the duplicate-run test on B. (5) The fix and the C decision
> are RUNS_TODO's reproducibility entry.

## 22. Setup D reaches a federated equilibrium that memorises and never generalises

`x_e50_long`, 6 runs, 65 slot-h, 2026-09-04. D at E=50 was 0/3 at 250,000 steps
with train at 100% and flat slopes — exactly the shape eight earlier boundaries
had before dissolving into censoring. It was re-run at **2,000,000 steps**, eight
times the budget.

| seed | `t_memo` | test @250k | @1M | @2M | train loss @250k → 2M | weight norm | drift/round |
|---|---|---|---|---|---|---|---|
| 42 | 3,500 | 78.9 | 80.8 | **80.5** | 0.067 → 0.062 | 101.6 → 101.8 | 8.00 → 7.96 |
| 123 | 2,500 | 82.3 | 80.2 | **82.3** | 0.065 → 0.061 | 101.5 → 101.6 | 8.01 → 7.98 |
| 456 | 3,000 | 80.1 | 83.0 | **82.6** | 0.077 → 0.069 | 101.2 → 101.5 | 8.03 → 8.00 |

Eight times the budget bought +1.6, +0.0 and +2.5 points; two final-quarter
slopes are negative. Every measurable quantity is stationary to within 1% over
1.75 million steps. **This is not the mechanism behind the project's other
genuine negative** (RUNS_TODO entry 3: B at wd=0, where fp32 train loss
underflowed to exactly 0.0 and nothing could move): here train loss never falls
below 0.055, the gradient is alive, decay at wd=1.0 is alive, each round's 50
local epochs drift the clients ~8 units and averaging pulls them back — and the
forces balance. A fixed point of federated AdamW on the quadratic MLP, with the
training set fit at 100% and a representation that generalises to 82%.

**The same signature appears on an independent axis.** D's Dirichlet stalls at
dir_α=1.0 (§20, 250,000 steps, E=5) show drift constant at 1.26–1.28, weight
norm 132–135 barely moving, train 100%, test 74.9–82.3% flat from ~175,000
steps. Two axes that share nothing but the setup reach the same state, which is
what promotes this from an observation about a cell to a property of the setup.

> **Consequences.** The K-axis story (§14.3) is that federation on AdamW setups
> slows *memorisation* until the clock runs out. This is different and sharper:
> memorisation completes and generalisation is *prevented* by a stable
> equilibrium. It is the "federated breakdown with memorisation intact" that
> `p1_k_collapse_budget`'s decision rule named, and it belongs in the paper as a
> positive result. It is a setup-D result: A (GD, wd=0) and E show no such mode
> on either axis, B and C fail by not memorising. Whether the equilibrium belongs
> to S₅ or to AdamW-on-a-quadratic-MLP is the A-vs-D confound PROGRESS records as
> a stated limitation, and this result makes that limitation expensive. SCAFFOLD
> Option I, once written, should be tested on this cell first.

## 21. exp4b on five setups — partial participation is free per round, except on MNIST

`t5_participation`, 45 specs, 30 run (15 banked controls), 134 slot-h,
2026-09-03/04. K=20, iid, E=5, f ∈ {1.0, 0.5, 0.25} → 20 / 10 / 5 clients per
round; rounds scale as 1/f so every cell reaches its control's total gradient
work. A is re-run at K=20 rather than reusing §18.3's K=50 ladder, so the panel
is matched.

**Read this axis in rounds.** `total_steps` accumulates E·f per round, so a run
at f=0.25 books a quarter of the work per round *by construction* and every
step-denominated quantity halves as f halves. A mid-sweep reading that called
that an acceleration was wrong. Medians over 3 runs, in rounds:

| setup | f=1.0 memo / first-cross / delay | f=0.5 | f=0.25 | final test |
|---|---|---|---|---|
| A | 740 / 2,660 / **1,920** | 760 / 2,680 / **1,920** | 800 / 2,720 / **1,920** | 100 / 100 / 100 |
| B | 720 / 18,740 / 18,020 | 720 / 17,320 / 16,600 | 800 / 10,880 / **10,080** | 100 / 100 / 100 |
| C | 1,460 / 1,580 / 120 | 1,560 / 1,600 / 40 | 1,200 / 1,200 / 0 | 100 / 100 / 100 |
| D | 2,960 / 18,860 / 15,900 | 3,000 / 21,360 / 18,360 | 2,960 / 18,640 / 15,680 | 89 / 89 / 93 |
| E | 260 / 2,240 / 1,980 | 240 / 2,280 / 2,040 | 320 / 2,560 / 2,240 | **94 / 89 / 72** |

**Memorisation is paced by aggregation rounds, not gradient steps, on every
setup** — flat in f within 10% everywhere. A's delay is 1,920 rounds at every f.
Two things move: **B's delay shortens 1.8×** at f=0.25 (one setup, with a 2–3×
seed spread, so moderate), and **E degrades** — held 3/3 → 1/3 → 0/3 and final
test 93.9 → 89.1 → 71.6%, the first cost of subsampling anywhere in the study.
E's cell is the degenerate K=20 rung (one full-batch step per local epoch); the
degeneracy is constant across f so the within-setup effect stands, but it is
not shown to survive a non-degenerate K (§10). C's delay is ≈0 throughout, so
its index is noise, and its numbers are withheld regardless (§23).

> §18.3's anchor result (flat in rounds at K=50) now holds at K=20 too, and on
> B, C and D. "Sampling noise averages out" (§18.4) generalises — with the MNIST
> exception recorded.

## 20. exp3a on five setups — heterogeneity has a threshold, a phase, and an architecture

`t3a_dirichlet_setups`, 87 specs, 72 run (15 banked 0.5 rungs), 137 slot-h,
2026-09-02/03. K=10, E=5, dir_α ∈ {0.01, 0.1, 0.5, 1, 10, 1000} at each setup's
exp3b working point and budget. Partitions were built and inspected before
launch: minimum shard at dir_α=0.01 is 116–184 (A), 298–337 (B), 386–515 (C),
194–398 (D) — an order of magnitude clear of the ≤2 that made §18.2's tail a
starvation artifact. E has 10 classes: 0.01 is infeasible and at 0.1 its minimum
shards are 5 / 25 / 56 against `batch_size=100`, so E's low rungs *are* in the
starvation regime and are labelled so. Held / median peak train / median
`t_first_cross`:

| dir_α | A | B | C | D | E |
|---|---|---|---|---|---|
| 0.01 | 3/3 · 100 · 14,100 | **0/3 · 20 · ∞** | **0/3 · 59 · ∞** | **0/3 · 5 · ∞** | — |
| 0.1 | 3/3 · 100 · 13,400 | **0/3 · 31 · ∞** | 2/3 · 97 · 29,950 | **0/3 · 60 · ∞** | 3/3 · 100 · 4,100 |
| 0.5 | 3/3 · 100 · 13,000 | 3/3 · 100 · 5,300 | 3/3 · 100 · 15,600 | **0/3 · 100 · ∞** | 3/3\* · 100 · 4,000 |
| 1.0 | 3/3 · 100 · 13,000 | 3/3 · 100 · 4,500 | 3/3 · 100 · 16,300 | **0/3 · 100 · ∞** | 3/3\* · 100 · 3,800 |
| 10 | 3/3 · 100 · 12,900 | 3/3 · 100 · 7,600 | 3/3 · 100 · 13,400 | 3/3 · 100 · 65,900 | 3/3\* · 100 · 4,400 |
| 1000 | 3/3 · 100 · 12,900 | 3/3 · 100 · 8,700 | 3/3 · 100 · 9,900 | 3/3 · 100 · 44,000 | 3/3\* · 100 · 4,100 |

\* E on `t_first_cross`; its `grokked` flags (1/3, 1/3, 1/3, 0/3) are §14.4's
sustain artifact — every seed crosses with train at 100%.

**Four responses to one manipulation.**

- **A is immune.** 18/18; `t_memo` 3,600–3,900 throughout; `t_first_cross` moves
  1.09× across a 100,000× span. **This contradicts §18.1** (2.01× on the same
  setup) and confirms §18.2: that ladder was K=20/50 at α=0.25, where the low
  rungs starved clients. Remove the starvation and heterogeneity per se costs
  the anchor nothing.
- **B and C fail by not memorising.** Below dir_α≈0.5 peak train collapses — B
  as a step (31% → 100% between 0.1 and 0.5, with no `t_memo` trend above it), C
  graded (59 → 97 → 100) with `t_memo` rising 2.5× monotonically as shards
  concentrate. These are optimisation failures and must not be plotted as delay.
  Across C's full ladder `t_first_cross` is flat within its seed spread, so
  **§17.2's "Dirichlet beats iid 1.58× on C" is withdrawn** as one concentration's
  noise (and C's numbers are withheld anyway, §23).
- **D fails the other way, and at a stricter threshold.** 0/3 below dir_α=10.
  At 0.01 it never trains (peak 3–5%); at 0.1 it half-memorises; at 0.5 and 1.0
  it memorises fully (`t_memo` 19,700 and 9,800) and freezes at 70–82% test —
  the equilibrium of §22, reached from a second axis.
- **E is unaffected** on `t_first_cross` (3,700–4,900 at every rung).

Two C cells at dir_α=0.1 report `t_memo=∞` with finite `t_first_cross`: test
crossed 95% while train peaked at 97–99%. Grokking without full memorisation is
real, and it leaves `delay` undefined at the most interesting rung (§10).

> **Consequences.** §18.1's "threshold" reading generalises in *form* — B, C and D
> each have one — but the threshold's position (0.5, 0.1–0.5, 10), its shape
> (step, graded, two-stage) and the phase it attacks are per-setup, and two
> setups have none. The claim cannot be stated once for the study. The two
> quadratic MLPs sit at opposite extremes, so architecture alone does not order
> it; D differs from A in task *and* optimiser, which is the A-vs-D confound.

## 19. The E axis on five setups — local work taxes a different phase on each architecture

`t5_local_epochs`, 63 specs, 48 run (15 banked controls), 65 slot-h, 2026-09-02.
K=10, iid, E ∈ {5, 10, 25, 50} on every setup, plus E=1 on A. **Compute-matched**:
rounds scale as 5/E so every rung does the same gradient work, and `eval_every`
and `checkpoint_every` scale with it so every rung is sampled on the same step
grid (a fixed cadence would read E=50 at ten times E=5's resolution — §13.4,
§14.4). B's and D's rungs were given 200,000 and 250,000 steps after an audit of
the controls' banked `t_first_cross`. Medians over 3 runs:

| setup | E=1 memo / fc | E=5 | E=10 | E=25 | E=50 |
|---|---|---|---|---|---|
| A | 3,700 / 12,600 | 3,700 / 12,900 | 3,700 / 13,300 | 3,700 / 14,800 | 4,000 / 23,000 |
| B | — | 1,300 / 55,900 | 2,500 / 64,500 | 6,100 / 66,800 | **12,700** / 59,800 |
| C | — | 3,200 / 7,800 | 6,400 / 9,100 | 23,900 / 23,600 | 32,500 / 31,400 (1/3) |
| D | — | 3,200 / 78,900 | 3,100 / 85,400 | 2,700 / 100,100 | 2,600 / **∞ (0/3)** |
| E | — | 600 / 5,200 | 800 / 5,100 | 1,200 / 4,600 | 1,900 / 4,900 |

**Two effects, and they split on different variables.**

1. **Memorisation cost tracks local work on transformers and not on full-batch
   MLPs.** B's `t_memo` ratio to E=5 is 1.0 : 1.9 : 4.7 : 9.8 against E ratios
   1 : 2 : 5 : 10, per-seed spread ±3%; C's is 1 : 2 : 7.5 : 10. A's and D's are
   flat (D slightly *falls*). E (MNIST, minibatched) is sub-linear, 3.2×. **This
   is not the decay clock**: D carries wd=1.0 against B's 0.1 and does not move.
   §14.3's mechanism stands for the K axis and does not explain the E axis.
2. **The delay is setup-dependent.** On A it grows monotonically and tightly,
   8,900 → 9,200 → 9,600 → 11,100 → 19,000 (2.1× over E = 1 → 50), on the one
   setup with no decay term to confound it. On B it does not move — every rung's
   `t_first_cross` sits inside B's known 46k–77k bimodal band. On C it collapses
   to zero from E=25 (`t_fc` ≈ `t_memo`). On D it grows, 75,700 → 81,600 → 97,500,
   then at E=50 the run enters §22's equilibrium.

**The E=1 identity rung is the equivalence check.** Under plain GD at momentum 0
FedAvg at E=1 is an algebraic identity with centralized training, and A's rung
reproduces the banked centralized run to the step — `t_memo` 3,700, `t_first_cross`
12,600 on seed 42 — on a different Flower version (1.30 against the corpus's
1.27), machine and CUDA build. New runs are comparable to the banked corpus.

> **Consequences.** §17.4's E column (drift/round 0.018 → 0.167 → 1.92) can carry
> "local work delays grokking" on the anchor only: on B, drift/round rises with E
> and the delay does not follow it. What the transformers pay for is memorisation,
> proportional to local work between aggregations — which is the same quantity
> §21 lowers by subsampling clients, with `t_memo` in rounds staying flat. C at
> E=50 is a training failure (peak train 86.6 / 86.9 on the two failing seeds) and
> its re-run at 1M steps is withheld under §23. E's E=10 rung reads 2/3 on
> `grokked` and 3/3 on `t_first_cross` (seed 123 dipped back seven times).

---

## 18. main's last two experiments, and what they do to the drift story

45 runs, 0 failures, 2026-08-13/14. exp3a and exp4b were the only v1 arms with
no v2 coverage. Both returned nulls in v1 — 100/100 and 72/72 grokked — but both
were measured at α≥0.30, K=10, the plane §4 has since shown is uniformly safe at
60/60, so neither could have found an effect had there been one. Asked at the
boundary they are new experiments, and they disagree with each other in a way
that sharpens §17.4.

### 18.1 exp3a — heterogeneity has a threshold, and §4's claim has a domain

α=0.25, E=5, 20,000 rounds, 3 seeds, against t2_boundary's banked iid arms.
Median `t_first_cross` as a ratio to the matched iid cell:

| K=20 (iid 29,800, 5/5) | dir_α=1000 | 10 | 1.0 | 0.1 | 0.01 |
|---|---|---|---|---|---|
| vs iid | 0.97× | 0.97× | 1.00× | **1.15×** | **2.01×** (2/3) |
| divergence | 0.00162 | 0.00129 | 0.00156 | 0.00204 | 0.00458 |

At K=50 the same shape, harder: 0.94× at dir_α=1.0, 1.14× at 0.1, **0/3** at 0.01.

**The partitioner control passed first.** dir_α=1000 at K=20 gives 28,800
[26,500–31,700] against iid's 29,800 [27,300–33,100]. At that concentration the
draw is numerically uniform, so this rung is the instrument checking itself, not
a data point — and it agrees.

**§4's "dirichlet tracks iid exactly at every K" is unaffected, and now has a
stated domain.** Both §4 and §17.2 ran their Dirichlet control at **dir_α=0.5**,
which sits in the flat region. The claim was true where it was measured. What
was missing is that the knob is flat across three orders of magnitude and only
bites below ~0.1 — so "structure, not heterogeneity" (§5.4, §17.2) stands as
argued, on a control that was in the right regime by luck rather than by design.

### 18.2 The tail of that ladder is starvation, not heterogeneity

dir_α=0.01 looked like the first breakdown in this project that survives
inspection: models memorise to 97–100% train, then sit at 1–5% test with a
**zero slope** at 2–3× the budget the iid arm needed. Five of six trajectories
flat. It was recorded here as exactly that, and it was wrong.

`dirichlet` moves two things at once and at low concentration moves them
together. Measured on the actual shards at K=50, the **median shard is
unchanged** between dir_α 0.1 and 0.01 (43 vs 41–51 samples) — so this is not a
bulk data-quantity effect — while classes-per-client falls 19 → 5 **and** the
size tail collapses, one client receiving a single sample.

`dirichlet_sizes` holds the geometry and removes the labels: the Dirichlet
partition is drawn through the same RNG to get its sizes (byte-identical per
seed, asserted in `TestPartitionDirichletSizes`), then every index is re-assigned
uniformly at random into shards of exactly those sizes.

**Failure tracks the smallest shard, seed for seed**, at dir_α=0.01:

| K | seed | min shard | concentrated | size control |
|---|---|---|---|---|
| 20 | 42 | **1** | censored | **censored** |
| 20 | 123 | 22 | 49,800 | 29,800 |
| 20 | 456 | 52 | 69,900 | 32,100 |
| 50 | 42 | **2** | censored | **censored** |
| 50 | 123 | **1** | censored | **censored** |
| 50 | 456 | 8 | censored | **47,800** |

Every seed whose smallest client held ≤2 samples dies in **both** arms; labels
made no difference. Every seed with ≥22 groks in both. The one intermediate case
is where label concentration decides.

> **So the breakdown is client starvation, not heterogeneity.** A client holding
> one sample poisons the aggregate, and no budget fixes it — which is why the
> flat-trajectory check did not catch it. This is the eighth apparent boundary in
> this project to dissolve, and the first to dissolve into something other than a
> budget artifact.

**What survives is clean, and it is a label effect.** With the geometry held
fixed and only the labels randomised:

| | dir_α=0.1, K=20 | K=50 | dir_α=0.01 survivors, K=20 |
|---|---|---|---|
| concentrated | 1.15× | 1.14× | 2.01× |
| **size control** | **1.00×** | **0.98×** | **1.04×** |

Label heterogeneity costs ~15% at dir_α=0.1 and roughly doubles time at 0.01.
It is real, it is measurable, and it **does not break grokking**.

**The instrument bounds the experiment.** Dirichlet-over-classes on mod-97 at
α=0.25 spreads ~2,350 samples over 97 classes, so concentrating labels
necessarily empties someone's shard: this partitioner *cannot* probe extreme
heterogeneity here without also testing starvation. Any future claim about
extreme non-IID on this dataset needs the size control beside it. Same species
as §16.1, where the per-client signature was the partition being tested.

### 18.3 exp4b — partial participation is free per round

α=0.25, K=50, iid. Rounds set to reach a common 100,000 `total_steps`, because
`total_steps` accumulates `E · samples_this_round / n_train_total` and a fixed
round count would starve the low-f cells by exactly the factor under test.
f=1.0 is t2_boundary's cell, deduped by content hash.

| f | clients/round | `t_first_cross` (steps) | → round | vs f=1.0 | divergence |
|---|---|---|---|---|---|
| 1.0 | 50 | 43,800 | 8,760 | 1.00× | 0.00193 |
| 0.6 | 30 | 25,920 | 8,640 | 0.99× | 0.00279 |
| 0.4 | 20 | 16,680 | 8,340 | 0.95× | 0.00342 |
| 0.2 | 10 | **8,260** | **8,260** | **0.94×** | **0.00355** |

**In rounds the axis is flat.** Sampling 10 of 50 clients reaches the bar in the
same number of communication rounds while doing a fifth of the gradient work —
so on the compute-matched axis it reads as a 5× speedup, and the honest
statement is that the marginal client contributes almost nothing at this cell.
v1's null reproduced at the boundary, with the quantitative statement v1 could
not make because its step axis was broken.

### 18.4 The refinement this forces on §17.4

§17.4 concluded that what matters is whether the drift *direction* is corrected
rather than its magnitude suppressed, on the SCAFFOLD-vs-FedProx contrast. Two
new axes say that is close but not quite it.

| axis | divergence | effect on grokking |
|---|---|---|
| exp3a, dir_α 1000 → 0.01 | ×3.7 | **2.01× slower** |
| exp4b, f 1.0 → 0.2 | ×1.8 | **none** (0.94×) |
| §18.2 control, geometry only | ×1.4 vs iid | **none** (1.00×) |

Three ways of raising client disagreement; one of them costs time. What
separates them is not magnitude and not direction-correction, but **what the
disagreement is made of**:

> **Systematic disagreement delays grokking; sampling noise does not.** When
> clients hold conflicting *labels* they pull toward genuinely different
> solutions and the delay scales with the conflict. When disagreement comes from
> subsampling clients (exp4b) or from unequal but unbiased shard sizes (§18.2),
> it averages out across rounds and costs nothing — even at 1.4–1.8× the
> divergence.

exp4b is the load-bearing one, because f moves disagreement **without touching
K, E or the data**, which is the independent test §17.4 said was missing. It
also explains the FedProx result without needing a separate story: FedProx
suppresses magnitude indiscriminately, including the systematic component that
carries the learning signal, which is why it loses to the baseline it is meant to
improve.

**Still not settled.** The damped-FedAvg control — FedAvg at FedProx's effective
step size — remains unwritten, and it is what separates "suppressed magnitude"
from "damped learning rate" in §17.1's FedProx failure.

---

## 17. The all-5 campaign — exp5, exp3b completed, tier X, and client drift

195 runs, 0 failures, finished 2026-08-11. Two chains on disjoint GPU pools
(`scripts/run_main_chain.sh`, `scripts/run_long_c_k50.sh`). Costs below are from
measured `wall_s`, not §8's fitted model, which over-costs non-anchor setups
~2.6×.

### 17.1 exp5 — v1's FedAdam claim survives calibration (`t3_algorithm_comparison`, 90 runs)

Every method fixed at its own calibrated server LR (§14.5), so this is the fair
comparison v1 could not make: v1 swept `server_lr` for FedAdam *alone* and then
reported it ~10× faster than the field. Median `t_first_cross`, 5 seeds:

| | H1 | H2 | H3 |
|---|---|---|---|
| FedAvg | 45,500 | 61,000 | 31,000 |
| **FedAdam** | **3,000** | 3,000 | 3,000 |
| **FedYogi** | 5,000 | **2,500** | **2,000** |
| SCAFFOLD | 4,500 | 5,000 | 4,000 |
| FedAvgM | 7,500 | 10,000 | 6,000 |
| FedProx | **0/5** | **0/5** | 345,000 |

Cells: H1 α=0.25 K=10 E=25 iid · H2 α=0.25 K=10 E=25 Dirichlet(0.1) · H3 α=0.30
K=10 E=50 Dirichlet(0.1). The ordering is identical on `t_grok`, so it does not
depend on which statistic is read.

**The claim survives, and §14.5 was too pessimistic about it.** That section
warned "the ordering may not survive" once every method was tuned. What
calibration changes is only *which* adaptive method wins — FedYogi ties or beats
FedAdam on two cells of three — not the 10–20× advantage over FedAvg, which is
larger here than v1 reported.

**FedProx at μ=0.01 is worse than doing nothing**: censored on two cells, and
11× slower than plain FedAvg on the third. It is the only method in the study
that loses to the baseline it is meant to improve. See §17.4 — this is not a
tuning accident, it is the shape of what the proximal term does.

**SCAFFOLD works**, and it is the only drift-correction method that does. Third
behind the two adaptive optimisers on every cell and ahead of FedAvgM.

### 17.2 exp3b completed — the partition claim narrows to two setups

The 54 deferred cells ran, plus coset. Every structured arm is now read against
an **exactly matched** iid baseline — same setup, K, α, width and budget — rather
than against a working point borrowed from another sweep. Ratio to that
baseline, on `t_first_cross`; **bold** beats random shards:

| setup | K | iid (steps) | coherent | incoherent | Dirichlet |
|---|---|---|---|---|---|
| **A** | 10 | 12,900 | **0.98×** operand | 1.09× target | 1.01× |
| **A** | 50 | 14,700 | **0.89×** operand | **1.92×** target | 1.02× |
| **B** | 10 | 32,500 | 2/3 operand | **0/3** target | **0.16×** |
| **B** | 20 | 9,100 | 2/3 operand | **0/3** target | **0.90×** |
| **C** | 5 | 8,100 | **0.53×** coset | — | — |
| **C** | 10 | 9,900 | — | 1.15× operand · 0/3 target | 1.58× |
| **C** | 50 | 2/3 | — | 0/3 | 0/3 |
| **D** | 5 | 27,100 | **0/3** coset | — | — |
| **D** | 10 | 78,900 | — | 0/3 operand · 0/3 target | 0/3 |
| **D** | 20 | 94,300 | — | 0/3 | 2/3 |
| **E** | 10 | 5,100 | — | 0/3 label_block | **0.78×** |
| **E** | 20 | 11,200 | — | 0/3 label_block | **0.52×** |

Partitions are grouped by what they *are*. The algebraically coherent split is
`operand` on modular arithmetic but **`coset`** on S₅ — sharding S₅ by
first-operand element is structured without being coherent, which is why it sits
with `target`.

**The headline holds on A and C, and on nothing else.** On the anchor operand
beats iid and the margin grows with fragmentation (0.98× at K=10, 0.89× at
K=50). On C the coherent split is **1.9× faster than random shards**, which is
the strongest single instance of the effect in the study and the first on a
second setup. On B the coherent split is *slower* and Dirichlet wins; on D every
structured partition fails outright.

> So §5.4's "coherent shards are strictly better than random ones" is **withdrawn
> as a general claim**. What survives: *where* coherent sharding helps it helps a
> lot and the margin grows with K, and incoherent structure (`target`) is the
> worst partition in the study — 0/3 on B, C and D alike, and 1.92× on A.

**C and D are the cleanest contrast the project has.** Same task, same optimiser,
same coherent partition, same K, differing only in architecture: coset makes the
transformer 1.9× faster and stops the quadratic MLP grokking at all, where random
shards grok 3/3. Whether coherence helps is a property of the setup, not of the
partition.

**A′ is still uninterpretable** and is excluded from the reading above: `target`
is its *fastest* partition at both K (0.56×, 0.57×), which no account predicts.
It sits at its cliff (§15.2) and carries §15.3's optimiser-restart confound.

**Two cells that are not what they look like.** D's coset seeds were still
climbing at the 250,000-step ceiling — +11.7 and +6.7 accuracy points per 100,000
steps on two of three — so the defensible claim is **≥10× slower than random
shards**, not "never groks". And C at K=50 is a **training** failure, not a
partition result: peak train accuracy is 12–15% on operand and 1.6% on target, so
the model never memorises. That is the decay clock at high K (§14.3), and it
**corrects §14.3's "C is 3/3 at every K, the only setup that never degrades"** —
that was measured at α=0.50 under iid. At α=0.40 with structured shards, C
degrades like every other AdamW setup.

### 17.3 Tier X — D's decay ladder finishes, and the step-size control

`x_d_wd_ladder` (45) and `x_d_lr_control` (18), both centralized setup D, ~1.4
slot-hours between them. Median `t_first_cross`:

| lr·λ band (wd) | 0.0 | 0.1 | 0.3 | **1.0** | 3.0 |
|---|---|---|---|---|---|
| grokked | 0/9 | 6/9 | 6/9 | **9/9** | 6/9 |
| median | — | 16,613 | 17,388 | **9,650** | 1,100 |

**The inherited wd=1.0 survives a fourth check**, now on the order parameter
rather than on a time: it is the only band that groks every seed. wd=3.0 is much
faster *when* it works (1,100) and fails a third of the time — decay accelerating
generalisation right up to the point where it outruns learning, which is §6.2's
boundary seen from the other side.

The lr control inverts the naive expectation: **lower is faster.** 2,750 at
lr=2.5e-4, 5,188 at 1e-3, 15,038 at 4e-3, all 6/6.

### 17.4 Client drift — right on every axis, wrong as a magnitude story

Does client drift delay grokking? Every federated run logs `mean_client_drift`
(how far each client moves locally) and `client_weight_divergence` (how much
clients disagree), so this is measurable on all 601 federated runs rather than
inferred from K and E.

**On every design axis, drift and delay move together.** Setup A, iid, α=0.30:

| K | 2 | 5 | 10 | 20 | 50 | | E | 1 | 5 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| drift/round | 0.063 | 0.113 | 0.167 | 0.245 | 0.417 | | drift/round | 0.018 | 0.167 | 1.92 |
| delay | 8,900 | 9,250 | 9,450 | 9,800 | 11,250 | | delay | — | 9,450 | 19,000 |

And with K and E held fixed so only the partition varies, the fastest partition
is the lowest-drift one in **4 of 4** cells. At K=50: operand 0.349 → 13,418;
iid 0.417 → 14,950; dirichlet 0.441 → 15,200; target 0.996 → 28,200.

**It has to be disagreement, not movement.** On setup D the failing arms move
*less* than the arm that groks — which is the datum §12 used to rule drift out,
and it was measuring the wrong quantity. On `client_weight_divergence` the
ordering is correct: at K=10, iid 0.0078 groks while operand 0.0119, dirichlet
0.0499 and target 0.2766 all fail.

**The intervention confirms it — and then falsifies it.** Same cells, same seeds,
only the aggregation rule changing:

| H2 | divergence | first crossing |
|---|---|---|
| FedAvg | 0.0188 | 61,000 |
| **SCAFFOLD** | 0.00010 (188× lower) | **5,000** |
| **FedProx** | 0.00037 (51× lower) | **censored** |

SCAFFOLD is the hypothesis confirmed by intervention. FedProx cuts divergence by
51× and never groks. **Reducing drift is neither necessary nor sufficient**, so
drift magnitude cannot be the causal variable — and their *movement* magnitudes
are nearly identical (0.043 vs 0.094), so that does not separate them either.

What does is direction. SCAFFOLD estimates the drift term and subtracts it,
preserving the descent direction; FedProx penalises movement away from the global
model, buying low drift by not learning. So the defensible statement is:

> **Client disagreement about the direction of progress delays grokking, and
> drift is its signature rather than its substance.** The method that corrects
> the direction wins; the method that merely suppresses the magnitude loses.

**REFINED by §18.4.** Two axes added since say "direction-correction" is close
but is not the discriminator. exp4b raises divergence 1.8× by subsampling
clients and costs nothing; §18.2's size control raises it 1.4× through unequal
shard sizes and costs nothing; exp3a raises it 3.7× through conflicting labels
and costs 2.01×. What separates them is what the disagreement is MADE OF —
systematic conflict delays grokking, sampling noise averages out. exp4b is the
independent test this section calls for, since f moves disagreement without
touching K, E or the data.

**The one test that could separate cause from correlate is a null, and it is
underpowered.** Within a single cell — everything fixed but the seed — the
seed that diverged more is no slower: median Kendall τ = **+0.020** across 36
cells, 18/36 positive. But within-cell divergence spread is 1.69× against 20,182×
between cells, so there is almost no signal to correlate against. Recorded
because it is the test that would have been quoted had it come out the other way.

**The control that would settle it** is FedAvg damped to FedProx's effective step
size, isolating "corrected direction" from "suppressed magnitude". ~15 runs on
the anchor; not written.

---

## 16. The mechanism behind the partition result, and setup D's internals

Post-hoc analysis of banked checkpoints. No new runs. Every table below
regenerates with:

```bash
venv/bin/python scripts/analyze_mechanism.py all      # or: confound | global | internals
```

### 16.1 The per-client channel cannot test the mechanism in the cell that matters

Readiness blocker #2 added per-client weight checkpoints specifically so the
frequency-consensus explanation for §5.4 could be tested post-hoc. On the cell
that carries the claim — `t2_boundary`, K=97, operand vs iid — **it cannot**.

`metrics/probes.py::client_signature` ships `W1[:, :p]`, the first-operand
block. The `operand` partition shards by first operand. At K=97 = p each client
therefore holds exactly one first-operand value, and trains exactly one column
of the matrix the analysis reads. Measured as each client's deviation energy
from the across-client mean, profiled by column:

| top-1 column's share of per-client deviation energy | r1000 | r5000 | r9000 | r13000 | r17000 | r20000 |
|---|---|---|---|---|---|---|
| operand | 0.989 | 0.989 | 0.988 | 0.988 | 0.988 | 0.988 |
| iid | 0.091 | 0.103 | 0.106 | 0.105 | 0.103 | 0.098 |

Stable across every checkpoint round. Any cross-client "shared basis" statistic
computed on this channel compares two structurally different objects, and would
report a large effect that is a restatement of the partition's definition. The
instrument and the treatment are the same variable.

**This does not invalidate the checkpoints** — it says they answer the question
on `iid`, `dirichlet` and `target` shards, and not on `operand` (or `coset`,
which shards the same axis). It is the seventh measurement artifact this project
has caught before it was claimed.

### 16.2 The mechanism holds on the GLOBAL model, and it precedes grokking

The global model is the same object in both arms, so it carries no such
confound. Per-neuron spectral IPR of the global `W1[:, :97]`
(`metrics/fourier.py::spectral_ipr`, the project's own instrument — values
reproduce the CSV's `final_ipr` exactly, max |Δ| = 0.0000 across all 10 runs).
`t2_boundary`, K=97, α=0.25, 5 seeds per arm. Higher = more periodic; 1/IPR is
the effective number of frequencies per neuron.

| round | 1,000 | 4,000 | 7,000 | 10,000 | 13,000 | 16,000 | 19,000 |
|---|---|---|---|---|---|---|---|
| iid | 0.0236 | 0.0352 | 0.0513 | 0.0667 | 0.0816 | 0.1005 | 0.1204 |
| **operand** | 0.0250 | 0.0505 | 0.0804 | 0.1067 | 0.1357 | 0.1775 | **0.2296** |
| iid 1/IPR | 42.4 | 28.4 | 19.5 | 15.0 | 12.3 | 10.0 | 8.3 |
| **operand** 1/IPR | 40.0 | 19.8 | 12.4 | 9.4 | 7.4 | 5.6 | **4.4** |

At the final round the arms are completely separated per seed — operand
0.2313–0.2839 against iid 0.1159–0.1632.

**The gap is not a consequence of grokking.** The earliest first crossing
anywhere in the cell is round 11,560 (operand seed 42, 57,800 steps at E=5), so
every round below it is pre-transition for all ten runs:

| round | iid range | operand range | ratio | |
|---|---|---|---|---|
| 2,000 | 0.0261–0.0305 | 0.0300–0.0412 | 1.19× | overlap |
| 4,000 | 0.0325–0.0421 | 0.0417–0.0642 | 1.43× | overlap |
| 6,000 | 0.0405–0.0546 | 0.0556–0.0857 | 1.52× | **separated** |
| 8,000 | 0.0485–0.0654 | 0.0731–0.1095 | 1.58× | **separated** |
| 10,000 | 0.0575–0.0774 | 0.0962–0.1387 | 1.60× | **separated** |

Complete separation ~5,600 rounds before any run reaches the bar. So coherent
shards build Fourier structure *first*, and generalise *because* of it — the
ordering the mechanism hypothesis predicts.

**Why this is a consensus measure and not merely a circuit-quality one.** Under
the operand partition each column of the first-operand block is trained by
exactly one client. A periodic pattern *across* columns therefore cannot be
produced by any single client — it requires the clients to agree. Under iid
every client touches every column, so no agreement is needed. The global
spectrum is thus a stricter test of consensus under operand than under iid, and
operand is the arm that wins.

**It is not confined to the sharded axis.** The second-operand block, which
`operand` does not shard, shows the same ordering — 1.20× at round 4,000 rising
to 1.43× at 10,000, separated by round 10,000. Weaker and later, consistent with
consensus acting first where the shards bite and propagating to the rest of the
circuit.

**And it predicts *which seeds* grok.** Within the iid arm alone, the two seeds
that eventually cross (at rounds 19,120 and 19,700) separate completely from the
three that never do, from round 6,000 onward:

| round | grokked (2) | censored (3) | |
|---|---|---|---|
| 4,000 | 0.0352, 0.0421 | 0.0325, 0.0334, 0.0386 | overlap |
| 6,000 | 0.0477, 0.0546 | 0.0405, 0.0418, 0.0468 | **separated** |
| 8,000 | 0.0613, 0.0654 | 0.0485, 0.0516, 0.0565 | **separated** |
| 10,000 | 0.0746, 0.0774 | 0.0575, 0.0604, 0.0667 | **separated** |

A within-arm result, so the partition is held fixed: spectral concentration at
round 10,000 orders the seeds by an outcome ~9,000 rounds away. n=5, so this is
a lead worth a dedicated cell, not a finished claim.

### 16.3 `d_internals`: the dip is masking, not a loss of structure

85 runs, 17 α rungs × 5 seeds, setup D centralized, previously unanalysed. The
manifest asked *why* α's effect is almost entirely the phase-1 plateau P(α).
Setup D's quadratic activation makes the split exact:
`logit = A[c,a] + 2T[c,a,b] + B[c,b]`, where A and B cannot compose, so T is the
compositional circuit.

**The non-compositional circuit is dead everywhere.** A+B scores 0.1–0.4% at
every α and every phase, at or below the 0.83% chance level. There is no
memorise-then-compose handoff on this setup: it is compositional from the start.

**The compositional circuit alone beats the full model**, at the plateau
(step 1,800) and by more at the dip (3,600):

| α | 0.3 | 0.4 | 0.45 | 0.5 | 0.6 |
|---|---|---|---|---|---|
| T accuracy − full-model test, plateau | 3.9 | 18.9 | **19.6** | 13.0 | 4.5 |
| same, at the dip | 9.0 | 33.8 | 29.4 | 19.8 | 7.4 |

Per-seed spread is ~1 point (α=0.4: 18.8, 18.9, 18.9, 19.2, 19.8), which for
this project is an unusually tight effect. The gap peaks mid-ladder, near where
P(α)'s sigmoid has its midpoint.

**So the dip is not the circuit degrading.** From 1,800 to 3,600 steps T's own
accuracy *rises* at every α ≥ 0.3 (α=0.4: 51.4 → 65.9%) while the full model's
test accuracy falls and T's share of the logit drops (0.797 → 0.698). The
marginal terms grow and mask a compositional circuit that never stops improving.
P(α) is T's accuracy at 1,800 steps minus that masking penalty.

**The pilot's `circ_units` lead does not replicate.** The manifest cites one
seed at α=0.55 going 54 units at 1,800 → 75 at 3,600. That seed reproduces
exactly; the other four go 95→55, 142→29, 89→68, 54→90. Across all 85 runs the
participation ratio rises in 30 and falls in 55, and spans 22–206 at fixed α.
The direction is seed noise, not a property of the dip.

**Group structure arrives late.** Total-variation distance of the S₅ isotypic
shares from the dimension-proportional null (dim²/120):

| α | 0.2 | 0.3 | 0.4 | 0.45 | 0.5 | 0.55 | 0.6 |
|---|---|---|---|---|---|---|---|
| plateau | 0.017 | 0.031 | 0.062 | 0.067 | 0.078 | 0.083 | 0.084 |
| dip | 0.016 | 0.040 | 0.070 | 0.079 | 0.078 | 0.082 | 0.086 |
| end | 0.066 | 0.099 | 0.100 | 0.114 | 0.121 | 0.118 | 0.116 |

Through phase 1 and the dip the decomposition is close to "no irrep selection at
all"; it roughly doubles by the end, moving weight off the 6-dimensional 311 and
onto the standard representation while suppressing both 1-dimensional irreps. So
representation-theoretic structure is a phase-2 phenomenon, where the
compositional circuit is a phase-1 one.

---

## 15. exp2 and exp3b — main's two central experiments, on six setups

449 runs. Both are ports of `main`'s own experiments, with their arms and seed
counts unchanged and their grids re-scoped against what Part 0 measured.

### 15.1 exp2 — does aggregation compensate for fragmenting the data?

main's three conditions verbatim (`experiments/exp2_aggregation.py`): `cent_full`
(one model, all the data), `cent_reduced` (one model, one client's 1/K shard),
`fl` (K clients, FedAvg). **No v2 manifest had ever carried the floor arm** —
`reduced_arm` was written, tested and plumbed with nothing calling it.

192 runs. K ∈ {2,5,10,20,50}, 3 seeds, each setup at its own working point.
FL as a multiple of its own centralized ceiling, on `t_first_cross`:

| K | A (GD) | B | D | E |
|---|---|---|---|---|
| 2 | **1.00×** | 1.25× | **1.01×** | 1.53× |
| 5 | 1.01× | 1.23× | 1.26× | 3.75× |
| 10 | 1.02× | 1.32× | 3.67× | 7.08× |
| 20 | 1.06× | 2.07× | 4.39× | 15.56× |
| 50 | **1.17×** | **0/3** | **0/3** | — |

**On the anchor, aggregation fully compensates.** Fifty clients cost 17% and
nothing else. That is main's exp2 conclusion, now with a floor arm and a K=2
control it never had.

**On the AdamW setups it degrades with K and then fails**, exactly as the decay
clock predicts (§14.3). A is the only setup with no decay clock and the only one
that holds.

**The floor reproduces main qualitatively**: one client's shard groks in **6 of
87** cells (7%); main reported 1 of 30.

**The K=2 control earned its place.** It is the cell closest to centralized, so
`cent_full` and `fl` should agree there. A (1.00×) and D (1.01×) agree exactly,
which validates the compute-matched step axis. Two setups did not, and they
failed differently:

- **C is not interpretable at n=3.** FL reads 4.5× *faster* than centralized, but
  centralized spans 11,200–43,600 across three seeds against FL's 6,500–15,400 —
  overlapping ranges. C's ~30× seed variance swamps the comparison. No number is
  reported for C.
- **A′ reads 13.3× at K=2**, consistently across seeds, with memorisation
  identical at 200. See §15.3 — this is expected, not a fault.

### 15.2 exp3b — does it matter HOW you shard?

main's exp3b (iid · operand · target) plus **dirichlet**, the control main could
not run: `t2_k_breakdown` found operand faster than iid at K=50 while dirichlet
tracked iid exactly, so without it *structure* and *heterogeneity* are confounded.

Run on the setups whose seed noise is small enough to resolve a ~10% effect —
within-cell spread at exp2's K=10 cells is 1.0–1.1× for A, A′ and E against
1.8–1.9× for B, C and D. B and C are deferred; **D's iid/operand cells ran
anyway**, since a 1.9× spread cannot hide a qualitative failure.

> **SUPERSEDED by §17.2.** The deferred cells ran on 2026-08-11, plus the coset
> arm. Read §17.2 instead of this section for the partition comparison: it uses
> exactly matched iid baselines per cell, where the table below borrows working
> points across sweeps.

Median `t_first_cross`:

| setup | K | iid | operand | target | dirichlet |
|---|---|---|---|---|---|
| **A** | 10 | 12,900 | **12,700** | 14,100 | 13,000 |
| **A** | 50 | 14,700 | **13,100** | **28,200** | 15,000 |
| D | 10 | 78,900 | **0/3** | — | — |
| D | 20 | 94,300 | **0/3** | — | — |
| E | 10 | 5,100 | — | — | **4,000** |
| E | 20 | 11,200 | — | — | **5,800** |

**The anchor replicates, and the control does its job.** Operand beats iid by
~1.5% at K=10 and **11% at K=50** — the gap grows with fragmentation. Dirichlet
tracks iid at both K. So the effect is **coherence, not heterogeneity**.

**`target` is ~2× WORSE than iid at K=50.** Sharding by output class is
structured and actively harmful. main ran `target` but had no dirichlet arm to
read it against, so this ordering was not available to it.

**D inverts it: iid groks at both K, operand fails 0/3.** Not slower — flat.
Before treating that as a contradiction, note that on S₅ the operand partition
shards by first-operand *element*, which is **not** algebraically coherent the
way a mod-p operand shard is. S₅'s coherent split is **coset**, and those cells
have not run. The available reading is therefore "incoherent structure is worse
than random", which *agrees* with the anchor rather than contradicting it — and
the coset arm is what decides between "structure helps" and "coherence helps".

**A′ shows no stable ordering** across K and is treated as uninterpretable: it
sits at its cliff (margin +0.01 against +0.10 for every other setup) and carries
the confound in §15.3.

**E prefers unstructured non-IID** — dirichlet beats iid at both K. `label_block`
fails 0/3, consistent with its shards being single-class.

### 15.3 A CORRECTION: the E=1 FedAvg identity covers stateless optimisers only

exp2's A′ column reads 13.3× at K=2, where federation should be nearly a no-op.
To test it I ran A′ at **E=1**, asserting the cell *must* reproduce the ceiling
because FedAvg at E=1 is an exact algebraic identity with centralized GD.

It did not — 0.0% test on all three seeds. **The assertion was wrong, not the
harness.**

`tests/test_fedavg_identity.py` proves that identity at `momentum=0.0,
weight_decay=0.0` — plain, **stateless** GD. It never claimed to cover AdamW. And
`training/federated.py` rebuilds the local optimiser every round when
`persist_local_opt_state=False`, which is standard FedAvg semantics. For GD that
is a no-op; for AdamW at E=1 **every round is a single cold-start Adam step**,
which under bias correction is closer to signSGD than to Adam. The cell must
*not* match the ceiling.

Two consequences:

1. **exp2 stands.** A′'s 13.3× is the optimiser-restart cost of faithful FedAvg
   with an adaptive optimiser, amplified by A′ sitting at its cliff.
2. **Every AdamW setup's federated cost conflates federation with optimiser
   restart** — A′, B, C, D, E alike. `persist_local_opt_state=True` exists to
   separate them.

**CORRECTION (2026-08-10): it has been used.** This section originally said no
run ever had. `s5_fl_probe` carries a paired arm — 12 banked runs, setup B,
α=0.30, K=10, iid, 3 seeds, at E=5 and E=50 — and it was sitting unread:

| E=5, 2,000 rounds | persist=False | persist=True |
|---|---|---|
| grokked | **3/3** | 2/3 |
| t_memo | 8,200 · 4,400 · 6,800 | ∞ · 5,900 · 4,000 |
| t_first_cross | 9,100 · 4,600 · 6,800 | ∞ · 6,600 · 4,900 |

Persisting Adam state across rounds does **not** recover the ceiling on B: one
seed fails outright, and the two that grok overlap the standard arm. So on the
setup that has the data the optimiser-restart cost is small, and the claim above
is weaker than stated for the family. The open case is **A′ specifically** —
where the 13.3× was observed — not every AdamW setup. That is ~3 runs, not ~12.

### 15.4 Weight decay is load-bearing on three setups, and finishing on the fourth

wd=0 controls, centralized, 3 seeds. Every cell **memorises by epoch 200–400**,
so none of these is a training failure:

| setup | wd=0 final test | chance | with decay |
|---|---|---|---|
| B | 3.2 · 5.9 · 2.5% | 0.88% | 100% |
| C | 0.3 · 0.4 · 0.4% | 0.83% | 100% |
| D | 1.5 · 1.2 · 1.2% | 0.83% | 95–98% |
| **E** | **81.0 · 74.2 · 80.3%** | 10% | 92–95% |

B, C and D memorise and then sit at chance indefinitely: decay is what causes
grokking. **E is different** — without decay it reaches 74–81%, far above chance
but short of its 90% bar, so decay *finishes* generalisation rather than gating
it. That matches Omnigrok, and it means "these setups need weight decay to grok"
is true for B/C/D and **false as stated for E**.

## 14. Part 0 — the campaign prerequisites, and the decay clock confirmed

129 runs. Four questions that every campaign manifest depends on, plus one
methodological finding that changes which statistic the project reports.

### 14.1 Setup C's cliff is at α≈0.3, not above 0.5 (`p0_c_alpha_width256`)

"C's working α is ≥0.5" has been carried since Gate A and used to argue C could
not be matched to the other setups. It rested on **one cell** — α=0.30, 0/5, at
hidden_width **128** with 40,000 epochs — while at that same width α=0.50 has a
KM median of 39,800. The ladder was cut off essentially *at the median for an
easier α*, so a harder α censoring under it was the expected outcome, not a
cliff. C had never been run below α=0.5 at width 256, the width the capacity
sweep actually selected.

At width 256, 100,000 epochs, 3 seeds:

| α | grokked | median first crossing | final test |
|---|---|---|---|
| 0.25 | 0/3 | — | 0.3–0.6% (chance) |
| 0.30 | **1/3** | 55,500 | 0.9 · 1.0 · **100** |
| 0.40 | **3/3** | 37,450 | 100% |
| 0.50 | 15/15 | ~15,150 | 100% |
| 0.60 | 3/3 | 5,300 | 100% |

**C's usable working point is α=0.40**, not ≥0.50 and not 0.30. The old claim is
withdrawn. α=0.30 is genuinely marginal at 1/3 rather than cliffed — its two
failures sit at chance for the full 100,000 epochs, but C's seed spread is
enormous (at α=0.6: 2,550 / 52,850 / 77,000 on t_grok, a 30× range), so a slow
tail running past budget is entirely consistent with what the grokking seed did
at 55,500.

**Every C run memorises at epoch 100–200**, at every α. So α does nothing to
memorisation on this setup; the entire α effect is on the delay.

### 14.2 Capacity: wider is not better, and for A′ it is fatal (`p0_capacity`)

Half / default / double width per setup, at its own working α. Read on
`t_first_cross`, for the reason in §14.4:

| setup | narrow | **default** | double |
|---|---|---|---|
| **A′** quad-MLP/mod-97 | **1,050** (128) | 3,900 (256) | **0/3 — never** (512) |
| **B** transformer/mod-113 | 3,750 (64) | **4,350** (128) | 3,500 (256) |
| **D** quad-MLP/S₅ | 0/3 (128) | 21,300 (256) | 18,550 (512) |
| **E** MLP/MNIST | 2,575 (100) | **725** (200) | 600 (400) |

- **A′ is 3.7× faster at half its default width, and fails outright at double.**
  There is an upper capacity limit, and A′'s inherited 256 sits above the optimum.
- **B's width is not binding at all** — all three widths cross at ~3,500–4,350.
  What does change is stability: post-crossing dips run [0,0,2] at width 64,
  [0,1,2] at 128 and [2,3,3] at 256, so wider is *less* stable. Nanda's 128 stands.
- **D needs ≥256**; 512 buys ~13%, not enough to justify moving off the default.
- **E's 200 stands.** Width 100 is unstable (dips 4 / 14 / 45); 200 and 400 have none.

### 14.3 THE DECAY CLOCK, CONFIRMED — D reproduces the collapse (`t1_setup_k_ladder`)

§13.7's mechanism hypothesis was that memorisation blows up with K on setups with
a decay clock (AdamW) and stays flat on those without (GD at wd=0), because
decoupled decay is applied per local step and is independent of shard size while
the learning signal from a 1/K shard is not. It predicted **D resembles B**.

It does, on a different architecture *and* a different task:

| setup D (quad-MLP, S₅, AdamW), iid, E=5, α=0.30 | K=5 | K=10 | K=20 | K=50 |
|---|---|---|---|---|
| t_memo | 1,400 | 3,200 | 15,300 | **never** |
| peak train | 100 | 100 | 100 | **69–74%** |
| grokked | 3/3 | 2/3 | 1/3 | **0/3** |

D at K=50 does not memorise inside 100,000 steps — the same failure B shows at
wd=1.0, on a setup that shares neither its architecture nor its task. Every
AdamW setup measured now shows memorisation slowing steeply with K: B, C, D and E
alike.

**And the delay grows with K too — B's is not flat.** §13.7 recorded that as
unproven on two K values with n=1 at one of them. The ladder resolves it:

| B, wd=0.1, iid, E=5 | centralized | K=5 | K=10 | K=20 | K=30 |
|---|---|---|---|---|---|
| t_memo | 150 | 800 | 1,300 | 3,600 | 5,900 |
| **delay** | 44,900 | 50,400 | 54,600 | **90,100** | 88,400 |

The delay doubles from centralized to K=20 and then plateaus. So the corrected,
unified statement is:

> **The delay grows with K on every setup measured.** What the decay clock
> controls is whether *memorisation* also blows up — flat under GD at wd=0,
> explosive under AdamW. Budget from both terms, and neither is constant.

**Setup C is the exception that sharpens it.** Its delay *collapses* with K —
3,500 at K=5, 4,600 at K=10, 600 at K=20, and **negative** at K=50, where the
first crossing (36,500) precedes memorisation (44,150). At high K, C stops
grokking and simply learns: test accuracy reaches the bar before train accuracy
reaches 99%. C is 3/3 at every K, the only setup that never degrades.

**Corrected by §17.2:** that is true at α=0.50 under iid, which is what this
ladder ran. At α=0.40 with structured shards C fails 0/3 at K=50 with peak train
accuracy of 12–15% — it never memorises. C degrades like every other AdamW setup;
this ladder simply did not reach the regime where it does.

### 14.4 `t_grok` depends on the BUDGET, not only the logging rate

§13.4 recorded that `t_grok` measures the logging rate on an unstable setup. It
is worse than that. Same config, same seed, same `log_every=50` — only the budget
differs:

| run | epochs | t_grok | t_first_cross |
|---|---|---|---|
| `b_decay_band` seed 123 | 50,000 | **4,350** | 4,350 |
| `capacity` seed 123 | 100,000 | **95,750** | 4,350 |

Because `t_grok` requires the bar to hold for the *rest of the run*, **a longer
budget can report a later t_grok for an identical trajectory prefix**. Extending
a run can therefore make its measured grokking time 22× worse.

This is not a corner case: it silently invalidated the first reading of §14.2,
where B appeared 13× faster at width 64 purely because the wider runs sampled
more dips. **`t_first_cross` is the statistic to compare across cells whenever
budgets differ**, and `t_grok` should only be compared within a fixed budget
*and* a fixed logging rate.

### 14.5 Server-LR calibration — and FedAdam's advantage does not survive it

`t3_server_lr_calibration`, 42/42. One representative heterogeneous cell (setup A,
α=0.30, K=10, E=5, dirichlet α=0.1), 3 seeds, read on `t_first_cross`:

| strategy | server_lr | momentum | grokked | first crossing |
|---|---|---|---|---|
| **FedAdam** | 0.01 | — | 3/3 | 2,100 |
| **FedAdam** | **0.1** | — | **3/3** | **600** |
| FedAdam | 0.3 / 1.0 | — | 0/3 | never — 1.0% test |
| **FedYogi** | 0.01 | — | 3/3 | 2,000 |
| **FedYogi** | **0.1** | — | **3/3** | **400** |
| FedYogi | 0.3 / 1.0 | — | 0/3 | never — 1.0% test |
| FedAvgM | 0.1 | 0.0 | **0/3** | never — 0.2% test |
| FedAvgM | 0.1 | 0.9 | 3/3 | 13,900 |
| FedAvgM | 0.3 | 0.9 | 3/3 | 5,000 |
| **FedAvgM** | **1.0** | **0.9** | **3/3** | **1,800** |

**The calibrated working points**, to be fixed before any algorithm comparison:

| | server_lr | momentum |
|---|---|---|
| FedAdam | 0.1 | — |
| FedYogi | 0.1 | — |
| FedAvgM | 1.0 | 0.9 |

Three things follow.

**FedAdam is not the best adaptive method once the others are tuned too.** At its
own best setting FedYogi reaches the bar at **400** against FedAdam's **600**.
v1's exp5 reported FedAdam ~10× faster than the field, but it swept `server_lr`
for FedAdam *alone* — so that headline compared a tuned method against untuned
ones. Fixing each at the points above is what makes the re-run fair, and on this
evidence the ordering may not survive.

**Both adaptive methods fall off a cliff above server_lr = 0.1.** At 0.3 and 1.0
they sit at chance (1.0% test, 0/3). The usable band is narrow, and 1.0 — the
natural default, and FedAvg's implicit value — is outside it for both.

**FedAvgM without momentum is worse than useless at low server_lr**: 0/3 at
(0.1, 0.0), against 3/3 at (0.1, 0.9). Its server LR and momentum are not
separable knobs and must be tuned as a pair.

**Caveat.** This cell has no FedAvg arm — FedAvg has no server-LR knob, so it was
not part of the sweep. These numbers calibrate each method against itself; they
do not establish a speedup over FedAvg. That comparison *is* exp5, and it now has
fair settings to run at.

## 13. Phase 1 — the setups, measured instead of inherited

Four sweeps, 96 runs, run to answer questions whose answers every downstream
manifest depends on. Each manifest's docstring in `scripts/build_manifests.py`
carries the decision rule it was written against; the verdicts below are those
rules applied.

### 13.1 The quadratic MLP does grok S₅ under plain GD (`p1_d_gd_probe`, 9/9)

Setup D is Gromov's architecture running Nanda's optimiser — inherited from the
S₅ side rather than the architecture side, and never checked: of 370 banked S₅
runs, **zero** used GD. That makes A vs D, which is meant to isolate the *task*,
move task and optimiser and loss together.

α=0.5, wd=0, MSE, 50,000 epochs, 3 seeds:

| lr | grokked | KM median T_grok |
|---|---|---|
| 5 | 0/3 | censored |
| 10 | 0/3 | censored |
| **50** | **3/3** | **22,600 [22,500, 23,200]** |

So the confound is closable — Gromov's exact configuration transfers to S₅ at
Gromov's exact learning rate. It costs ~3× the time AdamW needs at the same α
(22,600 against 7,200), which is the price of the clean contrast.

**Acted on by adding D′, not by moving D.** Run ids are content hashes, so
editing `SETUP_D` in place would orphan ~250 banked D runs — the Gate A ladder,
the 0.025-resolution tier-X ladder, and every D federated cell. This is the same
reasoning that produced A′ rather than a change to A. D′ is **gated**: its real
value is that wd=0 gives it no decay clock, which should make it immune to the
K≈30 collapse the way setup A is, and that only matters if §12 turns out to be a
genuine breakdown rather than a hyperparameter defect.

### 13.2 C's and D's decay: neither moves (`p1_cd_decay_band`, 30/30)

Both inherited wd=1.0 from B, which has a reason to carry it (B *is* the Nanda
replication) where they do not. Same log-spaced lr·λ ladder as `t0_wd_grid` and
`t0_mnist_wd_band`, 3 seeds, 100,000 epochs.

**D** (quad-MLP, S₅, α=0.30) — one band works and it is the inherited one:

| lr·λ | 1e-5 | 3e-5 | 1e-4 | 3e-4 | **1e-3** |
|---|---|---|---|---|---|
| grokked | 0/3 | 0/3 | 0/3 | 0/3 | **3/3** |
| T_grok | — | — | — | — | **21,300** |

**C** (transformer, S₅, α=0.50, width 256) — three bands work, and the tie-break
decides. Both statistics pick wd=1.0, by a wider margin on the sampling-robust one:

| lr·λ | grokked | KM median T_grok | median first crossing | dips after crossing |
|---|---|---|---|---|
| 1e-5 | 0/3 | censored | — | — |
| 3e-5 | 1/3 | censored | 97,100 | 0 |
| 1e-4 | 3/3 | 83,800 | 70,200 | 0, 1, 0 |
| 3e-4 | 3/3 | 96,750 | 42,650 | **13, 28, 2** |
| **1e-3** | **3/3** | **59,350** | **15,150** | 2, 1, 1 |

**Consequence: the re-ladder that Phase 2 was budgeted for does not happen.** The
plan reserved ~60 runs to re-measure C's and D's α ladders at a moved decay,
because the ladder sets the working point and every downstream budget. Neither
moved, so both Gate A ladders stand as written. This is Phase 1's largest saving.

**But C is unstable, and that is a new setup property.** Read the last two columns
together: higher decay reaches the bar much sooner and then oscillates, lower
decay is slow and steady. At wd=0.3 one seed drops below the bar 28 separate
times after first crossing it. C stays at wd=1.0 with `t_first_cross` as its
primary statistic — see §13.4 for why `t_grok` alone cannot be trusted here.

### 13.3 A′ cliffs *below* A, and groks ~45× faster (`p1_aprime_alpha`, 45/45)

A′ is A's architecture and task under AdamW (MSE, not CE, so A vs A′ moves one
variable). 5 seeds:

| α | 0.15 | 0.175 | **0.2** | 0.25 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|
| **A′** (AdamW) | 0/5 | 0/5 | **5/5 · 5,500** | 5/5 · 500 | 5/5 · 300 | 5/5 · 200 | 5/5 · 170 |
| **A** (GD) | 0/5 | — | **0/5** | 25,300 | 13,100 | 8,800 | 7,600 |
| ratio | | | A′ only | **51×** | 44× | 44× | 45× |

Two things, both new:

**The optimiser is worth a factor of ~45, flat across α.** Same architecture,
same data, same loss, same seeds — only GD → AdamW. The ratio is remarkably
constant, which says AdamW is rescaling the clock rather than changing the
transition.

**The cliff moves down.** A′ groks 5/5 at α=0.20 where A manages 0/5. So the data
threshold is *not* purely a property of the task family, as §11 suggested from
four setups that happened to agree — the optimiser shifts it too.

**A′'s only usable federated working point is α=0.20.** Above it there is no
delay left to disrupt: 500 steps at α=0.25 means E=5 federation gets 100 rounds
to act. This is the trap setup E already fell into, where the FL probe ran MNIST
at a working point with no delay at all and its censoring meant nothing. Only the
α=0.20 cell has a real gap (5,500, with a wide seed spread of 3,000–8,500).

### 13.4 `t_grok` measures the logging rate on an unstable setup

Found while reading §13.2. `compute_t_grok` requires the bar to hold for the rest
of the run — correct for a phase transition, but on a non-monotone curve every
extra sample point is another chance to observe a dip and push the answer later.

Setup C, one seed, **one trajectory**: `t_grok` = 15,200 at `log_every=200` and
59,350 at `log_every=50`. A 4× difference from the instrument alone, because the
coarse run never sampled a collapse to 20.2% at epoch 59,300. C's decay band was
about to be chosen by tie-breaking on exactly that number.

Two outcomes are now recorded per run alongside it, at the run's own
`grok_threshold`, and backfilled across all 814 banked rows:

- **`t_first_cross`** — first time the bar is reached, sustained or not. Not
  sampling-invariant either, but stable to within one logging interval rather
  than to within the instability's duration.
- **`post_grok_dips`** — logged points below the bar after that crossing. Zero
  means the transition held. The magnitude scales with `log_every`, so only the
  zero/non-zero distinction transfers between runs.

Together they separate *when did it generalise* from *when did it stop falling
over*, which on an unstable setup are different questions. This is the fifth
measurement artifact this project has caught reading as a scientific result.

### 13.5 B's decay band, and why it had to be measured (`p1_b_decay_band`, 15/15)

The third AdamW setup's decay, on the same 5-point lr·λ ladder as C and D.
α=0.30, 3 seeds, 50,000 epochs. Per-seed T_grok, since B's seed variance is
bimodal by design:

| lr·λ | wd | grokked | T_grok per seed | KM median |
|---|---|---|---|---|
| 1e-5 | 0.01 | 0/3 | never | censored |
| 3e-5 | 0.03 | 0/3 | never | censored |
| 1e-4 | 0.1 | 3/3 | 40,250 · 45,050 · 46,050 | **45,050** |
| 3e-4 | 0.3 | 3/3 | 18,000 · 20,000 · 49,350 | 20,000 |
| **1e-3** | **1.0** | **3/3** | 4,350 · 6,100 · 19,450 | **6,100** |

Three things, in order of importance:

**Every cell memorises at epoch 150.** Decay does nothing to memorisation on this
setup; the entire effect is on the generalisation timescale. That is the cleanest
statement of the AdamW weight-decay mechanism the project has produced.

**Weight decay accelerates grokking monotonically, and the inherited wd=1.0 is
optimal.** A third independent confirmation of §6.4 — and the third inherited
hyperparameter to survive being checked, after C's and D's. It also reproduces
Gate A independently (6,100 here against 6,600 there) including the bimodality:
one slow seed appears at *every* band.

**And it is the control the K-collapse arm was missing.** See below.

### 13.6 The K≈30 collapse — a graded training failure, and a budget trap avoided

`p1_k_collapse_wd`, complete at 18/18. α=0.30, E=5, 3 seeds, **10,000 steps**:

| K | wd | peak train | final test | t_memo | grokked |
|---|---|---|---|---|---|
| 20 | **0.1** | **100.0** | 0.23 / 0.45 / 0.46 | 3,500–3,900 | 0/3 |
| 20 | 1.0 | 44.9 / 100.0 | 7.3 / 99.8 | inf / 7,300 | 1/2 |
| 30 | **0.1** | **100.0** | 0.17 / 0.25 / 0.45 | 5,600–6,600 | 0/3 |
| 30 | 1.0 | 93.1 / 51.9 | 82.2 / 24.2 | inf | 0/2 |
| 50 | **0.1** | **77.6 / 79.2** | 0.34 / 0.38 | **inf** | 0/2 |

**The recovery is graded, not binary.** wd=0.1 restores memorisation *completely*
at K=20 and K=30 — 100% train, 3/3, against 42.8% at K=30 under the inherited
decay — and only *partially* at K=50, where peak train rises from ~3.6% to
~78% but never reaches the 99% memorisation bar. So the decision rule as written
("t_memo finite, 3/3 at K=50") is **not met**, but the knob is unambiguously the
right one: decay is what moves this failure, and it moves it a long way.

**The generalisation half of the question was about to be answered wrongly.**
Read alone, the wd=0.1 rows look decisive: memorised at 100% train, then sitting
at 0.2–0.5% test against a 0.88% chance level. Not slow generalisation — none.
That reads as a federated breakdown of grokking with memorisation intact, which
would be the headline of the whole campaign.

It is not. **§13.5 says centralized B at wd=0.1 needs 45,050 steps to generalise.
These runs were given 10,000** — with a *single* client the same configuration
would also show 100% train and no test accuracy at that budget. The cells are
censored by the clock, and say nothing about federation yet.

That is the **fifth** time a fixed budget has manufactured a boundary in this
project — after v1's headline claim, the E=1 probe cells, the first FL probe, and
setup C's Gate A verdict. It is the first time it was caught *before* being
claimed rather than after, and the only reason is that the control was run first.

`p1_k_collapse_budget` re-runs the arm at a budget keyed to that measured number:
K=20 at **200,000 steps (4.4×)** because it is the cheapest cell and therefore the
place to buy the most headroom, K=30 and K=50 at 100,000 (2.2×), plus a wd=1.0
control at the longer budget to check that "never memorises" is itself
budget-independent rather than the same error mirrored.

> **This was the gate on the whole campaign, and it opened.** wd=0.1 groks 3/3
> at K=20 — see §13.7, which also shows that the wd=1.0 control caught the same
> error mirrored, and that §12's headline table was itself censored.

**Cost note.** Measured on these runs, setup B federated is ~0.44–0.52 s/round
and **nearly flat in K** (886 s at K=20, 1,014 at K=30, 1,047 at K=50, all at
2,000 rounds). The fitted cost model in §8 comes from setup A and carries a
`1.291·K` term; it over-costs setup B by ~2.6× at K=50. Per-setup cost fits are
still owed.

### 13.7 THE GATE: the K≈30 collapse was never a collapse

`p1_k_collapse_budget`, 11/11, 4.9 h. Every cell of the diagnosis re-run at a
budget keyed to §13.5's measured centralized requirement. **The K axis reopens**,
and the reason it ever looked closed is the same reason four earlier claims in
this project were wrong.

**The gate cell.** wd=0.1, K=20, 200,000 steps, iid, E=5:

| seed | t_memo | T_grok | final test |
|---|---|---|---|
| 42 | 4,100 | 98,400 | 100.00 |
| 123 | 3,500 | 66,300 | 99.68 |
| 456 | 3,600 | 93,700 | 99.66 |

**3/3.** The same configuration was 0/3 at 10,000 steps. So the collapse is an
inherited-hyperparameter defect plus an under-budget, exactly as the decision
rule's first branch specified — **D′ is not needed**, and the A-vs-D optimiser
confound gets recorded as a stated limitation rather than fixed with a seventh
setup.

#### The decomposition that explains every cell

Recording `t_memo` separately from `t_grok` turns a confusing table into an
additive one. Restricted to iid, E=5, FedAvg, full participation, α=0.30:

**Corrected decay, wd=0.1:**

| | t_memo | T_grok | **delay** |
|---|---|---|---|
| centralized | 150 | 45,050 (3/3) | 44,900 |
| K=20 | 3,600 | 93,700 (3/3) | 90,100 |
| K=30 | 5,900 | 94,300 (1/3) | 88,400 |
| K=50 | 53,300 | censored (0/3) | — |

**Two separate things happen, and they are not equally well established.**

1. **Memorisation time explodes with K** — 150 → 3,600 → 5,900 → 53,300, i.e. 24×,
   39×, then 355× the centralized value. Three seeds at every K, and the effect is
   orders of magnitude beyond seed spread. **Solid.**
2. **The delay is large and roughly doubles under federation** — ~90,000 against
   the centralized 44,900. Established at K=20, 3 seeds.

**Whether the delay grows with K is NOT established, and this sweep cannot settle
it.** The per-seed delays are:

| K | seeds yielding a delay | delays |
|---|---|---|
| 20 | 3/3 | 62,800 · 90,100 · 94,300 |
| 30 | **1/3** | 88,700 |
| 50 | **0/3** | — |

Two K values, one of them a single seed, and the K=30 point merely falls inside
K=20's spread — which is itself a factor of 1.5, **larger than any K-dependence it
could detect**. Flat is consistent with this data; so is a mild increase.

So `T_grok(K) ≈ t_memo(K) + delay` describes the censored cells well: K=30 needs
≈ 5,900 + ~90,000 = ~95,900 against the 100,000 it got, which is why exactly one
seed made it at 94,300 and two did not; K=50 needs **≳ 143,000** and received
100,000, its three seeds memorising and sitting mid-delay at 2.4–14.0% test rather
than stuck. But 143,000 is a **lower bound, not a prediction** — if the delay does
grow with K, budgeting exactly to it recreates the censoring it was derived from.

**Inherited decay, wd=1.0** — same structure, opposite balance:

| | t_memo | T_grok | delay |
|---|---|---|---|
| centralized | 150 | 6,100 (3/3) | 5,950 |
| K=5 | 1,800 | 5,250 (1/1) | 3,450 |
| K=10 | 6,350 | 6,700 (8/9) | 350 |
| K=20 | 7,300 | 7,500 (1/3) | 200 |
| K=30 @ 10k | never | censored (0/3) | — |
| **K=30 @ 100k** | **12,900** | **13,200 (1/1)** | **300** |
| K=50 @ 100k | never | censored (0/1) | — |

At wd=1.0 the delay collapses to ~300 steps, so `T_grok ≈ t_memo` and the whole
question is whether the model memorises at all.

#### The correction this forces on §12

**The K=30 cell was a censored measurement, not a training failure.** §12's table
— peak train 100 / 98.2 / 42.8 / 5.9 / 5.0 at K = 10 / 20 / 30 / 40 / 50 — was
read off runs given **10,000 steps**. At K=30 memorisation takes **12,900**. Given
100,000 steps the same setup memorises and groks at 13,200. The "42.8%" that
anchored the entire collapse narrative is where that run had got to when the clock
stopped.

K=50 under the inherited decay does still fail at 100,000 steps — but at peak train
**37.8%**, against the 5.0% recorded at 10,000. It is slow, not stuck, and it is
the only cell in the diagnosis that still looks like a genuine wall.

> **There is no cliff at K≈30.** There is a steep, continuous slowdown of
> *memorisation* with client count, and a delay that federation roughly doubles
> and then leaves alone. Every "collapse" datum was that slowdown, measured
> against a clock that had already run out.

This is the **sixth** time a fixed budget has manufactured a boundary here, and
the third caught inside this one investigation — §13.6 (the wd=0.1 arm), the
wd=1.0 control added specifically to guard against the mirror-image error, and
§12's headline table itself. The control earned its two runs.

#### The decomposition is setup-dependent — A and B are opposites

Applying the same split to the anchor, which has 92 banked federated runs, gives
the reverse picture. Both at α=0.30, iid, E=5, FedAvg, 5 seeds (A) / 3 (B):

| | K=5 | K=10 | K=20 | K=50 |
|---|---|---|---|---|
| **A** (GD, wd=0) t_memo | 3,700 | 3,700 | 3,700 | 3,700 |
| **A** delay | 9,500 | 9,700 | 10,000 | 11,500 |
| **B** (AdamW, wd=0.1) t_memo | — | — | 3,600 | 53,300 |
| **B** delay | — | — | ~90,000 | — |

**A's memorisation is flat in K and its delay carries the entire K effect. B's
memorisation explodes.** Same axis, same statistic, opposite mechanism — and a
table of T_grok values shows one number for both, which is why this went unseen
until `t_memo` was recorded.

A's delay dependence is much stronger nearer the cliff. At α=0.25:

| K | 20 | 50 | 97 |
|---|---|---|---|
| t_memo | 3,100 | 3,100 | 4,100 |
| delay | 26,700 | 43,700 | ~92,950 |

Memorisation barely moves across a 5× change in K while the delay grows ~3.5×.
So on the anchor the delay is emphatically **not** flat in K — which is the
strongest available reason to treat B's apparent flatness (§13.7) as unproven
rather than established, and to treat 143,000 at K=50 as a floor.

**The mechanism this suggests** is the decay clock. A runs wd=0 and has none, so
shard size never fights memorisation. B's decoupled decay is applied per local
step and is independent of shard size, while the learning signal from a 1/K
shard is not — so smaller shards lose that race and memorisation is what gives
way. It predicts that D (AdamW) resembles B, and that any wd=0 setup resembles A.

`t1_setup_k_ladder` (39 runs) tests exactly that, reading t_memo(K) and delay(K)
per setup rather than T_grok(K).

#### What it costs the campaign

The K axis is open but it is not free, and budgets must now be set from
`t_memo(K) + delay` rather than from a multiple of the centralized number:

| setup B, wd=0.1, iid, E=5 | steps needed |
|---|---|
| K=20 | ~95,000 (measured, 3/3 at 200k) |
| K=30 | ~96,000 (1/3 at 100k) |
| K=50 | ~143,000 (predicted; 0/3 at 100k) |

K=97 extrapolates well past 200,000 steps, so the anchor's K ladder cannot be
reproduced on B at equal seed counts within a comparable budget. Either B's K
axis stops at 50, or the campaign accepts one expensive high-K cell per setup.
**No separate confirmation run is warranted.** Run ids are content hashes, so a
campaign cell specced at K=50 with headroom above 143,000 *is* the test — running
one first and the other later duplicates the work. The campaign will also yield
delay estimates across several K and several setups, which is a far better basis
for the K-dependence question than one more point here.

---

## 11. Gate A — each setup's own cliff

α=0.25 is the *anchor's* working point; nothing said the other setups shared it.
Centralized α ladders, 5 seeds, KM median:

| setup | 0.5 | 0.4 | 0.3 | 0.25 | 0.2 | 0.15 |
|---|---|---|---|---|---|---|
| **A** quad-MLP mod-97 | 7,600 | 8,800 | 13,100 | 25,300 | 0/5 | 0/5 |
| **B** transformer mod-113 | 800 | 1,400 | 6,600 | 12,400 | 2/5 | 0/5 |
| **D** quad-MLP S₅ | 7,200 | 12,600 | 21,300 | 36,200 | 0/5 | 0/5 |
| **C** transformer S₅ | 4/5 | 3/5 | 0/5 | 0/5 | 0/5 | 0/5 |

A reproduces v1's α=0.25 value (25,300 vs 25,133) — independent harness
agreement. B and D cliff at ~0.20 with clean monotone divergence; working point
α=0.30.

**Setup C's failure was censoring, not capacity.** The ladder gave C 40,000
epochs. The capacity sweep gave it 100,000 and got **12/12 even at the baseline
width 128**, KM median 51,200 — above the budget the ladder allowed. Width 256
then halves it to 21,600, so capacity is a real accelerant, but C should not have
been written off. At α=0.6 → 20,200 and α=0.7 → 4,800, both 5/5. C stays, at
α≥0.5 with ≥100k epochs, with the caveat that its working point sits far above
the others' 0.30 so matched-α comparison is unavailable.

That is the fourth time a fixed budget has manufactured a boundary in this
project — after v1's headline claim, the E=1 probe cells, and the first FL probe.

**Setup B's seed variance is intrinsic and bimodal**: 4,400 / 6,100 / 6,600 /
19,500 / 20,400 at α=0.30, against A's 12,600–13,400. Two clusters ~3× apart, so
more seeds narrow the interval by ~√2 and do not make it unimodal. B cannot
resolve effects below ~2–3× at 5 seeds.

**MNIST: delay and shardability are in opposition.** Every config with a large
memorise→generalise delay needs a large batch, which makes shards degenerate;
every shardable config has a small delay or none. Best compromise
(n_train=2000, batch=100): delay 500.

---

## 12. The K≈30 collapse on AdamW setups

> **SUPERSEDED by §13.7.** Everything in this section was measured at a 10,000-step
> budget. At K=30 memorisation alone takes 12,900 steps, so the "42.8% peak train"
> below is where that run had got to when the clock stopped, not a capability
> limit — the same setup memorises and groks at 13,200 given 100,000 steps. The
> section is kept because it is the reasoning the diagnosis was built on, and
> because the ruled-out mechanisms below still stand.

Setup B, one seed per cell, default hyperparameters — **peak** train accuracy:

| K | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|
| peak train | 100.0 | 98.2 | 42.8 | 5.9 | 5.0 |
| final train | 99.9 | 98.2 | 41.3 | 5.3 | 3.6 |

A smooth degradation, not a cliff — and a **training** failure, not a grokking
one. `peak ≈ final` throughout, so nothing memorises and then collapses; it never
learns. No budget fixes a model at 5% train accuracy. Setup A under plain GD at
wd=0 groks 5/5 at K=50 and is the only setup with no decay clock at all.

Ruled out: weight-norm collapse (norms flat or growing), client drift (the
failing D K=50 operand drifts *less* than the working D K=50 iid), and local step
size — lower lr is **worse**, 4.6% train at lr=1e-4.

The one knob that moves it is weight decay. At K=50 on setup B:

| lr | wd | lr·λ | train acc |
|---|---|---|---|
| 1e-3 | 1.0 | 1e-3 | ~3.6% ← inherited default |
| 1e-4 | 1.0 | 1e-4 | 4.6% |
| 3e-4 | 1.0 | 3e-4 | 3.4% |
| **1e-3** | **0.1** | **1e-4** | **70.2%** |

One seed per cell, so this is a lead rather than a result;
`manifests/p1_k_collapse_wd.jsonl` settles it at 3 seeds across K ∈ {20,30,50}.

**Do not lean on the numerical coincidence with §6.2.** lr·λ=1e-3 is also where
the anchor's corrected WD sweep found decay outrunning learning, but that sweep
was **GD**, where decay is coupled — and §6.4 is precisely the finding that the
optimiser is what discriminates. The federated weight-decay evidence stands on
its own; the matching number does not transfer.

### 12.1 Why this was invisible until now

`t_grok` is one number and grokking has two timescales, so "memorised but never
generalised" and "never trained" both recorded `inf`. `t_memo`, `delay` and
`peak_train_acc` are now recorded per run (and backfilled across all 720 banked
rows by `scripts/backfill_runs.py`), which is what makes the table above readable
as a training failure rather than a grokking one.

### 12.2 Related: D's operand partition fails, which inverts the headline

At K=10 on setup D, 50,000 steps: 99.8% train, **63% test**, 3/3 censored against
an 85 bar. Memorised, not generalising — while D at K=50 **iid** groks 3/3 at
~11,000. That is the opposite of §5.4, where the operand partition *rescues* K=97
on the anchor. Whether it is slow or stuck is unresolved and needs 5× budget
before "structure hurts D" is claimable.

Note also that on S₅ the operand partition shards by first-operand *element*,
which is not algebraically coherent the way a mod-p operand shard is. The
**coset** partition is S₅'s coherent one, so coset-vs-operand-vs-iid on D is what
actually separates "coherent shard" from "merely structured shard".

---

## 1. What groks, and when

Four setups, all confirmed. Delay = gap between memorisation and generalisation.

| setup | task | groks | memorise | generalise | delay |
|---|---|---|---|---|---|
| Quadratic MLP (Gromov) | mod-97 addition | yes | — | 7,600 | — |
| Nanda transformer | mod-113 addition | yes | 200 | 6,200 | 6,000 |
| Transformer | S₅ composition | yes | 200 | ~14,000 | ~14,000 |
| 3-layer MLP (Omnigrok) | MNIST-1k | yes | ~600 | 3,800 | 3,300 |

The quadratic MLP on modular addition is the anchor; every federated result below
uses it.

### Grok thresholds are dataset-dependent

`compute_t_grok` originally hardcoded a 95% test-accuracy bar. MNIST-1k peaks near
93%, so all 15 MNIST runs recorded `t_grok = inf` — "never grokked" — while their
histories plainly showed grokking. A measurement artifact reported as a scientific
null.

The bar is now a dataset property (`fedgrok.data.registry.grok_threshold`):

| dataset | bar | chance | ceiling |
|---|---|---|---|
| modular | 95.0 | ~1% | 100% |
| s5 | 85.0 | ~0.8% | ~92% |
| mnist | 90.0 | 10% | ~93% |

Modular is unchanged, and every prior modular cell was verified bit-identical after
the change. **S₅ was heading for the same silent failure**: at a 95% bar all 72
planned S₅ replication runs would have recorded `inf`.

`grok_threshold` is now stored per result row and is a survival cell key — a T_grok
is only interpretable next to the bar it was measured at.

---

## 2. The data threshold dominates everything

`α` is the fraction of the p²=9,409 possible pairs used for training. Centralized,
no federation involved (v1, modular addition):

| α | train examples | grokked | T_grok |
|---|---|---|---|
| ≤ 0.20 | ≤1,882 | **0 / 24** | never |
| 0.25 | 2,352 | 21/22 | **25,300** |
| 0.30 | 2,823 | 30/31 | 12,600 |
| 0.35 | 3,293 | 18/18 | 9,900 |
| 0.50 | 4,704 | 32/32 | 7,600 |

There is a **cliff between α=0.20 and α=0.25**, and T_grok diverges as it is
approached: 7,600 → 12,600 → 25,300 → never. (α=0.40 reads 12,100 at n=3 — noise.)

This is the most important fact about the setup. Federated effects are small next to
it, and the interesting regime is α=0.25, just above the cliff, where the problem is
already marginal and small penalties get amplified.

---

## 3. What federation costs

Like-for-like at α=0.3, E=5 (v2 `t2_k_breakdown`, 5 seeds/cell):

| | T_grok | vs centralized |
|---|---|---|
| Centralized | 12,600 | — |
| K=5 | 13,200 | +5% |
| K=10 | 13,400 | +6% |
| K=20 | 13,700 | +9% |
| K=50 | 15,200 | +21% |

**At E=1 the cost is exactly zero** — FedAvg with n_k/n weighting and full-batch
local GD is an algebraic identity with centralized GD, proven in
`tests/test_fedavg_identity.py` and observed in the wild: the T1 probe's IID and
operand runs returned identical test accuracies seed-for-seed (33.5 / 34.1 / 17.2).

**Local steps cost more than clients.** T1 probe, K=10, α=0.3:

| E | iid | operand |
|---|---|---|
| 1 | censored (budget, not federation) | censored |
| 5 | 12,900 [12,900, 13,700] | 12,700 [12,600, 13,400] |
| 50 | 23,000 [22,000, 25,000] | 17,000 [16,000, 18,000] |

E=5 → E=50 is +78% on the compute-matched step axis. The E=1 cells are **not**
breakdown evidence: they received 10,000 steps against the ~12,900 needed.

---

## 4. `t2_k_breakdown` — the α=0.3 control (60/60, complete)

K ∈ {5,10,20,50} × {iid, operand, dirichlet} × 5 seeds, α=0.3, E=5, 10k rounds.
KM median [95% CI]:

| K | iid | operand | dirichlet |
|---|---|---|---|
| 5 | 13200 [12700, 13500] | 13100 [12600, 13400] | 13400 [12700, 13600] |
| 10 | 13400 [13400, 13700] | 13200 [13200, 13500] | 13600 [13000, 13900] |
| 20 | 13700 [13200, 14100] | 13300 [12700, 13500] | 13900 [13200, 14200] |
| 50 | 15200 [14600, 16000] | **13700 [13000, 14000]** | 15400 [14700, 16100] |

**Every cell 5/5. Zero censoring.** The α=0.3 plane is uniformly safe.

**First sighting of the structure effect**: at K=50, operand is significantly faster
than iid (non-overlapping CIs). The gap scales with K — ~0 at K=5, ~400 steps at
K=20, ~1,500 at K=50. Dirichlet tracks iid exactly at every K, so this is
**structure, not heterogeneity**.

**Domain, measured later (§18.1):** this control ran at `dirichlet_alpha=0.5`.
The concentration knob is flat from 1000 down to ~1.0, costs ~15% at 0.1, and
below that stops testing heterogeneity at all (§18.2). The control was in the
right regime, so the claim stands as argued — but it is a statement about
moderate heterogeneity, not about all of it.

---

## 5. `t2_boundary` — the campaign that settled it (20/20, complete)

α=0.25, E=5, 20k rounds = 100k steps, 5 seeds. ~4.1 h/run at K=97.

| K | partition | grok | frac | KM median | 95% CI |
|---|---|---|---|---|---|
| 20 | iid | 5/5 | 1.00 | 29,800 | [27,300, 33,100] |
| 50 | iid | 5/5 | 1.00 | 46,800 | [40,700, 51,400] |
| 97 | iid | **2/5** | **0.40** | inf | [95,600, inf] |
| 97 | **operand** | **5/5** | 1.00 | **76,500** | [57,800, 77,500] |

Per seed at K=97 —
**iid:** 95,600 · 98,500 · and three censored at 100k ending 100% train with
**53% / 61% / 71% test**.
**operand:** 57,800 · 71,000 · 76,500 · 77,200 · 77,500, all reaching 100/100.

### 5.1 The harness sanity check passed

K=20 gives 29,800 [27,300, 33,100] against v1 exp2's 27,215 / 29,720 / 33,025. The
v2 rewrite reproduces exp2, so K=97 differences are attributable to federation
rather than to the refactor. K=50 grokking 5/5 confirms the 100k budget sufficed —
the other way this sweep could have returned nothing.

### 5.2 v1's breakdown claim does not survive

exp2 ran `t_max=50000`. Its α=0.25, E=5, iid ladder (3 seeds):

| K | 2 | 5 | 10 | 20 | 50 | 97 |
|---|---|---|---|---|---|---|
| T_grok | 23–27k | 24–28k | 25–29k | 27–33k | 41k, 44k, cens | all cens |

K=50 grokked at 41–44k against a 50k budget, so K=97's 0/3 was what continued
monotone delay looks like when it runs past the clock — the same trap as the E=1
probe cells, in the one place where censoring was supposed to *be* the signal.
**Two of five K=97 seeds do grok, at 95.6k and 98.5k.** Under a 50k budget all five
censor.

### 5.3 A partial transition is real but unresolved

2/5 at K=97 iid, with failures at 100% train and 53–71% test — memorised, not
generalised. **Caveat: both successes land within 5% of the 100k ceiling**, so the
budget is still marginal and those three may be mid-transition rather than stuck.

Separating "slower" from "broken" needs **more budget, not more seeds**. This
corrects the wave-2 plan, which called for deepening to 20 seeds — that would
sharpen a fraction whose denominator is itself budget-limited.

### 5.4 The headline: structure rescues K=97

operand 5/5 at 76,500 [57,800, 77,500] versus iid 2/5 at ~97,000 — non-overlapping,
and the difference between reliable grokking and marginal failure.

Combined with §4 (Dirichlet ≡ iid at every K), the claim is:

> **How you partition matters more than how far you fragment. Coherent shards are
> strictly better than random ones.**

**NARROWED by §17.2 (2026-08-11).** Tested on all six setups against exactly
matched iid baselines, this holds on the anchor and on C — where the coherent
split is 1.9× faster than random shards — and fails on B and D. The second
sentence is withdrawn as a general claim; the first survives, and `target`
being the worst partition in the study strengthens it.

Unlike the breakdown claim, this one is insensitive to where the budget was set.
The mechanism hypothesis — coherent shards let clients select a *shared* Fourier
basis that averaging reinforces, where iid clients pick conflicting sets that
partially cancel — is tested in **§16**, which finds it holds on the global
model and *precedes* grokking, and finds that the per-client channel cannot test
it on the operand arm at all.

(The snapshot inventory here was also understated: **211 banked runs** carry
per-client weights across all six setups — `aggregation`, `partitions`,
`setup_k_ladder`, `boundary` and `k_collapse_budget` — not the 20 boundary runs
this originally cited. `results/runs/*/checkpoints/`, 25 GB, gitignored.)

---

## 6. Weight decay

### 6.1 The v1 result was a values bug

exp5 ran `lr=50 × wd ∈ {0.01, 0.1, 1.0}`, i.e. `lr·λ ∈ {0.5, 5, 50}`. Both coupled
(SGD) and decoupled (AdamW) decay shrink weights by `(1 − lr·λ)` per step, so `lr·λ`
is the comparable quantity and `1/(lr·λ)` the decay timescale. Those settings halve
or invert every weight every step. All nine cells returning `T_grok = inf` was
arithmetically forced.

`core.utils.check_decay_stability` now rejects divergence (`lr·λ ≥ 2`) and the
subtler destructive case (`lr·λ > 0.1`).

### 6.2 Corrected sweep — `t0_wd_grid` (45 runs, α=0.5, p=97, 5 seeds)

| lr·λ | λ (GD, lr=50) | grokked | T_grok |
|---|---|---|---|
| 0 | 0 | 5/5 | 7,600 [7,500, 7,800] |
| 1e-5 | 2e-7 | 5/5 | 7,700 [7,500, 7,800] |
| 1e-4 | 2e-6 | 5/5 | 8,300 [8,200, 8,600] |
| 1e-3 | 2e-5 | **0/5** | censored |
| 1e-2 | 2e-4 | **0/5** | censored |

Grokking survives to `lr·λ = 1e-4` and dies at 1e-3, where decay outruns learning —
train accuracy never exceeds 8% and decays back to 1%, so memorisation never happens
and there is nothing to grok from. Within the usable band, decay is
**neutral-to-mildly-slowing**.

The AdamW arm at α=0.5 reaches 100/100 by epoch 200 at every λ (no delay at all), so
it cannot inform the question at this α.

### 6.3 MNIST — `t0_mnist_wd_band` (15 runs, 3 seeds)

| lr·λ | memorise | generalise | delay | grokked | peak test |
|---|---|---|---|---|---|
| 1e-5 | 600 | never | — | 0/3 | 89.2% |
| 3e-5 | 600 | 11,100 | ~10,500 | 3/3 | 91.1% |
| **1e-4** | 500 | **3,800** | **3,300** | 3/3 | **92.7%** |
| 3e-4 | 500 | 3,000 | 2,500 | 3/3 | 92.4% |
| 1e-3 | 800 | 3,200 | 2,400 | 3/3 | 91.5% |

**Weight decay accelerates grokking here, monotonically** — the published Omnigrok
result reproduced. Best band is `lr·λ = 1e-4`.

### 6.4 The discriminator is the optimizer, not the loss

The modular result (§6.2, neutral-to-slowing) and the MNIST result (§6.3,
accelerating) were initially reconciled by attributing acceleration to the
"AdamW/cross-entropy transformer regime". **That was wrong.** MNIST is MSE + AdamW
and accelerates; modular is MSE + GD and does not. The distinguishing variable is
**AdamW's decoupled decay vs GD's coupled decay**, not the loss function.

---

## 7. Task coverage

`t0_poly_pilot` (3 seeds, centralized, α=0.5, p=97) gated the operation set:

| task | grokked | T_grok | verdict |
|---|---|---|---|
| x² + y² | 3/3 | 7,000 [6,900, 7,000] | kept |
| x² + xy + y² | 0/3 | censored | **excluded** |

Consistent with Doshi et al. (2406.03495) separating learnable from non-learnable
modular polynomials for this architecture. Multiplication is excluded by design: on
nonzero residues it is cyclic Z_{p−1} under discrete log, testing the same Fourier
structure over a composite-order group — a confound, not a control.

---

## 8. Cost model

Fitted on 78 completed federated runs, and validated out-of-sample at K=97
(predicted 165 s for 200 rounds, measured 174 s):

```
minutes = (9.8 + 1.291·K + 0.418·E) × rounds / 10000
```

The K and E effects are cleanly additive. Reference points at 10k rounds, E=5:
K=5 → 22 min, K=20 → 33 min, K=50 → 75 min, K=97 → 2.25 h.

**Wall-clock is ~99% orchestration, not compute.** 50,000 centralized gradient steps
take **48 s**; the identical arithmetic federated across 5 clients takes **22 min** —
27× — all of it Flower/Ray shipping weights between client processes each round.
Cost therefore scales with *client count*, not with training length.

**Hardware provenance.** This model and every `wall_s` in this file were measured on
the 8× L4 box the project ran on until 2026-08-17. It has since moved to a single
RTX 3080 Laptop (8 GB, 16 cores). Because the cost is orchestration, **the figures
transfer**: re-measured 2026-08-19 on setup E at K=20, 235 ms/round here with four
runs sharing the card against 244 ms/round banked single-slot on the L4. What no
longer transfers is the *pool* — eight concurrent cards became one, so a sweep's
wall-clock is its slot-hours divided by ~4, not by ~8-16.

Practical consequence: order manifests **longest-job-first**. On `t2_boundary` that
cut wall-clock from ~7.2 h to ~5.9 h for identical work.

---

## 9. Run inventory

| | runs | grokked | censored |
|---|---|---|---|
| **v1** (`results/data/runs.csv`, log-recovered) | 870 | 554 | 316 |
| **v2** (`results/data/runs_v2.csv`) | 1,529 | 1,016 | 513 |

v1 by experiment: exp2 333 · exp5 153 · exp3a 100 · exp7 74 · exp4a 72 · exp4b 72 ·
exp4 36 · exp3b 30.

v2 by campaign — regenerate with
`summarize_runs.py results/data/runs_v2.csv --group group`:

| campaign | runs | status |
|---|---|---|
| `aggregation` | 192 | done — §15.1 |
| `aggregation_alpha2` | 30 | exp2's second α — 60 specs still outstanding |
| `b_wd_zero_alpha` | 15 | **done** — B's α ladder at wd=0; groks from α=0.60 (`RUNS_TODO` 2) |
| `b_wd01_alpha_ladder` | 12 | done — B's α ladder at wd=0.1, the low-decay reference |
| `wd_zero` | 9 | done — §15.4, the wd=0 controls |
| `b_wd_zero_a04_long` | 3 | **done** — α=0.40 to 1,000,000 steps; the gradient dies (`RUNS_TODO` 3) |
| `b_k20_wd_control` | 3 | done — the K=20 cell behind the 2026-08-20 decay decision |
| `central_anchor` | 131 | done — Gate A ladders, §11 |
| `partitions` | 108 | done — §17.2 (exp3b, complete incl. coset) |
| `d_alpha_high` | 90 | done — tier X |
| `algorithms` | 90 | **done — §17.1 (exp5, at calibrated server LRs)** |
| `dirichlet_band` | 24 | **done — §18.1 (exp3a, the concentration ladder)** |
| `size_control` | 12 | **done — §18.2 (the control that dissolved the breakdown)** |
| `participation` | 9 | **done — §18.3 (exp4b)** |
| `d_internals` | 85 | done — §16.3 |
| `k_fixed_total` | 54 | done — §4 |
| `fl_probe` | 48 | done — under-budgeted by construction |
| `d_wd_ladder` | 45 | **done — §17.3** |
| `aprime_alpha` | 45 | done — §13.3 |
| `wd_grid` | 45 | done — §6.2 |
| `server_lr_cal` | 42 | done — §14.5 |
| `d_alpha_fine` | 40 | done — tier X |
| `setup_k_ladder` | 39 | done — §14.3 |
| `mnist_fl` | 36 | done |
| `capacity` | 33 | done — §14.2 |
| `mnist_working_point` | 33 | done — §11 |
| `cd_decay_band` | 30 | done — §13.2 |
| `c_capacity` | 24 | done — §11 |
| `d_alpha_cliff` | 20 | done — tier X |
| `boundary` | 20 | done — §5, and the checkpoints behind §16.2 |
| `d_lr_control` | 18 | **done — §17.3** |
| `probe` | 18 | done — §3 |
| `mnist_wd_band` | 15 | done — §6.3 |
| `b_decay_band` | 15 | done — §13.5 |
| `c_alpha_w256` | 12 | done — §14.1 |
| `k_collapse_wd` | 12 | done — §13.6 |
| `k_collapse_budget` | 11 | done — §13.7 |
| `d_gd_probe` | 9 | done — §13.1 |
| `wd_zero` | 9 | done — §15.4 |
| `probe_rerun` | 9 | done |
| `poly_pilot` | 6 | done — §7 |
| plus 6 smaller diagnosis groups | 37 | `c_alpha` · `grok_confirm_fl` · `adam_restart` · `k50_hparam` · `k50_ladder` · `e1_identity` |

**v2 compute to date: ~965 machine-hours.** Checkpoints on disk: 25 GB across
383 run directories (gitignored) — **211 runs carry per-client weights**, spanning
all six setups, which is what §16 reads.

---

## 10. Caveats and known limitations

- **K=97 IID is unresolved** (§5.3). The budget is marginal; the 2/5 fraction has a
  budget-limited denominator.
- **RESOLVED — federated results now span all six setups** (exp2 §15.1, exp3b
  §17.2). The generality question this caveat raised has an answer, and it is
  partly negative: the structure effect replicates on C and fails on B and D.
- **D's coset cells are censored, not stuck** (§17.2). Two of three seeds were
  still gaining 7–12 accuracy points per 100,000 steps at the ceiling. "≥10×
  slower than random shards" is supported; "never groks" is not.
- **A′ has no coherent partition ordering** (§17.2) and is excluded from the
  partition reading: `target` is its fastest split at both K, which no account
  predicts. It sits at its cliff and carries §15.3's optimiser-restart confound.
- **exp5's server LRs were calibrated at E=5** (§14.5) and exp5's hard cells run
  E=25 and E=50. The calibration is assumed to transfer across the E axis and
  that has not been checked.
- **The drift result is observational** (§17.4). The within-cell test that could
  separate cause from correlate is a null *and* underpowered (1.69× within-cell
  spread against 20,182× between). The damped-FedAvg control is not written.
- **Five α=1.0 rows in `d_alpha_high` record `grokked=True` with `t_grok=0`.**
  They train on the whole grid, so there is no held-out set and `test_acc` is NaN
  throughout; NaN never compares below the bar, so the sustained-crossing scan
  returns the first step. `t_first_cross` reports `inf` for the same runs, which
  is correct. Exclude α=1.0 from any aggregate over that group.

  **Both halves are now fixed in code, so this cannot recur, but the five banked
  rows are unchanged and still need excluding.** `split_indices` rejects any α
  that empties either side of the split; `compute_t_grok` treats NaN as below the
  bar; `x_d_alpha_high` no longer emits the α=1.0 cells. The five result JSONs are
  left on disk as the record of what ran — deleting them or re-deriving them as
  censored are both open options, and neither is obviously right: they are not
  censored measurements, they are absent ones.
- **exp4b partial-participation T_grok values in `runs.csv` sit on the old inflated
  step axis** (~2.5× too high on `total_steps`); the Phase 0.6 fix changed that axis
  under `fraction_train < 1.0`. They will be corrected when exp4b is re-run.
- **v1 history filenames are not unique run keys.** Algorithm suffixes landed
  mid-campaign, so 61 paths were written by more than one run and any surviving JSON
  holds only the last. `harvest_logs.py` therefore takes algorithm identity from the
  log filename, and `runs.csv` — not the JSONs — is the v1 record.
- **Grid holes:** 6 empty exp3a α=0.20 logs; `exp3a_a0.30_dir10.0` crashed at round
  6,491 on a Ray actor failure. Tracked in `results/data/runs_skipped.csv`.
- **α=0.40 centralized reads 12,100** against 9,900 at α=0.35 and 7,600 at α=0.50 —
  n=3, almost certainly noise, but it is the one non-monotonicity in §2.
- **The harness is not run-to-run deterministic** (§23). Two runs of the same
  config and seed agree exactly on D and differ by up to 12.7 points of peak train
  accuracy on C — as much as changing the seed. Flower's `aggregate()` sums client
  weights in arrival order, nothing sorts them, and no deterministic-algorithm flag
  is set. Every banked run carries this; the sweep of 2–4 Sep is the first with
  duplicate cells to show it. Consequences: C's numbers are withheld; the
  statistics language should say *runs*, not *seeds*; the fix is in RUNS_TODO.
- **Setup E's participation degradation (§21) is measured at K=20**, which is
  MNIST's degenerate rung — one full-batch step per local epoch. The degeneracy
  is constant across f, so the within-setup effect stands, but it has not been
  shown to survive a non-degenerate K.
- **Two C cells at dir_α=0.1 report `t_memo=inf` with a finite `t_first_cross`**
  (§20): test crossed 95% while train peaked at 97–99% and never held the
  memorisation bar. Real (grokking without full memorisation), but `delay` is
  undefined there and the C delay curve has a hole at its most interesting rung.
- **Out of scope, stated deliberately:** DP-FedAvg, communication compression,
  Byzantine-robust aggregation, personalization, async/stragglers, FedBN,
  MOON/FedDecorr, and the LEAF/FLamby/FedScale benchmarks.
