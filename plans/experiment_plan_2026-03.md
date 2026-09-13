# Experimental Plan: Grokking in Federated Learning

> **Goal**: Produce a NeurIPS-quality empirical study characterising exactly when and why
> grokking survives — or fails — under federated learning.

---

## 1  Motivation

Prior work (Gromov 2023) shows that two-layer MLPs with quadratic activations exhibit
*grokking* on modular arithmetic: the network memorises the training set quickly, then
generalises long after memorisation, driven by the emergence of periodic Fourier features
in the first-layer weights.

Our preliminary experiments show that **grokking is remarkably robust to federated
training at α = 0.5** (50 % of all p² pairs used for training). Even K = 97 clients
(48 samples each), partial participation (f = 0.2), extreme local epochs (E = 200),
and non-IID target partitions all still grok to ≥ 94 % test accuracy.

This means the interesting regime is **not** the comfortable interior (α = 0.5) but the
**phase boundary** — the critical training fraction α_crit below which grokking fails.
The central question becomes:

> **Does federated learning shift the grokking phase boundary, and if so, why?**

---

## 2  Setup (fixed across all experiments unless stated otherwise)

| Symbol | Parameter | Value | Rationale |
|--------|-----------|-------|-----------|
| p | prime modulus | 97 | Gromov 2023 standard |
| — | task | (a + b) mod p | default; generality tested in appendix |
| — | activation | quadratic (x²) | theory-supported default |
| N | hidden width | 256 (validated in Exp 0) | well above Gromov Fig 4b plateau; 74 496 params |
| — | optimizer | SGD (no momentum) | no optimizer state → clean FL comparison |
| lr | learning rate | 50.0 | mean-field parameterisation default |
| λ | weight decay | 0.0 | isolates implicit regularisation |
| — | loss | MSE on one-hot targets | Gromov 2023 standard |
| — | training | full-batch | each "epoch" = 1 gradient step; no minibatch stochasticity |
| — | seeds | {42, 123, 456} | 3 seeds; report mean ± std |

### Derived quantities

| Quantity | Formula | Example (α = 0.5) |
|----------|---------|---------------------|
| total pairs | p² | 9 409 |
| training set | n_train = ⌊α · p²⌋ | 4 704 |
| test set | n_test = p² − n_train | 4 705 |
| per-client (IID) | n_client ≈ n_train / K | 941 (K = 5) |
| parameters | 3Np | 74 496 |
| param : data ratio | 3Np / n_train | 15.8 (α = 0.5) |

### Notation

| Symbol | Meaning |
|--------|---------|
| α | training fraction (fraction of p² used for training) |
| K | number of federated clients |
| E | local epochs per round (each = 1 full-batch gradient step on client data) |
| R | number of communication rounds |
| S = R × E | total gradient steps per client |
| f | participation fraction (fraction of K clients sampled per round) |
| μ | FedProx proximal coefficient (μ = 0 ⇒ FedAvg) |
| α_dir | Dirichlet concentration for non-IID partition |
| α_crit | critical training fraction below which grokking fails |

---

## 3  Formal Definitions

### 3.1  Grokking step

Let {t_i, a_i} be the sequence of (step, test accuracy) pairs logged during training.

**Definition.** The *grokking step* T_grok is the smallest logged step t_j such that
a_i ≥ 95 % for all i ≥ j (i.e., test accuracy reaches 95 % and never drops below it
for the remainder of training). If no such t_j exists within S_max steps, we record
T_grok = ∞ (labelled "no grokking").

**Rationale.** The 95 % threshold is conservative (grokking typically reaches > 99 %).
The "never drops below" condition avoids false positives from transient spikes. Using
all subsequent log points (rather than a fixed window) is stricter and eliminates an
arbitrary window-length parameter.

### 3.1b  Secondary metric: onset step T_50

**Definition.** T_50 is the smallest logged step t_j such that a_j ≥ 50 %.

**Rationale.** Near the phase boundary, grokking may be in-progress but incomplete
at S_max. T_50 captures the *onset* of generalization (departure from memorisation
at ~1/p ≈ 1 %) and provides a continuous signal even when T_grok = ∞. Report both
T_grok and T_50 for all experiments.

### 3.2  Total-step budget

#### Key constraint: centralized is computation-bound, FL is communication-bound

Centralized runs are cheap (~2 min for 50k steps, ~4 min for 100k steps) because
there is no per-round serialisation overhead. FL runs are expensive because
Flower/Ray simulation adds ~0.3–0.5 s per round regardless of model/data size.
FL runtime ≈ R × 0.4 s, where R = S_max / E.

#### Definitions

Let **T_base** = T_grok(α = 0.5, N = 256, centralized) — the baseline grokking step,
determined in Exp 0. Preliminary estimate: T_base ≈ 8 000 (based on N = 128 result).

Let **T_max** = max T_grok across all α that grok in Exp 1 — the slowest centralized
grokking observed. This is the right anchor for FL budgets because FL-IID matches
centralized step-for-step at α = 0.5; near the boundary, FL may be slower but should
not exceed ~2× centralized.

#### Per-experiment step budgets

| Experiment | S_max | Rationale | R (at E = 5) | Est. runtime/run |
|-----------|-------|-----------|-------------|-----------------|
| **Exp 0** (centralized) | 50 000 | Confirm grokking at N = 256; find T_base | — | ~2 min |
| **Exp 1** (centralized) | 100 000 | Generous headroom for slow grokking near α_crit. Gromov Fig 4a shows T_grok diverges as α → α_c; at α ≈ 0.2 grokking can take > 30 000 steps. At ~4 min/run, no reason to be stingy. | — | ~4 min |
| **Exp 2** cent. (a, b) | 100 000 | Same as Exp 1; centralized-reduced runs are even faster (tiny datasets) | — | ~4 min |
| **Exp 2** FL (c) | **adaptive**: min(50 000, 1.5 × T_max) | FL-IID ≈ centralized step-for-step; 1.5× headroom covers mild FL slowdown. If T_max = 30 000 → S_max = 45 000. If T_max = 8 000 → S_max = 12 000. | 9 000 (if 45k) | ~50 min (if 45k) |
| **Exp 3** (FL) | same as Exp 2 FL | Non-IID may delay grokking but early abort catches hopeless runs | same | same |
| **Exp 4a** (FL, fixed S) | same as Exp 2 FL | Fixed total steps; R varies with E. Higher E → fewer rounds → faster. | varies | E=5: ~50 min, E=50: ~5 min |
| **Exp 4b** (FL, fixed R) | R × E (variable) | Fixed R = 2 000; total steps increase with E. E = 25 → S = 50 000. | 2 000 (fixed) | ~12 min |
| **Exp 5** (FL) | **2 × T_max** | Rescue experiments need extra headroom — algorithms may grok later than FedAvg but still succeed. Cap at 80 000. | T_max × 2 / 5 | ~70 min (if T_max = 30k) |

#### Decision rule (set after Exp 1)

```
T_base = T_grok(α = 0.5, N = 256)                    # from Exp 0
T_max  = max{ T_grok(α) : α groks in Exp 1 }         # from Exp 1
S_FL   = min(50_000, ceil(1.5 × T_max / 1000) × 1000)  # round up to nearest 1k
S_rescue = min(80_000, 2 × T_max)
```

Example scenarios:

| Scenario | T_base | T_max | S_FL | S_rescue | R (E=5) | FL min/run |
|----------|--------|-------|------|----------|---------|-----------|
| Low α_crit (α_c ≈ 0.1) | 8 000 | 40 000 | 50 000 | 80 000 | 10 000 | ~55 min |
| Mid α_crit (α_c ≈ 0.2) | 8 000 | 25 000 | 38 000 | 50 000 | 7 600 | ~42 min |
| High α_crit (α_c ≈ 0.3) | 8 000 | 12 000 | 18 000 | 24 000 | 3 600 | ~20 min |

The "high α_crit" scenario is the lucky case — all FL runs are fast.

#### Early abort rules

1. **Memorisation failure**: If train_acc < 50 % by step min(2 × T_base, 15 000),
   abort. The model cannot even memorise — something is fundamentally broken
   (insufficient capacity, wrong LR, too little data).

2. **Generalisation hopeless**: If train_acc = 100 % and test_acc < 5 % (chance level
   ≈ 1/p ≈ 1 %) by step T_max, abort. The model memorised but shows zero
   generalisation signal — it will not grok within the remaining budget.

3. **T_50 decision**: If a run reaches S_max with T_50 achieved but T_grok = ∞
   (test_acc between 50–94 % and still rising), flag for manual review. These
   boundary cases are scientifically interesting and may warrant extension.

#### Logging frequency

| Context | Log interval | Log points at S_max = 50 000 |
|---------|-------------|------------------------------|
| Centralized | every 100 steps | 1 000 |
| FL | every round | R (variable; 2 000–10 000) |

#### E = 1 constraint

E = 1 means R = S_max rounds, which at S_FL = 45 000 would be 45 000 rounds ×
0.4 s = 5 hours per run — impractical. **E = 1 is tested only in Exp 4b** where
R is capped at 2 000 (S = 2 000 steps, ~12 min). The finding "E = 1 with
S = 2 000 doesn't grok" is informative: it establishes the minimum communication
budget, not the minimum compute.

### 3.3  Centralized-reduced baseline

To test the effect of federated **aggregation** (vs simply having less data), we
define the *centralized-reduced* baseline:

> Train a single centralized model on n_train / K data points for S_max steps.

Implementation: set α_eff = α / K in the centralized config with the same seed.
Because the dataset is constructed from a seeded permutation of all p² pairs, and
the training set is the first ⌊α · p²⌋ elements, setting α_eff = α / K yields the
first ⌊α_eff · p²⌋ elements — a strict subset of the original training set.

This is statistically equivalent to one FL client's data (same size, random subset of
the training pairs), enabling a direct comparison:

| Condition | Data | Aggregation? |
|-----------|------|-------------|
| Centralized-full | n_train | — |
| Centralized-reduced | n_train / K | — |
| FL (IID) | n_train / K per client | Yes (K clients averaged) |

---

## 4  Experiments

### Experiment 0 — Width Validation (prerequisite)

**Problem.** The Fourier solution for (a + b) mod p requires (p−1)/2 = 48 frequency
modes, but redundant neurons improve accuracy via destructive interference of
unwanted terms (Gromov 2023, §3.1, Eq. 15). Gromov's Figure 4b shows that for GD
at α = 0.5 and p = 97, test accuracy reaches only ~80 % at N = 100 and requires
**N ≈ 120–140 for near-100 % accuracy**. Our default N = 128 sits right at this
edge. We must verify that N = 128 is safely above the critical width *across the α
range we will test*, so that grokking failures in later experiments are attributable
to FL — not to insufficient model capacity.

**Note.** The prior width sweep (N ∈ {20, 40, 60, 80, 100, 120, 140}) was invalid:
most runs completed only 100 epochs (vs ~8 000 needed for grokking). This must be
rerun properly.

**Design.**

| Parameter | Values |
|-----------|--------|
| N | {100, 128, 200, 256} |
| α | {0.1, 0.3, 0.5} |
| seeds | {42} (single seed; extend to 3 if results are noisy) |
| S_max | 50 000 steps |

**Runs.** 4N × 3α = **12 runs** (fast: centralized, ~2 min each = ~24 min total).

**Step budget rationale.** At α = 0.5, N = 128: T_grok ≈ 8 200 steps. With N = 256
(more capacity), grokking should be similar or faster. At α = 0.1, grokking may take
~3–5× longer. S_max = 50 000 gives ~6× headroom over the baseline — sufficient to
observe grokking or confidently declare failure. This experiment also establishes
T_base for calibrating all subsequent budgets.

**Expected outcome.** N = 256 is our primary default (well above Gromov Fig 4b
plateau). This experiment confirms it works at low α, and documents the critical
width N_crit(α) for the paper. The comparison across widths also produces a
publishable figure (extending Gromov's Fig 4b to lower α values).

**LR validation.** The mean-field parametrisation (scaling absorbed into init)
ensures function-level dynamics are O(1) in N, so lr = 50 should transfer from
N = 128 to N = 256 without adjustment. If N = 256 shows significantly delayed
grokking compared to N = 200 at the same α, check whether lr scaling is needed
(try lr = 50 × 256/128 = 100). This is unlikely but cheap to test.

**Why this matters.** Without this step, any grokking failure at low α could be
confounded by width being too close to the theoretical minimum. This 30-minute
validation eliminates that confound.

---

### Experiment 1 — Centralized Grokking Phase Boundary

**Question.** At what training fraction α does centralized grokking fail?

**Hypothesis.** There exists a critical α_crit ∈ (0, 0.5) below which the model
memorises but never generalises. This boundary depends on the model capacity (N)
and the implicit regularisation from full-batch GD.

**Design.** Uses N* from Experiment 0.

| Parameter | Values |
|-----------|--------|
| α | {0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5} |
| seeds | {42, 123, 456} |
| S_max | 100 000 steps |

**Step budget rationale.** Centralized runs cost ~4 min at 100k steps — cheap enough
to be generous. Gromov Fig 4a shows T_grok diverges as α → α_c. At α ≈ 0.2,
GD grokking can exceed 30 000 steps. At α ≈ 0.1, possibly 50 000+. S_max = 100 000
gives 2× headroom even for the slowest grokkable settings. Any α that hasn't reached
T_50 by 100k steps is confidently in the "no grok" regime.

**After this experiment, compute:** T_max = max T_grok across all α that grok.
This calibrates all FL budgets (see §3.2).

**Controls.** α = 0.5 is the known-good baseline. Each α value gives a different
n_train, test set, and param-to-data ratio (shown for N = 256):

| α | n_train | n_test | params/data |
|---|---------|--------|-------------|
| 0.03 | 282 | 9 127 | 264.2 |
| 0.05 | 470 | 8 939 | 158.5 |
| 0.075 | 705 | 8 704 | 105.7 |
| 0.1 | 940 | 8 469 | 79.3 |
| 0.15 | 1 411 | 7 998 | 52.8 |
| 0.2 | 1 881 | 7 528 | 39.6 |
| 0.3 | 2 822 | 6 587 | 26.4 |
| 0.5 | 4 704 | 4 705 | 15.8 |

**Runs.** 8 × 3 = **24 runs** (~4 min each = **~96 min total**).

**Outputs.**
- α_crit (the phase boundary)
- Plot: grokking step vs α (with error bars). Sharp transition expected.
- For α near the boundary: full training dynamics (loss, acc, IPR, weight norms).

**This experiment is a prerequisite for all subsequent FL experiments.** It tells us
where to focus: the interesting FL experiments operate near α_crit.

---

### Experiment 2 — Aggregation Effect & FL Phase Boundary

**Question.** Does FL aggregation compensate for data fragmentation? Does FL shift
α_crit?

**Hypothesis.** Three possible outcomes:
1. FL α_crit ≈ centralized α_crit → aggregation perfectly compensates for
   fragmentation; FL is benign.
2. FL α_crit > centralized α_crit → FL needs more total data to grok;
   fragmentation hurts even with aggregation.
3. FL α_crit < centralized α_crit → aggregation acts as implicit regularisation;
   FL *helps* grokking (the surprising result).

**Design.** For each (α, K) pair, run three conditions:

| Condition | What | Data per model | Aggregation |
|-----------|------|----------------|-------------|
| **(a) Centralized-full** | Standard centralized | n_train = ⌊α · p²⌋ | — |
| **(b) Centralized-reduced** | One client's data size | n_reduced = ⌊(α/K) · p²⌋ | — |
| **(c) FL (IID)** | K clients, IID, f = 1.0 | n_train / K per client | FedAvg |

Sweep:

| Parameter | Values |
|-----------|--------|
| α | {α_crit − δ, α_crit, α_crit + δ, α_crit + 2δ, 0.5} where δ is chosen after Exp 1 to span the transition; fallback grid: {0.05, 0.1, 0.15, 0.2, 0.3, 0.5} |
| K | {2, 5, 10, 20, 50, 97} |
| E | 5 (default) |
| f | 1.0 (full participation) |
| seeds | {42, 123, 456} |

**Step budgets by condition:**
- **(a) Centralized-full:** S_max = 100 000 (~4 min/run). Reuse Exp 1 runs where α overlaps.
- **(b) Centralized-reduced:** S_max = 100 000 (~4 min/run). Very small datasets → even faster.
- **(c) FL:** S_max = S_FL (adaptive, from §3.2). R = S_FL / 5.
  If T_max = 30 000 → S_FL = 45 000, R = 9 000 → ~50 min/run.
  If T_max = 12 000 → S_FL = 18 000, R = 3 600 → ~20 min/run.

**Runs.** 6α × 6K × 3 conditions × 3 seeds = **324 runs.**
(Condition (a) doesn't depend on K, so reuse: 6α × 1 × 3 seeds = 18 unique.
Condition (b): 6α × 6K × 3 seeds = 108. Condition (c): same 108. Total unique: **234 runs.**)

**Key per-client data sizes (α = 0.1, n_train = 940, N = 256):**

| K | n_client | params/data (client) | Notes |
|---|----------|----------------------|-------|
| 2 | 470 | 158.5 | near-centralized |
| 5 | 188 | 396.3 | |
| 10 | 94 | 792.5 | |
| 20 | 47 | 1 585.0 | |
| 50 | 18 | 4 138.7 | fewer samples than output classes |
| 97 | 9 | 8 277.3 | extreme: ~9 samples per client |

K = 2 anchors the "near-centralized" end (each client has half the data). K = 97
is the natural ceiling for p = 97 — one client per residue class, with each client
seeing only ~n_train/97 ≈ 9 training pairs at α = 0.1. At α = 0.5, K = 97 still
gives ~48 samples per client (our breaking-point results confirmed grokking here).

**Note on extreme regimes.** The centralized-reduced condition at high K and low α
produces very small training sets. For example, α = 0.1 / K = 97 gives
n_reduced = ⌊0.001 × 9409⌋ = 9 samples — far fewer than the 97 output classes.
These settings will almost certainly fail to grok in centralized-reduced, which is
informative: if FL *does* grok with the same per-client data, it proves aggregation
is essential.

**Outputs.**
- **Figure 2 (main result):** For each α, plot grokking step vs K with three
  curves: centralized-full (flat), centralized-reduced (increasing),
  FL (the interesting one). Multiple panels for different α values.
- Alternatively: 2D heatmap of grokking step over (α, K) grid.
- The gap between centralized-reduced and FL curves quantifies the
  *aggregation benefit*. The gap between centralized-full and FL quantifies
  the *fragmentation cost*.

**Decision point.** If FL α_crit = centralized α_crit at all K values tested,
the conclusion is "FL does not affect the grokking phase boundary." The rest of
the experiments then focus on understanding *why* (mechanistic robustness) rather
than *where it breaks*.

---

### Experiment 3 — Heterogeneity at the Phase Boundary

**Question.** Does data heterogeneity shift the grokking phase boundary, and does
the *type* of heterogeneity matter?

**Hypothesis.** Operand-based partition disrupts grokking more than Dirichlet
(quantity skew) at matched imbalance levels, because it fragments the Fourier input
structure that the model must learn. Target-based partition disrupts the output space
but preserves input structure and may be less harmful.

**Design.** Use K values from Exp 2: a primary K (the most informative value,
default K = 10) and a secondary K (higher, default K = 50) to validate that
heterogeneity effects don't vanish or qualitatively change at different scales.
Use α values near α_crit from Exp 1.

**Sub-experiment 3a: Dirichlet sweep (continuous heterogeneity knob).**

Primary sweep at K_primary:

| Parameter | Values |
|-----------|--------|
| α | {α_crit − δ, α_crit, α_crit + δ, 0.3, 0.5} (~5 values) |
| α_dir | {0.01, 0.1, 0.5, 1.0, 10.0, 1000.0} (6 values, 1000 ≈ IID) |
| K | K_primary (from Exp 2, default 10) |
| E | 5, f = 1.0 |
| S_max | S_FL (adaptive, from §3.2) |
| seeds | {42, 123, 456} |

**Runs (primary).** 5 × 6 × 3 = **90 runs.**

K-validation at K_secondary (reduced sweep — only test the extremes to confirm
the phase boundary doesn't qualitatively shift):

| Parameter | Values |
|-----------|--------|
| α | {α_crit, α_crit + δ, 0.5} (3 values — boundary + comfortable) |
| α_dir | {0.1, 1.0, 1000.0} (3 values — hard, moderate, IID) |
| K | K_secondary (from Exp 2, default 50) |
| E | 5, f = 1.0 |
| S_max | S_FL |
| seeds | {42, 123, 456} |

**Runs (K-validation).** 3 × 3 × 3 = **27 runs.**

**Step budget rationale.** Same as Exp 2 FL. Non-IID may delay grokking vs IID,
but since S_FL already includes 1.5× headroom over centralized T_max, this should
be sufficient. Extreme heterogeneity (α_dir = 0.01) at low α will likely trigger
early abort (no memorisation), saving compute.

**Why K-validation matters.** At K = 50, each client has n_train/50 data points.
With non-IID on top of this data scarcity, the per-client distribution may be so
degenerate that grokking fails even at α = 0.5. Alternatively, more clients means
more diverse "views" of the data, which could make aggregation MORE effective. The
K-validation sub-sweep tests which effect dominates without requiring a full K ×
α_dir × α grid.

**Sub-experiment 3b: Structured partition (operand vs target vs IID).**

| Parameter | Values |
|-----------|--------|
| α | same 5 values as 3a primary |
| partition | {iid, operand, target} |
| K | K_primary |
| E | 5, f = 1.0 |
| seeds | {42, 123, 456} |

**Runs.** 5 × 3 × 3 = 45. IID runs overlap with 3a (α_dir = 1000), so
5 × 3 = 15 are reused. **30 new runs.**

**Combined Exp 3 unique runs:** 90 + 27 + 30 = **~147 runs.**
Estimated runtime: ~147 × ~50 min × 0.7 (early abort savings) ≈ **~86 hr sequential.**

**Outputs.**
- **Figure 3a:** Phase diagram heatmap — x-axis: Dirichlet α_dir (log scale),
  y-axis: training fraction α. Colour: grokking step (∞ = white/hatched).
  The phase boundary is a curve in this space.
- **Figure 3b:** Structured partition comparison — for each α, grouped bars
  showing grokking step for IID / operand / target. If operand partition is
  worse than Dirichlet-matched heterogeneity, this supports the Fourier
  fragmentation hypothesis.
- **Figure 3c (appendix):** K-validation — overlay K = 10 and K = 50 phase
  boundaries. If they overlap, heterogeneity effects are K-independent.
  If K = 50 boundary shifts right (needs more data), K amplifies heterogeneity.

---

### Experiment 4 — Optimization Fragmentation

**Question.** How do client drift and partial participation interact with data
heterogeneity to affect grokking? Is grokking compute-limited or
communication-limited?

**Why the previous design was insufficient.** Varying E and f under a single
heterogeneity setting only tests "communication efficiency" — which is obvious
(more communication helps). The core prediction of optimization fragmentation is:

> Under IID, client drifts are *correlated* (all clients optimise the same loss
> landscape), so aggregation of drifted models preserves global structure.
> Under non-IID, drifts are *conflicting* (clients move toward incompatible local
> optima), so aggregation after many local steps *destroys* the Fourier features
> that grokking requires.

**The E × heterogeneity interaction is therefore the primary test.** A secondary
effect is partial participation (f < 1): with f < 1, the global model is updated
from a random client subset, introducing aggregation noise. This noise compounds
with drift under non-IID.

**Design.** Three sub-experiments, each isolating one axis of optimisation
fragmentation. All use K from Exp 2 and α values from Exp 1.

---

**Sub-experiment 4a: Drift accumulation × heterogeneity (the key interaction).**

Fixed total steps S = S_FL. Full participation f = 1.0. Vary E and heterogeneity.

| E | R = S_FL / E | Per-round drift | Notes |
|---|-------------|----------------|-------|
| 5 | 9 000 | low | baseline |
| 10 | 4 500 | moderate | |
| 25 | 1 800 | high | |
| 50 | 900 | very high | |

(Example R values assume S_FL = 45 000.)

| Parameter | Values |
|-----------|--------|
| α | {α_crit + δ, 0.3, 0.5} (one near-boundary, two comfortable) |
| heterogeneity | {IID, non-IID*} |
| f | 1.0 (isolate drift from participation noise) |
| K | same as Exp 3 |
| S_max | S_FL |
| seeds | {42, 123, 456} |

*Non-IID setting chosen from Exp 3 — the mildest heterogeneity that showed
measurable grokking delay. Using the mildest (not hardest) ensures we can observe
the E interaction before grokking fails entirely.

**Runs.** 4E × 3α × 2het × 3 seeds = **72 runs.**
Average runtime ~23 min/run (weighted by E) → **~28 hr sequential.**

**Predictions.**
- IID: grokking step roughly constant across E (drift is correlated, aggregation
  of parallel trajectories is benign). Breaking-point results at α = 0.5 already
  suggest this.
- Non-IID: grokking step increases sharply with E, or grokking fails at high E.
  The critical E_max (maximum tolerable local epochs) is lower near α_crit.
- If confirmed, this demonstrates that **drift direction, not drift magnitude,
  determines whether grokking survives** — a mechanistic insight.

---

**Sub-experiment 4b: Partial participation × heterogeneity.**

Fixed total steps S = S_FL. Fixed E = 5. Vary f and heterogeneity.

| Parameter | Values |
|-----------|--------|
| α | same 3 values |
| f | {0.2, 0.4, 0.6, 1.0} |
| heterogeneity | {IID, non-IID*} (same as 4a) |
| E | 5 |
| K | same as Exp 3 |
| S_max | S_FL |
| seeds | {42, 123, 456} |

**Runs.** 4f × 3α × 2het × 3 seeds = **72 runs.**
R = S_FL / 5 for all → ~50 min/run → **~60 hr sequential.**
(f = 1.0 IID runs overlap with 4a baseline and Exp 2 — reuse where possible.)

**Predictions.**
- IID: grokking tolerates low f (our breaking-point results show f = 0.2 works
  at α = 0.5). The aggregation of random client subsets still averages to the
  full-data gradient in expectation.
- Non-IID: low f is more damaging because the subset of sampled clients may be
  unrepresentative of the full data distribution. The global model oscillates
  as different client subsets pull it in different directions.

---

**Sub-experiment 4c: Compute vs communication budget.**

Fixed R = 2 000 rounds, f = 1.0, IID only. Vary E (which varies total compute).
This answers: is grokking compute-limited or communication-limited?

| E | S = R × E | Grok expected? | Notes |
|---|-----------|----------------|-------|
| 1 | 2 000 | Unlikely (S ≪ T_base) | Establishes minimum-compute baseline |
| 5 | 10 000 | Maybe (S ≈ T_base) | Depends on α |
| 10 | 20 000 | Likely at α ≥ 0.3 | |
| 25 | 50 000 | Likely (S ≈ S_FL) | |

**Step budget rationale.** S grows with E at fixed R. E = 1 gives only S = 2 000
steps (well below T_base). E = 25 gives S = 50 000 (≈ S_FL). If E = 25 groks
but E = 5 doesn't, it's a compute limitation (10 000 steps insufficient), not a
communication problem (both have R = 2 000 rounds).

**Cross-comparison with 4a.** This is the critical logic:
- 4a at E = 50: S = S_FL, R = 900. Total compute is high but communication is low.
- 4c at E = 25: S = 50 000, R = 2 000. Both compute and communication are adequate.
- If 4a-E50 fails but 4c-E25 succeeds (at same α), the bottleneck is communication
  frequency, not total compute.

| Parameter | Values |
|-----------|--------|
| α | same 3 values |
| f | 1.0 |
| K | same as Exp 3 |
| seeds | {42, 123, 456} |

**Runs.** 4E × 3α × 3 seeds = **36 runs** (all at R = 2 000, ~12 min each = **~7 hr**).

---

**Combined Exp 4: 72 + 72 + 36 = ~180 runs** (minus overlaps ≈ **~160 unique runs**).

**Outputs.**
- **Figure 4a (main):** Grokking step vs E, two lines (IID vs non-IID), one panel
  per α. The gap between lines IS the optimization fragmentation effect. If the
  lines diverge at high E, drift direction matters. If they're parallel, it doesn't.
- **Figure 4b:** Grokking step vs f, two lines (IID vs non-IID). Shows whether
  partial participation compounds with heterogeneity.
- **Figure 4c:** Grokking step vs E at fixed R. Shows compute vs communication
  tradeoff. Mark the "sufficient compute" threshold.
- **Combined insight:** If IID is robust to both high E and low f, but non-IID
  breaks, the conclusion is: **optimization fragmentation only matters when
  combined with data heterogeneity**. This is a strong, specific, testable claim.

---

### Experiment 5 — Algorithm Comparison & Regularisation

**Question.** When FedAvg fails (or is delayed), can better algorithms or explicit
regularisation recover grokking?

**Hypotheses.**
- **FedProx** constrains client drift → preserves Fourier feature alignment → recovers
  grokking. Effective μ should increase with heterogeneity and local epochs.
- **FedAdam** (adaptive server-side optimiser) compensates for heterogeneous client
  updates and may converge faster.
- **Weight decay** provides explicit regularisation. In centralized training, wd > 0
  promotes grokking (Gromov 2023). In FL, it might serve the same role while also
  implicitly controlling client drift (penalising large weight deviations).

**Design.** Select 2–3 "hard" settings from Exps 2–4 where FedAvg either fails to grok
or is significantly delayed compared to centralized. Candidate settings (contingent on
Exps 1–4 results):
- **(H1)** α near α_crit, K = 20, IID, E = 5 (data scarcity + fragmentation)
- **(H2)** α near α_crit, K = 10, Dirichlet α_dir = 0.1 (data scarcity + heterogeneity)
- **(H3)** α = 0.3, K = 10, E = 25, f = 0.4 (communication-limited)

If no FedAvg failure is found in Exps 2–4, use the *most delayed* setting instead
and test whether algorithms can *accelerate* grokking.

For each hard setting, sweep:

| Algorithm | Parameters |
|-----------|-----------|
| FedAvg (baseline) | — |
| FedProx | μ ∈ {0.001, 0.01, 0.1, 1.0} |
| FedAdam | server_lr ∈ {0.01, 0.1, 1.0}, τ = 1e-3 |
| FedAvg + weight decay | λ ∈ {0.01, 0.1, 1.0} |

**Runs.** 3 settings × (1 + 4 + 3 + 3) algorithms × 3 seeds = **99 runs.**

**Step budget:** S_max = S_rescue = min(80 000, 2 × T_max) from §3.2. Rescue
experiments need extra headroom — a better algorithm may grok later than FedAvg
at the same setting but still succeed. At E = 5: R = S_rescue / 5.
If T_max = 30 000 → S_rescue = 60 000, R = 12 000 → ~67 min/run → **~110 hr sequential.**
If T_max = 12 000 → S_rescue = 24 000, R = 4 800 → ~27 min/run → **~45 hr sequential.**

**Outputs.**
- **Figure 5:** For each hard setting, bar chart of grokking step across algorithms.
  Include IPR at the grokking step to show that successful algorithms produce
  higher Fourier structure.
- Table with final test accuracy, grokking step, and IPR for all configurations.

**Implementation note.** FedAdam requires adding Flower's `FedAdam` strategy.
This is straightforward — Flower provides it natively; only the strategy construction
in `fed_train.py` needs to be parameterised.

---

### Experiment 6 — Mechanistic Analysis

**Question.** *Why* does grokking survive (or fail) under FL? What is the mechanism?

**Hypothesis.** Grokking requires all clients to converge on the same Fourier
features in W1. When client drift is small relative to the basin of attraction of
the Fourier solution, aggregation produces a coherent global model and grokking
proceeds. When drift is large enough that clients learn *incompatible*
representations, aggregation destroys structure and grokking fails.

**Design.** No additional runs — this is post-hoc analysis of runs from Exps 1–5.
However, it requires **new metrics** collected during training:

#### 6.1  New metrics to implement

| Metric | Definition | Where to compute |
|--------|-----------|-----------------|
| **Client drift** | d_k = ‖w_k^{after} − w_global^{before}‖_F averaged over all clients | Client fit() return value |
| **Client weight divergence** | σ_w = std({‖w_k‖_F}) across clients per round | Server-side after receiving updates |
| **Per-client IPR** | IPR computed on each client's local model before aggregation | Client fit() or server-side |
| **Fourier spectrum** | Full |W̃1(ν)|² spectrum (not just scalar IPR), saved at checkpoints | Server-side evaluate_fn |

These metrics have minimal computational overhead but require modifications to
`fed_train.py` to expose per-client information.

#### 6.2  Analysis plan

Select 4 representative runs:
1. **Healthy grok** — α = 0.5, IID, K = 5 (baseline)
2. **Boundary grok** — α ≈ α_crit, IID, K = 10 (barely works)
3. **Failed grok** — α < α_crit, or hard non-IID setting (doesn't grok)
4. **Rescued grok** — failed setting recovered by FedProx/FedAdam (from Exp 5)

For each, produce:

**Figure 6 (multi-panel):**
- **(a)** Test accuracy and train loss curves (4 runs overlaid)
- **(b)** IPR trajectories — show Fourier structure emergence
- **(c)** Mean client drift d̄ per round — show whether drift magnitude predicts failure
- **(d)** Scatter plot across ALL runs from Exps 2–5: mean client drift vs grokking step.
  Hypothesis: strong positive correlation (higher drift → later/no grokking).
- **(e)** Fourier spectra of W1 at three timepoints (before grok, during transition,
  after grok) for the healthy vs failed cases.

---

## 5  Summary of Run Counts & Compute Budget

All FL runtimes below assume the **mid-α_crit scenario** (T_max ≈ 30 000,
S_FL = 45 000, S_rescue = 60 000). See §3.2 for the adaptive formulas.

| Exp | Focus | S_max | Runs | FL | Cent. | Seq. runtime |
|-----|-------|-------|------|----|-------|-------------|
| 0 | Width validation | 50k (cent.) | 12 | 0 | 12 | ~24 min |
| 1 | Centralized boundary | 100k (cent.) | 24 | 0 | 24 | ~96 min |
| 2 | Aggregation & FL boundary | 100k (cent.) / S_FL (FL) | 234 | 108 | 126 | ~102 hr |
| 3 | Heterogeneity (+ K validation) | S_FL | ~147 | ~132 | ~15 | ~86 hr |
| 4a | Drift × heterogeneity | S_FL | 72 | 72 | 0 | ~28 hr |
| 4b | Participation × heterogeneity | S_FL | 72 | 72 | 0 | ~60 hr |
| 4c | Compute vs communication | R × E (variable) | 36 | 36 | 0 | ~7 hr |
| 5 | Algorithm rescue | S_rescue | 99 | 99 | 0 | ~110 hr |
| 6 | Mechanistic | — | 0 | 0 | 0 | — |
| **Total** | | | **~696** | **~519** | **~177** | **~395 hr** |

(After deducting overlaps — 4a/4b baseline runs shared with Exp 2/3, f = 1.0 IID
runs reused — effective unique runs ≈ **~650**.)

#### Runtime breakdown by scenario

| Scenario | T_max | S_FL | S_rescue | FL min/run (E=5) | Total seq. | 4-way parallel |
|----------|-------|------|----------|-----------------|-----------|---------------|
| High α_crit (≈ 0.3) | 12 000 | 18 000 | 24 000 | ~20 min | ~200 hr | ~50 hr ≈ **2.1 days** |
| Mid α_crit (≈ 0.2) | 30 000 | 45 000 | 60 000 | ~50 min | ~395 hr | ~99 hr ≈ **4.1 days** |
| Low α_crit (≈ 0.1) | 45 000 | 50 000 | 80 000 | ~56 min | ~450 hr | ~113 hr ≈ **4.7 days** |

**Early abort savings.** ~30–40 % of runs (extreme non-IID at low α, high K with
tiny per-client data, especially K = 50 and K = 97 at low α) will trigger early
abort. This reduces effective runtime by ~30 %, bringing the mid scenario to
~**2.9 days** at 4-way parallelism.

*Runtime assumptions: ~0.4 s Flower/Ray overhead per round. Centralized: ~2 min
per 50k steps, ~4 min per 100k steps. All runs within each experiment are
fully independent and parallelisable.*

---

## 6  Implementation Requirements

### 6.1  Code changes needed

| Change | File | Effort |
|--------|------|--------|
| Add `FedAdam` strategy option | `federated/train.py`, `federated/config.py` | Small — Flower provides `FedAdam` natively |
| Track client drift metric | `federated/train.py` (client fit + server callback) | Medium — return ‖Δw‖ from client, aggregate server-side |
| Track client weight divergence | `federated/train.py` (server aggregation callback) | Medium |
| Save Fourier spectrum at checkpoints | `core/metrics.py`, `federated/train.py` | Small |
| New sweep modes for Exps 1–5 | `main.py`, `fed_main.py` | Medium — parameterise from experiment tables above |
| Phase diagram visualisation | `federated/visualize.py` | Medium |

### 6.2  New config parameters

```python
# federated/config.py additions
strategy: str = "fedavg"        # "fedavg", "fedprox", "fedadam"
server_lr: float = 1.0          # server-side learning rate (FedAdam)
tau: float = 1e-3               # FedAdam adaptivity parameter
track_client_drift: bool = True # enables per-round drift logging
```

---

## 7  Execution Order & Dependencies

```
Exp 0 (width validation)
  │
  ├── determines N* (128 or 200)
  │
  ▼
Exp 1 (centralized boundary)          ← uses N*
  │
  ├── determines α_crit
  │
  ▼
Exp 2 (FL boundary + aggregation)     ← uses α values around α_crit
  │
  ├── determines best K for Exps 3–4
  ├── identifies whether FL shifts boundary
  │
  ▼
Exp 3 (heterogeneity)                 ← uses K from Exp 2, α near boundary
Exp 4 (communication)                 ← can run in parallel with Exp 3
  │
  ├── Exps 3–4 identify "hard" settings
  │
  ▼
Exp 5 (algorithm rescue)              ← uses hard settings from Exps 2–4
  │
  ▼
Exp 6 (mechanistic analysis)          ← post-hoc on all runs
```

**Parallelism opportunities:**
- Exp 1 runs are fast (centralized, small model) → run all 24 first
- Within Exp 2: all (α, K, condition) triples are independent → fully parallel
- Exps 3 and 4 are independent of each other → parallel
- Within Exp 5: all algorithm × setting triples are independent → parallel

---

## 8  Contingency Plans

### 8.1  If no centralized phase boundary exists (grokking at all α)

This would mean modular addition groks even with ~30 training pairs (α = 0.03).
If observed:
- Extend sweep to α ∈ {0.01, 0.02} (9–18 training samples)
- **Reduce p** (e.g. p = 23): Gromov notes α_c is a decreasing function of p, so
  smaller primes have higher α_crit. p = 23 also makes experiments ~16× faster
  (p² = 529 vs 9 409 pairs, model 3× smaller).
- Switch to a harder task (e.g., `x2_plus_y2`) which may have a higher α_crit
- The finding "grokking is universal for this architecture" is itself publishable

### 8.2  If FL never breaks grokking (even at α_crit)

This would mean federated aggregation is always sufficient to preserve grokking.
- The Exp 2 aggregation isolation becomes the key result: FL matches centralized-full
  even though each client has centralized-reduced data → aggregation is the mechanism
- Focus the paper on the *robustness* story rather than the *fragility* story
- The mechanistic analysis (Exp 6) explains WHY: the Fourier solution is a global
  attractor robust to noisy gradient estimates

### 8.3  If FL helps grokking (FL α_crit < centralized α_crit)

This is the most surprising and publishable finding:
- Aggregation acts as implicit regularisation (averaging noisy client models
  suppresses memorisation, promotes generalisation)
- Add experiment: vary K at α < centralized α_crit to find the *optimal* K
- Connect to existing theory on "implicit regularisation from distributed training"

---

## 9  Expected Paper Figures

| # | Content | Experiment | Role |
|---|---------|-----------|------|
| 1 | Training dynamics: centralized vs FedAvg-IID | Baseline (existing) | Reproduce + validate |
| 2 | Grokking step vs K: centralized-full / centralized-reduced / FL | Exp 2 | **Main result**: aggregation effect |
| 3 | Phase diagram: training fraction × heterogeneity | Exp 3 | Non-IID effects |
| 4 | Communication efficiency: grokking step vs local epochs | Exp 4 | Practical FL tradeoff |
| 5 | Algorithm rescue: bar chart across algorithms | Exp 5 | Practical contribution |
| 6 | Mechanistic: IPR + client drift + Fourier spectra | Exp 6 | Explains the "why" |

**Appendix figures:**
- Centralized phase boundary (α sweep) — Exp 1
- Structured partition comparison (operand vs target) — Exp 3b
- Full 2D heatmaps for all Exp 2 conditions
- Activation function generality (quadratic vs ReLU vs GELU)
- Task generality (addition vs other operations)

---

## 10  Reproducibility Checklist

- [ ] All random seeds recorded and fixed
- [ ] Exact software versions pinned (Python 3.12, PyTorch, Flower, Ray)
- [ ] All history JSON files saved with full metric trajectories
- [ ] Code versioned with git tag per experiment phase
- [ ] Hyperparameter tables in appendix match code defaults
- [ ] 3 seeds per configuration with mean ± std reported
- [ ] Grokking step definition applied uniformly across all experiments
- [ ] S_max = 50 000 sufficient for all observed grokking events (verify post-hoc)
- [ ] Centralized-reduced comparison uses strict subset of original training set (same seed permutation → verified by construction)

---

## 11  Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Flower simulation too slow at R > 5 000 | Medium | Default R = 6 000 (E = 5); E = 1 only in Exp 4b (R = 2 000); profile early and optimise if needed |
| α_crit is extremely low (< 0.03) | Low | Switch to harder tasks; adjust p |
| FedAdam diverges with lr = 50 client-side | Medium | FedAdam uses server-side adaptive LR; client LR stays at 50. If unstable, sweep client LR too |
| Optimizer state restart (SGD → no issue; AdamW → known limitation) | Noted | Primary experiments use SGD; note limitation for AdamW runs |
| Per-round metric logging slows simulation | Low | Drift metrics are cheap (one norm computation per client); Fourier spectra saved only at checkpoints (every 100 rounds) |
| 546 runs × infrastructure failure | Medium | Checkpoint partial results; idempotent run scripts; track completion in results directory |
