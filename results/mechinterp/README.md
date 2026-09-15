# Mechanistic-interpretability suite — outputs and findings

Produced 2026-09-15 by `scripts/mechinterp/` (run everything with `bash scripts/mechinterp/run_all.sh`;
every stage resumes). No banked run was modified. Twins are analysis instruments under `twins/`,
not banked into `results/data/runs_v2.csv`.

## What was measured, and on what

| Stage | Script | Coverage | Output |
|---|---|---|---|
| Logged dynamics | `histories.py` | all 1,694 runs | `history_moments.csv`, `history_series_crossaligned.csv`, `history_series_logstep.csv` |
| Checkpoint metrics | `checkpoints.py` | 656 runs with weights, ≤24 checkpoints each (9,708 rows, 273 metrics) | `ckpt/<id>.json`, `checkpoint_metrics.csv` |
| Spectra | `spectra.py` | 655 runs (one diverged run has no finite checkpoint) | `spectra/<id>.npz`, `spectra_clock.csv`, `spectra_in_power.csv`, `spectra_irreps.csv`, `spectra_sv_summary.csv` |
| Clients | `clients.py` | 478 federated runs with per-client snapshots (5,460 rows) | `client_metrics.csv` |
| Centralised twins | `twins.py train` | 32 runs, 10 families, same initial weights as the federated runs | `twins/rows`, `twins/runs` |
| Twin comparison | `twins.py compare` | every federated checkpoint with a twin checkpoint within 2.5% of its step (4,623 rows) | `twin_trajectories.csv` |
| Loss landscape | `landscape.py` | 1,382 linear paths | `landscape_paths.csv`, `landscape_barriers.csv` |
| Aggregation | `findings.py` | all of the above | `findings/*.csv`, `findings/FINDINGS.md` |

Column definitions are in each script's docstring (metric families by prefix: `fn_`, `w_`, `rep_`, `unit_`,
`add_`, `circ_`, `fin_/fU_/fV_/fout_`, `clock_`, `irr_`, `coset_`, `attn_`, `sharp_`, `cli_`, `tw_`).

### Checks that passed

- Initial weights rebuilt from the seed match the logged step-0 norms exactly (difference 0.0 on all 656 runs).
- Re-implemented transformer internals reproduce the model's logits exactly (max |Δ| = 0.0).
- Every twin matches its federated family on all model, data and optimiser fields (asserted before training),
  and the twins reproduce the banked centralised baselines: A α0.30 first crossing 12,600–13,300 (banked ≈12,900);
  B wd 1.0 3,900–5,900 (banked 4,000–6,200); D 20,000–21,500 (banked 21,500).

### Caveats that apply to the tables

- **Six setup-A runs diverged to NaN** (Dirichlet 0.01 and its shard-size control at K = 20/50). Their non-finite
  checkpoints carry `w_nonfinite = 1` and no other metrics. See finding 1.
- **Restricted accuracy is scale-free** (a tiny, correctly phased clock component scores 100%); read restricted
  and excluded *losses*.
- **Twin comparisons at matched step conflate progress with mechanism.** At a federated run's first crossing the
  twin may not have generalised yet (or may have long since), so `tw_pred_agree_te` at `cross` can be near 0 for
  cells that grok in fewer steps than centralised training (e.g. partial participation, whose `total_steps`
  scales with f). Use the `end` moment for "same solution?" questions.
- **`cli_dev_pair_cos` depends on the number of clients** (−1/(n−1) for independent zero-sum deviations). Its raw
  correlation with delay is mostly K; multiply by (n − 1) before comparing across cells.
- **`barrier_rel` explodes when endpoint losses are tiny** (grokked runs reach train loss ~1e-5). Read test
  accuracy along the path (`test_acc_mid`, `min_test_acc_path`).
- **Setup C is non-deterministic run to run** (RESULTS §23); its numbers are descriptive only.
- p-values in `findings/` use the Fisher z approximation for Spearman's ρ: a ranking of effects, not tests.

## Key numerical findings

### 1. The starvation "failures" on A are numerical divergence, not slow grokking

All six runs memorised (peak train 97.7–100%), then diverged abruptly: train loss 0.006 → 3.5 × 10⁷ within one
evaluation interval, weight norm 30.7 → 90.4, then NaN. First non-finite step: 4,800 / 5,100 / 5,700 / 17,200 /
26,600 / 47,900. Every later checkpoint is NaN. These runs should be excluded from any timing statistic rather
than treated as censored (they are recorded as `grokked = False` in the run table).

### 2. Under GD the federated solution is the centralised solution; under AdamW it is not

Federated final checkpoint vs its same-initialisation twin at the same step (`twin_trajectories.csv`, last step):

| Setup, cell | rel. weight distance | CKA (test reps) | same clock key freqs (Jaccard) | test predictions agree | midpoint of linear path: test acc |
|---|---|---|---|---|---|
| A, K = 2 / 5 / 10 / 20 / 50 (α 0.30) | 0.17 / 0.27 / 0.38 / 0.48 / 0.71 | 0.99 / 0.97 / 0.93 / 0.90 / 0.73 | 0.96 / 0.96 / 0.91 / 0.91 / 0.91 | 100% | 100% (A, all federated) |
| A, target partition K = 10 / 50 | 0.85 / 1.15 | 0.67 / 0.28 | 0.77 / 0.43 | 100% | |
| A, Dirichlet 1000 → 0.01 (K = 10) | 0.35 → 0.78 | 0.95 → 0.74 | 0.91 | 100% | |
| A′ (AdamW), every K | 1.23–1.31 | 0.19–0.45 | 0.72–0.87 | ≈100% | 93% |
| B wd 1.0, K = 2 / 5 / 10 / 20 | 1.36 / 1.35 / 1.45 / 1.47 | 0.44 / 0.45 / 0.00 / 0.00 | 0.40 / 0.67 / 0.00 / 0.14 | 100% | 2.7% (B, all) |
| C, K = 2 / 5 / 10 / 20 / 50 | 1.39 / 1.40 / 1.34 / 1.32 / 1.54 | 0.61 / 0.43 / 0.41 / 0.02 / 0.06 | dominant irrep same: yes / yes / yes / no / no | 100% | 7.4% (C, all) |
| D, K = 2 / 10 / 20 | 1.41 / 1.45 / 1.35 | 0.44 / 0.44 / 0.51 | dominant irrep same: yes / yes / no | 95–87% | 54% (D, all) |
| E, K = 2 / 10 / 20 | 1.51 / 1.56 / 1.52 | 0.72 / 0.69 / 0.89 | — | 84–95% | 80% (E, all) |

- On A (plain GD, no decay) federation perturbs the centralised trajectory by an amount that grows smoothly
  with K and with heterogeneity, keeps the same key frequencies, and stays in the same basin (test accuracy
  100% all the way along the straight line to the twin). Incoherent (target) sharding moves it furthest.
- On every AdamW setup the federated model ends >1.2 relative distance from its twin even at K = 2, and the two
  transformers pick **different circuits**: B's clock key frequencies overlap 0–0.67 with the twin's, C's dominant
  irrep differs from K = 20 on, and the straight line between federated and centralised transformers passes
  through a 3–7%-accuracy region. Both solutions generalise; they are different solutions. (No
  same-initialisation noise floor was measured; C is known to be non-deterministic run to run.)

### 3. How each architecture fails at high K, seen in the weights at first crossing

`findings/axis_trends.csv`, moment `cross`:

- **B (transformer, wd 0.1), K = 5 → 50:** the unembedding collapses to a single direction — effective rank
  57.5 → 6.6, stable rank 2.1 → 1.0, norm 21 → 90, share of its Fourier power at DC 0.48 → 0.997 (ρ = 0.96).
  High-K transformers put their output weight into a class-prior direction.
- **D (quadratic MLP, S₅), K = 5 → 50:** the output layer collapses too — W2 spectral norm 10.5 → 63.4, stable
  rank 44 → 1.7, representation participation ratio 158 → 41 — while train loss at crossing rises 9 × 10⁻⁶ → 0.25
  and sharpness 9 × 10⁻⁶ → 0.05 (all ρ = 0.96).
- **A (GD), K = 2 → 50:** the input layer grows and the output layer shrinks (W1 norm 32.5 → 35.4, W2 19.9 → 16.9),
  sharpness rises (lr·λ 0.53 → 0.72), and the fraction of strong neurons whose two operand blocks share a frequency
  falls 1.00 → 0.945. Dirichlet skew does the same (lr·λ 0.54 → 0.86). All well below GD's stability edge of 2.
- **A at α 0.25, K up to 97:** at crossing, hidden units are dominated by single-operand frequencies (marginal share
  0.35 → 0.95) and carry little two-operand structure (interaction share 0.65 → 0.05), with NC1 1.0 → 18.

### 4. What predicts the delay at the moment of memorisation (setup A)

Across A's federated runs, the structure already present in hidden units when the training set is memorised
predicts how long generalisation takes (`findings/delay_predictors.csv`):

| metric at t_memo | ρ with log delay (n = 52) | AUC, crossed vs not |
|---|---|---|
| `unit_sum_freq_concentration` | −0.94 | 0.98 |
| `rep_between_within_te` | −0.95 | 0.96 |
| `unit_fourier_sum_share` | −0.95 | 0.94 |
| `unit_interaction_share` | −0.72 | 0.94 |

Within the α 0.30 K ladder alone ρ = −0.95 (n = 15), but K varies along that ladder, so this is not independent
of K. It says the delay is set by how much sum-frequency (a + b) structure the units have when memorisation
completes, not by anything that happens later.

### 5. Which order parameters move before the model generalises

Lead of the half-change time over first crossing, as a fraction of the delay (`findings/history_leads.csv`,
all runs; positive = moves first):

| Setup | leads | lags |
|---|---|---|
| A | W1 norm 0.35 (77% of 333 runs) | Fourier IPR −0.28 (leads in 7%) |
| B | test loss 0.28 (89%) | embedding IPR −0.18 |
| C | coset accuracy 0.05 (95%) — barely | |
| D | compositional-circuit accuracy 0.72 (100% of 171), irrep structure 0.73 (83%), coset accuracy 0.50 (89%) | |
| E | weight norm 0.50 (95% of 118) | |

On A the Fourier IPR of the input weights sharpens *after* the bar is crossed, while the weight norm and the
hidden units' sum-frequency structure move before it. On D the compositional circuit forms long before the full
model generalises (the masking of RESULTS §24).

### 6. Memorising and generalising solutions are linearly connected on MLPs, not on transformers

Straight line from the checkpoint nearest t_memo to the final checkpoint (`landscape_barriers.csv`, federated):
test accuracy at the midpoint is 95% on A, 59% on A′, 69% on D and 84% on E, but 2.3% on B and 5.7% on C.

### 7. Clients disagree independently

Per-client deviations from the client mean have normalised pairwise cosine ≈ −1 on every setup (median −0.994 to
−1.000 after multiplying by n − 1): the deviations cancel in the average and share no common direction. After
that normalisation their correlation with delay on A drops from ρ = 0.81 to 0.42. Client-level metrics are in
`client_metrics.csv` and `findings/client_phases.csv`.
