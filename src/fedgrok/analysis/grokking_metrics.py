"""Grokking step detection and multi-seed result aggregation.

- T_grok: smallest step t_j such that test_acc >= threshold for ALL subsequent
  steps. Not comparable across budgets or logging rates -- extending a run can
  report a LATER t_grok on an identical trajectory prefix, because the bar has
  more chances to be broken. Use t_first_cross to compare cells.
- t_first_cross: first step at or above the threshold, sustained or not.
- T_50: first step where test_acc >= 50%.
- T_memo: first step where TRAIN accuracy >= 99%. The delay that defines
  grokking is t_grok - t_memo, and a cell that never memorised has failed to
  train rather than failed to grok -- a different diagnosis with a different fix.
"""

import math
from typing import List
import numpy as np


def _below(acc, threshold: float) -> bool:
    """Is `acc` below the bar, counting NaN as below?

    Written as `not (acc >= threshold)` rather than `acc < threshold` because the
    two differ on NaN, and only one of them is safe here. `nan < threshold` is
    False, so a NaN test curve finds NO point below the bar, and the
    sustained-crossing scan in `compute_t_grok` concludes the bar held from the
    first sample onwards -- reporting `grokked=True` at `t_grok = steps[0]` for a
    run that measured nothing at all.

    That is not hypothetical: `alpha=1.0` trains on the whole grid and leaves an
    empty test set, so `compute_accuracy` divides by zero and every test point is
    NaN. Five banked setup-D runs are recorded that way (RESULTS.md, known
    issues). A missing measurement must never read as the strongest possible
    positive result.

    The first-crossing helpers (`compute_t_50`, `compute_t_memo`,
    `compute_t_first_cross`) test `acc >= threshold` directly, which is already
    NaN-safe: NaN fails that comparison and the step is correctly skipped.
    """
    return not (acc >= threshold)


def compute_t_grok(steps: list, test_accs: list, threshold: float = 95.0) -> float:
    """Compute grokking step T_grok.

    Returns the smallest step where test accuracy reaches `threshold`
    and never drops below it for the remainder of training.
    Returns float('inf') if no such step exists, and also if the curve is NaN
    (no test set / diverged) -- see `_below`.
    """
    if not steps:
        return float("inf")

    n = len(steps)
    last_below = -1
    for i in range(n - 1, -1, -1):
        if _below(test_accs[i], threshold):
            last_below = i
            break

    if last_below == -1:
        return steps[0]
    if last_below == n - 1:
        return float("inf")

    return steps[last_below + 1]


def compute_t_50(steps: list, test_accs: list, threshold: float = 50.0) -> float:
    """Compute onset step T_50 — first step where test accuracy >= 50%."""
    for step, acc in zip(steps, test_accs):
        if acc >= threshold:
            return step
    return float("inf")


def compute_t_memo(steps: list, train_accs: list, threshold: float = 99.0) -> float:
    """Memorisation step T_memo — first step where TRAIN accuracy >= threshold.

    The partner of `compute_t_grok`, and the reason grokking has two timescales
    rather than one: the delay that defines the phenomenon is
    `t_grok - t_memo`, not `t_grok` alone. A cell that never memorises has not
    failed to grok — it has failed to train, which is a different diagnosis with
    a different fix, and the two are indistinguishable from `t_grok` on its own.

    Unlike T_grok this is a FIRST crossing, not a sustained one. Train accuracy
    is monotone in practice here, and requiring it to hold to the end of the run
    would discard the memorise-then-collapse trajectories (decay outrunning
    learning) that are exactly the ones worth telling apart.

    99% rather than the test-side bar because memorisation is near-total when it
    happens, and because it must not move with `grok_threshold`: T_memo has to
    mean the same thing across datasets for the delay to be comparable.
    """
    for step, acc in zip(steps, train_accs):
        if acc >= threshold:
            return step
    return float("inf")


def compute_t_first_cross(steps: list, test_accs: list, threshold: float) -> float:
    """First step test accuracy reaches `threshold`, sustained or not.

    `compute_t_grok` requires the bar to hold for the REST of the run, which is
    the right definition for a phase transition but makes the value depend on the
    LOGGING RATE whenever the curve is not monotone: every extra sample point is
    another chance to observe a dip and push the answer later. Measured on setup
    C, one seed, the identical trajectory scored 15,200 at log_every=200 and
    59,350 at log_every=50 -- a 4x difference from logging alone, because the
    coarse run never sampled a collapse to 20.2% at epoch 59,300.

    The first crossing is not sampling-invariant either, but it is stable to
    within one logging interval rather than to within the instability's duration.
    Recording both separates "when did it generalise" from "when did it stop
    falling over", which for an unstable setup are different questions.
    """
    for step, acc in zip(steps, test_accs):
        if acc >= threshold:
            return step
    return float("inf")


def count_post_cross_dips(steps: list, test_accs: list, threshold: float) -> int:
    """Logged points below `threshold` AFTER the first crossing.

    Zero means the transition held. Non-zero means the run generalised and then
    lost it at least once, which is a property of the setup worth carrying: on
    the S_5 transformer every seed dips (worst observed 9.4% from ~100%), on the
    S_5 quadratic MLP none do. The magnitude scales with the logging rate, so
    compare it only within a fixed `log_every` -- the zero/non-zero distinction
    is the part that transfers.
    """
    first = compute_t_first_cross(steps, test_accs, threshold)
    if first == float("inf"):
        return 0
    return sum(1 for step, acc in zip(steps, test_accs)
               if step > first and _below(acc, threshold))


def extract_grokking_results(history: dict, threshold: float = 95.0) -> dict:
    """Extract grokking metrics from a training history dict.

    Works with both centralized (key: 'epoch') and federated (key: 'total_steps').

    `threshold` is the test accuracy that counts as generalisation. It is
    dataset-dependent — pass `fedgrok.data.registry.grok_threshold(cfg)`, which
    documents the per-dataset values. The 95.0 default preserves the modular
    behaviour for callers that have no cfg to hand.
    """
    steps = history.get("total_steps", history.get("epoch", []))
    test_accs = history.get("test_acc", [])
    train_accs = history.get("train_acc", [])

    t_grok = compute_t_grok(steps, test_accs, threshold=threshold)
    t_50 = compute_t_50(steps, test_accs)
    t_memo = compute_t_memo(steps, train_accs)
    final_test_acc = test_accs[-1] if test_accs else 0.0
    final_train_acc = train_accs[-1] if train_accs else 0.0
    final_ipr = history.get("ipr", [0.0])[-1] if history.get("ipr") else 0.0

    return {
        "t_grok": t_grok,
        "t_50": t_50,
        "t_memo": t_memo,
        # Sampling-robust companion to t_grok, plus the instability it hides.
        "t_first_cross": compute_t_first_cross(steps, test_accs, threshold),
        "post_grok_dips": count_post_cross_dips(steps, test_accs, threshold),
        # The delay IS the phenomenon. inf if either end is missing: a run that
        # never memorised has no delay, and neither does one still censored.
        "delay": (t_grok - t_memo
                  if math.isfinite(t_grok) and math.isfinite(t_memo)
                  else float("inf")),
        "final_test_acc": final_test_acc,
        "final_train_acc": final_train_acc,
        # Peak train accuracy, because `final` hides the memorise-then-collapse
        # trajectory. When decay outruns learning the curve rises and falls, so a
        # run that reached 40% and decayed to 1% is recorded identically to one
        # that never left 1% -- and those are different failures. `t_memo` cannot
        # cover this either: at a 99% bar both are inf, as is a cell sitting at
        # 98.2%. Peak is the cheap scalar that orders them.
        #
        # Non-finite points are dropped rather than passed to max(), whose result
        # on a NaN-containing list depends on POSITION: max() keeps its running
        # value unless the next compares greater, and every comparison with NaN
        # is False -- so a leading NaN wins and a trailing one loses. The peak
        # over the finite points is the reading this field is for.
        "peak_train_acc": max((a for a in train_accs if math.isfinite(a)),
                              default=0.0),
        "final_ipr": final_ipr,
    }


def summarize_seeds(results: List[dict], budget: float = None) -> dict:
    """Aggregate grokking metrics across seeds with right-censoring.

    Seeds that did not grok within budget are right-censored (we know only
    T_grok > budget), not infinite. The headline is therefore the fraction
    grokked plus the Kaplan-Meier median (with a bootstrap CI), not a mean that
    goes to inf the moment one seed fails.

    Censoring time per non-grokked seed: its `steps_run` if present, else the
    passed `budget`, else the largest finite grok time observed (a conservative
    fallback). `t_grok_mean/std` are kept as descriptive stats over the seeds
    that DID grok — no longer inf-if-any-fail.

    See fedgrok.analysis.survival for the estimators.
    """
    from fedgrok.analysis.survival import summarize_survival

    n = len(results)
    t_groks = [r["t_grok"] for r in results]
    t_50s = [r["t_50"] for r in results]
    final_accs = [r["final_test_acc"] for r in results]

    finite_groks = [t for t in t_groks if t < float("inf")]
    finite_50s = [t for t in t_50s if t < float("inf")]
    fallback = max(finite_groks) if finite_groks else float("inf")

    durations, events = [], []
    for r in results:
        t = r["t_grok"]
        if t < float("inf"):
            durations.append(t)
            events.append(1)
        else:
            durations.append(r.get("steps_run") or budget or fallback)
            events.append(0)

    surv = summarize_survival(durations, events)

    return {
        **surv,                                          # n_seeds, n_grokked,
                                                         # fraction_grokked, KM median + CI
        # descriptive stats over the grokked seeds only (not inf-if-any-fail)
        "t_grok_mean": float(np.mean(finite_groks)) if finite_groks else float("inf"),
        "t_grok_std": float(np.std(finite_groks, ddof=1)) if len(finite_groks) > 1 else 0.0,
        "t_50_mean": float(np.mean(finite_50s)) if finite_50s else float("inf"),
        "t_50_std": float(np.std(finite_50s, ddof=1)) if len(finite_50s) > 1 else 0.0,
        "final_acc_mean": float(np.mean(final_accs)),
        "final_acc_std": float(np.std(final_accs, ddof=1)) if n > 1 else 0.0,
    }
