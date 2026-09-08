import pytest
from fedgrok.analysis.grokking_metrics import (
    compute_t_grok, compute_t_50, compute_t_memo, compute_t_first_cross,
    count_post_cross_dips, summarize_seeds, extract_grokking_results,
)


class TestTGrok:
    def test_clear_grokking(self):
        steps = list(range(0, 1000, 100))
        accs = [1.0, 1.0, 1.0, 10.0, 50.0, 96.0, 97.0, 98.0, 99.0, 99.5]
        assert compute_t_grok(steps, accs, threshold=95.0) == 500

    def test_no_grokking(self):
        steps = list(range(0, 1000, 100))
        accs = [1.0] * 10
        assert compute_t_grok(steps, accs, threshold=95.0) == float("inf")

    def test_transient_spike_rejected(self):
        steps = list(range(0, 800, 100))
        accs = [1.0, 1.0, 96.0, 80.0, 96.0, 97.0, 98.0, 99.0]
        assert compute_t_grok(steps, accs, threshold=95.0) == 400

    def test_single_point_at_end(self):
        steps = [0, 100, 200]
        accs = [1.0, 50.0, 96.0]
        assert compute_t_grok(steps, accs, threshold=95.0) == 200

    def test_empty_input(self):
        assert compute_t_grok([], [], threshold=95.0) == float("inf")

    def test_all_nan_is_censored_not_instant_grokking(self):
        """A NaN curve must not read as "grokked at step 0".

        `nan < threshold` is False, so the sustained-crossing scan used to find
        no point below the bar and conclude it held from the first sample. An
        empty test set (alpha=1.0) produces exactly this, and five banked
        setup-D runs are recorded grokked=True / t_grok=0 because of it.
        """
        steps = list(range(0, 500, 100))
        accs = [float("nan")] * 5
        assert compute_t_grok(steps, accs, threshold=95.0) == float("inf")
        # t_first_cross was already correct; the two must now agree.
        assert compute_t_first_cross(steps, accs, 95.0) == float("inf")

    def test_nan_after_crossing_breaks_the_sustained_bar(self):
        # Diverging to NaN after generalising is a failure, not a pass.
        steps = [0, 100, 200, 300]
        accs = [1.0, 96.0, 97.0, float("nan")]
        assert compute_t_grok(steps, accs, threshold=95.0) == float("inf")

    def test_nan_before_crossing_does_not_shift_t_grok(self):
        steps = [0, 100, 200, 300]
        accs = [float("nan"), 1.0, 96.0, 97.0]
        assert compute_t_grok(steps, accs, threshold=95.0) == 200

    def test_nan_counts_as_a_post_cross_dip(self):
        steps = [0, 100, 200, 300]
        accs = [1.0, 96.0, float("nan"), 97.0]
        assert count_post_cross_dips(steps, accs, 95.0) == 1


class TestT50:
    def test_onset_detected(self):
        steps = [0, 100, 200, 300]
        accs = [1.0, 30.0, 55.0, 90.0]
        assert compute_t_50(steps, accs) == 200

    def test_no_onset(self):
        steps = [0, 100, 200]
        accs = [1.0, 1.0, 1.0]
        assert compute_t_50(steps, accs) == float("inf")


class TestSummarizeSeeds:
    def test_aggregation(self):
        results = [
            {"t_grok": 500, "t_50": 300, "final_test_acc": 99.0},
            {"t_grok": 600, "t_50": 350, "final_test_acc": 98.5},
            {"t_grok": 550, "t_50": 320, "final_test_acc": 99.2},
        ]
        summary = summarize_seeds(results)
        assert summary["t_grok_mean"] == pytest.approx(550.0)
        assert summary["t_grok_std"] == pytest.approx(50.0, abs=0.1)
        assert summary["n_grokked"] == 3
        assert summary["n_seeds"] == 3

    def test_partial_grokking_is_censored_not_inf(self):
        """A non-grokked seed is right-censored, not infinite. The headline is
        fraction grokked + KM median; t_grok_mean is now the mean over the
        seeds that DID grok (not inf-if-any-fail — that was the bug)."""
        results = [
            {"t_grok": 500, "t_50": 300, "final_test_acc": 99.0, "steps_run": 50000},
            {"t_grok": float("inf"), "t_50": float("inf"), "final_test_acc": 1.0,
             "steps_run": 50000},
        ]
        summary = summarize_seeds(results)
        assert summary["n_grokked"] == 1
        assert summary["n_seeds"] == 2
        assert summary["fraction_grokked"] == 0.5
        # descriptive mean over grokked seeds is finite, not inf
        assert summary["t_grok_mean"] == pytest.approx(500.0)
        # KM median is well-defined (not inf) with one grok of two
        assert summary["t_grok_km_median"] == pytest.approx(500.0)

    def test_full_summary_reports_survival_keys(self):
        results = [{"t_grok": t, "t_50": t - 100, "final_test_acc": 99.0}
                   for t in (500, 600, 550)]
        summary = summarize_seeds(results)
        assert summary["fraction_grokked"] == 1.0
        assert "t_grok_km_median" in summary
        assert "t_grok_ci_low" in summary and "t_grok_ci_high" in summary


class TestDatasetAwareThreshold:
    """The grok bar must follow the dataset's achievable ceiling.

    A global 95% recorded every MNIST-1k run as `t_grok = inf` even though the
    histories show memorisation at ~600 epochs and generalisation thousands of
    epochs later — a measurement artifact reported as a scientific null.
    """

    def test_modular_bar_is_unchanged(self):
        """Every modular result ever recorded must be unaffected by this change."""
        from fedgrok.core.config import Config
        from fedgrok.data.registry import grok_threshold
        assert grok_threshold(Config()) == 95.0

    def test_mnist_and_s5_sit_below_their_ceilings(self):
        from fedgrok.core.config import Config
        from fedgrok.data.registry import grok_threshold
        assert grok_threshold(Config(dataset="mnist")) == 90.0   # ceiling ~93%
        assert grok_threshold(Config(dataset="s5")) == 85.0      # ceiling ~92%

    def test_mnist_curve_groks_at_its_own_bar_but_not_at_95(self):
        """The real MNIST shape: train memorises early, test climbs into the 90s."""
        steps = list(range(0, 10000, 1000))
        # test accuracy plateaus at 92.7% — the measured lr*wd=1e-4 band
        accs = [10.0, 78.0, 84.0, 88.0, 91.0, 92.0, 92.5, 92.7, 92.7, 92.7]
        assert compute_t_grok(steps, accs, threshold=95.0) == float("inf")
        assert compute_t_grok(steps, accs, threshold=90.0) == 4000

    def test_weakest_decay_band_is_honestly_censored(self):
        """lr*wd=1e-5 peaks at 89.2% — below 90, so censoring is correct here."""
        steps = list(range(0, 5000, 1000))
        accs = [10.0, 80.0, 86.0, 88.5, 89.2]
        assert compute_t_grok(steps, accs, threshold=90.0) == float("inf")

    def test_extract_passes_the_threshold_through(self):
        history = {"epoch": [0, 100, 200], "test_acc": [10.0, 91.0, 92.0],
                   "train_acc": [50.0, 100.0, 100.0]}
        assert extract_grokking_results(history)["t_grok"] == float("inf")
        assert extract_grokking_results(history, threshold=90.0)["t_grok"] == 100


class TestTMemoAndDelay:
    """Memorisation time, and the delay that is the actual phenomenon.

    `t_grok` alone cannot distinguish a model that memorised and never
    generalised from one that never trained at all -- both read `inf`. That
    ambiguity is not academic: the K>=30 AdamW cells sit at 1-5% TRAIN accuracy,
    which in a table of t_grok values is indistinguishable from a grokking
    failure, and the two have different causes and different fixes.
    """

    def test_memorisation_detected_at_first_crossing(self):
        steps = list(range(0, 1000, 100))
        train = [1.0, 40.0, 99.5, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        assert compute_t_memo(steps, train) == 200

    def test_never_memorised(self):
        steps = list(range(0, 500, 100))
        assert compute_t_memo(steps, [1.0, 3.0, 5.0, 4.0, 3.6]) == float("inf")

    def test_first_crossing_not_sustained_unlike_t_grok(self):
        """T_memo is deliberately a FIRST crossing, where T_grok is sustained.

        The memorise-then-collapse trajectory -- decay outrunning learning -- is
        exactly the one worth identifying, and a sustained-crossing rule would
        discard it by recording `inf`, i.e. "never memorised", for a run that
        plainly did.
        """
        steps = list(range(0, 500, 100))
        train = [1.0, 99.5, 60.0, 20.0, 1.0]
        assert compute_t_memo(steps, train) == 100
        assert compute_t_grok(steps, train, threshold=95.0) == float("inf")

    def test_delay_is_the_gap_between_the_two(self):
        steps = list(range(0, 1000, 100))
        train = [1.0, 100.0] + [100.0] * 8
        test = [1.0, 1.0, 1.0, 1.0, 1.0, 96.0, 97.0, 98.0, 99.0, 99.5]
        out = extract_grokking_results({"epoch": steps, "train_acc": train,
                                        "test_acc": test}, threshold=95.0)
        assert out["t_memo"] == 100
        assert out["t_grok"] == 500
        assert out["delay"] == 400

    def test_delay_is_inf_when_either_end_is_missing(self):
        steps = list(range(0, 400, 100))
        # memorised, never generalised
        memo_only = extract_grokking_results(
            {"epoch": steps, "train_acc": [1.0, 100.0, 100.0, 100.0],
             "test_acc": [1.0] * 4}, threshold=95.0)
        assert memo_only["t_memo"] == 100
        assert memo_only["delay"] == float("inf")
        # never trained at all -- the K>=40 case
        neither = extract_grokking_results(
            {"epoch": steps, "train_acc": [1.0, 4.0, 5.0, 3.0],
             "test_acc": [1.0] * 4}, threshold=95.0)
        assert neither["t_memo"] == float("inf")
        assert neither["delay"] == float("inf")


class TestPeakTrainAcc:
    def test_peak_separates_collapse_from_never_learning(self):
        """`final_train_acc` alone cannot order the two failure modes."""
        steps = list(range(0, 400, 100))
        collapsed = extract_grokking_results(
            {"epoch": steps, "train_acc": [1.0, 42.0, 20.0, 4.0],
             "test_acc": [1.0] * 4}, threshold=95.0)
        never = extract_grokking_results(
            {"epoch": steps, "train_acc": [1.0, 3.0, 4.0, 4.0],
             "test_acc": [1.0] * 4}, threshold=95.0)
        # indistinguishable on the recorded final value ...
        assert collapsed["final_train_acc"] == never["final_train_acc"] == 4.0
        # ... and on t_memo, which is inf for both at a 99% bar
        assert collapsed["t_memo"] == never["t_memo"] == float("inf")
        # ... but ordered by peak
        assert collapsed["peak_train_acc"] == 42.0
        assert never["peak_train_acc"] == 4.0

    def test_empty_history_is_zero_not_an_error(self):
        out = extract_grokking_results({"epoch": [], "train_acc": [],
                                        "test_acc": []}, threshold=95.0)
        assert out["peak_train_acc"] == 0.0


class TestFirstCrossAndDips:
    """Separating "when did it generalise" from "when did it stop falling over".

    `t_grok` requires the bar to hold for the REST of the run, which is right for
    a phase transition but makes the value depend on the LOGGING RATE whenever
    the curve is not monotone -- every extra sample point is another chance to
    observe a dip and push the answer later. Measured on setup C, one seed: the
    identical trajectory scored 15,200 at log_every=200 and 59,350 at
    log_every=50, because the coarse run never sampled the collapse in between.
    """

    def test_first_cross_ignores_a_later_dip_that_t_grok_punishes(self):
        steps = list(range(0, 800, 100))
        test = [1.0, 1.0, 96.0, 20.0, 96.0, 97.0, 98.0, 99.0]
        assert compute_t_first_cross(steps, test, 95.0) == 200
        # t_grok waits for the crossing that is never undone
        assert compute_t_grok(steps, test, threshold=95.0) == 400

    def test_never_crossed(self):
        steps = list(range(0, 400, 100))
        assert compute_t_first_cross(steps, [1.0, 2.0, 3.0, 4.0],
                                     95.0) == float("inf")

    def test_dips_counted_only_after_the_first_crossing(self):
        steps = list(range(0, 800, 100))
        # the two sub-bar points BEFORE the crossing must not count
        test = [1.0, 2.0, 96.0, 20.0, 96.0, 30.0, 98.0, 99.0]
        assert count_post_cross_dips(steps, test, 95.0) == 2

    def test_no_dips_when_the_transition_holds(self):
        steps = list(range(0, 500, 100))
        assert count_post_cross_dips(steps, [1.0, 96.0, 97.0, 98.0, 99.0],
                                     95.0) == 0

    def test_no_dips_when_it_never_crossed(self):
        """Zero must mean "held", not "never got there" -- read next to t_grok."""
        steps = list(range(0, 400, 100))
        assert count_post_cross_dips(steps, [1.0] * 4, 95.0) == 0

    def test_both_land_in_the_result_row_at_the_run_threshold(self):
        steps = list(range(0, 600, 100))
        test = [1.0, 91.0, 50.0, 91.0, 92.0, 93.0]
        out = extract_grokking_results(
            {"epoch": steps, "train_acc": [100.0] * 6, "test_acc": test},
            threshold=90.0)
        assert out["t_first_cross"] == 100
        assert out["post_grok_dips"] == 1
        assert out["t_grok"] == 300

    def test_sampling_rate_moves_t_grok_but_not_the_first_crossing(self):
        """The property that motivates the metric, as a regression test.

        One trajectory, two logging rates. The coarse run simply never samples
        the epoch where accuracy collapsed, so it reports grokking 200 steps
        earlier -- a difference in the instrument, not in the model.
        """
        fine_steps = [0, 100, 200, 300, 400, 500]
        fine_acc = [1.0, 96.0, 20.0, 96.0, 97.0, 98.0]
        coarse_steps = [0, 100, 300, 400, 500]          # the dip goes unsampled
        coarse_acc = [1.0, 96.0, 96.0, 97.0, 98.0]

        assert compute_t_grok(fine_steps, fine_acc, 95.0) == 300
        assert compute_t_grok(coarse_steps, coarse_acc, 95.0) == 100
        assert (compute_t_first_cross(fine_steps, fine_acc, 95.0)
                == compute_t_first_cross(coarse_steps, coarse_acc, 95.0) == 100)
        # and the dip count is what tells you the two are not the same run
        assert count_post_cross_dips(fine_steps, fine_acc, 95.0) == 1
        assert count_post_cross_dips(coarse_steps, coarse_acc, 95.0) == 0
