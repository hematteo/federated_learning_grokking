"""Tests for core/ package: Config, GrokNet, dataset, metrics, utils."""

import math
import pytest
import torch
import numpy as np

from fedgrok.core.config import Config
from fedgrok.models.groknet import GrokNet
from fedgrok.data.modular import (
    TASKS, make_dataset, build_encoded_grid, task_operands, DEGENERATE_TASKS,
)
from fedgrok.metrics.fourier import weight_norms, gradient_norms, compute_ipr, compute_accuracy
from fedgrok.core.utils import (
    get_device, make_optimizer, make_targets_onehot, check_decay_stability,
)


# ── Config ────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.p == 97
        assert cfg.task == "addition"
        assert cfg.alpha == 0.5
        assert cfg.hidden_width == 100
        assert cfg.activation == "quadratic"
        assert cfg.optimizer == "gd"
        assert cfg.lr == 50.0
        assert cfg.weight_decay == 0.0
        assert cfg.momentum == 0.0
        assert cfg.output_dir == "results/baselines/centralized"

    def test_no_dead_cli_default_machinery(self):
        """`apply_adamw_defaults` and its _*_set flags are gone.

        They were v1 argparse machinery: the CLI set `_lr_set` when the user
        passed --lr, and the method filled AdamW defaults for the rest. Nothing
        on the manifest path ever called it, so an adamw spec that omitted `lr`
        silently inherited GD's 50.0. It could not be wired in either -- a spec
        has no "was it set" flag -- so the check moved into build_config, where
        the spec is still visible. See TestAdamWNeedsAnExplicitLR.
        """
        cfg = Config(optimizer="adamw")
        assert not hasattr(cfg, "apply_adamw_defaults")
        for flag in ("_lr_set", "_wd_set", "_epochs_set"):
            assert not hasattr(cfg, flag)


class TestAdamWNeedsAnExplicitLR:
    """An AdamW spec must state its learning rate.

    Config's lr=50.0 is GD's (Gromov's) and diverges under AdamW, and nothing
    downstream catches the mismatch: check_decay_stability returns early at
    weight_decay=0, so the run proceeds, diverges to NaN and banks as an
    ordinary censored result.
    """

    def test_adamw_without_lr_raises(self):
        from fedgrok.manifest import build_config
        with pytest.raises(ValueError, match="must set `lr`"):
            build_config({"mode": "centralized", "optimizer": "adamw", "p": 7})

    def test_adamw_with_lr_is_fine(self):
        from fedgrok.manifest import build_config
        cfg = build_config({"mode": "centralized", "optimizer": "adamw",
                            "lr": 1e-3, "p": 7})
        assert cfg.lr == 1e-3

    def test_gd_may_rely_on_the_default(self):
        """GD's default IS 50.0, so omitting lr there is the intended usage."""
        from fedgrok.manifest import build_config
        assert build_config({"mode": "centralized", "p": 7}).lr == 50.0

    def test_the_guard_covers_federated_specs_too(self):
        from fedgrok.manifest import build_config
        with pytest.raises(ValueError, match="must set `lr`"):
            build_config({"mode": "federated", "optimizer": "adamw", "p": 7})

    def test_every_banked_adamw_manifest_spec_satisfies_it(self):
        """The guard closes a trap; it must not invalidate banked work."""
        import glob
        from fedgrok.manifest import load_manifest
        missing = [(m, s.get("id")) for m in glob.glob("manifests/*.jsonl")
                   for s in load_manifest(m)
                   if s.get("optimizer") == "adamw" and "lr" not in s]
        assert missing == []


# ── GrokNet ───────────────────────────────────────────────────────────────────

class TestGrokNet:
    def test_output_shape(self, small_model, random_batch):
        out = small_model(random_batch)
        assert out.shape == (10, 7)

    def test_no_bias_parameters(self, small_model):
        param_names = [name for name, _ in small_model.named_parameters()]
        assert param_names == ["W1", "W2"]

    def test_weight_shapes(self, small_model):
        assert small_model.W1.shape == (16, 14)  # (N, 2p)
        assert small_model.W2.shape == (7, 16)   # (P, N)

    def test_mean_field_init_scale(self):
        """W1 ~ N(0, 1/D), W2 ~ N(0, 1/N^2)."""
        torch.manual_seed(0)
        p, N = 97, 200
        model = GrokNet(2 * p, N, p)
        # W1 std should be close to 1/sqrt(D) = 1/sqrt(194) ≈ 0.0718
        w1_std = model.W1.data.std().item()
        assert abs(w1_std - 1.0 / math.sqrt(2 * p)) < 0.01
        # W2 std should be close to 1/N = 0.005
        w2_std = model.W2.data.std().item()
        assert abs(w2_std - 1.0 / N) < 0.002

    @pytest.mark.parametrize("activation", ["quadratic", "relu", "gelu", "abs", "quartic"])
    def test_all_activations_produce_output(self, activation):
        model = GrokNet(14, 16, 7, activation=activation)
        x = torch.randn(5, 14)
        out = model(x)
        assert out.shape == (5, 7)
        assert torch.isfinite(out).all()

    def test_invalid_activation_raises(self):
        model = GrokNet(14, 16, 7, activation="invalid")
        with pytest.raises(ValueError, match="Unknown activation"):
            model(torch.randn(1, 14))

    def test_quadratic_activation_is_squaring(self):
        model = GrokNet(14, 16, 7, activation="quadratic")
        h = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = model._activate(h)
        expected = torch.tensor([4.0, 1.0, 0.0, 1.0, 4.0])
        assert torch.allclose(result, expected)

    def test_deterministic_with_seed(self):
        torch.manual_seed(42)
        m1 = GrokNet(14, 16, 7)
        w1 = m1.W1.data.clone()

        torch.manual_seed(42)
        m2 = GrokNet(14, 16, 7)
        assert torch.equal(m1.W1.data, m2.W1.data)
        assert torch.equal(m1.W2.data, m2.W2.data)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TestSplitGuard:
    """alpha must leave both sides of the split non-empty.

    alpha=1.0 trains on the whole grid and leaves no test set. Nothing
    downstream fails loudly on that: compute_accuracy divides by zero, every
    test point is NaN, and NaN used to pass compute_t_grok's sustained-crossing
    scan, banking the run as grokked=True at t_grok=0.
    """

    @pytest.mark.parametrize("alpha", [1.0, 1.5, 0.0, -0.1])
    def test_degenerate_alpha_raises(self, alpha):
        from fedgrok.data.modular import split_indices
        with pytest.raises(ValueError, match="alpha"):
            split_indices(100, alpha, seed=0)

    def test_alpha_too_small_for_the_grid_raises(self):
        # 0.001 * 100 truncates to 0 training samples.
        from fedgrok.data.modular import split_indices
        with pytest.raises(ValueError, match="empty"):
            split_indices(100, 0.001, seed=0)

    def test_valid_alpha_is_unchanged(self):
        """The guard must not perturb the RNG stream banked runs depend on."""
        from fedgrok.data.modular import split_indices
        train, test = split_indices(100, 0.5, seed=0)
        expected = np.random.RandomState(0).permutation(100)
        assert np.array_equal(train, expected[:50])
        assert np.array_equal(test, expected[50:])

    def test_federated_split_leaves_the_rng_where_partitioners_expect_it(self):
        """split_indices draws exactly one permutation, guard or no guard.

        make_federated_datasets hands the SAME RandomState to the partitioners
        afterwards, so any change in how many draws the split consumes would
        silently re-assign every IID and Dirichlet shard.
        """
        from fedgrok.data.modular import split_indices
        rng = np.random.RandomState(0)
        split_indices(100, 0.5, rng=rng)
        after_split = rng.permutation(10)

        reference = np.random.RandomState(0)
        reference.permutation(100)          # the one draw the split is allowed
        assert np.array_equal(after_split, reference.permutation(10))


class TestDataset:
    def test_shapes(self, small_cfg):
        x_train, y_train, x_test, y_test = make_dataset(small_cfg)
        p = small_cfg.p
        n_total = p * p
        n_train = int(small_cfg.alpha * n_total)
        n_test = n_total - n_train

        assert x_train.shape == (n_train, 2 * p)
        assert y_train.shape == (n_train,)
        assert x_test.shape == (n_test, 2 * p)
        assert y_test.shape == (n_test,)

    def test_label_range(self, small_cfg):
        _, y_train, _, y_test = make_dataset(small_cfg)
        for y in [y_train, y_test]:
            assert y.min() >= 0
            assert y.max() < small_cfg.p
            assert y.dtype == torch.long

    def test_onehot_input_structure(self, small_cfg):
        """Each row of x should have exactly 2 ones (one-hot for n and m)."""
        x_train, _, _, _ = make_dataset(small_cfg)
        row_sums = x_train.sum(dim=1)
        assert torch.allclose(row_sums, torch.full_like(row_sums, 2.0))

    def test_no_train_test_overlap(self, small_cfg):
        """Train and test sets should partition the full p^2 dataset."""
        x_train, _, x_test, _ = make_dataset(small_cfg)
        n_total = small_cfg.p ** 2
        assert len(x_train) + len(x_test) == n_total

    def test_seed_reproducibility(self, small_cfg):
        d1 = make_dataset(small_cfg)
        d2 = make_dataset(small_cfg)
        for a, b in zip(d1, d2):
            assert torch.equal(a, b)

    def test_different_seed_gives_different_split(self, small_cfg):
        cfg2 = Config(p=small_cfg.p, seed=99, hidden_width=16)
        x1, _, _, _ = make_dataset(small_cfg)
        x2, _, _, _ = make_dataset(cfg2)
        # Different seeds should produce different train sets
        assert not torch.equal(x1, x2)

    @pytest.mark.parametrize("task_name", list(TASKS.keys()))
    def test_all_tasks_produce_valid_labels(self, task_name):
        cfg = Config(p=7, task=task_name, hidden_width=16)
        _, y_train, _, y_test = make_dataset(cfg)
        for y in [y_train, y_test]:
            assert y.min() >= 0
            assert y.max() < 7

    def test_addition_correctness(self):
        """Spot-check: for addition mod p, verify labels match (n+m) mod p.

        Reads the unsplit grid directly. This used to ask make_dataset for
        alpha=1.0 to get every pair onto the train side, which split_indices now
        rejects -- alpha=1.0 leaves no test set, and a run with no test set
        cannot measure grokking. The grid is what this test actually wants.
        """
        x, y, _nn, _mm = build_encoded_grid("addition", 5)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        p = 5
        for i in range(len(x)):
            # Decode n and m from one-hot
            n = x[i, :p].argmax().item()
            m = x[i, p:].argmax().item()
            expected = (n + m) % p
            assert y[i].item() == expected, f"n={n}, m={m}: got {y[i].item()}, expected {expected}"


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_weight_norms_returns_positive(self, small_model):
        wn = weight_norms(small_model)
        assert "weight_norm_layer1" in wn
        assert "weight_norm_layer2" in wn
        assert wn["weight_norm_layer1"] > 0
        assert wn["weight_norm_layer2"] > 0

    def test_weight_norms_matches_manual(self, small_model):
        wn = weight_norms(small_model)
        expected_w1 = small_model.W1.data.norm().item()
        expected_w2 = small_model.W2.data.norm().item()
        assert abs(wn["weight_norm_layer1"] - expected_w1) < 1e-6
        assert abs(wn["weight_norm_layer2"] - expected_w2) < 1e-6

    def test_gradient_norms_before_backward(self, small_model):
        """Before backward, gradients don't exist — should return empty."""
        gn = gradient_norms(small_model)
        assert gn == {}

    def test_gradient_norms_after_backward(self, trained_model_with_grads):
        gn = gradient_norms(trained_model_with_grads)
        assert "grad_norm_layer1" in gn
        assert "grad_norm_layer2" in gn
        assert gn["grad_norm_layer1"] > 0
        assert gn["grad_norm_layer2"] > 0

    def test_compute_ipr_returns_positive(self, small_model):
        ipr = compute_ipr(small_model)
        assert "ipr" in ipr
        assert ipr["ipr"] > 0

    def test_compute_ipr_bounded(self, small_model):
        """IPR should be between 0 and 1 (normalized)."""
        ipr_val = compute_ipr(small_model)["ipr"]
        assert 0 < ipr_val <= 1.0

    def test_compute_accuracy_perfect(self):
        logits = torch.tensor([[10.0, 0.0, 0.0],
                               [0.0, 10.0, 0.0],
                               [0.0, 0.0, 10.0]])
        targets = torch.tensor([0, 1, 2])
        assert compute_accuracy(logits, targets) == 100.0

    def test_compute_accuracy_zero(self):
        logits = torch.tensor([[10.0, 0.0, 0.0],
                               [10.0, 0.0, 0.0],
                               [10.0, 0.0, 0.0]])
        targets = torch.tensor([1, 2, 2])
        assert compute_accuracy(logits, targets) == 0.0

    def test_compute_accuracy_partial(self):
        logits = torch.tensor([[10.0, 0.0],
                               [0.0, 10.0],
                               [10.0, 0.0],
                               [0.0, 10.0]])
        targets = torch.tensor([0, 1, 1, 0])
        assert compute_accuracy(logits, targets) == 50.0

    def test_fourier_spectrum_shape(self, small_model):
        from fedgrok.metrics.fourier import fourier_spectrum
        spec = fourier_spectrum(small_model)
        assert "spectrum" in spec
        assert len(spec["spectrum"]) == small_model.N
        assert len(spec["spectrum"][0]) == small_model.P

    def test_fourier_spectrum_nonnegative(self, small_model):
        from fedgrok.metrics.fourier import fourier_spectrum
        spec = fourier_spectrum(small_model)
        for row in spec["spectrum"]:
            assert all(v >= 0 for v in row)


# ── Utils ─────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_get_device_returns_device(self):
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "mps", "cpu")

    def test_make_optimizer_sgd(self, small_model, small_cfg):
        opt = make_optimizer(small_model, small_cfg)
        assert isinstance(opt, torch.optim.SGD)

    def test_make_optimizer_adamw(self, small_model):
        cfg = Config(optimizer="adamw", lr=1e-4, weight_decay=1.0, hidden_width=16)
        opt = make_optimizer(small_model, cfg)
        assert isinstance(opt, torch.optim.AdamW)

    def test_make_optimizer_invalid(self, small_model):
        cfg = Config(optimizer="invalid", hidden_width=16)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            make_optimizer(small_model, cfg)

    def test_make_optimizer_respects_lr(self, small_model):
        cfg = Config(lr=123.0, hidden_width=16)
        opt = make_optimizer(small_model, cfg)
        assert opt.param_groups[0]["lr"] == 123.0

    def test_make_targets_onehot_shape(self):
        labels = torch.tensor([0, 3, 6])
        oh = make_targets_onehot(labels, 7)
        assert oh.shape == (3, 7)

    def test_make_targets_onehot_values(self):
        labels = torch.tensor([0, 2, 4])
        oh = make_targets_onehot(labels, 5)
        # Each row should be one-hot
        assert torch.equal(oh.sum(dim=1), torch.ones(3))
        assert oh[0, 0] == 1.0
        assert oh[1, 2] == 1.0
        assert oh[2, 4] == 1.0
        # All other entries should be 0
        assert oh[0, 1:].sum() == 0.0

    def test_make_targets_onehot_dtype(self):
        labels = torch.tensor([0, 1])
        oh = make_targets_onehot(labels, 3)
        assert oh.dtype == torch.float32

    def test_make_targets_onehot_follows_label_device(self):
        """One-hot must be built on the labels' device, not unconditionally CPU."""
        device = get_device()
        labels = torch.tensor([0, 2, 4], device=device)
        oh = make_targets_onehot(labels, 5)
        assert oh.device.type == labels.device.type


# ── Weight-decay stability guard ──────────────────────────────────────────────


class TestDecayStability:
    """Guards against the exp5 weight-decay defect.

    Both SGD (coupled) and AdamW (decoupled) shrink weights by (1 - lr*wd) per
    step, so lr*wd is the meaningful quantity. exp5 originally ran lr=50 with
    wd in [0.01, 1.0] -- lr*wd in [0.5, 50] -- destroying every model before it
    could learn, which is why all those cells reported T_grok=inf.
    """

    @pytest.mark.parametrize("weight_decay", [0.01, 0.1, 1.0])
    def test_rejects_original_exp5_values(self, weight_decay):
        """The three values that invalidated the original exp5 WD arms."""
        with pytest.raises(ValueError):
            check_decay_stability(50.0, weight_decay)

    def test_rejects_destructive_but_non_divergent(self):
        """lr*wd = 0.5 does not diverge, but halves every weight every step."""
        with pytest.raises(ValueError, match="Destructive"):
            check_decay_stability(50.0, 0.01)

    def test_rejects_divergent(self):
        with pytest.raises(ValueError, match="Divergent"):
            check_decay_stability(50.0, 1.0)

    @pytest.mark.parametrize("weight_decay", [0.0, 2e-7, 2e-6, 2e-5, 2e-4])
    def test_accepts_corrected_sweep(self, weight_decay):
        """The replacement sweep: lr*wd in {0, 1e-5, 1e-4, 1e-3, 1e-2} at lr=50."""
        check_decay_stability(50.0, weight_decay)

    @pytest.mark.parametrize("lr,weight_decay", [(1e-3, 1.0), (1e-4, 1.0)])
    def test_accepts_published_grokking_configs(self, lr, weight_decay):
        """Nanda/Power (lr*wd=1e-3) and Omnigrok (lr*wd=1e-4) must pass."""
        check_decay_stability(lr, weight_decay)

    def test_warns_on_aggressive_decay(self):
        with pytest.warns(RuntimeWarning, match="Aggressive"):
            check_decay_stability(50.0, 1e-3)  # lr*wd = 0.05 -> 20-step timescale

    def test_make_optimizer_enforces_guard(self):
        """The guard must fire through the optimizer factory, not just directly."""
        model = GrokNet(input_dim=14, hidden_width=8, output_dim=7)
        cfg = Config(optimizer="gd", lr=50.0, weight_decay=1.0)
        with pytest.raises(ValueError):
            make_optimizer(model, cfg)


# ── Task domains ──────────────────────────────────────────────────────────────


class TestTaskDomains:
    """Each task's input grid must contain only well-defined pairs.

    division previously mapped m=0 to a fabricated label 0, injecting p spurious
    samples that all shared one class and over-represented it ~2x.
    """

    def test_division_grid_excludes_zero_divisor(self):
        _, _, _, mm = build_encoded_grid("division", 7)
        assert mm.min() == 1

    def test_division_grid_has_p_times_p_minus_one_samples(self):
        p = 7
        _, labels, _, _ = build_encoded_grid("division", p)
        assert len(labels) == p * (p - 1)

    def test_division_labels_are_uniform(self):
        """Every residue appears exactly p-1 times once m=0 is excluded."""
        p = 7
        _, labels, _, _ = build_encoded_grid("division", p)
        counts = np.bincount(labels, minlength=p)
        assert set(counts.tolist()) == {p - 1}

    def test_division_raises_on_zero_divisor(self):
        with pytest.raises(ValueError, match="undefined at m=0"):
            TASKS["division"](3, 0, 7)

    @pytest.mark.parametrize("task", ["addition", "subtraction", "x2_plus_y2",
                                      "multiplication", "x2_y2_xy"])
    def test_unrestricted_tasks_use_the_full_grid(self, task):
        p = 7
        _, labels, _, _ = build_encoded_grid(task, p)
        assert len(labels) == p * p

    def test_multiplication_is_flagged_as_degenerate(self):
        """It is Z_{p-1} in disguise and keeps 2p-1 zero-product pairs."""
        assert "multiplication" in DEGENERATE_TASKS

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            task_operands("not_a_task", 7)


# ── Minibatching ──────────────────────────────────────────────────────────────


class TestMinibatching:
    """batch_size=0 is full-batch (Gromov default); >0 is minibatch SGD.

    Full-batch bit-identity is verified out of band against the pre-minibatch
    commit; here we check the field default and that the minibatch path runs and
    produces a well-formed, finite history.
    """

    def test_batch_size_defaults_to_full_batch(self):
        assert Config().batch_size == 0

    def test_minibatch_centralized_runs(self):
        from fedgrok.training.centralized import train
        cfg = Config(task="addition", p=17, alpha=0.5, seed=42, hidden_width=32,
                     loss="mse", optimizer="gd", lr=1.0, batch_size=32,
                     epochs=20, log_every=5, output_dir="/tmp/test_mb")
        history, _ = train(cfg)
        assert len(history["epoch"]) == 5           # epochs 0,5,10,15,20
        assert all(x == x for x in history["train_loss"])   # all finite (no NaN)

    def test_minibatch_and_full_batch_both_learn_direction(self):
        """Sanity: both paths reduce training loss over a short run."""
        from fedgrok.training.centralized import train
        for bs in (0, 32):
            cfg = Config(task="addition", p=17, alpha=0.5, seed=0, hidden_width=32,
                         loss="mse", optimizer="gd", lr=1.0, batch_size=bs,
                         epochs=40, log_every=10, output_dir="/tmp/test_mb2")
            history, _ = train(cfg)
            assert history["train_loss"][-1] < history["train_loss"][0]
