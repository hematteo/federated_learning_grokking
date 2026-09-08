"""Tests for federated/train.py: weight serialization, dataset cache, GrokClient, fed_train."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from fedgrok.models.groknet import GrokNet
from fedgrok.core.utils import make_targets_onehot, make_optimizer
from fedgrok.core.fed_config import FedConfig
from fedgrok.data.partition import make_federated_datasets
from fedgrok.training.federated import (
    _model_to_ndarrays,
    _ndarrays_to_state_dict,
    _make_model,
    _cfg_to_fit_config,
    _fit_config_to_cfg,
    _get_cached_datasets,
    _dataset_cache,
    _client_cache,
    GrokClient,
    fed_train,
)

SMALL_P = 7
# lr=1.0 is stable for p=7; default lr=50 is tuned for p=97 and diverges here
STABLE_LR = 1.0


# ── Weight serialization ─────────────────────────────────────────────────────

class TestWeightSerialization:
    def test_model_to_ndarrays_returns_list(self, small_model):
        arrays = _model_to_ndarrays(small_model)
        assert isinstance(arrays, list)
        assert len(arrays) == 2  # W1, W2

    def test_model_to_ndarrays_shapes(self, small_model):
        arrays = _model_to_ndarrays(small_model)
        assert arrays[0].shape == (16, 14)  # W1: (N, 2p)
        assert arrays[1].shape == (7, 16)   # W2: (P, N)

    def test_model_to_ndarrays_are_numpy(self, small_model):
        arrays = _model_to_ndarrays(small_model)
        for arr in arrays:
            assert isinstance(arr, np.ndarray)

    def test_ndarrays_to_state_dict_roundtrip(self, small_model):
        """model → ndarrays → state_dict should preserve weights exactly."""
        original_w1 = small_model.W1.data.clone()
        original_w2 = small_model.W2.data.clone()

        arrays = _model_to_ndarrays(small_model)
        state_dict = _ndarrays_to_state_dict(arrays, small_model)

        assert "W1" in state_dict
        assert "W2" in state_dict
        assert torch.allclose(state_dict["W1"], original_w1)
        assert torch.allclose(state_dict["W2"], original_w2)

    def test_full_roundtrip_through_model(self, small_model):
        """model → ndarrays → state_dict → new model: weights should match."""
        original_w1 = small_model.W1.data.clone()
        original_w2 = small_model.W2.data.clone()

        arrays = _model_to_ndarrays(small_model)

        new_model = GrokNet(14, 16, 7)
        state_dict = _ndarrays_to_state_dict(arrays, new_model)
        new_model.load_state_dict(state_dict)

        assert torch.allclose(new_model.W1.data, original_w1)
        assert torch.allclose(new_model.W2.data, original_w2)

    def test_roundtrip_preserves_forward_pass(self, small_model, random_batch):
        """Serialization roundtrip should produce identical model outputs."""
        original_out = small_model(random_batch)

        arrays = _model_to_ndarrays(small_model)
        new_model = GrokNet(14, 16, 7, activation=small_model.activation)
        state_dict = _ndarrays_to_state_dict(arrays, new_model)
        new_model.load_state_dict(state_dict)
        new_out = new_model(random_batch)

        assert torch.allclose(original_out, new_out, atol=1e-6)

    def test_model_to_ndarrays_on_cpu(self):
        """_model_to_ndarrays should always return CPU arrays."""
        model = GrokNet(14, 16, 7)
        # Even on MPS/CUDA, should work
        arrays = _model_to_ndarrays(model)
        for arr in arrays:
            assert isinstance(arr, np.ndarray)


# ── _make_model ───────────────────────────────────────────────────────────────

class TestMakeModel:
    def test_creates_groknet(self, small_fed_cfg):
        model = _make_model(small_fed_cfg)
        assert isinstance(model, GrokNet)

    def test_correct_dimensions(self, small_fed_cfg):
        model = _make_model(small_fed_cfg)
        assert model.D == 2 * small_fed_cfg.p
        assert model.N == small_fed_cfg.hidden_width
        assert model.P == small_fed_cfg.p

    def test_model_on_cpu(self, small_fed_cfg):
        model = _make_model(small_fed_cfg)
        assert next(model.parameters()).device == torch.device("cpu")


# ── Dataset cache ─────────────────────────────────────────────────────────────

class TestDatasetCache:
    def setup_method(self):
        """Clear the global cache before each test."""
        _dataset_cache.clear()

    def test_cache_hit(self):
        cfg = FedConfig(p=SMALL_P, num_clients=3, hidden_width=16, seed=42)
        result1 = _get_cached_datasets(cfg)
        result2 = _get_cached_datasets(cfg)
        # Should be the exact same objects
        assert result1 is result2

    def test_cache_miss_on_different_partition(self):
        cfg1 = FedConfig(p=SMALL_P, num_clients=3, partition="iid", hidden_width=16)
        cfg2 = FedConfig(p=SMALL_P, num_clients=3, partition="operand", hidden_width=16)
        result1 = _get_cached_datasets(cfg1)
        result2 = _get_cached_datasets(cfg2)
        assert result1 is not result2

    def test_cache_miss_on_different_num_clients(self):
        cfg1 = FedConfig(p=SMALL_P, num_clients=3, hidden_width=16)
        cfg2 = FedConfig(p=SMALL_P, num_clients=5, hidden_width=16)
        result1 = _get_cached_datasets(cfg1)
        result2 = _get_cached_datasets(cfg2)
        assert result1 is not result2

    def test_cache_miss_on_different_seed(self):
        cfg1 = FedConfig(p=SMALL_P, num_clients=3, hidden_width=16, seed=1)
        cfg2 = FedConfig(p=SMALL_P, num_clients=3, hidden_width=16, seed=2)
        result1 = _get_cached_datasets(cfg1)
        result2 = _get_cached_datasets(cfg2)
        assert result1 is not result2

    def test_cache_miss_on_different_dirichlet_alpha(self):
        cfg1 = FedConfig(p=SMALL_P, num_clients=3, partition="dirichlet",
                         dirichlet_alpha=0.1, hidden_width=16)
        cfg2 = FedConfig(p=SMALL_P, num_clients=3, partition="dirichlet",
                         dirichlet_alpha=1.0, hidden_width=16)
        result1 = _get_cached_datasets(cfg1)
        result2 = _get_cached_datasets(cfg2)
        assert result1 is not result2

    def test_cache_key_ignores_training_params(self):
        """Cache key should NOT include lr, optimizer, etc. — only data params."""
        cfg1 = FedConfig(p=SMALL_P, num_clients=3, hidden_width=16, lr=50.0)
        cfg2 = FedConfig(p=SMALL_P, num_clients=3, hidden_width=16, lr=0.001)
        result1 = _get_cached_datasets(cfg1)
        result2 = _get_cached_datasets(cfg2)
        # Same data params, different training params → should be same object
        assert result1 is result2


# ── GrokClient ────────────────────────────────────────────────────────────────

class TestMinibatchShuffleSeeding:
    """Client minibatch order must be reproducible.

    torch.manual_seed(cfg.seed) in fed_train runs in the DRIVER process; every
    client is a separate Ray actor whose default generator is seeded
    non-deterministically. The shuffles in GrokClient.fit were therefore the one
    uncontrolled RNG source in the project, affecting all 94 banked setup-E
    (MNIST) federated runs -- asymmetrically, since centralized MNIST *is* seeded.
    """

    def test_same_run_client_and_round_gives_the_same_shuffle(self):
        from fedgrok.training.federated import _shuffle_seed
        assert _shuffle_seed(42, 3, 7) == _shuffle_seed(42, 3, 7)

    @pytest.mark.parametrize("a,b", [
        ((42, 0, 1), (42, 1, 0)),      # client vs round must not alias
        ((42, 0, 0), (43, 0, 0)),      # run seed separates
        ((42, 0, 0), (42, 1, 0)),      # client separates
        ((42, 0, 0), (42, 0, 1)),      # round separates
    ])
    def test_axes_do_not_alias(self, a, b):
        from fedgrok.training.federated import _shuffle_seed
        assert _shuffle_seed(*a) != _shuffle_seed(*b)

    def test_seed_is_in_range_for_torch_generator(self):
        from fedgrok.training.federated import _shuffle_seed
        for args in [(0, 0, 0), (999999, 50, 200000)]:
            torch.Generator().manual_seed(_shuffle_seed(*args))

    def test_minibatch_client_fit_is_reproducible(self):
        """Two identical fit() calls must produce identical weights."""
        _dataset_cache.clear(); _client_cache.clear()
        cfg = FedConfig(p=SMALL_P, num_clients=3, local_epochs=2, batch_size=4,
                        hidden_width=16, partition="iid", seed=42, lr=STABLE_LR)
        fit_config = _cfg_to_fit_config(cfg, server_round=1)
        params = _model_to_ndarrays(_make_model(cfg))

        first, _, _ = GrokClient(partition_id=0).fit(
            [p.copy() for p in params], fit_config)
        _client_cache.clear()          # force a cold client, as a re-run would be
        second, _, _ = GrokClient(partition_id=0).fit(
            [p.copy() for p in params], fit_config)
        for a, b in zip(first, second):
            assert np.array_equal(a, b)

    def test_full_batch_path_does_not_touch_the_global_rng(self):
        """Banked full-batch runs must stay bit-identical.

        The shuffle generator is private and created only on the minibatch
        branch, so a full-batch fit must leave the process RNG exactly where it
        found it -- otherwise every banked full-batch federated run becomes
        unreproducible.
        """
        _dataset_cache.clear(); _client_cache.clear()
        cfg = FedConfig(p=SMALL_P, num_clients=3, local_epochs=2, batch_size=0,
                        hidden_width=16, partition="iid", seed=42, lr=STABLE_LR)
        fit_config = _cfg_to_fit_config(cfg, server_round=1)
        params = _model_to_ndarrays(_make_model(cfg))

        client = GrokClient(partition_id=0)
        # Warm the cache first: building the client's model draws from the global
        # RNG, which is expected and happens once per client. The steady state --
        # every round after that -- is what must leave the stream alone.
        client.fit(params, fit_config)

        torch.manual_seed(1234)
        before = torch.get_rng_state()
        client.fit(params, _cfg_to_fit_config(cfg, server_round=2))
        assert torch.equal(before, torch.get_rng_state())


class TestGrokClient:
    def _make_client_config(self, partition_id=0, partition="iid"):
        """Create a fit config dict mimicking what Flower sends."""
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, local_epochs=2,
            hidden_width=16, partition=partition, seed=42, lr=STABLE_LR,
        )
        return cfg, _cfg_to_fit_config(cfg, server_round=1)

    def _make_initial_parameters(self, cfg):
        """Create initial model parameters as ndarrays."""
        model = _make_model(cfg)
        return _model_to_ndarrays(model)

    def test_fit_returns_correct_structure(self):
        _dataset_cache.clear()
        cfg, fit_config = self._make_client_config()
        params = self._make_initial_parameters(cfg)

        client = GrokClient(partition_id=0)
        result = client.fit(params, fit_config)

        assert len(result) == 3
        updated_params, num_samples, metrics = result
        assert isinstance(updated_params, list)
        assert len(updated_params) == 2  # W1, W2
        assert isinstance(num_samples, int)
        assert num_samples > 0
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_fit_modifies_weights(self):
        """After local training, weights should be different from initial."""
        _dataset_cache.clear()
        cfg, fit_config = self._make_client_config()
        params = self._make_initial_parameters(cfg)

        client = GrokClient(partition_id=0)
        updated_params, _, _ = client.fit(params, fit_config)

        # At least one weight array should have changed
        changed = any(
            not np.allclose(orig, updated, atol=1e-10)
            for orig, updated in zip(params, updated_params)
        )
        assert changed, "Weights should change after local training"

    def test_fit_preserves_shapes(self):
        _dataset_cache.clear()
        cfg, fit_config = self._make_client_config()
        params = self._make_initial_parameters(cfg)

        client = GrokClient(partition_id=0)
        updated_params, _, _ = client.fit(params, fit_config)

        for orig, updated in zip(params, updated_params):
            assert orig.shape == updated.shape

    def test_different_partitions_produce_different_updates(self):
        """Clients with different data should produce different weight updates."""
        _dataset_cache.clear()
        cfg, fit_config = self._make_client_config()
        params = self._make_initial_parameters(cfg)

        client0 = GrokClient(partition_id=0)
        client1 = GrokClient(partition_id=1)

        updated0, _, _ = client0.fit(params, fit_config)
        updated1, _, _ = client1.fit(params, fit_config)

        # Different partitions should produce different updates
        different = any(
            not np.allclose(a, b, atol=1e-10)
            for a, b in zip(updated0, updated1)
        )
        assert different, "Different clients should produce different weight updates"

    def test_fit_returns_correct_num_samples(self):
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, local_epochs=1,
            hidden_width=16, partition="iid", seed=42, lr=STABLE_LR,
        )
        fit_config = _cfg_to_fit_config(cfg, server_round=1)
        params = self._make_initial_parameters(cfg)

        # Get the actual partition size
        client_data, _, _, _, _ = make_federated_datasets(cfg)
        expected_size = len(client_data[0][1])

        client = GrokClient(partition_id=0)
        _, num_samples, _ = client.fit(params, fit_config)
        assert num_samples == expected_size

    def test_evaluate_returns_stub(self):
        client = GrokClient(partition_id=0)
        loss, num_samples, metrics = client.evaluate([], {})
        assert loss == 0.0
        assert num_samples == 0
        assert metrics == {}

    def test_fit_loss_is_finite(self):
        _dataset_cache.clear()
        cfg, fit_config = self._make_client_config()
        params = self._make_initial_parameters(cfg)

        client = GrokClient(partition_id=0)
        _, _, metrics = client.fit(params, fit_config)
        assert np.isfinite(metrics["loss"])
        assert 0 <= metrics["accuracy"] <= 100


# ── fed_train integration ─────────────────────────────────────────────────────

class TestFedTrain:
    def setup_method(self):
        _dataset_cache.clear()

    def test_returns_history_and_model(self):
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, model = fed_train(cfg)
        assert isinstance(history, dict)
        assert isinstance(model, GrokNet)

    def test_history_has_correct_keys(self):
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, _ = fed_train(cfg)
        # The core series every federated run must produce. Asserted as a SUBSET,
        # not equality: setup-dependent series (architecture-agnostic weight
        # norms, and the per-setup mechanistic probe) are added on top, and which
        # ones appear depends on the dataset/model. Equality here would mean any
        # new measurement breaks a test about the old ones.
        required_keys = {
            "round", "total_steps", "sequential_steps", "n_participating",
            "train_loss", "test_loss",
            "train_acc", "test_acc",
            "weight_norm_layer1", "weight_norm_layer2",
            "ipr",
            "mean_client_drift", "client_weight_divergence",
            # architecture-agnostic, so present on every setup
            "weight_norm_total", "weight_norm_first", "weight_norm_last",
        }
        assert required_keys <= set(history.keys()), (
            f"missing: {sorted(required_keys - set(history))}"
        )
        # Every series must stay in lockstep with `round`, or plots misalign.
        n = len(history["round"])
        ragged = {k: len(v) for k, v in history.items()
                  if isinstance(v, list) and len(v) != n}
        assert not ragged, f"ragged history series: {ragged}"

    def test_history_length_matches_rounds(self):
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=3, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, _ = fed_train(cfg)
        # Round 0 (initial eval) + rounds 1..3 = 4 entries
        assert len(history["round"]) == 4

    def test_total_steps_computation(self):
        local_epochs = 2
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=3, local_epochs=local_epochs,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, _ = fed_train(cfg)
        for i, (rnd, steps) in enumerate(zip(history["round"], history["total_steps"])):
            assert steps == rnd * local_epochs

    def test_final_model_not_initial(self):
        """Final model should have different weights than random init."""
        torch.manual_seed(42)
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=3, local_epochs=2,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, final_model = fed_train(cfg)

        # Create a fresh model with same init to compare
        torch.manual_seed(42)
        init_model = _make_model(cfg)

        # Weights should differ after training
        w1_changed = not torch.allclose(final_model.W1.data.cpu(), init_model.W1.data, atol=1e-8)
        w2_changed = not torch.allclose(final_model.W2.data.cpu(), init_model.W2.data, atol=1e-8)
        assert w1_changed or w2_changed, "Final model weights should differ from init"

    def test_losses_are_finite(self):
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, _ = fed_train(cfg)
        for loss in history["train_loss"] + history["test_loss"]:
            assert np.isfinite(loss)

    def test_accuracies_are_bounded(self):
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, _ = fed_train(cfg)
        for acc in history["train_acc"] + history["test_acc"]:
            assert 0 <= acc <= 100

    @pytest.mark.parametrize("partition", ["iid", "operand", "target", "dirichlet"])
    def test_all_partitions_run(self, partition):
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=1,
            hidden_width=16, partition=partition, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, model = fed_train(cfg)
        assert len(history["round"]) == 3  # round 0 + 2 rounds

    def test_saves_history_json(self, tmp_path):
        import json
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir=str(tmp_path),
        )
        fed_train(cfg)

        # Find the saved JSON
        json_files = list(tmp_path.glob("history_*.json"))
        assert len(json_files) == 1

        with open(json_files[0]) as f:
            saved = json.load(f)
        assert "round" in saved
        assert "test_acc" in saved

    def test_saves_weights_when_requested(self, tmp_path):
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir=str(tmp_path),
            save_weights=True,
        )
        fed_train(cfg)

        pt_files = list(tmp_path.glob("weights_*.pt"))
        assert len(pt_files) == 1

        # Verify saved weights can be loaded
        state_dict = torch.load(pt_files[0], weights_only=True)
        assert "W1" in state_dict
        assert "W2" in state_dict


# ── Step accounting ──────────────────────────────────────────────────────────


class TestStepAccounting:
    """total_steps must reflect actual gradient work, not rounds * E.

    The old accounting was `server_round * local_epochs`, which is correct only
    at full participation. Under partial participation it over-counted, since
    only a fraction of the clients (and hence of the data) trained each round.
    """

    def _run(self, fraction_train, tmp_path):
        from fedgrok.training.federated import fed_train, _dataset_cache
        _dataset_cache.clear()
        cfg = FedConfig(
            task="addition", p=7, alpha=0.5, seed=42, hidden_width=16,
            num_clients=5, num_rounds=5, local_epochs=3, lr=1.0,
            fraction_train=fraction_train, partition="iid",
            output_dir=str(tmp_path),
        )
        history, _ = fed_train(cfg)
        return history

    def test_full_participation_matches_rounds_times_epochs(self, tmp_path):
        """At C=1.0 the compute-matched axis coincides with rounds * E."""
        history = self._run(1.0, tmp_path)
        expected = [r * 3 for r in history["round"]]
        assert history["total_steps"] == pytest.approx(expected)
        assert history["sequential_steps"] == expected

    def test_partial_participation_counts_less_than_sequential(self, tmp_path):
        """At C<1.0 fewer samples are touched per round, so total_steps lags."""
        history = self._run(0.4, tmp_path)
        assert history["total_steps"][-1] < history["sequential_steps"][-1]
        # sequential_steps is participation-independent
        assert history["sequential_steps"] == [r * 3 for r in history["round"]]

    def test_participating_client_count_is_recorded(self, tmp_path):
        history = self._run(0.4, tmp_path)
        # Round 0 is the pre-training evaluation of the initial parameters.
        assert history["n_participating"][0] == 0
        assert all(n == 2 for n in history["n_participating"][1:])

    def test_total_steps_is_monotonic(self, tmp_path):
        history = self._run(0.4, tmp_path)
        steps = history["total_steps"]
        assert all(b >= a for a, b in zip(steps, steps[1:]))


# ── eval_every consistency ───────────────────────────────────────────────────


class TestEvalEvery:
    """eval_every must only subsample the curve, never change the dynamics.

    This is the permanent guard on the per-round caching: if the warm client
    model or the reused eval model leaked state across rounds, or if the
    step-accumulation were gated behind the eval-round check, the subsampled run
    would disagree with the full run at the rounds they share.
    """

    def _run(self, eval_every):
        from fedgrok.training.federated import fed_train, _dataset_cache
        _dataset_cache.clear()
        cfg = FedConfig(
            task="addition", p=17, alpha=0.5, seed=42, hidden_width=32,
            num_clients=4, num_rounds=20, local_epochs=5, lr=1.0,
            fraction_train=1.0, partition="dirichlet", dirichlet_alpha=0.3,
            eval_every=eval_every, output_dir="/tmp/test_eval_every",
        )
        history, _ = fed_train(cfg)
        return history

    def test_subsampled_matches_full_at_shared_rounds(self):
        full = self._run(1)
        sub = self._run(5)

        full_by_round = {r: i for i, r in enumerate(full["round"])}
        assert set(sub["round"]) <= set(full["round"])

        for j, r in enumerate(sub["round"]):
            i = full_by_round[r]
            for key in ("test_acc", "train_loss", "weight_norm_layer1",
                        "total_steps", "ipr"):
                assert sub[key][j] == pytest.approx(full[key][i], abs=1e-5), (
                    f"{key} at round {r}: eval_every=5 gave {sub[key][j]}, "
                    f"eval_every=1 gave {full[key][i]}"
                )

    def test_final_round_is_always_logged(self):
        """The last round must appear even when it is not a multiple of eval_every."""
        sub = self._run(7)  # 20 is not a multiple of 7
        assert sub["round"][-1] == 20


# ── Loss propagates to clients ────────────────────────────────────────────────


class TestLossPropagation:
    """cfg.loss and cfg.model must survive the fit-config round-trip to clients.

    _cfg_to_fit_config serialises the config to a scalar dict that Flower ships
    to each client, which rebuilds it with _fit_config_to_cfg. If loss/model are
    dropped there, a CE run would silently train its clients with the default
    MSE — the exact silent-field-drop the hand-maintained dict invites.
    """

    def test_loss_and_model_survive_fit_config_roundtrip(self):
        from fedgrok.training.federated import _cfg_to_fit_config, _fit_config_to_cfg
        cfg = FedConfig(p=97, loss="ce", model="groknet", optimizer="adamw",
                        lr=1e-3, num_clients=4, partition="iid")
        rebuilt = _fit_config_to_cfg(_cfg_to_fit_config(cfg, server_round=1))
        assert rebuilt.loss == "ce"
        assert rebuilt.model == "groknet"

    def test_client_uses_ce_target_shape(self):
        """A CE client's prepared target must be class indices, not one-hot."""
        from fedgrok.training.federated import _get_warm_client, _dataset_cache, _client_cache
        _dataset_cache.clear()
        _client_cache.clear()
        cfg = FedConfig(p=7, loss="ce", num_clients=2, partition="iid",
                        hidden_width=16, seed=42)
        # 5-tuple: the last slot is the (optionally compiled) training forward.
        _, _, y_target, y_local, _ = _get_warm_client(cfg, 0, torch.device("cpu"))
        assert y_target.dim() == 1            # class indices, not (n, p) one-hot
        assert y_target.dtype == torch.int64
        assert torch.equal(y_target, y_local)


# ── Fit-config round-trips every field ────────────────────────────────────────


class TestFitConfigCompleteness:
    """The generated fit-config must carry EVERY FedConfig field to the client.

    This is the permanent guard against the class of bug where a new config
    field (loss, model, batch_size, dataset, ...) is added but not threaded
    through the hand-maintained serialisation, silently mis-training clients.
    """

    def test_every_field_round_trips(self):
        import dataclasses
        from fedgrok.training.federated import _cfg_to_fit_config, _fit_config_to_cfg

        # A config with a non-default value in every field we can safely set.
        cfg = FedConfig(
            p=53, task="subtraction", dataset="s5", group_n=5, alpha=0.3, seed=7,
            model="transformer", hidden_width=64, n_layers=2, init_scale=3.0,
            activation="relu", loss="ce", optimizer="adamw", lr=1e-3,
            weight_decay=1.0, momentum=0.0, batch_size=64,
            num_clients=8, num_rounds=10, local_epochs=3, fraction_train=0.5,
            partition="dirichlet", dirichlet_alpha=0.1, proximal_mu=0.01,
            strategy="fedprox", server_lr=0.1, tau=1e-2, eval_every=5,
        )
        rebuilt = _fit_config_to_cfg(_cfg_to_fit_config(cfg, server_round=1))

        for field in dataclasses.fields(FedConfig):
            assert getattr(rebuilt, field.name) == getattr(cfg, field.name), (
                f"field {field.name!r} did not survive the fit-config round-trip"
            )

    def test_server_round_is_added_but_not_a_config_field(self):
        from fedgrok.training.federated import _cfg_to_fit_config, _fit_config_to_cfg
        fc = _cfg_to_fit_config(FedConfig(p=17), server_round=42)
        assert fc["server_round"] == 42
        # and reconstruction ignores it (not a FedConfig field)
        assert _fit_config_to_cfg(fc).p == 17


# ── Client placement knobs (VRAM) ────────────────────────────────────────────
# These control how many client processes hold a CUDA context at once. They are
# env vars rather than Config fields on purpose: a Config field would be hashed
# into the run id and orphan every banked run. The tests below pin that
# property, since it is the one that is expensive to get wrong.

class TestClientPlacement:
    """FEDGROK_CLIENT_CPU / FEDGROK_GPU_CLIENT_CAP."""

    def test_cap_unset_uses_every_client(self, monkeypatch):
        from fedgrok.training.federated import _gpu_client_cap
        monkeypatch.delenv("FEDGROK_GPU_CLIENT_CAP", raising=False)
        # Unset must reproduce the original 1/K allocation exactly.
        assert _gpu_client_cap(50) == 50

    def test_cap_limits_concurrency(self, monkeypatch):
        from fedgrok.training.federated import _gpu_client_cap
        monkeypatch.setenv("FEDGROK_GPU_CLIENT_CAP", "8")
        assert _gpu_client_cap(50) == 8

    def test_cap_never_exceeds_client_count(self, monkeypatch):
        from fedgrok.training.federated import _gpu_client_cap
        monkeypatch.setenv("FEDGROK_GPU_CLIENT_CAP", "64")
        # A cap above K would hand each client more than a whole GPU.
        assert _gpu_client_cap(10) == 10

    def test_cap_rejects_zero_and_negative(self, monkeypatch):
        from fedgrok.training.federated import _gpu_client_cap
        for bad in ("0", "-1"):
            monkeypatch.setenv("FEDGROK_GPU_CLIENT_CAP", bad)
            with pytest.raises(ValueError, match="positive integer"):
                _gpu_client_cap(10)

    def test_blank_cap_is_treated_as_unset(self, monkeypatch):
        from fedgrok.training.federated import _gpu_client_cap
        monkeypatch.setenv("FEDGROK_GPU_CLIENT_CAP", "   ")
        assert _gpu_client_cap(10) == 10

    def test_client_cpu_flag_forces_cpu(self, monkeypatch):
        from fedgrok.training.federated import _client_device, _clients_on_cpu
        monkeypatch.setenv("FEDGROK_CLIENT_CPU", "1")
        assert _clients_on_cpu() is True
        assert _client_device().type == "cpu"

    def test_client_cpu_accepts_word_forms(self, monkeypatch):
        from fedgrok.training.federated import _clients_on_cpu
        for truthy in ("1", "true", "TRUE", "yes"):
            monkeypatch.setenv("FEDGROK_CLIENT_CPU", truthy)
            assert _clients_on_cpu() is True
        for falsy in ("", "0", "no", "false"):
            monkeypatch.setenv("FEDGROK_CLIENT_CPU", falsy)
            assert _clients_on_cpu() is False

    def test_client_device_follows_get_device_when_unset(self, monkeypatch):
        from fedgrok.training.federated import _client_device
        from fedgrok.core.utils import get_device
        monkeypatch.delenv("FEDGROK_CLIENT_CPU", raising=False)
        assert _client_device() == get_device()
