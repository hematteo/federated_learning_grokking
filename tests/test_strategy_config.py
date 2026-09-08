import pytest
from fedgrok.core.fed_config import FedConfig


class TestFedConfigStrategy:
    def test_default_strategy_is_fedavg(self):
        cfg = FedConfig()
        assert cfg.strategy == "fedavg"

    def test_fedadam_params(self):
        cfg = FedConfig(strategy="fedadam", server_lr=0.1, tau=1e-3)
        assert cfg.strategy == "fedadam"
        assert cfg.server_lr == 0.1
        assert cfg.tau == 1e-3

    def test_track_client_drift_default_true(self):
        cfg = FedConfig()
        assert cfg.track_client_drift is True

    def test_weight_decay_in_fl(self):
        cfg = FedConfig(weight_decay=0.1)
        assert cfg.weight_decay == 0.1

    def test_strategy_preserved_through_inheritance(self):
        cfg = FedConfig(strategy="fedprox", proximal_mu=0.1)
        assert cfg.strategy == "fedprox"
        assert cfg.proximal_mu == 0.1


# ── Server-side strategies (Phase 4) ──────────────────────────────────────────


class TestServerStrategies:
    def test_new_strategy_fields_default(self):
        cfg = FedConfig()
        assert cfg.server_momentum == 0.0
        assert cfg.feddyn_alpha == 0.01

    @pytest.mark.parametrize("strategy", ["fedavgm", "fedyogi", "scaffold"])
    def test_new_strategies_accepted(self, strategy):
        assert FedConfig(strategy=strategy).strategy == strategy

    @pytest.mark.parametrize("strategy", ["feddyn", "fedAvgM", "fed_adam", ""])
    def test_unimplemented_or_misspelled_strategy_raises(self, strategy):
        """An unrecognised strategy must not quietly become FedAvg.

        `feddyn` was in FedConfig's Literal and had no branch in
        _build_strategy, so it fell through to plain FedAvg and banked a result
        row labelled `strategy: "feddyn"`. Typos did the same. The Literal is a
        type annotation and is not enforced at runtime, so _build_strategy is
        the only place this can be caught.
        """
        from fedgrok.training.federated import _build_strategy
        from flwr.common import ndarrays_to_parameters
        import numpy as np

        init = ndarrays_to_parameters([np.zeros((2, 2), dtype=np.float32)])
        cfg = FedConfig(num_clients=4)
        cfg.strategy = strategy          # bypass the annotation, as a manifest can
        with pytest.raises(ValueError, match="Unknown strategy"):
            _build_strategy(cfg, init, evaluate_fn=None)

    def test_build_strategy_dispatches_native(self):
        from flwr.server.strategy import FedAvg, FedAvgM, FedYogi, FedAdam
        from fedgrok.training.federated import _build_strategy
        from flwr.common import ndarrays_to_parameters
        import numpy as np

        init = ndarrays_to_parameters([np.zeros((2, 2), dtype=np.float32)])
        cases = {
            "fedavg": FedAvg, "fedprox": FedAvg,   # fedprox uses FedAvg + client-side term
            "fedavgm": FedAvgM, "fedyogi": FedYogi, "fedadam": FedAdam,
        }
        for name, cls in cases.items():
            cfg = FedConfig(strategy=name, num_clients=4)
            strat = _build_strategy(cfg, init, evaluate_fn=None)
            assert isinstance(strat, cls), f"{name} -> {type(strat).__name__}"

    def test_fedavgm_neutral_settings_reduce_to_fedavg(self):
        """FedAvgM with lr=1, momentum=0 must reproduce FedAvg exactly."""
        import warnings, logging
        warnings.filterwarnings("ignore")
        logging.getLogger("flwr").setLevel(logging.ERROR)
        from fedgrok.training.federated import fed_train, _dataset_cache, _client_cache

        def run(strategy, **kw):
            _dataset_cache.clear(); _client_cache.clear()
            cfg = FedConfig(task="addition", p=17, alpha=0.5, seed=42,
                            hidden_width=32, num_clients=4, num_rounds=12,
                            local_epochs=1, lr=1.0, partition="iid",
                            strategy=strategy, eval_every=12,
                            output_dir="/tmp/test_strat", **kw)
            h, _ = fed_train(cfg)
            return h["test_acc"][-1], h["train_loss"][-1]

        avg = run("fedavg")
        avgm = run("fedavgm", server_lr=1.0, server_momentum=0.0)
        assert avgm == pytest.approx(avg, abs=1e-4)
