"""Tests for federated/config.py: FedConfig defaults, inheritance, config roundtrip."""

import pytest

from fedgrok.core.config import Config
from fedgrok.core.fed_config import FedConfig
from fedgrok.training.federated import _cfg_to_fit_config, _fit_config_to_cfg


class TestFedConfig:
    def test_inherits_from_config(self):
        assert issubclass(FedConfig, Config)

    def test_defaults(self):
        cfg = FedConfig()
        assert cfg.num_clients == 5
        assert cfg.num_rounds == 2000
        assert cfg.local_epochs == 5
        assert cfg.fraction_train == 1.0
        assert cfg.partition == "iid"
        assert cfg.dirichlet_alpha == 0.5

    def test_overrides_config_defaults(self):
        """FedConfig overrides hidden_width=128 and output_dir."""
        cfg = FedConfig()
        assert cfg.hidden_width == 128  # Config default is 100
        assert cfg.output_dir == "results/baselines/federated"  # Config default is results/baselines/centralized

    def test_inherits_base_fields(self):
        cfg = FedConfig()
        assert cfg.p == 97
        assert cfg.task == "addition"
        assert cfg.alpha == 0.5
        assert cfg.optimizer == "gd"
        assert cfg.lr == 50.0

    def test_custom_values(self):
        cfg = FedConfig(
            p=53, num_clients=10, num_rounds=500,
            local_epochs=3, partition="operand",
            dirichlet_alpha=0.1, hidden_width=64,
        )
        assert cfg.p == 53
        assert cfg.num_clients == 10
        assert cfg.num_rounds == 500
        assert cfg.local_epochs == 3
        assert cfg.partition == "operand"
        assert cfg.dirichlet_alpha == 0.1
        assert cfg.hidden_width == 64


class TestConfigRoundtrip:
    """Test _cfg_to_fit_config and _fit_config_to_cfg preserve all fields."""

    def test_basic_roundtrip(self, small_fed_cfg):
        fit_config = _cfg_to_fit_config(small_fed_cfg, server_round=5)
        reconstructed = _fit_config_to_cfg(fit_config)

        assert reconstructed.p == small_fed_cfg.p
        assert reconstructed.task == small_fed_cfg.task
        assert reconstructed.alpha == small_fed_cfg.alpha
        assert reconstructed.seed == small_fed_cfg.seed
        assert reconstructed.num_clients == small_fed_cfg.num_clients
        assert reconstructed.partition == small_fed_cfg.partition
        assert reconstructed.dirichlet_alpha == small_fed_cfg.dirichlet_alpha
        assert reconstructed.local_epochs == small_fed_cfg.local_epochs
        assert reconstructed.lr == small_fed_cfg.lr
        assert reconstructed.optimizer == small_fed_cfg.optimizer
        assert reconstructed.weight_decay == small_fed_cfg.weight_decay
        assert reconstructed.momentum == small_fed_cfg.momentum
        assert reconstructed.hidden_width == small_fed_cfg.hidden_width
        assert reconstructed.activation == small_fed_cfg.activation

    def test_roundtrip_with_nondefault_values(self):
        cfg = FedConfig(
            p=53, task="multiplication", alpha=0.7, seed=99,
            num_clients=10, partition="dirichlet", dirichlet_alpha=0.1,
            local_epochs=3, lr=0.001, optimizer="adamw",
            weight_decay=0.5, momentum=0.9,
            hidden_width=64, activation="relu",
        )
        fit_config = _cfg_to_fit_config(cfg, server_round=42)
        reconstructed = _fit_config_to_cfg(fit_config)

        assert reconstructed.p == 53
        assert reconstructed.task == "multiplication"
        assert reconstructed.alpha == 0.7
        assert reconstructed.seed == 99
        assert reconstructed.num_clients == 10
        assert reconstructed.partition == "dirichlet"
        assert reconstructed.dirichlet_alpha == 0.1
        assert reconstructed.local_epochs == 3
        assert reconstructed.lr == 0.001
        assert reconstructed.optimizer == "adamw"
        assert reconstructed.weight_decay == 0.5
        assert reconstructed.momentum == 0.9
        assert reconstructed.hidden_width == 64
        assert reconstructed.activation == "relu"

    def test_fit_config_includes_server_round(self, small_fed_cfg):
        fit_config = _cfg_to_fit_config(small_fed_cfg, server_round=42)
        assert fit_config["server_round"] == 42

    def test_fit_config_types_are_serializable(self, small_fed_cfg):
        """All values in fit_config should be basic Python types (str, int, float)."""
        fit_config = _cfg_to_fit_config(small_fed_cfg, server_round=1)
        for key, val in fit_config.items():
            assert isinstance(val, (int, float, str)), \
                f"Key {key} has type {type(val)}, expected basic Python type"

    def test_roundtrip_preserves_float_precision(self):
        """Ensure floats survive dict→FedConfig conversion without loss."""
        cfg = FedConfig(lr=1.23456789e-5, alpha=0.123456789, dirichlet_alpha=0.999)
        fit_config = _cfg_to_fit_config(cfg, server_round=1)
        reconstructed = _fit_config_to_cfg(fit_config)
        assert reconstructed.lr == pytest.approx(cfg.lr)
        assert reconstructed.alpha == pytest.approx(cfg.alpha)
        assert reconstructed.dirichlet_alpha == pytest.approx(cfg.dirichlet_alpha)
