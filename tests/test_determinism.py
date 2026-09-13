"""Run-to-run determinism of the federated harness.

RESULTS 23 found two runs of one config and seed differing by up to 12.7 points
of peak train accuracy on setup C. The cause was float summation order:
Flower's aggregate walks the results list in arrival order and nothing sorted
it. `_SortedResultsMixin` sorts by client id; these tests pin (a) that the
aggregate is then a pure function of the client updates, and (b) that two
complete runs of one small spec produce identical histories.
"""

import logging
import random
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")
logging.getLogger("flwr").setLevel(logging.ERROR)

from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from fedgrok.core.fed_config import FedConfig
from fedgrok.training import scaffold as sc
from fedgrok.training.federated import (
    _build_strategy, _client_cache, _dataset_cache, fed_train,
)


class _Proxy:
    def __init__(self, cid):
        self.cid = cid


def _results(n_clients, rng):
    """n client results whose float32 sum is order-sensitive."""
    out = []
    for cid in range(n_clients):
        arrays = [rng.standard_normal((37, 53)).astype(np.float32) * 10.0 ** rng.integers(-3, 4),
                  rng.standard_normal(200).astype(np.float32)]
        res = FitRes(status=Status(code=Code.OK, message=""),
                     parameters=ndarrays_to_parameters(arrays),
                     num_examples=int(rng.integers(50, 500)), metrics={})
        out.append((_Proxy(str(cid)), res))
    return out


def _aggregate(strategy, results):
    params, _ = strategy.aggregate_fit(1, list(results), [])
    return [a.tobytes() for a in parameters_to_ndarrays(params)]


class TestSortedFedAggregation:
    def _strategy(self, order):
        cfg = FedConfig(p=17, num_clients=12, aggregation_order=order)
        return _build_strategy(cfg, ndarrays_to_parameters([np.zeros(1)]), None)

    def test_cid_order_is_shuffle_invariant(self):
        rng = np.random.default_rng(0)
        results = _results(12, rng)
        strategy = self._strategy("cid")
        reference = _aggregate(strategy, results)
        shuffler = random.Random(1)
        for _ in range(20):
            shuffled = list(results)
            shuffler.shuffle(shuffled)
            assert _aggregate(strategy, shuffled) == reference

    def test_arrival_order_keeps_flowers_behaviour(self):
        """With sorting off the sum follows the list -- the pre-2026-09-09
        semantics -- so the two orders can (and here do) differ in the bits."""
        rng = np.random.default_rng(0)
        results = _results(12, rng)
        strategy = self._strategy("arrival")
        assert strategy.sort_results is False
        forward = _aggregate(strategy, results)
        backward = _aggregate(strategy, list(reversed(results)))
        stock = FedAvg(initial_parameters=ndarrays_to_parameters([np.zeros(1)]))
        assert forward == _aggregate(stock, results)
        assert backward == _aggregate(stock, list(reversed(results)))
        # Sorted, the same two lists agree.
        sorted_strategy = self._strategy("cid")
        assert _aggregate(sorted_strategy, results) == _aggregate(sorted_strategy, list(reversed(results)))

    def test_partition_id_wins_over_random_node_ids(self):
        """Simulation cids are random node ids; the partition id in the fit
        metrics is what fixes the order across runs."""
        rng = np.random.default_rng(3)
        results = _results(8, rng)
        for pid, (proxy, res) in enumerate(results):
            proxy.cid = str(rng.integers(1, 2**62))
            res.metrics["partition_id"] = pid
        strategy = self._strategy("cid")
        reference = _aggregate(strategy, results)
        relabelled = [(_Proxy(str(rng.integers(1, 2**62))), res) for _, res in results]
        random.Random(4).shuffle(relabelled)
        assert _aggregate(strategy, relabelled) == reference

    def test_every_strategy_class_is_sorted(self):
        params = ndarrays_to_parameters([np.zeros(1)])
        for strategy in ("fedavg", "fedprox", "fedadam", "fedyogi", "fedavgm"):
            cfg = FedConfig(p=17, strategy=strategy)
            built = _build_strategy(cfg, params, None)
            assert isinstance(built, sc._SortedResultsMixin), strategy
            assert built.sort_results is True

    def test_default_is_cid(self):
        assert FedConfig(p=17).aggregation_order == "cid"


def _run_fed(tmp_path, **kw):
    _dataset_cache.clear()
    _client_cache.clear()
    cfg = FedConfig(
        task="addition", p=17, alpha=0.5, seed=42, hidden_width=32,
        num_clients=6, num_rounds=6, local_epochs=3, lr=1.0,
        partition="dirichlet", dirichlet_alpha=0.3, eval_every=1,
        output_dir=str(tmp_path), **kw,
    )
    history, model = fed_train(cfg)
    return history, [p.detach().cpu().numpy().copy() for p in model.parameters()]


class TestFedRunToRunIdentity:
    def test_two_runs_of_one_spec_are_identical(self, tmp_path):
        """The regression test RUNS_TODO's reproducibility entry asked for.
        Exact equality, not approx: the point is that nothing in the harness
        draws from an unseeded or order-dependent source any more."""
        h1, w1 = _run_fed(tmp_path / "a")
        h2, w2 = _run_fed(tmp_path / "b")
        for key in ("test_acc", "train_acc", "train_loss", "weight_norm"):
            if key in h1:
                assert h1[key] == h2[key], key
        for a, b in zip(w1, w2):
            assert np.array_equal(a, b)
