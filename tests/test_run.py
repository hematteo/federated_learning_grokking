"""Tests for the single-run entry point (fedgrok.run).

Kept to cheap centralized runs (no Ray) so the suite stays fast; the federated
path shares the same code below extract_grokking_results and is covered by the
integration tests.
"""

import json
import os

import pytest

from fedgrok.run import run_spec, result_path
from fedgrok.manifest import run_id


CHEAP = {
    "mode": "centralized", "task": "addition", "p": 7, "alpha": 0.5,
    "seed": 42, "hidden_width": 16, "lr": 1.0, "epochs": 100, "log_every": 50,
}


def test_run_spec_writes_result_and_returns_row(tmp_path):
    row = run_spec(CHEAP, results_root=str(tmp_path / "runs"),
                   histories_root=str(tmp_path / "hist"))

    # Result JSON exists at the resolved path.
    assert os.path.exists(result_path(CHEAP, str(tmp_path / "runs")))

    # Row carries id, outcomes, and config, schema-aligned with runs.csv.
    for key in ("id", "mode", "task", "p", "alpha", "seed", "t_grok",
                "final_acc", "grokked", "censored", "steps_run"):
        assert key in row
    assert row["mode"] == "centralized"
    assert row["id"] == run_id(CHEAP)
    assert row["steps_run"] == 100.0


def test_result_json_is_valid_and_inf_serialised(tmp_path):
    run_spec(CHEAP, results_root=str(tmp_path / "runs"),
             histories_root=str(tmp_path / "hist"))
    with open(result_path(CHEAP, str(tmp_path / "runs"))) as handle:
        data = json.load(handle)
    # A tiny non-grokking run: t_grok must round-trip as the string "inf".
    assert data["t_grok"] == "inf"
    assert data["censored"] is True


def test_result_json_is_parseable_by_a_strict_parser(tmp_path):
    """No bare NaN / Infinity tokens -- those are Python extensions, not JSON.

    `final_ipr` is NaN on every non-modular run (IPR is GrokNet-specific), and
    for a long time that was written as a bare `NaN`, which Python reads back
    happily and jq / JS / Go / Rust reject. It went unnoticed because the whole
    internal pipeline is Python. This asserts the file is portable.
    """
    run_spec(CHEAP, results_root=str(tmp_path / "runs"),
             histories_root=str(tmp_path / "hist"))
    text = open(result_path(CHEAP, str(tmp_path / "runs"))).read()

    def _reject(token):
        raise AssertionError(f"non-standard JSON constant in result file: {token}")

    json.loads(text, parse_constant=_reject)


def test_non_finite_floats_serialise_as_strings(tmp_path):
    from fedgrok.run import _write_json_atomic

    path = str(tmp_path / "row.json")
    _write_json_atomic(path, {
        "pos": float("inf"), "neg": float("-inf"), "nan": float("nan"),
        "nested": [float("nan"), {"deep": float("inf")}], "ok": 1.5,
    })
    text = open(path).read()
    assert "NaN" not in text and "Infinity" not in text

    data = json.loads(text, parse_constant=lambda t: pytest.fail(f"bare {t}"))
    assert data == {"pos": "inf", "neg": "-inf", "nan": "nan",
                    "nested": ["nan", {"deep": "inf"}], "ok": 1.5}


def test_result_path_is_stable_without_explicit_id():
    # No "id" in the spec -> result_path must still resolve deterministically.
    p1 = result_path(CHEAP, "results/data/runs")
    p2 = result_path(dict(CHEAP), "results/data/runs")
    assert p1 == p2 and p1.endswith(".json")


def test_tags_do_not_change_the_run_id():
    tagged = dict(CHEAP, tier="T0", group="pilot", experiment="exp2")
    assert run_id(tagged) == run_id(CHEAP)


class TestIncompleteRunGuard:
    """A run that stops early must not be banked as a completed one.

    Flower's `run_simulation` can return NORMALLY after a Ray actor failure, so a
    starved simulation exits 0 and writes a plausible result. This is the worst
    shape of bug for this project: two K=50 cells stopped at 157 and 234 of 2,000
    rounds and recorded 2.9% train accuracy, which reads exactly like the
    large-K training collapse under investigation and would have been quoted as
    evidence for it.
    """

    def test_completion_is_measured_in_rounds_not_steps(self):
        """Steps would false-positive on every partial-participation cell.

        Under fraction_train < 1 the compute-matched `total_steps` axis is
        legitimately below rounds x E, by design.
        """
        from fedgrok.core.fed_config import FedConfig
        from fedgrok.run import _completion
        cfg = FedConfig(num_rounds=2_000, local_epochs=5, fraction_train=0.2)
        history = {"round": [0, 1_000, 2_000],
                   "total_steps": [0.0, 1_000.0, 2_000.0]}   # 1/5 of rounds x E
        reached, configured = _completion(history, cfg, "federated")
        assert (reached, configured) == (2_000.0, 2_000.0)

    def test_centralized_completion_reads_the_final_epoch(self):
        from fedgrok.core.config import Config
        from fedgrok.run import _completion
        cfg = Config(epochs=40_000)
        assert _completion({"epoch": [0, 20_000, 40_000]}, cfg, "centralized") \
            == (40_000.0, 40_000.0)

    def test_truncated_federated_run_raises_and_writes_nothing(self, tmp_path,
                                                               monkeypatch):
        import fedgrok.training.federated as fed
        from fedgrok.run import IncompleteRun, run_spec

        def _stops_early(cfg):
            # 234 of 2,000 rounds -- the observed failure, with a plausible
            # history attached so nothing else can catch it first.
            return {"round": [0, 234], "total_steps": [0.0, 1_171.0],
                    "train_acc": [1.0, 2.9], "test_acc": [1.0, 0.5]}, None

        monkeypatch.setattr(fed, "fed_train", _stops_early)
        spec = {"mode": "federated", "task": "addition", "p": 97,
                "num_clients": 50, "num_rounds": 2_000, "local_epochs": 5}
        with pytest.raises(IncompleteRun, match="234/2000"):
            run_spec(spec, results_root=str(tmp_path / "runs"),
                     histories_root=str(tmp_path / "hist"))
        # Nothing banked, so the launcher's resume repeats the cell.
        assert not list((tmp_path / "runs").glob("*.json"))

    def test_complete_run_is_unaffected(self, tmp_path, monkeypatch):
        import fedgrok.training.centralized as cent
        from fedgrok.run import run_spec

        def _finishes(cfg):
            return {"epoch": [0, 100], "train_acc": [1.0, 100.0],
                    "test_acc": [1.0, 100.0]}, None

        monkeypatch.setattr(cent, "train", _finishes)
        spec = {"mode": "centralized", "task": "addition", "p": 97,
                "epochs": 100}
        row = run_spec(spec, results_root=str(tmp_path / "runs"),
                       histories_root=str(tmp_path / "hist"))
        assert row["steps_run"] == 100.0
        assert len(list((tmp_path / "runs").glob("*.json"))) == 1
