"""Run a single spec and write its result. The atomic unit of a sweep.

This is the unified run entry point (it supersedes the old main.py and
fed_main.py): one spec -> one training run -> one result JSON. The launcher
invokes it once per run in its own subprocess, with CUDA_VISIBLE_DEVICES already
set, so GPU selection is not this module's concern.

    python -m fedgrok.run --spec-file spec.json
    python -m fedgrok.run --spec '{"mode":"federated","task":"addition",...}'

The result JSON schema matches results/data/runs.csv (the archived-log harvest)
so old and new runs form one table, plus a few v2-only fields (id, strategy,
num_rounds, eval_every, steps_run).
"""

import argparse
import dataclasses
import json
import math
import os
import tempfile
import time

from fedgrok.manifest import build_config, run_id, TAG_KEYS
from fedgrok.analysis.grokking_metrics import extract_grokking_results
from fedgrok.data.registry import grok_threshold


DEFAULT_RESULTS_DIR = "results/data/runs"


def _run_dir(spec, results_root):
    return os.path.join(results_root, spec["id"])


def result_path(spec, results_root=DEFAULT_RESULTS_DIR):
    return os.path.join(results_root, run_id(spec) + ".json")


def _steps_run(history):
    """Last recorded step count (total_steps for FL, epoch for centralized)."""
    steps = history.get("total_steps") or history.get("epoch") or []
    return float(steps[-1]) if steps else 0.0


class IncompleteRun(RuntimeError):
    """A run returned normally but did not reach its configured horizon."""


def _completion(history, cfg, mode):
    """(reached, configured) in the run's own natural unit -- rounds or epochs.

    NOT steps. Under partial participation `total_steps` is legitimately below
    rounds x E, so a step-based check would flag every fraction_train < 1 cell.
    Rounds and epochs are exact: the centralized loop is `range(epochs + 1)` and
    logs the final epoch, and `evaluate_fn` forces an eval at
    `server_round == num_rounds`, so a complete run always records its horizon.
    """
    if mode == "federated":
        seq = history.get("round", [])
        return (float(seq[-1]) if seq else 0.0), float(cfg.num_rounds)
    seq = history.get("epoch", [])
    return (float(seq[-1]) if seq else 0.0), float(cfg.epochs)


def run_spec(spec: dict, results_root: str = DEFAULT_RESULTS_DIR,
             histories_root: str = "results/runs") -> dict:
    """Execute one spec, write its result JSON, and return the result row."""
    spec = dict(spec)
    spec["id"] = run_id(spec)
    mode = spec["mode"]

    # Each run's full per-round history lands in its own directory.
    spec.setdefault("output_dir", os.path.join(histories_root, spec["id"]))
    cfg = build_config(spec)

    # Drop the spec next to the history BEFORE training. Checkpoints are bare
    # state_dicts, so without this there is no way to know which architecture or
    # data split a .pt file belongs to -- which is why the post-hoc analyzer used
    # to hardcode GrokNet and alpha=0.5, and was silently wrong on every run that
    # did not match. Written up front so it exists even if the run then crashes.
    os.makedirs(spec["output_dir"], exist_ok=True)
    _write_json_atomic(os.path.join(spec["output_dir"], "spec.json"), spec)

    t0 = time.time()
    if mode == "centralized":
        from fedgrok.training.centralized import train
        history, _ = train(cfg)
    elif mode == "federated":
        from fedgrok.training.federated import fed_train
        history, _ = fed_train(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    wall_s = time.time() - t0

    # Did it actually finish? Flower's run_simulation can return NORMALLY after
    # a Ray actor failure, so a starved simulation exits 0, writes a plausible
    # result, and is banked as a completed run. Observed: two K=50 cells stopped
    # at 157 and 234 of 2,000 rounds and were recorded as 2.9% train accuracy --
    # which reads exactly like the training collapse under investigation, and
    # would have been quoted as evidence for it.
    #
    # The cause is resource oversubscription, not a bug: clients reserve
    # num_cpus=1 each, so 12 concurrent runs x 50 clients demands 600 CPUs on a
    # 64-core box. It therefore depends on what else is running, which is why the
    # same cell completed in an earlier, less crowded sweep. Lower --per-gpu or
    # make num_cpus fractional for large-K sweeps.
    #
    # Raising rather than recording a flag is deliberate: the launcher then
    # counts it as a failure and, because no result JSON is written, resume
    # re-runs it. A flag would leave the row in place and rely on every future
    # analysis remembering to filter on it. The history is still on disk under
    # output_dir for diagnosis.
    reached, configured = _completion(history, cfg, mode)
    if configured > 0 and reached < configured:
        unit = "rounds" if mode == "federated" else "epochs"
        raise IncompleteRun(
            f"{spec['id']}: reached {reached:.0f}/{configured:.0f} {unit} "
            f"({100 * reached / configured:.1f}%) but returned without error. "
            f"Nothing is written, so this run is not banked and resume will "
            f"repeat it. If this is a large-K federated cell, reduce --per-gpu: "
            f"each client reserves num_cpus=1, so K x concurrent-runs must stay "
            f"within the core count."
        )

    threshold = grok_threshold(cfg)
    metrics = extract_grokking_results(history, threshold=threshold)
    t_grok = metrics["t_grok"]

    # One row, schema-compatible with results/data/runs.csv. Config fields are
    # taken from the resolved dataclass so defaults are explicit; tags from spec.
    cfg_dict = dataclasses.asdict(cfg)

    # Start from the WHOLE resolved config, then name the important fields
    # explicitly below for column ordering. Hand-listing the fields was a
    # recurring bug: every time the dataclass gained an axis, the results table
    # silently lost it -- dataset/model/loss went missing for 123 runs,
    # persist_local_opt_state for the A/B that was meant to settle a confound,
    # and n_heads/d_mlp for a capacity sweep whose entire point was those two
    # numbers. Deriving the row from the dataclass makes that structurally
    # impossible; a new field is recorded the moment it exists.
    _skip = {"output_dir", "save_weights"}
    auto = {k: v for k, v in cfg_dict.items()
            if not k.startswith("_") and k not in _skip}

    row = {
        **auto,
        "id": spec["id"],
        "mode": mode,
        # grouping tags (may be absent)
        **{k: spec[k] for k in ("tier", "group", "experiment", "setting",
                                "algorithm", "manifest", "setup",
                                "arm", "reduced_from_k") if k in spec},
        # config — setup identity first. Without dataset/model/loss a t1
        # replication row cannot say whether it is the groknet or the
        # transformer on S5; they were separable only by hidden_width, and
        # only by accident.
        "dataset": cfg.dataset, "model": cfg.model, "loss": cfg.loss,
        "batch_size": cfg.batch_size, "init_scale": cfg.init_scale,
        "n_layers": cfg.n_layers, "group_n": cfg.group_n,
        "task": cfg.task, "optimizer": cfg.optimizer, "p": cfg.p,
        "hidden_width": cfg.hidden_width, "alpha": cfg.alpha, "seed": cfg.seed,
        "lr": cfg.lr, "weight_decay": cfg.weight_decay,
        # `alpha` is the data-fraction axis for the grid datasets only; MNIST's
        # is n_train (alpha is ignored there), and `epochs` is the centralized
        # censoring time. Without these a centralized row cannot say how much
        # data it saw or how long it had to grok.
        "activation": cfg.activation, "momentum": cfg.momentum,
        "n_train": cfg.n_train, "n_test": cfg.n_test,
        "epochs": cfg.epochs, "log_every": cfg.log_every,
        "num_clients": cfg_dict.get("num_clients"),
        "local_epochs": cfg_dict.get("local_epochs"),
        "num_rounds": cfg_dict.get("num_rounds"),
        "fraction_train": cfg_dict.get("fraction_train"),
        "partition": cfg_dict.get("partition"),
        "dirichlet_alpha": cfg_dict.get("dirichlet_alpha"),
        "coset_subgroup": cfg_dict.get("coset_subgroup"),
        "proximal_mu": cfg_dict.get("proximal_mu"),
        "strategy": cfg_dict.get("strategy"),
        "server_lr": cfg_dict.get("server_lr"),
        "server_momentum": cfg_dict.get("server_momentum"),
        "tau": cfg_dict.get("tau"),
        "eval_every": cfg_dict.get("eval_every"),
        "checkpoint_every": cfg_dict.get("checkpoint_every"),
        # The A/B arm for the AdamW optimizer-restart confound. Without it the two
        # arms are indistinguishable in the results table -- the same defect that
        # dropping dataset/model/loss caused, one axis over.
        "persist_local_opt_state": cfg_dict.get("persist_local_opt_state"),
        "feddyn_alpha": cfg_dict.get("feddyn_alpha"),
        "checkpoint_client_weights": cfg_dict.get("checkpoint_client_weights"),
        # outcomes — grok_threshold is recorded because it varies by dataset,
        # so a t_grok is only interpretable next to the bar it was measured at.
        "grok_threshold": threshold,
        "t_grok": t_grok,
        "t_50": metrics["t_50"],
        # Memorisation time and the delay between it and grokking. Recorded
        # because t_grok alone cannot distinguish "did not generalise" from "did
        # not train" -- the K>=30 AdamW cells sit at 1-5% TRAIN accuracy, which
        # reads identically to a grokking failure in a table of t_grok values.
        "t_memo": metrics["t_memo"],
        "delay": metrics["delay"],
        "t_first_cross": metrics["t_first_cross"],
        "post_grok_dips": metrics["post_grok_dips"],
        "peak_train_acc": metrics["peak_train_acc"],
        "final_acc": metrics["final_test_acc"],
        "final_train_acc": metrics["final_train_acc"],
        "final_ipr": metrics["final_ipr"],
        "grokked": bool(t_grok != float("inf")),
        "censored": bool(t_grok == float("inf")),
        "steps_run": _steps_run(history),
        "wall_s": round(wall_s, 1),
    }

    _write_json_atomic(result_path(spec, results_root), row)
    return row


def _write_json_atomic(path, obj):
    """Write JSON via a temp file + rename, so a crash never leaves a partial
    result that resume would mistake for a completed run.

    Non-finite floats become STRINGS: inf -> "inf", -inf -> "-inf", nan -> "nan".

    Python's json emits bare `Infinity` / `NaN` tokens for these, which are not in
    the JSON grammar and which no other parser accepts. `inf` was handled from the
    start; `nan` was not, and `final_ipr` is NaN on every non-modular run (IPR is
    GrokNet-specific), so 1,038 of the 1,529 banked result files were unparseable
    by anything but Python. That went unnoticed because the whole internal
    pipeline IS Python, and it matters now the results are packaged for a public
    data host. `scripts/repair_result_json.py` fixes files already on disk.

    allow_nan=False makes it loud rather than silent if a non-finite value ever
    reaches json.dump without passing through _san.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _san(o):
        if isinstance(o, float) and not math.isfinite(o):
            return "nan" if math.isnan(o) else ("inf" if o > 0 else "-inf")
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_san(v) for v in o]
        return o

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(_san(obj), handle, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--spec", help="Spec as a JSON string")
    g.add_argument("--spec-file", help="Path to a JSON file holding one spec")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--histories-root", default="results/runs")
    args = parser.parse_args()

    spec = json.loads(args.spec) if args.spec else json.load(open(args.spec_file))
    row = run_spec(spec, results_root=args.results_root,
                   histories_root=args.histories_root)

    grok = "grok" if row["grokked"] else "CENSORED"
    print(f"\n[{row['id']}] {grok}  T_grok={row['t_grok']}  "
          f"final_acc={row['final_acc']:.1f}%  ({row['wall_s']}s)")


if __name__ == "__main__":
    main()
