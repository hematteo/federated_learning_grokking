"""Federated Averaging via Flower simulation for grokking experiments.

Uses Flower's run_simulation with:
  - Custom NumPyClient that does local training on its partition
  - FedAvg strategy with centralized evaluation (evaluate_fn)
  - on_fit_config_fn to pass hyperparameters to clients

Each client reconstructs its data partition from the config + partition-id
(dataset is tiny so this is fast). The server evaluates the global model
on the full test set after each round.
"""

import dataclasses
import json
import os
import time
import warnings
from collections import OrderedDict

import numpy as np
import torch

import flwr as fl
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdam, FedAvgM, FedYogi
from flwr.simulation import run_simulation

from fedgrok.core.fed_config import FedConfig
from fedgrok.data.partition import make_federated_datasets
from fedgrok.core.registry import build_model, build_loss
from fedgrok.data.registry import dataset_dims
from fedgrok.core.utils import make_optimizer, get_device
from fedgrok.metrics.fourier import (
    weight_norms, compute_ipr, compute_accuracy, fourier_spectrum,
    dft_applicable, weight_norms_applicable, weight_norm_report,
)
from fedgrok.metrics.probes import client_signature, mechanistic_probe, probe_keys


# ── Dataset cache ────────────────────────────────────────────────────────────
# Avoids rebuilding the dataset on every client fit() call. Each unique config
# produces one cached entry; clients and server share the same CPU tensors.

_dataset_cache = {}


def _data_key(cfg):
    """Everything that determines the DATA a client sees.

    The key used to be (p, task, alpha, seed, num_clients, partition,
    dirichlet_alpha), which is only sufficient in a single-dataset study. Across
    setups it collides outright: a modular config and an S5 config produce the
    identical tuple (S5 leaves p/task at their defaults), as do two MNIST configs
    differing only in n_train. Whichever ran first then serves its data to the
    second. That is loud when the dims differ and silent when they do not.
    """
    return (cfg.dataset, cfg.p, cfg.task, cfg.alpha, cfg.seed,
            cfg.num_clients, cfg.partition, getattr(cfg, "dirichlet_alpha", None),
            cfg.group_n, cfg.coset_subgroup, cfg.n_train, cfg.n_test)


def _client_key(cfg, partition_id):
    """Data identity plus everything determining the cached MODEL and TARGET.

    `_client_cache` stores a built model and a loss-specific target tensor, so it
    needs more than `_data_key`. Loss especially: an MSE entry served to a CE run
    hands one-hot floats to CrossEntropyLoss, which accepts them as soft labels
    and trains without error on the wrong objective.
    """
    return (_data_key(cfg), cfg.model, cfg.loss, cfg.hidden_width, cfg.n_layers,
            cfg.activation, cfg.init_scale, cfg.n_heads, cfg.d_mlp, partition_id)


def _get_cached_datasets(cfg):
    """Return federated datasets, caching by config to avoid redundant work."""
    cache_key = _data_key(cfg)
    if cache_key not in _dataset_cache:
        _dataset_cache[cache_key] = make_federated_datasets(cfg)
    return _dataset_cache[cache_key]


# ── Per-client warm state ────────────────────────────────────────────────────
# fit() runs once per client per round. Rebuilding the model, moving it to the
# device, and re-copying the client's data to the device every round is pure
# overhead — the data never changes and the model's shape never changes; only
# its parameter *values* change each round. In Flower's simulation each client
# is a Ray actor (a persistent process), so a module-level dict keyed by the
# client's identity survives across that client's rounds — the same mechanism
# _dataset_cache relies on. We cache the on-device model and data tensors and,
# each round, copy the incoming global parameters into the model in place.

_client_cache = {}


# ── Client placement: the two knobs that set a run's VRAM ────────────────────
# Each Flower client is a separate Ray actor PROCESS, so every concurrently
# scheduled client pays a full CUDA context. Measured on an RTX 3080: ~226 MiB
# per process before any model or data exists. That overhead, not the workload,
# is what a federated run's VRAM is made of -- setup B's transformer is 0.9 MB
# and a K=50 client's data slice is 0.09 MB, so fifty clients need ~180 MB of
# actual memory against ~11 GB of context.
#
# Both knobs are environment variables and deliberately NOT Config fields.
# `manifest.py` hashes the spec to make the run id, so an experiment axis added
# here would have to appear in specs and would re-id every banked run. Where a
# client's tensors live is a scheduling decision, not an experiment axis: it
# changes what the machine can hold, not what is computed.
#
#   FEDGROK_CLIENT_CPU=1     Clients train on CPU; the server still evaluates on
#                            GPU. Removes the VRAM ceiling outright.
#   FEDGROK_GPU_CLIENT_CAP=N At most N clients hold a CUDA context at once, so
#                            VRAM stops scaling with K. FedAvg is synchronous --
#                            every selected client trains from the same global
#                            weights and all must return before aggregation --
#                            so running K clients in waves of N computes the
#                            same thing as running them all at once.

def _clients_on_cpu():
    return os.environ.get("FEDGROK_CLIENT_CPU", "").strip().lower() in ("1", "true", "yes")


def _client_device():
    """Device for client-side training. The server's device is chosen separately."""
    if _clients_on_cpu():
        return torch.device("cpu")
    return get_device()


def _compile_clients():
    """Compile the client forward with CUDA graphs (see the placement note)."""
    return os.environ.get("FEDGROK_COMPILE", "").strip().lower() in ("1", "true", "yes")


def _gpu_client_cap(num_clients):
    """How many clients may hold a CUDA context concurrently (default: all K)."""
    raw = os.environ.get("FEDGROK_GPU_CLIENT_CAP", "").strip()
    if not raw:
        return num_clients
    cap = int(raw)
    if cap < 1:
        raise ValueError(
            f"FEDGROK_GPU_CLIENT_CAP must be a positive integer, got {raw!r}"
        )
    return min(cap, num_clients)


def _get_warm_client(cfg, partition_id, device):
    """Return a cached (model, x_local, y_local_oh, y_local) for this client.

    Built once per (dataset, client) and reused across rounds. The model's
    parameter values are overwritten with the round's global weights by the
    caller, so its initialisation is irrelevant after the first round.
    """
    key = _client_key(cfg, partition_id)
    entry = _client_cache.get(key)
    if entry is None:
        client_data, _, _, _, _ = _get_cached_datasets(cfg)
        x_local, y_local = client_data[partition_id]
        x_local = x_local.to(device)
        y_local = y_local.to(device)
        # Loss target: one-hot for MSE, class indices for CE (build_loss owns it).
        y_local_target = build_loss(cfg).prepare_target(y_local, dataset_dims(cfg)[1])

        model = _make_model(cfg).to(device)
        # A compiled forward for the TRAINING steps only. It shares parameters
        # with `model`, so the optimizer, the weight copy-in and every
        # state_dict path below still operate on the eager module and are
        # unaffected. `reduce-overhead` is the mode that matters here: it backs
        # the forward with CUDA graphs, which is what removes the per-launch
        # cost that dominates a 103-sample step. Metrics stay eager on purpose --
        # they run under model.eval(), and switching modes on a compiled module
        # forces a recapture every round.
        train_fwd = torch.compile(model, mode="reduce-overhead") if _compile_clients() else model
        entry = (model, x_local, y_local_target, y_local, train_fwd)
        _client_cache[key] = entry
    return entry


_optimizer_cache = {}


def _get_warm_optimizer(cfg, model, partition_id):
    """This client's optimizer, persisted across rounds (see persist_local_opt_state).

    Lives in the same Ray actor as _client_cache, so the Adam moment estimates
    survive between this client's rounds instead of being re-initialised.

    Keyed on _client_key PLUS the optimizer's own hyperparameters. _client_key
    alone describes the data and the model shape, which is all _client_cache
    needs, but an optimizer is also made of lr / weight_decay / optimizer /
    momentum -- none of which are in it. Two runs in one process differing only
    in lr would otherwise share the first one's optimizer and train at the wrong
    learning rate. A sweep gives each run its own subprocess, so the path that
    reaches this is the test suite and the multi-run experiment scripts.
    """
    key = (_client_key(cfg, partition_id),
           cfg.optimizer, cfg.lr, cfg.weight_decay, cfg.momentum)
    if key not in _optimizer_cache:
        _optimizer_cache[key] = make_optimizer(model, cfg)
    return _optimizer_cache[key]


def _load_ndarrays_into(model, ndarrays):
    """Copy a list of parameter ndarrays into `model` in place, on its device.

    Equivalent to load_state_dict(ndarrays) but without reallocating tensors or
    moving anything across devices; the order matches _model_to_ndarrays.

    The length check is not defensive padding: zip() stops at the shorter
    sequence, so a mismatch would leave the model's trailing tensors holding the
    PREVIOUS round's values and train on from there. Nothing downstream notices.
    """
    entries = model.state_dict()
    if len(entries) != len(ndarrays):
        raise ValueError(
            f"Parameter count mismatch: model has {len(entries)} state_dict "
            f"entries, received {len(ndarrays)} arrays. Copying would silently "
            f"leave the trailing tensors at their previous values."
        )
    with torch.no_grad():
        for param, arr in zip(entries.values(), ndarrays):
            param.copy_(torch.from_numpy(arr).to(param.device))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _model_to_ndarrays(model):
    """Extract model weights as list of numpy arrays (always on CPU)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _ndarrays_to_state_dict(ndarrays, model):
    """Convert numpy arrays back to a state dict matching model's keys."""
    keys = list(model.state_dict().keys())
    return OrderedDict({k: torch.from_numpy(v) for k, v in zip(keys, ndarrays)})


def _make_model(cfg):
    """Create a fresh model from config (on CPU), via the model registry."""
    return build_model(cfg)


# Field names of FedConfig, computed once. Every field is a Flower Scalar
# (int/float/str/bool), so the whole config round-trips through the fit-config
# dict without hand-listing fields — which is what previously dropped loss,
# model and batch_size and silently mis-trained clients.
_FEDCONFIG_FIELDS = {f.name for f in dataclasses.fields(FedConfig)}


def _cfg_to_fit_config(cfg: FedConfig, server_round: int):
    """Serialize the whole config into a dict Flower can send to clients.

    Generated from the dataclass fields so no field can be silently dropped.
    """
    config = dataclasses.asdict(cfg)
    config["server_round"] = server_round
    return config


def _fit_config_to_cfg(config: dict) -> FedConfig:
    """Reconstruct FedConfig from the dict sent by the server."""
    kwargs = {k: v for k, v in config.items() if k in _FEDCONFIG_FIELDS}
    return FedConfig(**kwargs)


def _shuffle_seed(seed: int, partition_id: int, server_round: int) -> int:
    """Deterministic seed for one client's minibatch shuffle in one round.

    Explicit arithmetic rather than `hash(...)`: hashing a tuple of ints happens
    to be stable across processes today (PYTHONHASHSEED randomises str/bytes, not
    ints), but that is an implementation detail, not something to rest run
    reproducibility on. The multipliers are distinct primes so the three axes
    cannot alias -- client 1 at round 0 must not draw client 0's round-1 shuffle.
    """
    mixed = seed * 1_000_003 + partition_id * 10_007 + server_round
    return mixed % (2**31 - 1)


def compute_drift(w_before: list, w_after: list) -> float:
    """Compute Frobenius norm of weight difference: ||w_after - w_before||_F."""
    total = 0.0
    for wb, wa in zip(w_before, w_after):
        diff = wa - wb
        total += float(np.sum(diff ** 2))
    return float(np.sqrt(total))


# ── Flower Client ────────────────────────────────────────────────────────────

class GrokClient(NumPyClient):
    """Flower client that trains GrokNet on a local data partition."""

    def __init__(self, partition_id: int):
        self.partition_id = partition_id

    def fit(self, parameters, config):
        cfg = _fit_config_to_cfg(config)
        device = _client_device()

        # Warm model + on-device data (built once per client, reused each round).
        # y_local_target is one-hot (MSE) or class indices (CE) per the loss.
        model, x_local, y_local_target, y_local, train_fwd = _get_warm_client(
            cfg, self.partition_id, device)

        # Overwrite the model with this round's global weights, in place.
        _load_ndarrays_into(model, parameters)

        # Optimizer state (momentum, Adam moments) is by default NOT preserved
        # across rounds — a fresh optimizer restarts it, which is the standard
        # FedAvg/FedProx semantics. For SGD at momentum=0 that is a genuine no-op,
        # which is why the anchor setup's E axis is clean. For AdamW it is not:
        # every round becomes E bias-corrected cold-start Adam steps, so an E
        # sweep confounds "more local drift" with "more optimizer restarts". Set
        # persist_local_opt_state=True to hold the state across rounds and
        # measure the difference. Default False keeps every existing run exact.
        if cfg.persist_local_opt_state:
            optimizer = _get_warm_optimizer(cfg, model, self.partition_id)
        else:
            optimizer = make_optimizer(model, cfg)
        loss_fn = build_loss(cfg).loss_fn

        # FedProx: snapshot the global weights for the proximal term.
        proximal_mu = cfg.proximal_mu
        if proximal_mu > 0:
            global_params = [w.detach().clone() for w in model.parameters()]

        # SCAFFOLD: fetch this round's server control variate c and this client's
        # c_i; the g - c_i + c correction is applied after each backward.
        scaffold_cv_bytes = config.get("scaffold_cv")
        scaffold = scaffold_cv_bytes is not None
        if scaffold:
            from fedgrok.training import scaffold as _sc
            shapes = [w.shape for w in parameters]
            server_cv = _sc._cv_from_bytes(scaffold_cv_bytes, shapes)
            cv_key = _client_key(cfg, self.partition_id)
            client_cv = _sc.get_client_cv(cv_key, parameters)

        def _step(xb, yb):
            """One gradient step on a batch (+ FedProx / SCAFFOLD corrections)."""
            loss = loss_fn(train_fwd(xb), yb)
            if proximal_mu > 0:
                prox = sum(
                    torch.sum((p - gp) ** 2)
                    for p, gp in zip(model.parameters(), global_params)
                )
                loss = loss + (proximal_mu / 2.0) * prox
            optimizer.zero_grad()
            loss.backward()
            if scaffold:
                _sc.apply_correction(model, server_cv, client_cv)
            optimizer.step()

        model.train()
        bs = cfg.batch_size
        n_local = x_local.shape[0]
        n_steps = 0
        # The minibatch order needs its own seeded stream. `fed_train` calls
        # torch.manual_seed(cfg.seed), but that runs in the DRIVER process; every
        # client is a separate Ray actor whose default generator is seeded
        # non-deterministically at first use, so the shuffles below were the one
        # uncontrolled RNG source in the project. They affected all 94 banked
        # setup-E (MNIST) federated runs, and asymmetrically -- centralized MNIST
        # *is* seeded, so the arm those are compared against carried a noise
        # source it did not.
        #
        # A private CPU generator rather than torch.manual_seed, for two reasons.
        # It leaves the actor's global stream untouched, so the full-batch path
        # stays bit-identical to every banked run -- which is why this sits
        # inside the branch and not above it. And a CPU generator draws the same
        # permutation whichever device the client trains on, so FEDGROK_CLIENT_CPU
        # no longer changes what is computed.
        if bs and bs > 0:
            perm_gen = torch.Generator().manual_seed(_shuffle_seed(
                int(cfg.seed), int(self.partition_id),
                int(config.get("server_round", 0))))
        for _ in range(cfg.local_epochs):
            if bs and bs > 0:
                # A local epoch is a shuffled minibatch pass. randperm is only
                # reached here, so the full-batch path stays RNG-identical.
                perm = torch.randperm(n_local, generator=perm_gen).to(device)
                for i in range(0, n_local, bs):
                    idx = perm[i:i + bs]
                    _step(x_local[idx], y_local_target[idx])
                    n_steps += 1
            else:
                # Full-batch local GD: one local epoch = one step.
                _step(x_local, y_local_target)
                n_steps += 1

        # Return updated weights (moved to CPU) and metrics
        model.eval()
        with torch.no_grad():
            out = model(x_local)
            local_loss = loss_fn(out, y_local_target).item()
            local_acc = compute_accuracy(out, y_local)

        updated_weights = _model_to_ndarrays(model)
        drift = compute_drift(parameters, updated_weights)
        weight_norm = float(sum(np.sum(w**2) for w in updated_weights) ** 0.5)
        # IPR is GrokNet-specific; NaN on other architectures.
        local_ipr = compute_ipr(model)["ipr"] if dft_applicable(model, cfg) else float("nan")
        _client_probe = mechanistic_probe(cfg)

        # Extract this client's signature matrix for per-client mechanistic
        # analysis (only at checkpoint rounds). Serialized as bytes since Flower
        # metrics only support Scalar types. The matrix is architecture-specific
        # (see metrics.probes.client_signature) — it used to be W1[:, :cfg.p]
        # behind a GrokNet gate, so no non-modular run captured anything.
        server_round = int(config.get("server_round", 0))
        ckpt_every = int(config.get("checkpoint_every", 0))
        is_checkpoint_round = (ckpt_every > 0 and server_round > 0
                               and server_round % ckpt_every == 0)

        metrics_dict = {"loss": local_loss, "accuracy": local_acc, "drift": drift,
             "weight_norm": weight_norm, "ipr": local_ipr}
        if config.get("checkpoint_client_weights", False) and is_checkpoint_round:
            sig_name, sig = client_signature(model, cfg)
            if sig is None:
                warnings.warn(
                    f"checkpoint_client_weights is on but model {cfg.model!r} "
                    f"exposes no signature matrix; per-client weights will not "
                    f"be saved for this run.", RuntimeWarning, stacklevel=2)
            else:
                metrics_dict["w1_first_p"] = sig.astype(np.float32).tobytes()
                metrics_dict["w1_shape_0"] = sig.shape[0]
                metrics_dict["w1_shape_1"] = sig.shape[1]
                metrics_dict["w1_name"] = sig_name

        # Per-client mechanistic probe, every round rather than only at
        # checkpoints — cheap, and it makes the per-client story available even
        # on runs that never enable checkpointing.
        for key, value in _client_probe(model, x_local, y_local, cfg).items():
            metrics_dict[f"client_{key}"] = value

        # SCAFFOLD: update this client's control variate and report Δc_i so the
        # server can update c. lr is the local learning rate; n_steps the number
        # of local gradient steps actually taken (full-batch or minibatch).
        if scaffold:
            new_cv, delta_cv = _sc.client_cv_update(
                parameters, updated_weights, server_cv, client_cv,
                lr=cfg.lr, n_steps=n_steps,
            )
            _sc.set_client_cv(cv_key, new_cv)
            metrics_dict["scaffold_dc"] = _sc._cv_bytes(delta_cv)

        return (
            updated_weights,
            len(y_local),
            metrics_dict,
        )

    def evaluate(self, parameters, config):
        # Server-side evaluation handles global metrics; skip client eval
        return 0.0, 0, {}


# ── Strategy builder ─────────────────────────────────────────────────────────

def _build_strategy(cfg, init_params, evaluate_fn, fit_metrics_aggregation_fn=None,
                    scaffold_ctx=None):
    """Build Flower strategy based on config.strategy field."""
    # SCAFFOLD ships the server control variate c to clients in the fit config;
    # the on_fit_config closure reads the live c from scaffold_ctx each round.
    def _on_fit_config(rnd):
        config = _cfg_to_fit_config(cfg, rnd)
        if scaffold_ctx is not None:
            from fedgrok.training import scaffold as _sc
            config["scaffold_cv"] = _sc._cv_bytes(scaffold_ctx["server_cv_box"][0])
        return config

    common_kwargs = dict(
        fraction_fit=cfg.fraction_train,
        fraction_evaluate=0.0,
        min_fit_clients=max(1, int(cfg.num_clients * cfg.fraction_train)),
        min_available_clients=cfg.num_clients,
        initial_parameters=init_params,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=_on_fit_config,
    )
    if fit_metrics_aggregation_fn is not None:
        common_kwargs["fit_metrics_aggregation_fn"] = fit_metrics_aggregation_fn

    if cfg.strategy == "scaffold":
        # SCAFFOLD's Option-II control variate is c_i+ = c_i - c + (x - y_i)/(eta*K),
        # which inverts x - y_i = eta * sum(g). That identity holds for SGD. Under
        # AdamW the update is eta * sum(m_hat / (sqrt(v_hat) + eps)) plus decoupled
        # decay, so dividing by eta*K recovers a per-coordinate PRECONDITIONED sum,
        # not the gradient sum -- the resulting c_i is wrong by a factor that varies
        # per coordinate. It would still produce plausible numbers, and SCAFFOLD is
        # the load-bearing "is drift the mechanism?" arm, so fail loudly instead.
        if cfg.optimizer == "adamw":
            raise ValueError(
                "SCAFFOLD is not valid with optimizer='adamw': its control-variate "
                "estimator (x - y_i)/(lr * n_steps) assumes SGD, so under Adam's "
                "per-coordinate preconditioning c_i is systematically wrong. Use "
                "optimizer='gd', or compare drift correction with FedProx, whose "
                "proximal term makes no such assumption."
            )
        from fedgrok.training.scaffold import ScaffoldStrategy
        return ScaffoldStrategy(
            **common_kwargs,
            server_cv_box=scaffold_ctx["server_cv_box"],
            num_total_clients=cfg.num_clients,
            param_shapes=scaffold_ctx["param_shapes"],
        )

    if cfg.strategy == "fedadam":
        # Server-side adaptive optimiser (Adam) on the pseudo-gradient.
        return FedAdam(**common_kwargs, eta=cfg.server_lr, tau=cfg.tau)
    if cfg.strategy == "fedyogi":
        # Server-side Yogi — Adam's sign-based sibling; often stronger than Adam.
        return FedYogi(**common_kwargs, eta=cfg.server_lr, tau=cfg.tau)
    if cfg.strategy == "fedavgm":
        # Server-side heavy-ball momentum on the pseudo-gradient. This IS
        # DiLoCo's outer optimiser, so it bridges to the local-SGD-at-scale line.
        return FedAvgM(
            **common_kwargs,
            server_learning_rate=cfg.server_lr,
            server_momentum=cfg.server_momentum,
        )
    # "fedavg" and "fedprox" both use the plain FedAvg strategy; FedProx's
    # proximal term is applied client-side in GrokClient.fit().
    if cfg.strategy in ("fedavg", "fedprox"):
        return FedAvg(**common_kwargs)

    # Anything else is a mistake, and it must not be a silent one. This used to
    # fall through to FedAvg, so `strategy="feddyn"` -- which FedConfig's Literal
    # advertised and no branch implements -- ran plain FedAvg and banked a result
    # row labelled `strategy: "feddyn"`. A typo ("fedAvgM", "fed_adam") did the
    # same. The Literal is a type annotation and is not enforced at runtime, so
    # this is the only place the check can happen.
    raise ValueError(
        f"Unknown strategy {cfg.strategy!r}. Implemented: fedavg, fedprox, "
        f"fedadam, fedavgm, fedyogi, scaffold. FedDyn is NOT implemented -- it "
        f"needs per-client state like SCAFFOLD; use fedprox for a "
        f"proximal-regularisation arm."
    )


# ── Flower Server + Evaluation ───────────────────────────────────────────────

def fed_train(cfg: FedConfig):
    """Run FedAvg via Flower simulation. Returns history dict and final model."""
    torch.manual_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Drop cached state from any previous run in this process. Both caches are
    # keyed on the full data/model identity, so a stale entry cannot be
    # mis-served -- but clearing frees their GPU memory when many runs share one
    # process (the test suite, and experiments/exp_mechanistic_checkpoints.py,
    # which calls fed_train five times). _dataset_cache used to be left alone
    # here, so it persisted across runs under a key that could not tell two
    # setups apart.
    _client_cache.clear()
    _dataset_cache.clear()
    _optimizer_cache.clear()

    # Precompute global data (single call, also populates cache for clients)
    client_data, x_train_full, y_train_full, x_test, y_test = _get_cached_datasets(cfg)

    # Move evaluation data to device, then prepare loss targets there.
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    x_train_full = x_train_full.to(device)
    y_train_full = y_train_full.to(device)

    loss_spec = build_loss(cfg)
    n_classes = dataset_dims(cfg)[1]
    y_test_target = loss_spec.prepare_target(y_test, n_classes)
    y_train_full_target = loss_spec.prepare_target(y_train_full, n_classes)

    # Print partition sizes
    print(f"Clients: {cfg.num_clients}, partition: {cfg.partition}, "
          f"samples per client: {[len(y) for _, y in client_data]}")

    # History (captured by closure in evaluate_fn)
    history = {
        "round": [], "total_steps": [], "sequential_steps": [],
        "n_participating": [],
        "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": [],
        "weight_norm_layer1": [], "weight_norm_layer2": [],
        "ipr": [],
        "mean_client_drift": [],
        "client_weight_divergence": [],
    }

    # Mutable container to capture final model parameters from evaluate_fn
    _final_ndarrays = [None]

    loss_fn = loss_spec.loss_fn
    start_time = time.time()

    # Mutable container for inter-callback communication (closure-shared).
    # Flower calls aggregate_fit (hence _aggregate_fit_metrics) before
    # evaluate_fn within a round, so evaluate_fn reads this round's values.
    # "client_probes" must be initialised here, not only in _aggregate_fit_metrics:
    # evaluate_fn runs at round 0, before any client has fit. Seeding it with the
    # probe's declared key names at NaN keeps every history series the same
    # length -- otherwise the client-probe series would start at round 1 and be
    # one entry shorter than `round`, which silently misaligns any x/y plot of
    # the two.
    _client_probe_seed = {
        f"client_{name}_{stat}": float("nan")
        for name in probe_keys(cfg) for stat in ("mean", "std")
    }
    _round_metrics = {"mean_drift": 0.0, "weight_divergence": 0.0,
                      "n_participating": 0, "samples_this_round": 0,
                      "client_probes": dict(_client_probe_seed)}
    _client_w1_cache = {"data": None}

    # Running count of centralized-equivalent gradient steps (see evaluate_fn).
    _step_accum = {"total": 0.0}
    n_train_total = len(y_train_full)

    # Setup-appropriate mechanistic probe, resolved once (see metrics/probes.py).
    _probe_fn = mechanistic_probe(cfg)

    def _aggregate_fit_metrics(metrics_list):
        """Aggregate per-client fit metrics from GrokClient.fit()."""
        drifts = [m.get("drift", 0.0) for _, m in metrics_list]
        w_norms = [m.get("weight_norm", 0.0) for _, m in metrics_list]

        _round_metrics["mean_drift"] = float(np.mean(drifts)) if drifts else 0.0
        _round_metrics["weight_divergence"] = float(np.std(w_norms)) if len(w_norms) > 1 else 0.0

        # Per-client probe values -> mean and spread across clients. The spread
        # is the interesting one: it says whether clients are converging on the
        # same circuit or diverging, which is the structured-vs-random question.
        _round_metrics["client_probes"] = dict(_client_probe_seed)
        probe_names = {k for _, m in metrics_list for k in m if k.startswith("client_")}
        for name in sorted(probe_names):
            vals = [float(m[name]) for _, m in metrics_list if name in m]
            if vals:
                _round_metrics["client_probes"][f"{name}_mean"] = float(np.mean(vals))
                _round_metrics["client_probes"][f"{name}_std"] = float(np.std(vals))

        # Actual work done this round, taken from what clients reported rather
        # than from fraction_train. This is exact under partial participation
        # and under Dirichlet partitions, where shards have unequal sizes.
        _round_metrics["n_participating"] = len(metrics_list)
        _round_metrics["samples_this_round"] = sum(int(n) for n, _ in metrics_list)

        # Capture per-client W1 if available (deserialize from bytes)
        w1_list = []
        for _, m in metrics_list:
            w1_bytes = m.get("w1_first_p")
            if w1_bytes is not None:
                shape = (int(m["w1_shape_0"]), int(m["w1_shape_1"]))
                w1_list.append(np.frombuffer(w1_bytes, dtype=np.float32).reshape(shape))
        _client_w1_cache["data"] = w1_list if w1_list else None

        return {
            "mean_drift": _round_metrics["mean_drift"],
            "weight_divergence": _round_metrics["weight_divergence"],
        }

    # One eval model, reused across rounds (was rebuilt + moved to device every
    # round). evaluate_fn runs in the server process, so this is a plain closure
    # variable, not a Ray-actor cache. It is assigned below, AFTER init_model, so
    # its construction does not consume the torch RNG draw that seeds the initial
    # global weights — otherwise the whole run starts from a different point.
    _eval_model_box = [None]

    def evaluate_fn(server_round, parameters, config):
        """Centralized evaluation after each aggregation round."""
        _final_ndarrays[0] = parameters  # capture for final model reconstruction

        # total_steps must accumulate on EVERY round, independent of eval_every,
        # or skipped rounds would silently drop their gradient work. This is the
        # compute-matched x-axis: each round the participating clients compute E
        # local steps over `samples_this_round` examples, i.e. an equivalent of
        # E * samples_this_round / n_train_total full-training-set steps. At full
        # participation this reduces to E per round (the old `round * E`); it
        # over-counted only under partial participation.
        _step_accum["total"] += (
            cfg.local_epochs * _round_metrics["samples_this_round"] / n_train_total
        )

        is_checkpoint_round = (
            cfg.checkpoint_every > 0 and server_round > 0
            and server_round % cfg.checkpoint_every == 0
        )
        # Round 0 (initial state), the final round, checkpoint rounds, and every
        # eval_every-th round are logged; the rest skip the forward passes.
        is_eval_round = (
            server_round % max(1, cfg.eval_every) == 0
            or server_round == cfg.num_rounds
            or is_checkpoint_round
        )
        if not is_eval_round:
            return 0.0, {}

        model = _eval_model_box[0]
        _load_ndarrays_into(model, parameters)

        model.eval()
        with torch.no_grad():
            out_test = model(x_test)
            test_loss = loss_fn(out_test, y_test_target).item()
            test_acc = compute_accuracy(out_test, y_test)
            out_train = model(x_train_full)
            train_loss = loss_fn(out_train, y_train_full_target).item()
            train_acc = compute_accuracy(out_train, y_train_full)

        # Two independent capabilities — see the note in training/centralized.py.
        # Frobenius norms are basis-free and so are valid on S5; only the DFT
        # needs a cyclic group.
        if weight_norms_applicable(model):
            wn = weight_norms(model)
            wn1, wn2 = wn["weight_norm_layer1"], wn["weight_norm_layer2"]
        else:
            wn1 = wn2 = float("nan")
        ipr_val = compute_ipr(model)["ipr"] if dft_applicable(model, cfg) else float("nan")
        report = weight_norm_report(model)
        probe = _probe_fn(model, x_test, y_test, cfg)

        history["round"].append(server_round)
        # sequential_steps = rounds * E is the depth of the update chain,
        # independent of participation; total_steps (accumulated above) is the
        # compute-matched axis. The two diverge once fraction_train < 1.
        history["total_steps"].append(_step_accum["total"])
        history["sequential_steps"].append(server_round * cfg.local_epochs)
        history["n_participating"].append(_round_metrics["n_participating"])
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["weight_norm_layer1"].append(wn1)
        history["weight_norm_layer2"].append(wn2)
        history["ipr"].append(ipr_val)
        history["mean_client_drift"].append(_round_metrics["mean_drift"])
        history["client_weight_divergence"].append(_round_metrics["weight_divergence"])
        # Additive keys, kept in lockstep by setdefault: round 0 is always an
        # eval round, so every key is created on the first pass.
        for key, value in {**report, **probe,
                           **_round_metrics["client_probes"]}.items():
            history.setdefault(key, []).append(value)

        # Save checkpoint if requested
        if is_checkpoint_round:
            ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            # Global model checkpoint
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_round{server_round}.pt")
            torch.save(model.state_dict(), ckpt_path)

            # Global Fourier spectrum (GrokNet-specific; skipped otherwise)
            if dft_applicable(model, cfg):
                spec = fourier_spectrum(model)
                spec_path = os.path.join(ckpt_dir, f"spectrum_round{server_round}.pt")
                torch.save(spec["spectrum"], spec_path)

            # Per-client W1 (if available)
            if _client_w1_cache["data"] is not None:
                client_path = os.path.join(ckpt_dir, f"client_w1_round{server_round}.pt")
                torch.save(_client_w1_cache["data"], client_path)

        if server_round % max(1, cfg.num_rounds // 10) == 0:
            elapsed = time.time() - start_time
            print(
                f"[Round {server_round:>4d}/{cfg.num_rounds}]  "
                f"train_loss={train_loss:.6f}  test_loss={test_loss:.6f}  "
                f"train_acc={train_acc:.1f}%  test_acc={test_acc:.1f}%  "
                f"ipr={ipr_val:.4f}  ({elapsed:.1f}s)"
            )

        return test_loss, {"test_accuracy": test_acc, "train_accuracy": train_acc}

    # Initial model for FedAvg. Its RNG draw MUST come before the eval model's,
    # since these initial weights are the run's starting point.
    init_model = _make_model(cfg)
    init_ndarrays = _model_to_ndarrays(init_model)
    init_params = ndarrays_to_parameters(init_ndarrays)

    # Now safe to build the reusable eval model (params get overwritten each
    # round, so its own initial values are irrelevant — only the RNG order was).
    _eval_model_box[0] = _make_model(cfg).to(device)

    # SCAFFOLD: server control variate c (starts at 0) + per-client c_i store,
    # shared with the strategy and the on_fit_config closure via scaffold_ctx.
    scaffold_ctx = None
    if cfg.strategy == "scaffold":
        from fedgrok.training import scaffold as _sc
        _sc.reset_client_cv()
        scaffold_ctx = {
            "server_cv_box": [_sc.zeros_like_params(init_ndarrays)],
            "param_shapes": [a.shape for a in init_ndarrays],
        }

    # ── Build Flower apps ────────────────────────────────────────────────

    def client_fn(context: Context):
        partition_id = context.node_config["partition-id"]
        return GrokClient(partition_id=partition_id).to_client()

    # Capture cfg in closure for on_fit_config_fn
    fed_cfg = cfg

    def server_fn(context: Context):
        strategy = _build_strategy(fed_cfg, init_params, evaluate_fn,
                                   fit_metrics_aggregation_fn=_aggregate_fit_metrics,
                                   scaffold_ctx=scaffold_ctx)
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=fed_cfg.num_rounds),
        )

    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)

    # ── Run simulation ───────────────────────────────────────────────────
    prox_info = f", proximal_mu={cfg.proximal_mu}" if cfg.proximal_mu > 0 else ""
    print(f"\nStarting Flower simulation: {cfg.num_rounds} rounds, "
          f"{cfg.num_clients} clients, {cfg.local_epochs} local epochs{prox_info}\n")

    # Allocate fractional CUDA GPUs across clients so all K can be co-scheduled
    # on one device; MPS is used via PyTorch directly (not managed by Ray), so
    # num_gpus stays 0 for MPS. The launcher pins one device per run with
    # CUDA_VISIBLE_DEVICES, so "one GPU" here means that run's assigned device.
    #
    # The divisor is the concurrency cap, not K, so FEDGROK_GPU_CLIENT_CAP can
    # hold VRAM flat as K grows -- see the client-placement note above. Left
    # unset the divisor is K, which is the original behaviour exactly.
    num_gpus = 0.0
    if torch.cuda.is_available() and not _clients_on_cpu():
        num_gpus = 1.0 / _gpu_client_cap(cfg.num_clients)

    # Ray init tuning: the dashboard and the metrics exporter agent are pure
    # overhead for a short-lived single-run simulation. The exporter in
    # particular floods the logs with "Failed to establish connection to the
    # metrics exporter agent ... repeated Nx across cluster" retries. Disabling
    # both removes that traffic. log_to_driver=False keeps client stdout out of
    # the server log.
    backend_config = {
        "client_resources": {"num_cpus": 1, "num_gpus": num_gpus},
        "init_args": {
            "include_dashboard": False,
            "log_to_driver": False,
            "_metrics_export_port": None,
        },
    }

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config=backend_config,
    )

    # ── Reconstruct final trained model ──────────────────────────────────
    final_model = _make_model(cfg)
    if _final_ndarrays[0] is not None:
        state_dict = _ndarrays_to_state_dict(_final_ndarrays[0], final_model)
        final_model.load_state_dict(state_dict)
    final_model.to(device)

    # ── Save results ─────────────────────────────────────────────────────
    os.makedirs(cfg.output_dir, exist_ok=True)
    dirichlet_suffix = f"_dir{cfg.dirichlet_alpha}" if cfg.partition == "dirichlet" else ""
    prox_suffix = f"_mu{cfg.proximal_mu}" if cfg.proximal_mu > 0 else ""
    adam_suffix = f"_adam_tau{cfg.tau}" if cfg.strategy == "fedadam" else ""
    slr_suffix = f"_slr{cfg.server_lr}" if cfg.strategy == "fedadam" else ""
    wd_suffix = f"_wd{cfg.weight_decay}" if cfg.weight_decay > 0 else ""
    # The setup must be in the name. Without it an S5 or MNIST run inherits the
    # Config defaults for task/p and is written as "..._addition_p97_...", which
    # is not just confusing but collides with a genuine mod-97 addition run when
    # both share an output_dir. (The v2 sweep path gives every run its own
    # directory keyed by run id, so this is a naming fix, not a data-loss one --
    # but the legacy experiments/ scripts do share directories.)
    setup_prefix = f"{cfg.dataset}_{cfg.model}_{cfg.loss}"
    task_part = f"{cfg.task}_p{cfg.p}" if cfg.dataset == "modular" else (
        f"n{cfg.group_n}" if cfg.dataset == "s5" else f"n{cfg.n_train}")
    tag = (f"fed_{setup_prefix}_{task_part}_{cfg.optimizer}_N{cfg.hidden_width}"
           f"_a{cfg.alpha}_K{cfg.num_clients}_le{cfg.local_epochs}"
           f"_ft{cfg.fraction_train}_{cfg.partition}{dirichlet_suffix}"
           f"{prox_suffix}{adam_suffix}{slr_suffix}{wd_suffix}_s{cfg.seed}")
    history_path = os.path.join(cfg.output_dir, f"history_{tag}.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"\nHistory saved to {history_path}")

    if cfg.save_weights:
        weights_path = os.path.join(cfg.output_dir, f"weights_{tag}.pt")
        torch.save(final_model.state_dict(), weights_path)
        print(f"Weights saved to {weights_path}")

    return history, final_model
