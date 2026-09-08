"""Modular arithmetic dataset over Z_p.

Each sample is a pair (n, m) encoded as two concatenated one-hot vectors (size 2p).
The target is f(n, m) mod p, encoded as a one-hot vector (size p).

This module is the single source of truth for both the input grid and the
train/test split.  `fedgrok.data.partition` composes the same two functions so
that a federated run and a centralized run with the same config see exactly the
same split.
"""

import torch
import numpy as np
from fedgrok.core.config import Config


# ── Task functions ──────────────────────────────────────────────────────────

def _division(n, m, p):
    """(n / m) mod p via Fermat's little inverse.  Requires m != 0."""
    if m == 0:
        raise ValueError(
            "division is undefined at m=0; it is defined on the p*(p-1) grid. "
            "Use build_encoded_grid(), which applies the domain restriction."
        )
    return (n * pow(int(m), int(p - 2), int(p))) % p


TASKS = {
    "addition":        lambda n, m, p: (n + m) % p,
    "subtraction":     lambda n, m, p: (n - m) % p,
    "division":        _division,
    "x2_plus_y2":      lambda n, m, p: (n**2 + m**2) % p,
    "x_plus_y_squared": lambda n, m, p: ((n + m)**2) % p,
    "multiplication":  lambda n, m, p: (n * m) % p,
    "x2_y2_xy":        lambda n, m, p: (n**2 + m**2 + n * m) % p,
    "x3_xy2_y":        lambda n, m, p: (n**3 + n * m**2 + m) % p,
}

# Tasks whose second operand must be non-zero.  Division by zero is undefined in
# Z_p; the original implementation mapped m=0 to a fabricated label 0, which
# injected p spurious samples all sharing a single class.  Power et al. define
# division on the p*(p-1) grid, which is what we use.
NONZERO_SECOND_OPERAND = {"division"}

# Tasks with a known structural caveat, surfaced here so they are not chosen for
# a sweep by accident:
#
#   multiplication  On the non-zero residues this is the cyclic group Z_{p-1}
#                   under discrete log, so it exercises the *same* Fourier
#                   machinery as addition, just over a composite-order group
#                   (p-1 = 96 for p=97, which is 2^5 * 3 and has a rich subgroup
#                   lattice).  It is therefore a confound rather than a control
#                   when used as a "different operation".  It also keeps all
#                   2p-1 zero-product pairs, over-representing class 0 ~2x.
DEGENERATE_TASKS = {
    "multiplication": "cyclic Z_{p-1} in disguise; retains 2p-1 zero-product pairs",
}


def task_operands(task: str, p: int):
    """Return the (n, m) index arrays of all valid input pairs for `task`.

    Applies the task's domain restriction, so division yields p*(p-1) pairs
    while the others yield p*p.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}. Options: {sorted(TASKS)}")

    ns = np.arange(p)
    ms = np.arange(1, p) if task in NONZERO_SECOND_OPERAND else np.arange(p)
    nn, mm = np.meshgrid(ns, ms, indexing="ij")
    return nn.flatten(), mm.flatten()


def build_encoded_grid(task: str, p: int):
    """Build the full one-hot encoded dataset for `task` over Z_p.

    Returns:
        x       float32 (n_samples, 2p) — concatenated one_hot(n), one_hot(m)
        labels  int     (n_samples,)    — f(n, m) mod p
        nn, mm  int     (n_samples,)    — the operand indices, kept because the
                                          operand-based federated partition
                                          needs the first operand per sample
    """
    task_fn = TASKS[task]
    nn, mm = task_operands(task, p)
    labels = np.array([task_fn(int(n), int(m), p) for n, m in zip(nn, mm)])

    n_samples = len(nn)
    x = np.zeros((n_samples, 2 * p), dtype=np.float32)
    x[np.arange(n_samples), nn] = 1.0
    x[np.arange(n_samples), p + mm] = 1.0

    return x, labels, nn, mm


def split_indices(n_samples: int, alpha: float, seed: int = None, rng=None):
    """Return (train_idx, test_idx) for a random train/test split.

    The single source of truth for the split.  Both make_dataset and
    fedgrok.data.partition.make_federated_datasets call this, so a centralized
    run and a federated run with the same config train on identical data.

    Pass `rng` instead of `seed` when the caller needs to keep drawing from the
    same stream afterwards.  The federated path does this: its partitioners
    consume the *same* RandomState after the permutation has advanced it, so
    handing them a fresh one would silently change every IID and Dirichlet
    assignment relative to previously published runs.
    """
    if rng is None:
        if seed is None:
            raise ValueError("split_indices requires either `seed` or `rng`")
        rng = np.random.RandomState(seed)

    # alpha must leave BOTH sides non-empty. alpha >= 1.0 trains on the whole
    # grid and leaves no held-out set, which fails loudly nowhere downstream:
    # compute_accuracy divides by zero, every test point becomes NaN, and NaN
    # used to pass the sustained-crossing scan in compute_t_grok because
    # `nan < threshold` is False. The run was then banked as `grokked=True,
    # t_grok=0` -- a run that measured nothing, recorded as the strongest
    # possible result. Five setup-D rows in `x_d_alpha_high` are exactly that
    # (RESULTS.md, known issues). compute_t_grok is guarded too now, but the
    # spec should never have been runnable in the first place.
    #
    # The permutation is drawn AFTER the check, and exactly once either way, so
    # the RNG stream the federated partitioners inherit is unchanged.
    n_train = int(alpha * n_samples)
    if n_train <= 0 or n_train >= n_samples:
        raise ValueError(
            f"alpha={alpha} gives n_train={n_train} of {n_samples} samples, "
            f"leaving one side of the split empty. alpha is the TRAINING "
            f"fraction and must satisfy 0 < alpha < 1; a run with no held-out "
            f"set cannot measure generalisation, which is the quantity every "
            f"grok metric is defined on."
        )

    perm = rng.permutation(n_samples)
    return perm[:n_train], perm[n_train:]


def make_dataset(cfg: Config):
    """Build full dataset and split into train/test.

    Returns:
        x_train, y_train, x_test, y_test  (all torch tensors on CPU)
        x are float tensors of shape (num_samples, 2p)  — concatenated one-hots
        y are long  tensors of shape (num_samples,)      — class labels (0..p-1)
    """
    x, labels, _, _ = build_encoded_grid(cfg.task, cfg.p)
    train_idx, test_idx = split_indices(len(labels), cfg.alpha, cfg.seed)

    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(labels[train_idx]).long()
    x_test = torch.from_numpy(x[test_idx])
    y_test = torch.from_numpy(labels[test_idx]).long()

    return x_train, y_train, x_test, y_test
