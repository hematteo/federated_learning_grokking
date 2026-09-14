"""Single-operand (additive) split of a two-operand model's logits, by least squares.

WHY THIS EXISTS. `quadratic_circuits` splits a quadratic GrokNet's logit EXACTLY
into A[c,a] + 2T[c,a,b] + B[c,b], which is what lets RESULTS §16.3 and §24 say
whether a failure is a missing compositional circuit or a circuit hidden under
single-operand terms. The transformer (setups B and C) has no such identity --
but the question is the same, and on a complete a x b grid the least-squares
additive fit has a closed form:

    L[a, b, :] = mu + f[a, :] + g[b, :] + R[a, b, :]

with f and g the centred row and column means of L over the grid and R the
residual -- everything that genuinely depends on both operands. Scoring the model
with f, g or both removed is then the same ablation `quadratic_circuits` performs
with A and B, for any architecture that takes the concatenated one-hot pair.

CHECKED AGAINST THE EXACT SPLIT. On setup D the two agree to a fraction of a
point (interaction-only accuracy 97.8 against T-only 97.7 on the iid K=10 cell,
94.6 against 94.4 under the coset partition; RESULTS §24.4). They are not
identical by construction: T's own row and column means are absorbed into f and
g here, so this fit puts slightly LESS energy in the interaction term than the
exact split does. That direction is conservative for the masking reading.

The grid must be complete (every (a, b) pair present). `build_sn_grid` and the
modular addition grid are; the modular division task drops m = 0 and is refused
rather than fitted approximately.
"""

import torch

from fedgrok.data.registry import build_dataset, dataset_dims, dataset_grid
from fedgrok.metrics.fourier import compute_accuracy


def additive_split(logits):
    """(mu, f, g, R) for logits of shape (G, G, P) on the complete grid.

    All four broadcast to (G, G, P) and sum to `logits` exactly. This is the
    least-squares additive fit: on a complete grid the row and column means are
    the orthogonal projection onto {f[a] + g[b]}.
    """
    if logits.dim() != 3 or logits.shape[0] != logits.shape[1]:
        raise ValueError(f"expected (G, G, P) logits, got {tuple(logits.shape)}")
    mu = logits.mean(dim=(0, 1), keepdim=True)
    f = logits.mean(dim=1, keepdim=True) - mu
    g = logits.mean(dim=0, keepdim=True) - mu
    return mu, f, g, logits - mu - f - g


def centred_energy(term):
    """Mean over examples of the squared class-centred logit -- see
    `quadratic_circuits._centred_energy` for why centring is the right measure."""
    flat = term.reshape(-1, term.shape[-1])
    centred = flat - flat.mean(dim=1, keepdim=True)
    return float((centred ** 2).sum(dim=1).mean())


def grid_logits(model, cfg, batch=2048):
    """Logits on the complete grid as (G, G, P), with the labels as (G, G).

    Pairs are placed by the argmax of the two one-hot blocks, so this does not
    depend on the order `dataset_grid` enumerates them in.
    """
    x, labels, _ = dataset_grid(cfg)
    x = torch.as_tensor(x, dtype=torch.float32)
    labels = torch.as_tensor(labels)
    _, n_classes = dataset_dims(cfg)
    width = x.shape[1] // 2
    G = width
    if x.shape[0] != G * G:
        raise ValueError(
            f"{cfg.dataset}/{cfg.task}: the grid has {x.shape[0]} pairs, not "
            f"{G * G}. The additive fit needs every (a, b) pair present."
        )
    ia, ib = x[:, :width].argmax(dim=1), x[:, width:].argmax(dim=1)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(x[i:i + batch].to(device)).cpu()
                         for i in range(0, x.shape[0], batch)])
    if was_training:
        model.train()
    L = torch.empty(G, G, out.shape[1], dtype=out.dtype)
    L[ia, ib] = out
    Y = torch.empty(G, G, dtype=labels.dtype)
    Y[ia, ib] = labels
    return L, Y


def heldout_mask(cfg):
    """(G, G) boolean mask of the pairs in cfg's held-out split."""
    _, _, x_test, _ = build_dataset(cfg)
    width = x_test.shape[1] // 2
    ia, ib = x_test[:, :width].argmax(dim=1), x_test[:, width:].argmax(dim=1)
    mask = torch.zeros(width, width, dtype=torch.bool)
    mask[ia, ib] = True
    return mask


def additive_report(model, cfg, mask=None):
    """Ablation scores of `model` on cfg's held-out pairs (or on `mask`).

    acc_full            the model as trained
    acc_interaction     R alone: both single-operand terms removed
    acc_no_a / acc_no_b one single-operand term removed
    acc_additive        mu + f + g alone: the ceiling reachable without composing
    share_a/b/interaction   centred logit energy of each term over that of the
                        full logit (the terms are not orthogonal, so the three
                        need not sum to one)
    """
    L, Y = grid_logits(model, cfg)
    mu, f, g, R = additive_split(L)
    if mask is None:
        mask = heldout_mask(cfg)
    full = L
    scored = {
        "acc_full": full,
        "acc_interaction": R,
        "acc_no_a": L - f,
        "acc_no_b": L - g,
        "acc_additive": (mu + f + g).expand_as(L),
    }
    out = {k: compute_accuracy(v[mask], Y[mask]) for k, v in scored.items()}
    total = centred_energy(full) or 1.0
    out["share_a"] = centred_energy(f.expand_as(L)) / total
    out["share_b"] = centred_energy(g.expand_as(L)) / total
    out["share_interaction"] = centred_energy(R) / total
    return out
