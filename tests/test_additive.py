"""The additive (single-operand) split: exact on a complete grid, and it agrees
with the exact quadratic split on the model where both are defined."""
import torch

from fedgrok.core.config import Config
from fedgrok.core.registry import build_model
from fedgrok.metrics import quadratic_circuits as qc
from fedgrok.metrics.additive import (
    additive_report, additive_split, grid_logits, heldout_mask,
)


def test_split_recovers_a_planted_decomposition():
    torch.manual_seed(0)
    G, P = 12, 7
    mu = torch.randn(1, 1, P)
    f = torch.randn(G, 1, P); f = f - f.mean(0, keepdim=True)
    g = torch.randn(1, G, P); g = g - g.mean(1, keepdim=True)
    R = torch.randn(G, G, P)
    R = R - R.mean(0, keepdim=True) - R.mean(1, keepdim=True) + R.mean((0, 1), keepdim=True)
    L = mu + f + g + R
    mu_, f_, g_, R_ = additive_split(L)
    assert torch.allclose(mu_, mu, atol=1e-5)
    assert torch.allclose(f_, f, atol=1e-5)
    assert torch.allclose(g_, g, atol=1e-5)
    assert torch.allclose(R_, R, atol=1e-5)
    assert torch.allclose(mu_ + f_ + g_ + R_, L, atol=1e-6)


def _s4_groknet(seed=0):
    cfg = Config(dataset="s5", group_n=4, model="groknet", hidden_width=16,
                 activation="quadratic", alpha=0.5, seed=seed)
    torch.manual_seed(seed)
    return build_model(cfg), cfg


def test_single_operand_model_has_no_interaction():
    """Zero the second-operand block: the logit depends on `a` alone, so the
    fit must put everything in f and nothing in g or R."""
    model, cfg = _s4_groknet()
    with torch.no_grad():
        model.W1[:, model.W1.shape[1] // 2:] = 0.0
    rep = additive_report(model, cfg)
    assert rep["share_b"] < 1e-5
    assert rep["share_interaction"] < 1e-5
    # share_a is not 1: the grand mean mu carries class structure of its own
    # and is centred separately, exactly as A's mean is inside the exact split.
    assert rep["share_a"] > 0.05
    # With f removed nothing class-varying is left, so both ablations are at
    # chance (24 classes); the exact argmax of a near-zero residual is noise.
    assert rep["acc_no_a"] < 15.0 and rep["acc_interaction"] < 15.0
    assert rep["acc_no_b"] == rep["acc_full"]
    assert rep["acc_additive"] == rep["acc_full"]


def test_agrees_with_the_exact_quadratic_split():
    """On a quadratic GrokNet the additive fit's marginal energy equals the
    exact A + B energy up to T's own row/column means, which are small for a
    random network; the full-model accuracy is identical by construction."""
    model, cfg = _s4_groknet(seed=1)
    L, Y = grid_logits(model, cfg)
    mask = heldout_mask(cfg)
    rep = additive_report(model, cfg, mask=mask)
    x = torch.zeros(int(mask.sum()), 2 * L.shape[0])
    ia, ib = mask.nonzero(as_tuple=True)
    x[torch.arange(len(ia)), ia] = 1.0
    x[torch.arange(len(ib)), L.shape[0] + ib] = 1.0
    exact = qc.circuit_report(model, x, Y[mask])
    assert abs(rep["acc_full"] - qc.compute_accuracy(model(x), Y[mask])) < 1e-9
    assert abs(rep["share_interaction"] - exact["circ_share_interaction"]) < 0.15
    assert rep["acc_interaction"] > 0.0
