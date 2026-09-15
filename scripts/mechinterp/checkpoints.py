"""Per-checkpoint mechanistic metrics for every run with saved weights.

    venv/bin/python scripts/mechinterp/checkpoints.py                    # all runs, resumable
    venv/bin/python scripts/mechinterp/checkpoints.py --ids fede_x cent_y   # a few
    venv/bin/python scripts/mechinterp/checkpoints.py --shard 0/2           # split across processes
    venv/bin/python scripts/mechinterp/checkpoints.py --collect             # per-run JSON -> CSV

One JSON per run under results/mechinterp/ckpt/<id>.json; `--collect` flattens them
into results/mechinterp/checkpoint_metrics.csv (one row per checkpoint).

METRIC FAMILIES (column prefixes)

  fn_      function: train/test loss, accuracy, generalisation gap, median logit
           margin, softmax entropy, Gini of per-class test accuracy.
  w_<M>_   weight geometry per matrix M: Frobenius and spectral norm, stable rank
           (||M||_F^2 / sigma_1^2), entropy effective rank, Gini of singular values,
           of |entries| and of row (neuron) norms, participation ratio of row norms
           over rows, distance and cosine to the matrix at initialisation.
  w_all_   the same totals over every matrix; w_all_rel_norm = norm / init norm.
  rep_     penultimate representation: effective rank and participation ratio of the
           centred feature covariance (test split), neural-collapse NC1
           tr(Sigma_W Sigma_B^+)/C on the train split, between/within variance
           ratio on train and test, fraction of dead units (ReLU models).
  unit_    hidden units over the full input grid (grid setups): energy share of each
           unit's activation that depends on both operands (additive split per unit),
           and on modular tasks the 2-D DFT energy shares of unit activations at
           sum frequencies (w, w), difference frequencies (w, -w), single-operand
           frequencies and the rest, plus mean per-unit concentration on its
           dominant sum frequency.
  add_     additive split of the logits on the grid (metrics/additive): accuracy of
           the interaction term alone, of the single-operand terms alone, and
           centred-energy shares, on train and held-out pairs.
  circ_    exact quadratic circuit split (metrics/quadratic_circuits; quadratic MLP).
  fin_ / fU_ / fV_ / fout_   Z_p DFT of input and output weights (modular tasks):
           mean spectral IPR over rows, DC share, share at the top frequency, number
           of frequencies covering 90% of non-DC power, Gini and entropy-effective
           number of frequencies; for the quadratic MLP the fraction of strong
           neurons whose U and V (and W2) dominant frequencies agree.
  clock_   logit "clock" circuit (modular tasks): share of class-varying logit energy
           in the components exp(i w (a + b - c)), number of frequencies covering 90%
           of it, top frequency share, Jaccard overlap of this checkpoint's key set
           with the final checkpoint's; restricted / excluded loss and accuracy
           (Nanda et al.) with the FINAL checkpoint's key frequencies, on train and
           test pairs. Read the LOSSES: restricted accuracy is scale-free, so a tiny
           but correctly phased clock component already scores 100%.
  irr_<M>_ S5 isotypic energy fractions and structure score of input / output blocks,
           dominant irrep and its share (metrics/irreps).
  coset_   coset attribution (Stander et al.) for the S4 cosets and for the A5 cosets.
  attn_    transformer: attention from the answer position to the first operand, and
           its entropy, averaged over heads and the grid.
  sharp_   top Hessian eigenvalue of the training loss by power iteration and the full
           gradient norm, on a subset of checkpoints (every checkpoint nearest an event
           plus a spread); sharp_lr_lambda = lr * lambda (2 is GD's stability edge).
"""
import argparse
import dataclasses
import glob
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import common as C                                                   # noqa: E402
from fedgrok.core.registry import build_loss                         # noqa: E402
from fedgrok.metrics import irreps, quadratic_circuits               # noqa: E402
from fedgrok.metrics.nonabelian import coset_attribution             # noqa: E402

OUTDIR = os.path.join(C.OUT, "ckpt")
MAX_CKPTS = 24
SHARP_CKPTS = 8


# ── small numerics ───────────────────────────────────────────────────────────

def gini(x):
    x = np.sort(np.abs(np.asarray(x, dtype=np.float64)).ravel())
    if x.size == 0 or x.sum() <= 0:
        return math.nan
    n = x.size
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)


def eff_rank_from(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[v > 1e-12]
    if v.size == 0:
        return math.nan
    p = v / v.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def participation(values):
    v = np.asarray(values, dtype=np.float64)
    s2 = (v ** 2).sum()
    return float(v.sum() ** 2 / s2) if s2 > 0 else math.nan


def n_cover(shares, frac=0.9):
    s = np.sort(np.asarray(shares, dtype=np.float64))[::-1]
    if s.sum() <= 0:
        return math.nan
    return int(np.searchsorted(np.cumsum(s) / s.sum(), frac) + 1)


# ── model internals ──────────────────────────────────────────────────────────

def weight_mats(model):
    """{short name: 2-D tensor with rows = units/output directions}."""
    if hasattr(model, "W1"):
        return {"W1": model.W1.data, "W2": model.W2.data}
    if hasattr(model, "W_E"):
        h, d = model.n_heads, model.d_model
        return {"WE": model.W_E.data, "WU": model.W_U.data.t(),
                "WQ": model.W_Q.data.permute(0, 2, 1).reshape(-1, d),
                "WK": model.W_K.data.permute(0, 2, 1).reshape(-1, d),
                "WV": model.W_V.data.permute(0, 2, 1).reshape(-1, d),
                "WO": model.W_O.data.reshape(-1, d),
                "Win": model.W_in.data.t(), "Wout": model.W_out.data}
    return {f"L{i}": layer.weight.data for i, layer in enumerate(model.layers)}


def transformer_internals(model, x):
    h = model._embed(x)
    q = torch.einsum("bsd,hde->bhse", h, model.W_Q)
    k = torch.einsum("bsd,hde->bhse", h, model.W_K)
    v = torch.einsum("bsd,hde->bhse", h, model.W_V)
    attn = (torch.einsum("bhse,bhte->bhst", q, k) / math.sqrt(model.d_head)).softmax(-1)
    z = torch.einsum("bhst,bhte->bhse", attn, v)
    h = h + torch.einsum("bhse,hed->bsd", z, model.W_O)
    mlp = torch.relu(h @ model.W_in)
    h = h + mlp @ model.W_out
    return {"logits": h[:, -1, :] @ model.W_U, "feat": h[:, -1, :], "units": mlp[:, -1, :],
            "attn": attn}


def forward_parts(model, x, batch=4096):
    """logits, penultimate features and hidden units, batched, no grad."""
    outs = {"logits": [], "feat": [], "units": [], "attn": []}
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb = x[i:i + batch]
            if hasattr(model, "W1"):
                hid = model._activate(xb @ model.W1.t())
                parts = {"logits": hid @ model.W2.t(), "feat": hid, "units": hid}
            elif hasattr(model, "W_E"):
                parts = transformer_internals(model, xb)
            else:
                a = xb
                for layer in model.layers[:-1]:
                    a = torch.relu(layer(a))
                parts = {"logits": model.layers[-1](a), "feat": a, "units": a}
            for k2, v in parts.items():
                outs[k2].append(v)
    return {k2: torch.cat(v) for k2, v in outs.items() if v}


# ── metric families ──────────────────────────────────────────────────────────

def function_metrics(cfg, d, parts_tr, parts_te):
    loss_fn = build_loss(cfg).loss_fn
    m = {}
    for split, parts, y, t in (("tr", parts_tr, d["ytr"], d["ttr"]), ("te", parts_te, d["yte"], d["tte"])):
        L = parts["logits"]
        m[f"fn_loss_{split}"] = float(loss_fn(L, t))
        m[f"fn_acc_{split}"] = float((L.argmax(1) == y).float().mean() * 100)
        correct = L.gather(1, y[:, None]).squeeze(1)
        other = L.clone()
        other[torch.arange(len(y)), y] = -float("inf")
        m[f"fn_margin_{split}"] = float((correct - other.max(1).values).median())
        p = L.float().softmax(1)
        m[f"fn_entropy_{split}"] = float(-(p * (p + 1e-12).log()).sum(1).mean())
    m["fn_gap_loss"] = m["fn_loss_te"] - m["fn_loss_tr"]
    m["fn_gap_acc"] = m["fn_acc_tr"] - m["fn_acc_te"]
    pred = parts_te["logits"].argmax(1)
    y = d["yte"]
    counts = torch.bincount(y, minlength=d["n_classes"]).float()
    hits = torch.bincount(y[pred == y], minlength=d["n_classes"]).float()
    mask = counts > 0
    m["fn_class_acc_gini"] = gini((hits[mask] / counts[mask]).cpu().numpy())
    return m


def weight_metrics(model, model0):
    m = {}
    mats, mats0 = weight_mats(model), weight_mats(model0)
    tot2 = tot02 = dist2 = dot = 0.0
    for name, M in mats.items():
        M0 = mats0[name]
        S = torch.linalg.svdvals(M.float()).cpu().numpy()
        rows = M.norm(dim=1).cpu().numpy()
        fro = float(M.norm())
        m[f"w_{name}_fro"] = fro
        m[f"w_{name}_spec"] = float(S[0])
        m[f"w_{name}_stable_rank"] = float((S ** 2).sum() / S[0] ** 2) if S[0] > 0 else math.nan
        m[f"w_{name}_eff_rank"] = eff_rank_from(S)
        m[f"w_{name}_sv_gini"] = gini(S)
        m[f"w_{name}_entry_gini"] = gini(M.cpu().numpy())
        m[f"w_{name}_row_gini"] = gini(rows)
        m[f"w_{name}_row_pr"] = participation(rows) / len(rows)
        diff = float((M - M0).norm())
        m[f"w_{name}_dist_init"] = diff
        m[f"w_{name}_rel_dist_init"] = diff / float(M0.norm())
        m[f"w_{name}_cos_init"] = float((M * M0).sum() / (M.norm() * M0.norm() + 1e-12))
        tot2 += fro ** 2
        tot02 += float(M0.norm()) ** 2
        dist2 += diff ** 2
        dot += float((M * M0).sum())
    m["w_all_norm"] = tot2 ** 0.5
    m["w_all_rel_norm"] = (tot2 / tot02) ** 0.5 if tot02 > 0 else math.nan
    m["w_all_dist_init"] = dist2 ** 0.5
    m["w_all_rel_dist_init"] = (dist2 / tot02) ** 0.5 if tot02 > 0 else math.nan
    m["w_all_cos_init"] = dot / ((tot2 * tot02) ** 0.5 + 1e-12)
    return m


def representation_metrics(cfg, d, parts_tr, parts_te):
    m = {}
    for split, parts, y in (("tr", parts_tr, d["ytr"]), ("te", parts_te, d["yte"])):
        X = parts["feat"].double()
        Xc = X - X.mean(0, keepdim=True)
        S = torch.linalg.svdvals(Xc).cpu().numpy()
        if split == "te":
            m["rep_eff_rank_te"] = eff_rank_from(S)
            m["rep_pr_te"] = participation(S ** 2)
            units = parts["units"]
            if not hasattr(parts, "W1") and cfg.model in ("mlp", "transformer"):
                m["rep_dead_frac_te"] = float(((units > 0).float().mean(0) == 0).float().mean())
            unit_energy = (units.double() ** 2).mean(0).cpu().numpy()
            m["rep_unit_gini_te"] = gini(unit_energy)
            m["rep_unit_pr_te"] = participation(unit_energy) / len(unit_energy)
        classes = torch.unique(y)
        mu = X.mean(0)
        means = torch.stack([X[y == c].mean(0) for c in classes])
        within = torch.zeros(X.shape[1], X.shape[1], dtype=X.dtype, device=X.device)
        for i, c in enumerate(classes):
            D = X[y == c] - means[i]
            within += D.t() @ D
        within /= X.shape[0]
        B = means - mu
        between = B.t() @ B / len(classes)
        tw, tb = float(torch.trace(within)), float(torch.trace(between))
        m[f"rep_between_within_{split}"] = tb / tw if tw > 0 else math.nan
        if split == "tr":
            # SVD-based pseudo-inverse: the between-class covariance is rank <= C-1 and
            # often numerically degenerate, where eigh fails to converge.
            try:
                m["rep_nc1_tr"] = float(torch.trace(within @ torch.linalg.pinv(between, rcond=1e-10))
                                        / len(classes))
            except RuntimeError:
                m["rep_nc1_tr"] = math.nan
    return m


def grid_parts(model, d):
    return forward_parts(model, d["xgrid"])


def unit_structure_metrics(cfg, d, gparts):
    """Hidden units over the complete grid: two-operand dependence and, on Z_p, frequency."""
    G = d["G"]
    U = gparts["units"].reshape(G, G, -1).double()            # (a, b, units)
    mu = U.mean((0, 1), keepdim=True)
    f = U.mean(1, keepdim=True) - mu
    g = U.mean(0, keepdim=True) - mu
    R = U - mu - f - g
    ef = (f ** 2).sum((0, 1)) * G                              # per-unit energy, broadcast over b
    eg = (g ** 2).sum((0, 1)) * G
    er = (R ** 2).sum((0, 1))
    tot = ef + eg + er
    strong = tot > 0.01 * tot.max()
    m = {"unit_interaction_share": float(er.sum() / tot.sum()) if tot.sum() > 0 else math.nan,
         "unit_frac_interacting": float(((er / tot.clamp(min=1e-30)) > 0.5)[strong].float().mean())
         if strong.any() else math.nan}
    if cfg.dataset == "modular":
        Fq = torch.fft.fft2(U.permute(2, 0, 1).float())          # (units, ka, kb)
        P = (Fq.abs() ** 2).double()
        P[:, 0, 0] = 0
        total = P.sum((1, 2))
        k = torch.arange(G, device=P.device)
        ka, kb = torch.meshgrid(k, k, indexing="ij")
        marg = (ka == 0) ^ (kb == 0)
        summ = (ka == kb) & (ka != 0)
        diff = (ka == (G - kb) % G) & (ka != 0) & (kb != 0)
        wsum = total.sum()
        if wsum > 0:
            m["unit_fourier_sum_share"] = float(P[:, summ].sum() / wsum)
            m["unit_fourier_diff_share"] = float(P[:, diff].sum() / wsum)
            m["unit_fourier_marginal_share"] = float(P[:, marg].sum() / wsum)
            m["unit_fourier_other_share"] = 1 - (m["unit_fourier_sum_share"] + m["unit_fourier_diff_share"]
                                                 + m["unit_fourier_marginal_share"])
            diag = P[:, summ.nonzero(as_tuple=True)[0], summ.nonzero(as_tuple=True)[1]]   # (units, G-1)
            w = torch.arange(1, (G - 1) // 2 + 1, device=P.device)
            comb = diag[:, w - 1] + diag[:, (G - w) - 1]               # combine w and -w
            conc = comb.max(1).values / total.clamp(min=1e-30)
            m["unit_sum_freq_concentration"] = float((conc * total).sum() / wsum)
            m["unit_n_distinct_sum_freqs"] = int(len(torch.unique(comb[strong.to(comb.device)].argmax(1))))
    return m


def freq_profile(M, G):
    """Z_p DFT of rows of M (rows x G)."""
    Fq = torch.fft.fft(M.double(), dim=1)
    P = Fq.abs() ** 2
    w = torch.arange(1, (G - 1) // 2 + 1, device=M.device)
    comb = P[:, w] + P[:, G - w]                                  # (rows, W)
    dc = P[:, 0]
    agg = comb.sum(0).cpu().numpy()
    mags = Fq.abs()
    ipr = ((mags / mags.norm(dim=1, keepdim=True).clamp(min=1e-12)) ** 4).sum(1).mean()
    out = {"ipr": float(ipr), "dc_share": float(dc.sum() / P.sum()),
           "top1_share": float(agg.max() / agg.sum()) if agg.sum() > 0 else math.nan,
           "nfreq90": n_cover(agg), "freq_gini": gini(agg), "freq_eff_n": eff_rank_from(agg)}
    dominant = (comb.argmax(1) + 1).cpu().numpy()
    return out, dominant, agg


def fourier_weight_metrics(model, cfg):
    m = {}
    G = cfg.p
    if hasattr(model, "W1"):
        U, V, W2 = model.W1.data[:, :G], model.W1.data[:, G:], model.W2.data
        pu, du, _ = freq_profile(U, G)
        pv, dv, _ = freq_profile(V, G)
        po, do, _ = freq_profile(W2.t(), G)
        for pre, prof in (("fU_", pu), ("fV_", pv), ("fout_", po)):
            m.update({pre + k: v for k, v in prof.items()})
        strength = (U.norm(dim=1) * V.norm(dim=1) * W2.norm(dim=0)).cpu().numpy()
        strong = strength >= np.quantile(strength, 0.5)
        m["fin_uv_agree_strong"] = float((du == dv)[strong].mean())
        m["fin_uvw_agree_strong"] = float(((du == dv) & (dv == do))[strong].mean())
        m["fin_n_unique_freq_strong"] = int(len(np.unique(du[strong])))
    else:
        pe, de, _ = freq_profile(model.W_E.data.t(), G)          # (d_model, p)
        po, do, _ = freq_profile(model.W_U.data, G)              # (d_model, p)
        m.update({"fin_" + k: v for k, v in pe.items()})
        m.update({"fout_" + k: v for k, v in po.items()})
    return m


def clock_decomposition(L, G):
    """FFT of (G, G, G) logits, per-frequency clock energy and the class-varying total."""
    Fq = torch.fft.fftn(L.float())
    P = Fq.abs() ** 2
    varying = float(P[:, :, 1:].sum())
    w = torch.arange(1, (G - 1) // 2 + 1, device=L.device)
    e = (P[w, w, (G - w) % G] + P[G - w, G - w, w]).double().cpu().numpy()
    return Fq, e, varying


def key_set(e, frac=0.9, cap=10 ** 6):
    """Frequencies covering `frac` of clock energy, largest first. Uncapped: the
    quadratic MLP on Z_97 spreads its clock over ~44 of 48 frequencies, and a cap
    would leave most of the circuit inside the 'excluded' logits."""
    order = np.argsort(e)[::-1]
    cum = np.cumsum(e[order]) / max(e.sum(), 1e-30)
    n = min(int(np.searchsorted(cum, frac) + 1), cap)
    return sorted(int(order[i]) + 1 for i in range(n))


def clock_metrics(cfg, d, gL, key_final):
    G = d["G"]
    L = gL.reshape(G, G, -1)
    Fq, e, varying = clock_decomposition(L, G)
    keys = key_set(e)
    m = {"clock_share": float(e.sum() / varying) if varying > 0 else math.nan,
         "clock_nfreq90": len(keys),
         "clock_top1_share": float(e.max() / e.sum()) if e.sum() > 0 else math.nan,
         "clock_freq_gini": gini(e),
         "clock_top5": " ".join(str(int(i) + 1) for i in np.argsort(e)[::-1][:5]),
         "clock_key_jaccard_final": (len(set(keys) & set(key_final)) / len(set(keys) | set(key_final)))
         if key_final else math.nan}
    if key_final:
        mask = torch.zeros(G, G, G, dtype=torch.bool, device=L.device)
        for w in key_final:
            mask[w, w, (G - w) % G] = True
            mask[G - w, G - w, w] = True
        c0 = torch.zeros_like(mask)
        c0[:, :, 0] = True
        restricted = torch.fft.ifftn(Fq * (mask | c0)).real
        excluded = torch.fft.ifftn(Fq * (~mask)).real
        loss_fn = build_loss(cfg).loss_fn
        for split, (ia, ib), y, t in (("tr", d["tr_ab"], d["ytr"], d["ttr"]),
                                      ("te", d["te_ab"], d["yte"], d["tte"])):
            for name, T in (("restricted", restricted), ("excluded", excluded)):
                logits = T[ia, ib].to(t.dtype if t.is_floating_point() else torch.float32)
                m[f"clock_{name}_loss_{split}"] = float(loss_fn(logits, t))
                m[f"clock_{name}_acc_{split}"] = float((logits.argmax(1) == y).float().mean() * 100)
    return m, keys


def additive_logit_metrics(d, gL):
    G = d["G"]
    L = gL.reshape(G, G, -1).double()
    mu = L.mean((0, 1), keepdim=True)
    f = L.mean(1, keepdim=True) - mu
    g = L.mean(0, keepdim=True) - mu
    R = L - mu - f - g
    Y = d["ygrid"].reshape(G, G)

    def cen_energy(T):
        T = T.expand_as(L).reshape(-1, L.shape[-1])
        T = T - T.mean(1, keepdim=True)
        return float((T ** 2).sum(1).mean())
    tot = cen_energy(L) or 1.0
    m = {"add_share_a": cen_energy(f) / tot, "add_share_b": cen_energy(g) / tot,
         "add_share_interaction": cen_energy(R) / tot}
    for split, (ia, ib) in (("tr", d["tr_ab"]), ("te", d["te_ab"])):
        y = Y[ia, ib]
        for name, T in (("interaction", R), ("additive", (mu + f + g).expand_as(L)),
                        ("no_a", L - f), ("no_b", L - g)):
            m[f"add_acc_{name}_{split}"] = float((T[ia, ib].argmax(1) == y).float().mean() * 100)
    return m


def s5_metrics(model, cfg, d):
    m = {}
    if hasattr(model, "W1"):
        G = model.W1.shape[1] // 2
        blocks = {"U": model.W1.data[:, :G], "V": model.W1.data[:, G:], "out": model.W2.data.t()}
    else:
        blocks = {"E": model.W_E.data.t(), "out": model.W_U.data}
    for name, M in blocks.items():
        fr = irreps.fractions(M)
        base = dict(zip(fr.keys(), irreps.random_baseline_fractions(5)))
        m[f"irr_{name}_structure"] = irreps.structure_score(M)
        top = max(fr, key=fr.get)
        m[f"irr_{name}_dominant"] = top
        m[f"irr_{name}_dominant_share"] = fr[top]
        m[f"irr_{name}_dominant_over_random"] = fr[top] / base[top]
        if name in ("U", "E"):
            for k, v in fr.items():
                m[f"irr_{name}_{k}"] = v
    for sub in ("s_nm1", "a_n"):
        c = coset_attribution(model, d["xte"], d["yte"], dataclasses.replace(cfg, coset_subgroup=sub))
        m[f"coset_{sub}_acc_te"] = c["coset_accuracy"] * 100
        m[f"coset_{sub}_purity_te"] = c["coset_purity"]
    return m


def circuit_metrics(model, d):
    m = {}
    for split, x, y in (("tr", d["xtr"], d["ytr"]), ("te", d["xte"], d["yte"])):
        rep = quadratic_circuits.circuit_report(model, x, y)
        m.update({f"circ_{k[5:]}_{split}": v for k, v in rep.items()})
    return m


def attention_metrics(gparts):
    A = gparts["attn"]                                           # (grid, heads, 2, 2)
    to_a = A[:, :, 1, 0]
    ent = -(A[:, :, 1, :] * (A[:, :, 1, :] + 1e-12).log()).sum(-1)
    return {"attn_to_a_mean": float(to_a.mean()), "attn_to_a_head_spread": float(to_a.mean(0).std()),
            "attn_entropy_mean": float(ent.mean())}


def sharpness(model, cfg, d, iters=20):
    loss_fn = build_loss(cfg).loss_fn
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(d["xtr"]), d["ttr"])
    grads = torch.autograd.grad(loss, params, create_graph=True)
    gnorm = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads)))
    v = [torch.randn_like(p) for p in params]
    norm = torch.sqrt(sum((x ** 2).sum() for x in v))
    v = [x / norm for x in v]
    lam = math.nan
    for _ in range(iters):
        hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        lam = float(sum((a * b).sum() for a, b in zip(hv, v)))
        norm = torch.sqrt(sum((x ** 2).sum() for x in hv))
        if norm <= 0:
            break
        v = [x / norm for x in hv]
    return {"sharp_lambda_max": lam, "sharp_grad_norm": gnorm, "sharp_lr_lambda": lam * float(cfg.lr)}


# ── per run ──────────────────────────────────────────────────────────────────

def analyse_run(rid, row):
    t0 = time.time()
    cfg = C.config(rid)
    h = C.history(rid)
    cks = C.checkpoints(rid, h)
    if not cks:
        return None
    chosen = C.select_checkpoints(cks, row, MAX_CKPTS)
    d = C.data(cfg)
    model0 = C.init_model(cfg)
    out = {"run": C.descriptors(row), "init_check": C.init_check(model0, h), "n_ckpts_total": len(cks),
           "checkpoints": []}
    is_grid, modular = "G" in d, cfg.dataset == "modular"
    s5 = cfg.dataset == "s5"
    quad = quadratic_circuits.applicable(model0) and is_grid
    # Sharpness on events plus a spread.
    ev = {int(np.argmin([abs(c[0] - C.fnum(row.get(k))) for c in chosen]))
          for k in ("t_memo", "t_first_cross") if math.isfinite(C.fnum(row.get(k)))}
    spread = set(np.linspace(0, len(chosen) - 1, SHARP_CKPTS - len(ev)).round().astype(int).tolist())
    sharp_idx = ev | spread

    key_final = None
    # Final checkpoint first, so every earlier checkpoint is scored with its key frequencies.
    order = list(range(len(chosen)))[::-1]
    records = {}
    for i in order:
        step, unit, n, path = chosen[i]
        model = C.load_model(cfg, path)
        rec = {"step": step, "unit": unit, "index": n}
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            # Diverged (the Dirichlet-0.01 starvation runs on A go NaN): record the fact,
            # not a decomposition of NaNs -- the SVD fallback on them never returns.
            rec["w_nonfinite"] = 1
            records[i] = rec
            del model
            continue
        rec["w_nonfinite"] = 0
        ptr, pte = forward_parts(model, d["xtr"]), forward_parts(model, d["xte"])
        rec.update(function_metrics(cfg, d, ptr, pte))
        rec.update(weight_metrics(model, model0))
        rec.update(representation_metrics(cfg, d, ptr, pte))
        if is_grid:
            gp = grid_parts(model, d)
            rec.update(unit_structure_metrics(cfg, d, gp))
            rec.update(additive_logit_metrics(d, gp["logits"]))
            if "attn" in gp:
                rec.update(attention_metrics(gp))
            if modular:
                rec.update(fourier_weight_metrics(model, cfg))
                cm, keys = clock_metrics(cfg, d, gp["logits"], key_final)
                if key_final is None:                   # the final checkpoint: rescore with its own keys
                    key_final = keys
                    cm, _ = clock_metrics(cfg, d, gp["logits"], key_final)
                rec.update(cm)
            del gp
        if s5:
            rec.update(s5_metrics(model, cfg, d))
        if quad:
            rec.update(circuit_metrics(model, d))
        if i in sharp_idx:
            model.train()
            rec.update(sharpness(model, cfg, d))
        records[i] = rec
        del model, ptr, pte
    out["checkpoints"] = [records[i] for i in sorted(records)]
    out["key_freqs_final"] = key_final
    out["seconds"] = round(time.time() - t0, 1)
    return out


def collect():
    rows = []
    for f in sorted(glob.glob(os.path.join(OUTDIR, "*.json"))):
        j = json.load(open(f))
        base = {**j["run"], "init_check": j.get("init_check"),
                "key_freqs_final": " ".join(map(str, j.get("key_freqs_final") or []))}
        tm, tf = C.fnum(base.get("t_memo")), C.fnum(base.get("t_first_cross"))
        for rec in j["checkpoints"]:
            r = {**base, **{k: (math.nan if v is None else v) for k, v in rec.items()}}
            r["step_over_t_memo"] = rec["step"] / tm if math.isfinite(tm) and tm > 0 else math.nan
            r["step_over_t_cross"] = rec["step"] / tf if math.isfinite(tf) and tf > 0 else math.nan
            rows.append(r)
    path = C.write_csv(os.path.join(C.OUT, "checkpoint_metrics.csv"), rows)
    print(f"  {len(rows)} checkpoint rows -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--shard")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect()
        return
    rows = C.load_rows()
    ids = a.ids or [rid for rid in rows if glob.glob(os.path.join(C.RUNS, rid, "checkpoints", "ckpt_*.pt"))]
    ids = C.shard_filter(ids, a.shard)
    os.makedirs(OUTDIR, exist_ok=True)
    done = fail = 0
    for k, rid in enumerate(ids):
        path = os.path.join(OUTDIR, f"{rid}.json")
        if os.path.exists(path) and not a.redo:
            continue
        try:
            res = analyse_run(rid, rows[rid])
            if res is not None:
                C.write_json(path, res)
                done += 1
                print(f"[{k + 1}/{len(ids)}] {rid} {rows[rid]['setup']} {len(res['checkpoints'])} ckpts "
                      f"{res['seconds']}s init_check={res['init_check']}", flush=True)
        except Exception:
            fail += 1
            print(f"[{k + 1}/{len(ids)}] {rid} FAILED\n{traceback.format_exc()}", flush=True)
        torch.cuda.empty_cache()
    print(f"done {done}, failed {fail}")


if __name__ == "__main__":
    main()
