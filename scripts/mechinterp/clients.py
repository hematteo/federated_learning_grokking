"""What the clients do to the circuit: per-client snapshots against the global model.

    venv/bin/python scripts/mechinterp/clients.py      # -> results/mechinterp/client_metrics.csv

Federated runs with checkpoint_client_weights save, at each checkpoint round R, every
participating client's signature matrix after its local training in round R
(metrics/probes.client_signature: the quadratic MLP's first-operand block U, the
transformer's token embedding W_E, the MNIST MLP's first layer), next to the global
checkpoint aggregated from them. One row per (run, checkpoint round).

GENERIC (every setup)
  cli_n                     clients in the snapshot (f * K under partial participation)
  cli_agg_gap               ||mean of clients - global|| / ||global||: zero for plain
                            FedAvg on equal shards; unequal shards, server momentum and
                            adaptive server optimisers move the global off the mean
  cli_rel_spread            RMS client deviation from the client mean / ||client mean||
  cli_cos_global_mean/min   cosine of each client's matrix with the global one
  cli_dev_pair_cos          mean pairwise cosine of client deviations from their mean
                            (-1/(n-1) for independent zero-sum deviations)
  cli_dev_eff_rank_frac     entropy effective rank of the stacked deviations / (n - 1):
                            1 if clients move in independent directions, ~0 if one
  cli_coherence             ||mean_k (C_k - global)||^2 / mean_k ||C_k - global||^2
MODULAR (A, A', B), frequency over the operand / token index
  cli_key_n                 global key frequencies (covering 90% of the signature's
                            non-DC power)
  cli_dev_key_share         share of the deviations' non-DC power at those frequencies
  cli_dev_key_share_uniform the share a spectrally white deviation would put there
  cli_dev_key_over_uniform  the ratio: >1 means clients disagree ON the circuit
  cli_key_agreement         energy of the mean deviation over mean deviation energy at
                            the key frequencies (1 identical, 1/n independent, 0 cancel)
  cli_neuron_freq_agree     quadratic MLP: fraction of the strongest half of neurons whose
                            dominant frequency in each client equals the global one
S5 (C, D), isotypic decomposition over the group index
  cli_dev_dominant_irrep_share   share of deviation energy in the GLOBAL dominant irrep
  cli_global_dominant_irrep_share  that irrep's share in the global signature
  cli_dev_irrep_structure        structure score of the deviations
"""
import glob
import math
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import common as C                                         # noqa: E402
from fedgrok.metrics import irreps                         # noqa: E402
from fedgrok.metrics.probes import client_signature        # noqa: E402

_R = re.compile(r"client_w1_round(\d+)\.pt$")


def eff_rank(S):
    S = S[S > 1e-12]
    if S.size == 0:
        return math.nan
    p = S / S.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def power_over(M, axis_last=True):
    """|DFT|^2 of rows of M over the last axis, combined over +-w, shape (rows, W)."""
    G = M.shape[1]
    P = np.abs(np.fft.fft(M, axis=1)) ** 2
    w = np.arange(1, (G - 1) // 2 + 1)
    return P[:, w] + P[:, G - w]


def run_rows(rid, r):
    cfg = C.config(rid)
    h = C.history(rid)
    r2s = {int(a): float(b) for a, b in zip(h["round"], h["total_steps"])} if h else {}
    out = []
    for f in sorted(glob.glob(os.path.join(C.RUNS, rid, "checkpoints", "client_w1_round*.pt")),
                    key=lambda p: int(_R.search(p).group(1))):
        R = int(_R.search(f).group(1))
        gpath = os.path.join(C.RUNS, rid, "checkpoints", f"ckpt_round{R}.pt")
        if not os.path.exists(gpath):
            continue
        clients = [np.asarray(c, dtype=np.float64) for c in torch.load(f, map_location="cpu", weights_only=False)]
        if len(clients) < 2:
            continue
        model = C.load_model(cfg, gpath).cpu()
        _, glob_sig = client_signature(model, cfg)
        Gm = np.asarray(glob_sig, dtype=np.float64)
        Cs = np.stack(clients)                                  # (n, rows, cols)
        if Cs.shape[1:] != Gm.shape:
            continue
        n = len(Cs)
        mean_c = Cs.mean(0)
        dev = Cs - mean_c
        rec = {**C.descriptors(r), "round": R, "step": r2s.get(R, math.nan), "cli_n": n}
        gnorm = np.linalg.norm(Gm)
        rec["cli_agg_gap"] = float(np.linalg.norm(mean_c - Gm) / gnorm) if gnorm > 0 else math.nan
        rec["cli_rel_spread"] = float(np.sqrt((dev ** 2).sum((1, 2)).mean()) / max(np.linalg.norm(mean_c), 1e-12))
        cos = [(c * Gm).sum() / (np.linalg.norm(c) * gnorm + 1e-12) for c in Cs]
        rec["cli_cos_global_mean"] = float(np.mean(cos))
        rec["cli_cos_global_min"] = float(np.min(cos))
        flat = dev.reshape(n, -1)
        norms = np.linalg.norm(flat, axis=1, keepdims=True).clip(1e-12)
        U = flat / norms
        pc = U @ U.T
        rec["cli_dev_pair_cos"] = float((pc.sum() - n) / (n * (n - 1)))
        rec["cli_dev_eff_rank_frac"] = eff_rank(np.linalg.svd(flat, compute_uv=False)) / (n - 1)
        upd = (Cs - Gm).reshape(n, -1)
        rec["cli_coherence"] = float((upd.mean(0) ** 2).sum() / max((upd ** 2).sum(1).mean(), 1e-30))

        if cfg.dataset == "modular":
            # Rows indexed by feature, columns by operand/token.
            def as_rows(M):
                return M if hasattr(model, "W1") else M.T              # W_E is (tokens, d)
            Gr = as_rows(Gm)
            PG = power_over(Gr).sum(0)
            order = np.argsort(PG)[::-1]
            k = int(np.searchsorted(np.cumsum(PG[order]) / PG.sum(), 0.9) + 1)
            keys = order[:k]
            PD = np.stack([power_over(as_rows(d)) for d in dev])        # (n, rows, W)
            tot = PD.sum()
            rec["cli_key_n"] = k
            rec["cli_dev_key_share"] = float(PD[:, :, keys].sum() / tot) if tot > 0 else math.nan
            rec["cli_dev_key_share_uniform"] = k / PG.size
            rec["cli_dev_key_over_uniform"] = rec["cli_dev_key_share"] / rec["cli_dev_key_share_uniform"]
            Fd = np.fft.fft(np.stack([as_rows(d) for d in dev]), axis=2)  # (n, rows, G)
            G = Fd.shape[2]
            idx = np.concatenate([keys + 1, G - (keys + 1)])
            mean_e = (np.abs(Fd[:, :, idx].mean(0)) ** 2).sum()
            e_mean = (np.abs(Fd[:, :, idx]) ** 2).sum() / n
            rec["cli_key_agreement"] = float(mean_e / e_mean) if e_mean > 0 else math.nan
            if hasattr(model, "W1"):
                gdom = power_over(Gr).argmax(1)
                strength = np.linalg.norm(Gr, axis=1)
                strong = strength >= np.quantile(strength, 0.5)
                agree = [(power_over(c).argmax(1) == gdom)[strong].mean() for c in Cs]
                rec["cli_neuron_freq_agree"] = float(np.mean(agree))
        elif cfg.dataset == "s5":
            as_rows = (lambda M: M) if hasattr(model, "W1") else (lambda M: M.T)
            gfr = irreps.fractions(torch.as_tensor(as_rows(Gm)))
            top = max(gfr, key=gfr.get)
            dstack = np.concatenate([as_rows(d) for d in dev], axis=0)
            dfr = irreps.fractions(torch.as_tensor(dstack))
            rec["cli_global_dominant_irrep"] = top
            rec["cli_global_dominant_irrep_share"] = gfr[top]
            rec["cli_dev_dominant_irrep_share"] = dfr[top]
            rec["cli_dev_irrep_structure"] = irreps.structure_score(torch.as_tensor(dstack))
        out.append(rec)
    return out


def main():
    rows = C.load_rows()
    ids = [rid for rid in sorted(rows) if rows[rid]["mode"] == "federated"
           and glob.glob(os.path.join(C.RUNS, rid, "checkpoints", "client_w1_round*.pt"))]
    allrows = []
    for n, rid in enumerate(ids):
        try:
            allrows += run_rows(rid, rows[rid])
        except Exception as e:                                   # keep going; report
            print(f"  {rid} FAILED: {e!r}", flush=True)
        if n % 50 == 0:
            print(f"  {n}/{len(ids)}", flush=True)
    print("  wrote", C.write_csv(os.path.join(C.OUT, "client_metrics.csv"), allrows), len(allrows), "rows")


if __name__ == "__main__":
    torch.set_num_threads(4)
    C.DEVICE = torch.device("cpu")
    main()
