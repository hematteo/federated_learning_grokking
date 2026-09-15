"""Full spectra per checkpoint, for plotting how structure forms over training.

    venv/bin/python scripts/mechinterp/spectra.py            # -> results/mechinterp/spectra/<id>.npz
    venv/bin/python scripts/mechinterp/spectra.py --collect  # -> spectra_*.csv (long format)

checkpoints.py reduces each spectrum to scalars (IPR, number of frequencies covering
90%, Gini, dominant irrep). This keeps the vectors, on the same checkpoints
(common.select_checkpoints), so figures can show them evolving.

Per run, an .npz with `steps` and:
  sv_<M>            singular values of each weight matrix (common across setups)
  rownorm_<M>       sorted row (neuron) norms of each matrix -- Lorenz curves / Gini
  modular:  clock_energy   (ckpts, W) per-frequency energy of exp(i w (a+b-c)) in the logits,
                           normalised by the class-varying logit energy
            in_power       (ckpts, W) input-weight power per frequency (U, or W_E over tokens)
            v_power        (ckpts, W) the quadratic MLP's second-operand block
            out_power      (ckpts, W) output-weight power per frequency
            unit_sum_power (ckpts, W) hidden-unit activation power at sum frequency (w, w)
  s5:       irrep_in / irrep_out (ckpts, 7) isotypic energy fractions of the input / output blocks
            irrep_names

--collect writes spectra_clock.csv, spectra_in_power.csv, spectra_irreps.csv (fractions
per frequency / irrep, long format with run descriptors) and spectra_sv_summary.csv
(top 32 normalised singular values per matrix).
"""
import argparse
import glob
import math
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import common as C                                              # noqa: E402
import checkpoints as K                                         # noqa: E402
from fedgrok.metrics import irreps                              # noqa: E402

SD = os.path.join(C.OUT, "spectra")


def pw(M, G):
    P = torch.fft.fft(M.double(), dim=1).abs() ** 2
    w = torch.arange(1, (G - 1) // 2 + 1, device=M.device)
    return (P[:, w] + P[:, G - w]).sum(0).cpu().numpy()


def run(rid, row):
    cfg = C.config(rid)
    cks = C.select_checkpoints(C.checkpoints(rid), row, K.MAX_CKPTS)
    if not cks:
        return None
    d = C.data(cfg)
    arrs = {"steps": np.array([c[0] for c in cks])}
    acc = {}
    cks = [c for c in cks if all(torch.isfinite(v).all() for v in torch.load(c[3], map_location="cpu").values())]
    if not cks:                      # diverged throughout (A's Dirichlet-0.01 starvation runs)
        return None
    arrs = {"steps": np.array([c[0] for c in cks])}
    for step, unit, n, path in cks:
        model = C.load_model(cfg, path)
        for name, M in K.weight_mats(model).items():
            acc.setdefault(f"sv_{name}", []).append(torch.linalg.svdvals(M.float()).cpu().numpy())
            acc.setdefault(f"rownorm_{name}", []).append(np.sort(M.norm(dim=1).cpu().numpy()))
        if cfg.dataset == "modular":
            G = d["G"]
            gp = K.grid_parts(model, d)
            _, e, varying = K.clock_decomposition(gp["logits"].reshape(G, G, -1), G)
            acc.setdefault("clock_energy", []).append(e / varying if varying > 0 else e * np.nan)
            if hasattr(model, "W1"):
                acc.setdefault("in_power", []).append(pw(model.W1.data[:, :G], G))
                acc.setdefault("v_power", []).append(pw(model.W1.data[:, G:], G))
                acc.setdefault("out_power", []).append(pw(model.W2.data.t(), G))
            else:
                acc.setdefault("in_power", []).append(pw(model.W_E.data.t(), G))
                acc.setdefault("out_power", []).append(pw(model.W_U.data, G))
            U = gp["units"].reshape(G, G, -1).permute(2, 0, 1).float()
            P = (torch.fft.fft2(U).abs() ** 2).sum(0)
            w = torch.arange(1, (G - 1) // 2 + 1, device=P.device)
            acc.setdefault("unit_sum_power", []).append((P[w, w] + P[G - w, G - w]).double().cpu().numpy())
            del gp
        elif cfg.dataset == "s5":
            if hasattr(model, "W1"):
                Gs = model.W1.shape[1] // 2
                bin_, bout = model.W1.data[:, :Gs], model.W2.data.t()
            else:
                bin_, bout = model.W_E.data.t(), model.W_U.data
            fi, fo = irreps.fractions(bin_), irreps.fractions(bout)
            arrs["irrep_names"] = np.array(list(fi.keys()))
            acc.setdefault("irrep_in", []).append(np.array(list(fi.values())))
            acc.setdefault("irrep_out", []).append(np.array([fo[k] for k in fi]))
        del model
    for k, v in acc.items():
        arrs[k] = np.stack(v)
    return arrs


def collect():
    rows = C.load_rows()
    clock, inp, irr, sv = [], [], [], []
    for f in sorted(glob.glob(os.path.join(SD, "*.npz"))):
        rid = os.path.basename(f)[:-4]
        if rid not in rows:
            continue
        z = np.load(f, allow_pickle=False)
        base = {"id": rid, "setup": rows[rid]["setup"], "mode": rows[rid]["mode"], "group": rows[rid]["group"],
                "axis": rows[rid]["axis"]}
        steps = z["steps"]
        for i, st in enumerate(steps):
            if "clock_energy" in z:
                for w, v in enumerate(z["clock_energy"][i], start=1):
                    clock.append({**base, "step": st, "freq": w, "clock_energy_share": float(v)})
                tot = z["in_power"][i].sum()
                for w, v in enumerate(z["in_power"][i], start=1):
                    inp.append({**base, "step": st, "freq": w, "in_power_share": float(v / tot) if tot > 0 else math.nan})
            if "irrep_in" in z:
                for name, a, b in zip(z["irrep_names"], z["irrep_in"][i], z["irrep_out"][i]):
                    irr.append({**base, "step": st, "irrep": str(name), "in_fraction": float(a), "out_fraction": float(b)})
            for key in [k for k in z.files if k.startswith("sv_")]:
                s = z[key][i]
                s = s / s.sum() if s.sum() > 0 else s
                for j, v in enumerate(s[:32]):
                    sv.append({**base, "step": st, "matrix": key[3:], "rank": j + 1, "sv_share": float(v)})
    for name, data in (("spectra_clock.csv", clock), ("spectra_in_power.csv", inp),
                       ("spectra_irreps.csv", irr), ("spectra_sv_summary.csv", sv)):
        if data:
            print("  wrote", C.write_csv(os.path.join(C.OUT, name), data), len(data), "rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--shard")
    a = ap.parse_args()
    if a.collect:
        collect()
        return
    rows = C.load_rows()
    ids = [rid for rid in rows if glob.glob(os.path.join(C.RUNS, rid, "checkpoints", "ckpt_*.pt"))]
    ids = C.shard_filter(ids, a.shard)
    os.makedirs(SD, exist_ok=True)
    for k, rid in enumerate(ids):
        out = os.path.join(SD, f"{rid}.npz")
        if os.path.exists(out):
            continue
        try:
            arrs = run(rid, rows[rid])
            if arrs is not None:
                np.savez_compressed(out, **arrs)
                print(f"[{k + 1}/{len(ids)}] {rid}", flush=True)
        except Exception:
            print(f"[{k + 1}/{len(ids)}] {rid} FAILED\n{traceback.format_exc()}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
