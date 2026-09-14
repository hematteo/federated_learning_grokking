"""Regenerate RESULTS.md section 25: the order parameters through the federated axes. No training.

    venv/bin/python scripts/analyze_order_parameters.py ipr     # 25.1  setup A, Fourier IPR at memorisation / crossing / end, every axis
    venv/bin/python scripts/analyze_order_parameters.py drift   # 25.2  setup A, where client updates live in frequency space (per-client snapshots)
    venv/bin/python scripts/analyze_order_parameters.py embed   # 25.3  setup B, embedding IPR against memorisation
    venv/bin/python scripts/analyze_order_parameters.py s5      # 25.4  C and D, coset attribution lead and irrep structure
    venv/bin/python scripts/analyze_order_parameters.py all [--json DIR]

THE INSTRUMENTS are the ones the training loops already log every eval round --
`ipr` (metrics/fourier.spectral_ipr on GrokNet's W1[:, :p]), `embed_ipr` (the same
DFT on the transformer's W_E over the token index), `coset_accuracy` (Stander
et al.'s coset attribution, metrics/nonabelian) and `irrep_structure_u` /
`irrep_u_*` (the S_5 isotypic energy profile of U, metrics/irreps) -- plus the
per-client first-layer snapshots the federated loop saves at checkpoint rounds.
The `drift` pass takes a few minutes: it loads every per-client snapshot on A.
"""
import argparse, csv, glob, json, os, re, statistics, sys, collections
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedgrok.metrics import irreps

CSV = "results/data/runs_v2.csv"
P = 97


def rows():
    return list(csv.DictReader(open(CSV)))


def history(r):
    fs = glob.glob(f"results/runs/{r['id']}/history_*.json")
    return json.load(open(fs[0])) if fs else None


def first(steps, vals, bar):
    for s, v in zip(steps, vals):
        if v is not None and v >= bar:
            return s
    return None


def at(steps, vals, t):
    if t is None:
        return None
    i = min(range(len(steps)), key=lambda j: abs(steps[j] - t))
    return vals[i]


def med(rs, k):
    v = [x[k] for x in rs if x.get(k) is not None]
    return statistics.median(v) if v else None


def f3(v): return "—" if v is None else f"{v:.3f}"
def f0(v): return "—" if v is None else f"{v:,.0f}"
def f1(v): return "—" if v is None else f"{v:.1f}"


def is_setup_a(r):
    return r["setup"] == "A" or (r["setup"] == "" and r["dataset"] == "modular"
                                 and r["model"] == "groknet" and r["optimizer"] == "gd")


def cellkey(r):
    if r["mode"] == "federated":
        return (r["group"], float(r["alpha"]), int(float(r["num_clients"])), int(float(r["local_epochs"])),
                r["partition"], r["dirichlet_alpha"] if r["partition"] in ("dirichlet", "dirichlet_sizes") else "",
                float(r["fraction_train"]), r["strategy"])
    return ("centralised " + r["group"], float(r["alpha"]), 1, 0, "", "", 1.0, "")


def cellname(k):
    g, al, K, E, part, da, fr, st = k
    if K == 1:
        return f"{g} α{al}"
    return f"{g} α{al} K{K} E{E} {part} {da} f{fr} {st}".replace("  ", " ").strip()


def sort_key(k):
    return (k[0], k[1], k[4], str(k[5]), k[2], k[3], k[6], k[7])


# ── 25.1 ─────────────────────────────────────────────────────────────────────
def ipr():
    """Setup A: IPR at init / memorisation / first crossing / end, and when it reaches half its peak."""
    cells = collections.defaultdict(list)
    for r in filter(is_setup_a, rows()):
        h = history(r)
        if not h or not h.get("ipr") or h["ipr"][0] is None:
            continue
        steps = h.get("total_steps") or h.get("epoch")
        tm = first(steps, h["train_acc"], 99); tf = first(steps, h["test_acc"], 95)
        peak = max(v for v in h["ipr"] if v is not None)
        cells[cellkey(r)].append(dict(
            ipr0=h["ipr"][0], ipr_memo=at(steps, h["ipr"], tm), ipr_fc=at(steps, h["ipr"], tf),
            ipr_end=h["ipr"][-1], peak=peak, t_half=first(steps, h["ipr"], 0.5 * peak), tf=tf,
            held=r["grokked"] == "True"))
    print("25.1  setup A: per-neuron spectral IPR of W1[:, :97] at init, memorisation, first crossing and end; "
          "t_half = step it first reaches half its peak\n")
    print(f"{'cell':<62}{'n':>3}{'held':>6} | {'init':>6}{'@memo':>7}{'@cross':>7}{'end':>6}{'peak':>6} | {'t_half':>8}{'t_cross':>9}")
    out = {}
    for k in sorted(cells, key=sort_key):
        rs = cells[k]
        if len(rs) < 2:
            continue
        out[cellname(k)] = rs
        print(f"{cellname(k):<62}{len(rs):>3}{sum(x['held'] for x in rs):>4}/{len(rs)} | "
              f"{f3(med(rs,'ipr0')):>6}{f3(med(rs,'ipr_memo')):>7}{f3(med(rs,'ipr_fc')):>7}{f3(med(rs,'ipr_end')):>6}"
              f"{f3(med(rs,'peak')):>6} | {f0(med(rs,'t_half')):>8}{f0(med(rs,'tf')):>9}")
    return out


# ── 25.2 ─────────────────────────────────────────────────────────────────────
def _ck(rid, kind):
    fs = glob.glob(f"results/runs/{rid}/checkpoints/{kind}*.pt")
    return dict(sorted((int(re.search(r"(\d+)\.pt", f).group(1)), f) for f in fs))


def _spec(W):
    """Power spectrum along the operand axis, DC removed, folded to the 48 frequency pairs of Z_97."""
    p = np.abs(np.fft.fft(W, axis=-1)) ** 2
    p[..., 0] = 0
    return p[..., 1:49] + p[..., 49:][..., ::-1]


def drift():
    """Setup A: per neuron, do the clients' per-round changes land on the neuron's own frequency?"""
    cells = collections.defaultdict(list)
    for r in filter(lambda r: is_setup_a(r) and r["mode"] == "federated" and r["strategy"] == "fedavg", rows()):
        g = _ck(r["id"], "ckpt_round"); c = _ck(r["id"], "client_w1_round")
        common = sorted(set(g) & set(c))
        if not common:
            continue
        E = float(r["local_epochs"]); tf = None if r["t_first_cross"] == "inf" else float(r["t_first_cross"])
        recs = []
        for rnd in common:
            Wg = torch.load(g[rnd], map_location="cpu", weights_only=True)["W1"][:, :P].numpy()
            Wc = np.stack(torch.load(c[rnd], map_location="cpu", weights_only=False))
            dev = Wc - Wg[None]
            pg, pd = _spec(Wg), _spec(dev)
            dom = pg.argmax(-1)
            conc_g = np.take_along_axis(pg, dom[:, None], 1)[:, 0] / pg.sum(-1)
            conc_d = np.take_along_axis(pd, dom[None, :, None], 2)[:, :, 0].sum(0) / pd.sum((0, 2))
            F = np.fft.fft(dev, axis=-1)
            Fdom = np.take_along_axis(F, (dom + 1)[None, :, None], 2)[:, :, 0]
            agree = (np.abs(Fdom.mean(0)) ** 2) / (np.abs(Fdom) ** 2).mean(0)
            recs.append(dict(pre=(tf is None or rnd * E < tf), cg=float(np.median(conc_g)),
                             cd=float(np.median(conc_d)), agree=float(np.median(agree))))
        cells[cellkey(r)].append(dict(recs=recs, held=r["grokked"] == "True"))
    print("\n25.2  setup A, FedAvg. Per neuron: share of spectral power at the neuron's dominant frequency in the GLOBAL "
          "W1 (conc_g; uniform = 0.021) and in the clients' per-round deviations from it (conc_d).\n"
          "      agree = |mean_k dev_k(ν*)|² / mean_k |dev_k(ν*)|² at that frequency: 1 = clients move it "
          "identically, 1/K = independently, 0 = their moves cancel.\n")
    print(f"{'cell':<54}{'n':>3}{'held':>6} | {'pre: conc_g':>11}{'conc_d':>8}{'agree':>7} | {'post: conc_g':>12}{'conc_d':>8}{'agree':>7}{'1/K':>7}")
    out = {}
    for k in sorted(cells, key=lambda k: (k[4], str(k[5]), k[2], k[3], k[6], k[1], k[0])):
        rs = cells[k]
        pre = [x for s in rs for x in s["recs"] if x["pre"]]
        post = [x for s in rs for x in s["recs"] if not x["pre"]]
        pm = lambda arr, key: f3(statistics.median([x[key] for x in arr])) if arr else "—"
        out[cellname(k)] = dict(pre=pre, post=post)
        print(f"{cellname(k):<54}{len(rs):>3}{sum(s['held'] for s in rs):>4}/{len(rs)} | {pm(pre,'cg'):>11}{pm(pre,'cd'):>8}"
              f"{pm(pre,'agree'):>7} | {pm(post,'cg'):>12}{pm(post,'cd'):>8}{pm(post,'agree'):>7}{1/k[2]:>7.3f}")
    return out


# ── 25.3 ─────────────────────────────────────────────────────────────────────
def embed():
    """Setup B: embedding IPR at memorisation / crossing / end, and time-resolved against train accuracy."""
    cells = collections.defaultdict(list)
    for r in [x for x in rows() if x["setup"] == "B"]:
        h = history(r)
        if not h or not h.get("embed_ipr") or h["embed_ipr"][0] is None:
            continue
        steps = h.get("total_steps") or h.get("epoch"); e = h["embed_ipr"]
        tm = first(steps, h["train_acc"], 99); tf = first(steps, h["test_acc"], 95)
        k = cellkey(r)
        k = k[:1] + (k[1],) + k[2:]
        cells[(k[0], float(r["weight_decay"])) + k[1:]].append(dict(
            e0=e[0], e_memo=at(steps, e, tm), e_fc=at(steps, e, tf), e_end=e[-1],
            peak=max(v for v in e if v is not None), tm=tm, tf=tf, peak_train=max(h["train_acc"]),
            held=r["grokked"] == "True", h=h, steps=steps))
    print("\n25.3  setup B: IPR of the DFT of W_E over the token index, at init / memorisation / first crossing / end\n")
    print(f"{'cell':<66}{'n':>3}{'held':>6}{'pk_train':>9} | {'init':>6}{'@memo':>7}{'@cross':>7}{'end':>6}{'peak':>6} | {'t_memo':>8}{'t_cross':>9}")
    out = {}
    for k in sorted(cells, key=lambda k: (k[0], k[1], k[2], k[5], str(k[6]), k[3], k[4], k[7])):
        rs = cells[k]
        if len(rs) < 2:
            continue
        name = f"wd{k[1]} " + cellname((k[0],) + k[2:])
        out[name] = [{kk: v for kk, v in x.items() if kk not in ("h", "steps")} for x in rs]
        print(f"{name:<66}{len(rs):>3}{sum(x['held'] for x in rs):>4}/{len(rs)}{med(rs,'peak_train'):>9.1f} | "
              f"{f3(med(rs,'e0')):>6}{f3(med(rs,'e_memo')):>7}{f3(med(rs,'e_fc')):>7}{f3(med(rs,'e_end')):>6}{f3(med(rs,'peak')):>6} | "
              f"{f0(med(rs,'tm')):>8}{f0(med(rs,'tf')):>9}")
    # time-resolved: train accuracy / embedding IPR at fixed steps
    GRID = [200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000]
    SHOW = [("centralised α0.3, wd 1.0", ("centralised central_anchor", 1.0, 0.3, 1)),
            ("iid K=10, E=5, wd 1.0", ("aggregation", 1.0, 0.3, 10)),
            ("iid K=20, E=5, wd 1.0", ("aggregation", 1.0, 0.3, 20)),
            ("iid K=50, E=5, wd 1.0", ("aggregation", 1.0, 0.3, 50)),
            ("target K=10, wd 1.0", ("partitions", 1.0, 0.3, 10, "target")),
            ("Dirichlet 0.1 K=10, wd 1.0", ("dirichlet_setups", 1.0, 0.3, 10, "dirichlet", "0.1")),
            ("iid K=20, E=5, wd 0.1", ("k_collapse_budget", 0.1, 0.3, 20)),
            ("iid K=50, E=5, wd 0.1", ("k_collapse_budget", 0.1, 0.3, 50))]
    print("\n      train accuracy / embedding IPR at fixed steps, medians over runs\n")
    print(f"{'cell':<30}" + "".join(f"{s:>12,}" for s in GRID))
    for name, sel in SHOW:
        rs = [x for k, v in cells.items() for x in v
              if k[0] == sel[0] and k[1] == sel[1] and k[2] == sel[2] and k[3] == sel[3]
              and (len(sel) < 5 or k[5] == sel[4]) and (len(sel) < 6 or str(k[6]) == sel[5])]
        line = f"{name:<30}"
        for s in GRID:
            vals = []
            for x in rs:
                if s > x["steps"][-1]:
                    continue
                i = min(range(len(x["steps"])), key=lambda j: abs(x["steps"][j] - s))
                if x["h"]["embed_ipr"][i] is not None:
                    vals.append((x["h"]["train_acc"][i], x["h"]["embed_ipr"][i]))
            line += f"{'':>12}" if not vals else f"{statistics.median(v[0] for v in vals):>6.0f}/{statistics.median(v[1] for v in vals):.3f}"
        print(line)
    return out


# ── 25.4 ─────────────────────────────────────────────────────────────────────
KEEP = {"aggregation", "local_epochs", "e50_long", "dirichlet_setups", "partitions",
        "participation_setups", "setup_k_ladder", "centralised aggregation"}


def s5():
    """C and D: coset-attribution lead over exact accuracy; D's irrep excess; C's embedding structure vs the crossing."""
    BASE = dict(zip(irreps.IRREP_NAMES, irreps.random_baseline_fractions(5)))
    out = {}
    for setup in ("D", "C"):
        cells = collections.defaultdict(list)
        for r in [x for x in rows() if x["setup"] == setup]:
            k = cellkey(r)
            if k[0] not in KEEP or (k[0] == "centralised aggregation" and k[1] != 0.3):
                continue
            h = history(r)
            if not h or "coset_accuracy" not in h:
                continue
            steps = h.get("total_steps") or h.get("epoch")
            cos = [None if v is None else 100 * v for v in h["coset_accuracy"]]
            tm = first(steps, h["train_acc"], 99); tf = first(steps, h["test_acc"], 85); tc = first(steps, cos, 85)
            rec = dict(tm=tm, tf=tf, tc=tc, lead=(tf - tc) if (tf and tc) else None, cos_end=cos[-1],
                       pur_end=h["coset_purity"][-1], test_end=h["test_acc"][-1], held=r["grokked"] == "True")
            if setup == "D" and "irrep_structure_u" in h:
                su = h["irrep_structure_u"]
                rec.update(su_memo=at(steps, su, tm), su_fc=at(steps, su, tf), su_end=su[-1])
                end = {n: h[f"irrep_u_{n}"][-1] for n in irreps.IRREP_NAMES if f"irrep_u_{n}" in h}
                s = sum(end.values()) or 1
                rec.update(ex41=(end["41"] / s) / BASE["41"], ex11111=(end["11111"] / s) / BASE["11111"],
                           ex5=(end["5"] / s) / BASE["5"])
            cells[k].append(rec)
        print(f"\n25.4  setup {setup}: coset attribution. t_c = first step coset accuracy ≥ 85%; lead = t_cross − t_c "
              f"(positive: the coset level arrives first)\n")
        hdr = f"{'cell':<50}{'n':>3}{'held':>6} | {'t_memo':>8}{'t_c':>9}{'t_cross':>9}{'lead':>8}{'coset@end':>10}{'purity':>7}{'test@end':>9}"
        if setup == "D":
            hdr += f" | {'sU@memo':>8}{'sU@cross':>9}{'sU@end':>7}{'[4,1]×':>7}{'[1⁵]×':>7}{'[5]×':>6}"
        print(hdr)
        for k in sorted(cells, key=sort_key):
            rs = cells[k]
            if len(rs) < 2:
                continue
            out[f"{setup} {cellname(k)}"] = rs
            line = (f"{cellname(k):<50}{len(rs):>3}{sum(x['held'] for x in rs):>4}/{len(rs)} | {f0(med(rs,'tm')):>8}{f0(med(rs,'tc')):>9}"
                    f"{f0(med(rs,'tf')):>9}{f0(med(rs,'lead')):>8}{f1(med(rs,'cos_end')):>10}{f3(med(rs,'pur_end')):>7}{f1(med(rs,'test_end')):>9}")
            if setup == "D":
                line += f" | {f3(med(rs,'su_memo')):>8}{f3(med(rs,'su_fc')):>9}{f3(med(rs,'su_end')):>7}{f3(med(rs,'ex41')):>7}{f3(med(rs,'ex11111')):>7}{f3(med(rs,'ex5')):>6}"
            print(line)
    # C: embedding structure at checkpoints before / after the crossing
    print("\n25.4  setup C: irrep structure of W_E at checkpoints before and after the model's first crossing\n")
    cells = collections.defaultdict(list)
    for r in [x for x in rows() if x["setup"] == "C" and x["mode"] == "federated"]:
        cks = sorted(_ck(r["id"], "ckpt_round").items())
        if not cks:
            continue
        E = float(r["local_epochs"]); tf = None if r["t_first_cross"] == "inf" else float(r["t_first_cross"])
        pts = []
        for rnd, f in cks:
            WE = torch.load(f, map_location="cpu", weights_only=True)["W_E"].T.float()
            pts.append(dict(step=rnd * E, s=irreps.structure_score(WE)))
        cells[cellkey(r)].append(dict(pts=pts, tf=tf, held=r["grokked"] == "True"))
    print(f"{'cell':<50}{'held':>6}{'ckpt every':>11} | {'structure pre-cross':>20}{'n':>4}{'post':>8}{'n':>4}")
    for k in sorted(cells, key=sort_key):
        rs = cells[k]
        pre = [p["s"] for s in rs for p in s["pts"] if s["tf"] is None or p["step"] < s["tf"]]
        post = [p["s"] for s in rs for p in s["pts"] if s["tf"] is not None and p["step"] >= s["tf"]]
        out[f"C ckpt {cellname(k)}"] = dict(pre=pre, post=post)
        print(f"{cellname(k):<50}{sum(s['held'] for s in rs):>4}/{len(rs)}{rs[0]['pts'][0]['step']:>11,.0f} | "
              f"{f3(statistics.median(pre) if pre else None):>20}{len(pre):>4}{f3(statistics.median(post) if post else None):>8}{len(post):>4}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["ipr", "drift", "embed", "s5", "all"])
    ap.add_argument("--json", help="directory to write each pass's raw numbers into")
    a = ap.parse_args()
    for what in (["ipr", "drift", "embed", "s5"] if a.what == "all" else [a.what]):
        res = globals()[what]()
        if a.json:
            os.makedirs(a.json, exist_ok=True)
            json.dump(res, open(os.path.join(a.json, f"{what}.json"), "w"), default=float)
