"""Aggregate the mechanistic-interpretability tables into numerical findings.

    venv/bin/python scripts/mechinterp/findings.py      # -> results/mechinterp/findings/

Inputs (each optional; a missing table skips its analyses):
  checkpoint_metrics.csv   checkpoints.py --collect
  history_moments.csv      histories.py
  client_metrics.csv       clients.py
  twin_trajectories.csv    twins.py compare

OUTPUTS (all long-format CSV unless noted)

  ckpt_moments_by_cell.csv   median / quartiles of every checkpoint metric at four moments
                             per run -- init (first checkpoint), memo (checkpoint nearest
                             t_memo), cross (nearest t_first_cross) and end (last) -- by
                             setup, family (data fraction, weight decay), axis and cell.
                             A moment is used only if the nearest checkpoint lies within
                             x0.67-x1.5 of the event time.
  axis_trends.csv            Spearman correlation between the axis variable and each
                             metric at each moment, within each (setup, family, axis):
                             K -> log K, E -> log E, f -> f, dirichlet -> -log10 alpha_dir
                             (increasing heterogeneity), centralised_alpha -> alpha.
                             Includes the medians at the lowest and highest axis value.
  categorical_contrasts.csv  partitions and algorithms: each level's median against the
                             reference level (iid / fedavg) at the same setup, family and
                             K, with a rank-biserial effect size.
  delay_predictors.csv       across runs of a setup that memorised and crossed: Spearman of
                             each metric at memo (and at cross) with log delay and with
                             log t_first_cross; and, across all runs that memorised, the
                             AUC with which the metric at memo separates runs that crossed
                             from runs that did not.
  lead_lag_ckpt.csv          per setup and metric, from checkpoint trajectories: median
                             lead of the metric's half-change time over t_first_cross, as
                             a fraction of the delay, and the fraction of runs it leads in.
  history_leads.csv          the same from the dense logged histories (history_moments).
  client_phases.csv          client metrics by setup, axis, cell and phase (before t_memo,
                             during the delay, after t_first_cross).
  client_delay_correlations.csv  run-median client metrics during the delay vs log delay.
  twin_summary.csv           federated-vs-twin similarity by setup, family, axis, cell and
                             moment, plus axis trends of each similarity at crossing.
  FINDINGS.md                the strongest effects above, in words and numbers.

p-values use the Fisher z approximation for Spearman's rho, z = atanh(rho) sqrt(n-3):
adequate for ranking effects, not for claims at small n.
"""
import collections
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

FD = os.path.join(C.OUT, "findings")
PREFIXES = ("fn_", "w_", "rep_", "unit_", "add_", "circ_", "fin_", "fU_", "fV_", "fout_", "clock_",
            "irr_", "coset_", "attn_", "sharp_")
TEXT_COLS = {"clock_top5", "irr_U_dominant", "irr_V_dominant", "irr_E_dominant", "irr_out_dominant",
             "cli_global_dominant_irrep", "key_freqs_final", "unit", "twin_id"}


# ── io and stats ─────────────────────────────────────────────────────────────

def load(name):
    path = os.path.join(C.OUT, name)
    if not os.path.exists(path):
        print(f"  (missing {name}, skipped)")
        return None
    return list(csv.DictReader(open(path)))


def num(v):
    return C.fnum(v)


def ranks(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ties
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / counts)[inv]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 6:
        return math.nan, math.nan, n
    rx, ry = ranks(x[ok]), ranks(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return math.nan, math.nan, n
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rho_c = max(min(rho, 0.999999), -0.999999)
    z = math.atanh(rho_c) * math.sqrt(n - 3)
    return rho, math.erfc(abs(z) / math.sqrt(2)), n


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return math.nan
    r = ranks(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() + len(pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def q(vals, p):
    v = np.asarray([x for x in vals if math.isfinite(x)], float)
    return float(np.quantile(v, p)) if v.size else math.nan


def family(r):
    frac = r["alpha"] if r["dataset"] != "mnist" else "n" + r["n_train"]
    return f"{frac}|wd{r['weight_decay']}"


def cell(r):
    if r["mode"] != "federated":
        return f"central {r['group']} a{r['alpha']}"
    part = r["partition"] + (f"{r['dirichlet_alpha']}" if r["partition"].startswith("dirichlet") else "")
    return (f"{r['group']} K{r['num_clients']} E{r['local_epochs']} f{r['fraction_train']} {part} "
            f"{r['strategy']}").replace(".0 ", " ")


def axis_value(r):
    ax = r.get("axis")
    try:
        if ax == "K":
            return math.log(num(r["num_clients"]))
        if ax == "E":
            return math.log(num(r["local_epochs"]))
        if ax == "f":
            return num(r["fraction_train"])
        if ax == "dirichlet":
            return -math.log10(num(r["dirichlet_alpha"]))
        if ax == "centralised_alpha":
            return num(r["alpha"])
    except (ValueError, TypeError):
        return math.nan
    return math.nan


def metric_columns(rows):
    cols = []
    for k in rows[0].keys():
        if k.startswith(PREFIXES) and k not in TEXT_COLS:
            cols.append(k)
    return cols


# ── checkpoint moments ───────────────────────────────────────────────────────

def run_moments_from_ckpts(ck):
    by_run = collections.defaultdict(list)
    for r in ck:
        by_run[r["id"]].append(r)
    moments = {}
    for rid, recs in by_run.items():
        recs.sort(key=lambda r: num(r["step"]))
        steps = np.array([num(r["step"]) for r in recs])
        m = {"init": recs[0], "end": recs[-1]}
        for name, key in (("memo", "t_memo"), ("cross", "t_first_cross")):
            t = num(recs[0][key])
            if math.isfinite(t) and t > 0:
                j = int(np.argmin(np.abs(steps - t)))
                if 0.67 <= steps[j] / t <= 1.5:
                    m[name] = recs[j]
        moments[rid] = m
    return by_run, moments


def ckpt_moments_by_cell(ck, cols, moments):
    groups = collections.defaultdict(list)
    for rid, m in moments.items():
        r0 = m["init"]
        for name, rec in m.items():
            groups[(r0["setup"], family(r0), r0["axis"], cell(r0), name)].append(rec)
    out = []
    for (setup, fam, axis, cl, moment), recs in sorted(groups.items()):
        for c in cols:
            vals = [num(r[c]) for r in recs]
            vals = [v for v in vals if math.isfinite(v)]
            if not vals:
                continue
            out.append({"setup": setup, "family": fam, "axis": axis, "cell": cl, "moment": moment,
                        "metric": c, "n": len(vals), "median": q(vals, .5), "q25": q(vals, .25),
                        "q75": q(vals, .75)})
    return out


def axis_trends(cols, moments, value_key=None):
    groups = collections.defaultdict(list)
    for rid, m in moments.items():
        r0 = m["init"]
        if r0["axis"] in ("K", "E", "f", "dirichlet", "centralised_alpha"):
            groups[(r0["setup"], family(r0), r0["axis"])].append(m)
    out = []
    for (setup, fam, axis), ms in sorted(groups.items()):
        for moment in ("init", "memo", "cross", "end"):
            recs = [(axis_value(m["init"]), m[moment]) for m in ms if moment in m]
            if len(recs) < 6:
                continue
            xs = np.array([a for a, _ in recs])
            if len(set(np.round(xs, 6))) < 3:
                continue
            lo, hi = xs.min(), xs.max()
            for c in cols:
                ys = [num(rec[c]) for _, rec in recs]
                rho, p, n = spearman(xs, ys)
                if not math.isfinite(rho):
                    continue
                out.append({"setup": setup, "family": fam, "axis": axis, "moment": moment, "metric": c,
                            "n": n, "spearman": rho, "p_approx": p,
                            "median_at_axis_min": q([y for x, y in zip(xs, ys) if x == lo], .5),
                            "median_at_axis_max": q([y for x, y in zip(xs, ys) if x == hi], .5),
                            "n_levels": len(set(np.round(xs, 6)))})
    return out


def categorical_contrasts(cols, moments):
    out = []
    specs = (("partition", "partition", "iid"), ("algorithm", "strategy", "fedavg"))
    for axis, field, ref in specs:
        groups = collections.defaultdict(lambda: collections.defaultdict(list))
        for rid, m in moments.items():
            r0 = m["init"]
            if r0["axis"] != axis and not (axis == "partition" and r0["axis"] == "K" and r0["partition"] == "iid"):
                continue
            key = (r0["setup"], family(r0), r0["num_clients"],
                   r0["local_epochs"] if axis == "algorithm" else "", r0["partition"] if axis == "algorithm" else "")
            level = r0[field] + (r0["dirichlet_alpha"] if field == "partition" and r0["partition"].startswith("dirichlet") else "")
            groups[key][level].append(m)
        for key, levels in sorted(groups.items()):
            if ref not in levels:
                continue
            for level, ms in levels.items():
                if level == ref:
                    continue
                for moment in ("memo", "cross", "end"):
                    a = [m[moment] for m in ms if moment in m]
                    b = [m[moment] for m in levels[ref] if moment in m]
                    if len(a) < 2 or len(b) < 2:
                        continue
                    for c in cols:
                        va = [num(x[c]) for x in a]
                        vb = [num(x[c]) for x in b]
                        va, vb = [v for v in va if math.isfinite(v)], [v for v in vb if math.isfinite(v)]
                        if len(va) < 2 or len(vb) < 2:
                            continue
                        au = auc(va, vb) if min(len(va), len(vb)) >= 3 else math.nan
                        out.append({"setup": key[0], "family": key[1], "axis": axis, "K": key[2], "E": key[3],
                                    "partition": key[4], "level": level, "reference": ref, "moment": moment,
                                    "metric": c, "n_level": len(va), "n_ref": len(vb),
                                    "median_level": q(va, .5), "median_ref": q(vb, .5),
                                    "rank_biserial": 2 * au - 1 if math.isfinite(au) else math.nan})
    return out


def delay_predictors(cols, moments):
    out = []
    by_setup = collections.defaultdict(list)
    for rid, m in moments.items():
        by_setup[m["init"]["setup"]].append(m)
    for setup, ms in sorted(by_setup.items()):
        crossed = [m for m in ms if "memo" in m and math.isfinite(num(m["init"]["t_first_cross"]))
                   and math.isfinite(num(m["init"]["t_memo"]))]
        delay = [math.log(max(num(m["init"]["t_first_cross"]) - num(m["init"]["t_memo"]), 1.0)) for m in crossed]
        tfc = [math.log(num(m["init"]["t_first_cross"])) for m in crossed]
        memorised = [m for m in ms if "memo" in m]
        pos = [m for m in memorised if math.isfinite(num(m["init"]["t_first_cross"]))]
        neg = [m for m in memorised if not math.isfinite(num(m["init"]["t_first_cross"]))]
        for c in cols:
            rec = {"setup": setup, "metric": c, "n_crossed": len(crossed), "n_memorised_not_crossed": len(neg)}
            for moment in ("memo", "cross"):
                ys = [num(m[moment][c]) if moment in m else math.nan for m in crossed]
                rec[f"rho_{moment}_vs_log_delay"], rec[f"p_{moment}_delay"], rec[f"n_{moment}"] = spearman(ys, delay)
                rec[f"rho_{moment}_vs_log_tcross"], _, _ = spearman(ys, tfc)
            rec["auc_memo_crossed_vs_not"] = auc([num(m["memo"][c]) for m in pos], [num(m["memo"][c]) for m in neg])
            if any(math.isfinite(v) for v in rec.values() if isinstance(v, float)):
                out.append(rec)
    return out


def lead_lag(cols, by_run):
    per = collections.defaultdict(list)
    for rid, recs in by_run.items():
        tm, tc = num(recs[0]["t_memo"]), num(recs[0]["t_first_cross"])
        if not (math.isfinite(tm) and math.isfinite(tc) and tc > tm):
            continue
        steps = np.array([num(r["step"]) for r in recs])
        after = steps >= tm * 0.67
        if after.sum() < 4:
            continue
        for c in cols:
            v = np.array([num(r[c]) for r in recs])
            ok = after & np.isfinite(v)
            if ok.sum() < 4:
                continue
            s, y = steps[ok], v[ok]
            total = y[-1] - y[0]
            if abs(total) < 1e-9 * max(1.0, abs(y[0])):
                continue
            frac = (y - y[0]) / total
            idx = np.nonzero(frac >= 0.5)[0]
            if not idx.size:
                continue
            t50 = s[idx[0]]
            per[(recs[0]["setup"], c)].append((tc - t50) / (tc - tm))
    out = []
    for (setup, c), leads in sorted(per.items()):
        leads = np.array(leads)
        out.append({"setup": setup, "metric": c, "n_runs": len(leads), "median_lead_frac_delay": float(np.median(leads)),
                    "frac_runs_leading": float((leads > 0).mean())})
    return out


# ── other tables ─────────────────────────────────────────────────────────────

def history_leads(hm):
    out = []
    cols = [k for k in hm[0] if k.endswith("_lead_frac_delay")]
    groups = collections.defaultdict(list)
    for r in hm:
        groups[(r["setup"], r["axis"])].append(r)
        groups[(r["setup"], "all")].append(r)
    for (setup, axis), rs in sorted(groups.items()):
        for c in cols:
            v = [num(r[c]) for r in rs]
            v = [x for x in v if math.isfinite(x)]
            if len(v) < 3:
                continue
            out.append({"setup": setup, "axis": axis, "series": c.replace("_lead_frac_delay", ""), "n": len(v),
                        "median_lead_frac_delay": q(v, .5), "q25": q(v, .25), "q75": q(v, .75),
                        "frac_leading": float(np.mean(np.array(v) > 0))})
    return out


def client_tables(cl):
    cols = [k for k in cl[0] if k.startswith("cli_") and k not in TEXT_COLS]
    groups = collections.defaultdict(list)
    run_delay = collections.defaultdict(list)
    for r in cl:
        st, tm, tc = num(r["step"]), num(r["t_memo"]), num(r["t_first_cross"])
        if math.isfinite(tm) and st <= tm:
            ph = "pre_memo"
        elif math.isfinite(tc) and st >= tc:
            ph = "post_cross"
        else:
            ph = "delay_or_stalled"
        groups[(r["setup"], family(r), r["axis"], cell(r), ph)].append(r)
        if ph == "delay_or_stalled" and math.isfinite(tc):
            run_delay[r["id"]].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        for c in cols:
            v = [num(r[c]) for r in rs]
            v = [x for x in v if math.isfinite(x)]
            if not v:
                continue
            out.append({"setup": key[0], "family": key[1], "axis": key[2], "cell": key[3], "phase": key[4],
                        "metric": c, "n": len(v), "median": q(v, .5), "q25": q(v, .25), "q75": q(v, .75)})
    corr = []
    by_setup = collections.defaultdict(list)
    for rid, rs in run_delay.items():
        by_setup[rs[0]["setup"]].append(rs)
    for setup, runs in sorted(by_setup.items()):
        delay = [math.log(max(num(rs[0]["t_first_cross"]) - num(rs[0]["t_memo"]), 1.0)) for rs in runs]
        for c in cols:
            meds = [q([num(r[c]) for r in rs], .5) for rs in runs]
            rho, p, n = spearman(meds, delay)
            if math.isfinite(rho):
                corr.append({"setup": setup, "metric": c, "n_runs": n, "spearman_vs_log_delay": rho, "p_approx": p})
    return out, corr


def twin_tables(tw):
    cols = [k for k in tw[0] if k.startswith("tw_") and k not in TEXT_COLS]
    by_run = collections.defaultdict(list)
    for r in tw:
        by_run[r["id"]].append(r)
    groups = collections.defaultdict(list)
    at_cross = []
    for rid, recs in by_run.items():
        recs.sort(key=lambda r: num(r["step"]))
        steps = np.array([num(r["step"]) for r in recs])
        picks = {"first": recs[0], "end": recs[-1]}
        for name, key in (("memo", "t_memo"), ("cross", "t_first_cross")):
            t = num(recs[0][key])
            if math.isfinite(t) and t > 0:
                j = int(np.argmin(np.abs(steps - t)))
                if 0.67 <= steps[j] / t <= 1.5:
                    picks[name] = recs[j]
        for name, rec in picks.items():
            groups[(rec["setup"], family(rec), rec["axis"], cell(rec), name)].append(rec)
        if "cross" in picks:
            at_cross.append(picks["cross"])
    out = []
    for key, rs in sorted(groups.items()):
        for c in cols:
            v = [num(r[c]) for r in rs]
            v = [x for x in v if math.isfinite(x)]
            if v:
                out.append({"setup": key[0], "family": key[1], "axis": key[2], "cell": key[3], "moment": key[4],
                            "metric": c, "n": len(v), "median": q(v, .5), "q25": q(v, .25), "q75": q(v, .75)})
    fam = collections.defaultdict(list)
    for r in at_cross:
        if r["axis"] in ("K", "E", "f", "dirichlet"):
            fam[(r["setup"], family(r), r["axis"])].append(r)
    for (setup, f, axis), rs in sorted(fam.items()):
        xs = [axis_value(r) for r in rs]
        if len(set(np.round(xs, 6))) < 3:
            continue
        for c in cols:
            rho, p, n = spearman(xs, [num(r[c]) for r in rs])
            if math.isfinite(rho):
                out.append({"setup": setup, "family": f, "axis": axis, "cell": "TREND at cross", "moment": "cross",
                            "metric": c, "n": n, "median": rho, "q25": p, "q75": math.nan})
    return out


# ── report ───────────────────────────────────────────────────────────────────

def fmt(v, d=3):
    return "—" if not isinstance(v, (int, float)) or not math.isfinite(v) else f"{v:.{d}g}"


def report(trends, preds, leads, hleads, ccorr, twins, contrasts):
    L = ["# Mechanistic-interpretability findings (auto-generated)", "",
         "Generated by `scripts/mechinterp/findings.py` from the tables in `results/mechinterp/`. "
         "Spearman p-values use the Fisher z approximation; treat them as a ranking of effects.", ""]
    if trends:
        L += ["## Strongest axis trends at first crossing (|rho| >= 0.6, n >= 9, p < 0.01)", ""]
        by = collections.defaultdict(list)
        for t in trends:
            if t["moment"] == "cross" and abs(t["spearman"]) >= 0.6 and t["n"] >= 9 and t["p_approx"] < 0.01:
                by[(t["setup"], t["family"], t["axis"])].append(t)
        for key, ts in sorted(by.items()):
            ts.sort(key=lambda t: -abs(t["spearman"]))
            L.append(f"**{key[0]} {key[1]} axis {key[2]}** ({ts[0]['n']} runs)")
            for t in ts[:12]:
                L.append(f"- `{t['metric']}` rho {t['spearman']:+.2f}: {fmt(t['median_at_axis_min'])} → "
                         f"{fmt(t['median_at_axis_max'])}")
            L.append("")
    if preds:
        L += ["## What at memorisation predicts the delay (|rho| >= 0.5, n >= 12)", ""]
        by = collections.defaultdict(list)
        for p in preds:
            rho = p.get("rho_memo_vs_log_delay")
            if isinstance(rho, float) and math.isfinite(rho) and abs(rho) >= 0.5 and p["n_memo"] >= 12:
                by[p["setup"]].append(p)
        for setup, ps in sorted(by.items()):
            ps.sort(key=lambda p: -abs(p["rho_memo_vs_log_delay"]))
            L.append(f"**{setup}**")
            for p in ps[:12]:
                L.append(f"- `{p['metric']}` rho {p['rho_memo_vs_log_delay']:+.2f} (n {p['n_memo']}); "
                         f"AUC crossed-vs-not {fmt(p['auc_memo_crossed_vs_not'], 2)}")
            L.append("")
    if leads:
        L += ["## Metrics that change before generalisation (checkpoint trajectories; median lead >= 0.3 of the delay, leads in >= 70% of runs)", ""]
        by = collections.defaultdict(list)
        for l in leads:
            if l["n_runs"] >= 6 and l["median_lead_frac_delay"] >= 0.3 and l["frac_runs_leading"] >= 0.7:
                by[l["setup"]].append(l)
        for setup, ls in sorted(by.items()):
            ls.sort(key=lambda l: -l["median_lead_frac_delay"])
            L.append(f"**{setup}**: " + "; ".join(f"`{l['metric']}` {l['median_lead_frac_delay']:.2f}"
                                                    for l in ls[:15]))
        L.append("")
    if hleads:
        L += ["## Logged order parameters: lead over first crossing as a fraction of the delay (all runs)", ""]
        for h in hleads:
            if h["axis"] == "all":
                L.append(f"- {h['setup']} `{h['series']}`: median {h['median_lead_frac_delay']:.2f} "
                         f"(IQR {h['q25']:.2f}–{h['q75']:.2f}), leads in {100 * h['frac_leading']:.0f}% of {h['n']} runs")
        L.append("")
    if ccorr:
        L += ["## Client metrics during the delay vs log delay (|rho| >= 0.4)", ""]
        for c in sorted(ccorr, key=lambda c: (c["setup"], -abs(c["spearman_vs_log_delay"]))):
            if abs(c["spearman_vs_log_delay"]) >= 0.4:
                L.append(f"- {c['setup']} `{c['metric']}` rho {c['spearman_vs_log_delay']:+.2f} (n {c['n_runs']})")
        L.append("")
    if twins:
        L += ["## Federated vs centralised twin at first crossing (medians by cell)", ""]
        for t in twins:
            if t["moment"] == "cross" and t["cell"] != "TREND at cross" and t["metric"] in (
                    "tw_cka_te", "tw_pred_agree_te", "tw_rel_dist", "tw_clock_key_jaccard", "tw_irrep_same_dominant"):
                L.append(f"- {t['setup']} {t['family']} {t['cell']} `{t['metric']}` {fmt(t['median'])} (n {t['n']})")
        L.append("")
    if contrasts:
        L += ["## Largest partition / algorithm contrasts at first crossing (|rank-biserial| = 1, n >= 3 each)", ""]
        for c in contrasts:
            if c["moment"] == "cross" and isinstance(c["rank_biserial"], float) and abs(c["rank_biserial"]) == 1:
                if c["metric"].startswith(("clock_share", "unit_interaction_share", "circ_share_interaction",
                                           "irr_U_structure", "irr_E_structure", "w_all_rel_norm", "rep_nc1",
                                           "sharp_lambda_max", "fin_ipr", "fU_ipr")):
                    L.append(f"- {c['setup']} {c['family']} K{c['K']} {c['level']} vs {c['reference']} "
                             f"`{c['metric']}` {fmt(c['median_level'])} vs {fmt(c['median_ref'])}")
        L.append("")
    return "\n".join(L)


def main():
    os.makedirs(FD, exist_ok=True)
    trends = preds = leads = hl = ccorr = tws = contrasts = []
    ck = load("checkpoint_metrics.csv")
    if ck:
        cols = metric_columns(ck)
        by_run, moments = run_moments_from_ckpts(ck)
        C.write_csv(os.path.join(FD, "ckpt_moments_by_cell.csv"), ckpt_moments_by_cell(ck, cols, moments))
        trends = axis_trends(cols, moments)
        C.write_csv(os.path.join(FD, "axis_trends.csv"), trends)
        contrasts = categorical_contrasts(cols, moments)
        C.write_csv(os.path.join(FD, "categorical_contrasts.csv"), contrasts)
        preds = delay_predictors(cols, moments)
        C.write_csv(os.path.join(FD, "delay_predictors.csv"), preds)
        leads = lead_lag(cols, by_run)
        C.write_csv(os.path.join(FD, "lead_lag_ckpt.csv"), leads)
        print(f"  checkpoint findings: {len(moments)} runs, {len(cols)} metrics")
    hm = load("history_moments.csv")
    if hm:
        hl = history_leads(hm)
        C.write_csv(os.path.join(FD, "history_leads.csv"), hl)
    cl = load("client_metrics.csv")
    if cl:
        phases, ccorr = client_tables(cl)
        C.write_csv(os.path.join(FD, "client_phases.csv"), phases)
        C.write_csv(os.path.join(FD, "client_delay_correlations.csv"), ccorr)
    tw = load("twin_trajectories.csv")
    if tw:
        tws = twin_tables(tw)
        C.write_csv(os.path.join(FD, "twin_summary.csv"), tws)
    open(os.path.join(FD, "FINDINGS.md"), "w").write(report(trends, preds, leads, hl, ccorr, tws, contrasts))
    print("  wrote", FD)


if __name__ == "__main__":
    main()
