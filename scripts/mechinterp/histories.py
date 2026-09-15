"""Order-parameter dynamics from the logged histories of every run (no checkpoints).

    venv/bin/python scripts/mechinterp/histories.py        # -> results/mechinterp/history_*.csv

Every training loop logs, at each eval, the function (train/test loss and accuracy),
the weights (layer norms or total norm) and a setup-specific probe (Fourier IPR on
the modular quadratic MLP, embedding IPR on the modular transformer, coset accuracy,
the exact quadratic circuit and the S5 irrep profile on S5), plus client drift and
divergence on federated runs. This reads those series for all runs.

OUTPUTS

  history_moments.csv   one row per run. For each series S:
      S@init, S@memo, S@cross, S@grok, S@end      value at step 0, t_memo, t_first_cross,
                                                  t_grok and the last logged step
      S@delay25 / @delay50 / @delay75             at 25/50/75% of the way from t_memo
                                                  to t_first_cross
      S_min, S_max, S_argmin_step, S_argmax_step
      S_t50_change                                first step after t_memo at which S has
                                                  covered half its change from t_memo
                                                  to the end of the run
      S_lead_over_cross                           t_first_cross - S_t50_change (positive:
                                                  S moves before the model generalises)
      S_lead_frac_delay                           the lead as a fraction of the delay
    and run-level derived quantities:
      norm_decay_rate_delay   -d log(weight norm)/d step between t_memo and t_first_cross
      lr_times_wd             the decoupled/coupled decay timescale's inverse
      norm_ratio_cross_memo, norm_ratio_end_init
      drift_pre_memo / drift_delay / drift_post  median mean_client_drift per window
      div_pre_memo / div_delay / div_post        the same for client_weight_divergence
      post_cross_test_min     lowest test accuracy after first crossing (instability)
      train_loss_min, train_loss_zero_step       float32 loss underflow (the wd = 0 stall)

  history_series_crossaligned.csv   grokked runs: every series resampled at 48 points of
      u = step / t_first_cross, log-spaced over [0.02, 20]
  history_series_logstep.csv        all runs: resampled at 48 log-spaced steps between
      the first logged step and the budget
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

SERIES = ["train_acc", "test_acc", "train_loss", "test_loss", "weight_norm_layer1", "weight_norm_layer2",
          "weight_norm_total", "weight_norm_first", "weight_norm_last", "ipr", "embed_ipr",
          "coset_accuracy", "coset_purity", "circ_share_interaction", "circ_share_marginal",
          "circ_acc_interaction", "circ_acc_marginal", "circ_units", "irrep_structure_u",
          "irrep_structure_v", "mean_client_drift", "client_weight_divergence", "grad_norm_layer1",
          "grad_norm_layer2"]
# Series whose half-change time is a meaningful "order parameter moved" event.
LEAD_SERIES = ["weight_norm_layer1", "weight_norm_total", "ipr", "embed_ipr", "coset_accuracy",
               "circ_acc_interaction", "circ_share_interaction", "irrep_structure_u", "test_loss"]


def clean(arr):
    return np.array([np.nan if (v is None or (isinstance(v, float) and not math.isfinite(v))) else float(v)
                     for v in arr], dtype=float)


def value_at(steps, vals, t):
    if not math.isfinite(t) or len(steps) == 0:
        return math.nan
    ok = np.isfinite(vals)
    if not ok.any():
        return math.nan
    s, v = steps[ok], vals[ok]
    if t <= s[0]:
        return float(v[0])
    if t >= s[-1]:
        return float(v[-1])
    return float(np.interp(t, s, v))


def window_median(steps, vals, lo, hi):
    m = (steps > lo) & (steps <= hi) & np.isfinite(vals)
    return float(np.median(vals[m])) if m.any() else math.nan


def half_change_step(steps, vals, t0):
    """First step >= t0 at which vals has covered half of its change from t0 to the end."""
    if not math.isfinite(t0):
        return math.nan
    ok = np.isfinite(vals) & (steps >= t0)
    if ok.sum() < 3:
        return math.nan
    s, v = steps[ok], vals[ok]
    total = v[-1] - v[0]
    if abs(total) < 1e-12:
        return math.nan
    frac = (v - v[0]) / total
    idx = np.nonzero(frac >= 0.5)[0]
    return float(s[idx[0]]) if idx.size else math.nan


def run_moments(r, h):
    steps = clean(C.steps_of(h))
    out = C.descriptors(r)
    tm, tc, tg = C.fnum(r["t_memo"]), C.fnum(r["t_first_cross"]), C.fnum(r["t_grok"])
    end = float(steps[-1]) if len(steps) else math.nan
    out["lr_times_wd"] = C.fnum(r["lr"]) * C.fnum(r["weight_decay"])
    out["delay_first_cross"] = tc - tm if math.isfinite(tc) and math.isfinite(tm) else math.nan
    series = {k: clean(h[k]) for k in SERIES if k in h and len(h[k]) == len(steps)}
    for k, v in series.items():
        if not np.isfinite(v).any():
            continue
        out[f"{k}@init"] = value_at(steps, v, steps[0])
        out[f"{k}@memo"] = value_at(steps, v, tm)
        out[f"{k}@cross"] = value_at(steps, v, tc)
        out[f"{k}@grok"] = value_at(steps, v, tg)
        out[f"{k}@end"] = value_at(steps, v, end)
        if math.isfinite(tm) and math.isfinite(tc) and tc > tm:
            for q in (25, 50, 75):
                out[f"{k}@delay{q}"] = value_at(steps, v, tm + (tc - tm) * q / 100)
        ok = np.isfinite(v)
        out[f"{k}_min"] = float(np.nanmin(v))
        out[f"{k}_max"] = float(np.nanmax(v))
        out[f"{k}_argmin_step"] = float(steps[ok][np.argmin(v[ok])])
        out[f"{k}_argmax_step"] = float(steps[ok][np.argmax(v[ok])])
        if k in LEAD_SERIES:
            t50 = half_change_step(steps, v, tm)
            out[f"{k}_t50_change"] = t50
            if math.isfinite(t50) and math.isfinite(tc):
                out[f"{k}_lead_over_cross"] = tc - t50
                if math.isfinite(tm) and tc > tm:
                    out[f"{k}_lead_frac_delay"] = (tc - t50) / (tc - tm)
    norm_key = next((k for k in ("weight_norm_total", "weight_norm_layer1") if k in series
                     and np.isfinite(series[k]).any()), None)
    if norm_key:
        n = series[norm_key]
        a, b = value_at(steps, n, tm), value_at(steps, n, tc)
        if math.isfinite(tm) and math.isfinite(tc) and tc > tm and a > 0 and b > 0:
            out["norm_decay_rate_delay"] = -(math.log(b) - math.log(a)) / (tc - tm)
            out["norm_ratio_cross_memo"] = b / a
        n0, ne = value_at(steps, n, steps[0]), value_at(steps, n, end)
        out["norm_ratio_end_init"] = ne / n0 if n0 > 0 else math.nan
        out["norm_key"] = norm_key
    for key, pre in (("mean_client_drift", "drift"), ("client_weight_divergence", "div")):
        if key in series:
            v = series[key]
            lo_m = tm if math.isfinite(tm) else end
            out[f"{pre}_pre_memo"] = window_median(steps, v, 0, lo_m)
            if math.isfinite(tm):
                out[f"{pre}_delay"] = window_median(steps, v, tm, tc if math.isfinite(tc) else end)
            if math.isfinite(tc):
                out[f"{pre}_post"] = window_median(steps, v, tc, end)
    if "test_acc" in series and math.isfinite(tc):
        after = series["test_acc"][steps > tc]
        out["post_cross_test_min"] = float(np.nanmin(after)) if after.size else math.nan
    if "train_loss" in series:
        tl = series["train_loss"]
        out["train_loss_min"] = float(np.nanmin(tl))
        zero = np.nonzero(tl == 0)[0]
        out["train_loss_zero_step"] = float(steps[zero[0]]) if zero.size else math.nan
    return out


def resample(r, h, kind):
    steps = clean(C.steps_of(h))
    if len(steps) < 3:
        return []
    if kind == "crossaligned":
        tc = C.fnum(r["t_first_cross"])
        if not (math.isfinite(tc) and tc > 0):
            return []
        u = np.logspace(np.log10(0.02), np.log10(20), 48)
        grid = u * tc
    else:
        lo = steps[steps > 0][0] if (steps > 0).any() else 1.0
        grid = np.logspace(np.log10(lo), np.log10(steps[-1]), 48)
        u = grid
    rows = []
    series = {k: clean(h[k]) for k in SERIES if k in h and len(h[k]) == len(steps)}
    for i, (x, t) in enumerate(zip(u, grid)):
        if t > steps[-1] * 1.0001 or t < steps[0]:
            continue
        row = {"id": r["id"], "setup": r["setup"], "mode": r["mode"], "group": r["group"],
               "axis": r["axis"], "point": i, "x": float(x), "step": float(t)}
        for k, v in series.items():
            row[k] = value_at(steps, v, t)
        rows.append(row)
    return rows


def main():
    rows = C.load_rows()
    moments, cross, logs = [], [], []
    for n, (rid, r) in enumerate(sorted(rows.items())):
        h = C.history(rid)
        if not h or not C.steps_of(h):
            continue
        moments.append(run_moments(r, h))
        cross += resample(r, h, "crossaligned")
        logs += resample(r, h, "logstep")
        if n % 200 == 0:
            print(f"  {n}/{len(rows)}", flush=True)
    for name, data in (("history_moments.csv", moments), ("history_series_crossaligned.csv", cross),
                       ("history_series_logstep.csv", logs)):
        print("  wrote", C.write_csv(os.path.join(C.OUT, name), data), len(data), "rows")


if __name__ == "__main__":
    main()
