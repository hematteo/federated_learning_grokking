"""Compare IPR onset: 2.5× threshold vs changepoint detection (PELT)."""

import json, os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import ruptures as rpt

def smooth(arr, window=100):
    if len(arr) < window: return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")

def load_json(path):
    with open(path) as f:
        return json.load(f)

def compute_t_grok(steps, test_accs, threshold=0.95):
    for s, a in zip(steps, test_accs):
        if a >= threshold: return s
    return float("inf")

def ipr_changepoint(ipr_smooth):
    """Find the first changepoint in IPR using PELT (L2 cost).

    Returns the index of the first regime shift.
    """
    # PELT with rbf kernel, pen=auto via BIC-like penalty
    # We want to find where IPR transitions from flat to rising
    signal = ipr_smooth.reshape(-1, 1)
    algo = rpt.Pelt(model="l2", min_size=50).fit(signal)
    # Penalty: log(n) * variance gives BIC-like behaviour
    pen = np.log(len(signal)) * np.var(signal)
    changepoints = algo.predict(pen=pen)
    # Remove the final point (always = len(signal))
    changepoints = [c for c in changepoints if c < len(signal)]
    if changepoints:
        return changepoints[0]  # first regime shift
    return None

# ── Load all grokking runs ──
EXP3A_DIR = "results/exp3_heterogeneity/exp3a"
alphas = [0.25, 0.3, 0.35, 0.5]
dir_alphas = [0.01, 0.1, 0.5, 1.0, 10.0, 1000.0]
seeds = [42, 123, 456]
runs = []
for alpha in alphas:
    for dir_alpha in dir_alphas:
        for seed in seeds:
            pattern = f"*_a{alpha}_K10_*_dir{dir_alpha}_s{seed}.json"
            matches = sorted(glob.glob(os.path.join(EXP3A_DIR, pattern)))
            if not matches: continue
            h = load_json(matches[0])
            steps = np.array(h["total_steps"])
            test_accs = np.array(h["test_acc"])
            t_grok = compute_t_grok(steps.tolist(), test_accs.tolist())
            if t_grok == float("inf"): continue
            runs.append({"h": h, "alpha": alpha, "steps": steps, "t_grok": t_grok})

print(f"Loaded {len(runs)} grokking runs")

# ── Single-run demo first ──
demo_h = load_json("results/exp3_heterogeneity/exp3a/history_fed_addition_gd_p97_N256_a0.5_K10_le5_ft1.0_dirichlet_dir1000.0_s42.json")
demo_steps = np.array(demo_h["total_steps"])
demo_ipr = np.array(demo_h["ipr"])
demo_ipr_s = smooth(demo_ipr, window=100)
demo_drift = smooth(np.array(demo_h["mean_client_drift"]), window=100)

# Changepoint on smoothed IPR
demo_cp_idx = ipr_changepoint(demo_ipr_s)
print(f"Demo run: changepoint at index {demo_cp_idx}, step {int(demo_steps[demo_cp_idx])}")

# Threshold method for comparison
demo_be = max(10, len(demo_ipr) // 10)
demo_baseline = np.mean(demo_ipr[:demo_be])
demo_thresh_val = demo_baseline * 2.5
demo_thresh_idx = None
for i in range(demo_be, len(demo_ipr)):
    if demo_ipr[i] > demo_thresh_val:
        demo_thresh_idx = i
        break

# Drift peak
demo_dwu = max(10, len(demo_drift) // 20)
demo_dpeak_idx = demo_dwu + np.argmax(demo_drift[demo_dwu:])

# ── Compute both methods for all runs ──
t_ipr_thresh = []
t_ipr_cp = []
t_drift_peaks = []
t_groks = []
run_alphas = []
n_cp_failed = 0

for run in runs:
    h = run["h"]
    steps = run["steps"]
    ipr = np.array(h["ipr"])
    ipr_s = smooth(ipr, window=100)
    drift = smooth(np.array(h["mean_client_drift"]), window=100)

    # Threshold
    be = max(10, len(ipr) // 10)
    ib = np.mean(ipr[:be])
    it = ib * 2.5
    thresh_idx = None
    for i in range(be, len(ipr)):
        if ipr[i] > it:
            thresh_idx = i
            break

    # Changepoint
    cp_idx = ipr_changepoint(ipr_s)

    # Drift peak
    dwu = max(10, len(drift) // 20)
    dpeak_idx = dwu + np.argmax(drift[dwu:])

    if thresh_idx is not None and cp_idx is not None:
        t_ipr_thresh.append(float(steps[thresh_idx]))
        t_ipr_cp.append(float(steps[cp_idx]))
        t_drift_peaks.append(float(steps[dpeak_idx]))
        t_groks.append(run["t_grok"])
        run_alphas.append(run["alpha"])
    else:
        n_cp_failed += 1

print(f"Valid runs: {len(t_ipr_cp)}, changepoint failed: {n_cp_failed}")

t_ipr_thresh = np.array(t_ipr_thresh)
t_ipr_cp = np.array(t_ipr_cp)
t_drift_peaks = np.array(t_drift_peaks)
t_groks = np.array(t_groks)

alphas_unique = sorted(set(run_alphas))
alpha_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas_unique)))
alpha_cmap = dict(zip(alphas_unique, alpha_colors))
colors = [alpha_cmap[a] for a in run_alphas]

# ── Figure ──
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
fig.suptitle("Changepoint detection (PELT) vs 2.5× threshold for IPR onset",
             fontsize=14, fontweight="bold")

# ── (a) Single-run: IPR with both methods annotated ──
ax = fig.add_subplot(gs[0, 0])
ax.plot(demo_steps, demo_ipr, color="#ddd", lw=0.5, alpha=0.6, label="Raw IPR")
ax.plot(demo_steps, demo_ipr_s, color="#7B1FA2", lw=1.8, label="Smoothed IPR")
ax.axhline(demo_thresh_val, ls=":", color="#FF6F00", lw=1, alpha=0.6)
if demo_thresh_idx is not None:
    ax.axvline(demo_steps[demo_thresh_idx], ls="--", color="#FF6F00", lw=1.5,
               label=f"2.5× threshold = step {int(demo_steps[demo_thresh_idx])}")
if demo_cp_idx is not None:
    ax.axvline(demo_steps[demo_cp_idx], ls="-", color="#D32F2F", lw=2,
               label=f"PELT changepoint = step {int(demo_steps[demo_cp_idx])}")
ax.axvline(demo_steps[demo_dpeak_idx], ls="-", color="#1565C0", lw=2, alpha=0.7,
           label=f"Drift peak = step {int(demo_steps[demo_dpeak_idx])}")
ax.set_ylabel("IPR"); ax.set_xlabel("Steps")
ax.legend(fontsize=7.5, loc="center right")
ax.set_title("(a) Single run: IPR onset methods compared", fontsize=10, loc="left", fontweight="bold")

# ── (b) Single-run: drift with peak annotated ──
ax = fig.add_subplot(gs[0, 1])
ax.plot(demo_steps, demo_drift, color="#1565C0", lw=1.5)
ax.plot(demo_steps[demo_dpeak_idx], demo_drift[demo_dpeak_idx], "v",
        color="#D32F2F", markersize=12, zorder=5, label=f"Drift peak = step {int(demo_steps[demo_dpeak_idx])}")
if demo_cp_idx is not None:
    ax.axvline(demo_steps[demo_cp_idx], ls="-", color="#D32F2F", lw=2, alpha=0.5,
               label=f"IPR changepoint = step {int(demo_steps[demo_cp_idx])}")
ax.set_ylabel("Mean Client Drift"); ax.set_xlabel("Steps")
ax.legend(fontsize=8)
ax.set_title("(b) Drift peak + IPR changepoint on same timeline", fontsize=10, loc="left", fontweight="bold")

# ── (c) Scatter: IPR changepoint vs drift peak ──
ax = fig.add_subplot(gs[1, 0])
ax.scatter(t_ipr_cp, t_drift_peaks, c=colors, s=40, alpha=0.7, edgecolors="k", linewidths=0.3, zorder=3)
lims = [0, max(t_ipr_cp.max(), t_drift_peaks.max()) * 1.1]
ax.plot(lims, lims, "k--", alpha=0.4, lw=1, label="Simultaneous")
ax.fill_between(lims, lims, [lims[1]]*2, alpha=0.05, color="blue")
ax.fill_between(lims, [0]*2, lims, alpha=0.05, color="red")
ax.set_xlabel(r"$T_{IPR}$ changepoint (steps)")
ax.set_ylabel(r"$T_{drift\ peak}$ (steps)")
ax.set_xlim(lims); ax.set_ylim(lims)
r_cp, p_cp = stats.pearsonr(t_ipr_cp, t_drift_peaks)
lag_cp = t_drift_peaks - t_ipr_cp
n_after = np.sum(lag_cp > 0)
ax.text(0.97, 0.03,
        f"r = {r_cp:.3f}\nDrift after: {n_after}/{len(lag_cp)}",
        transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
ax.legend(fontsize=7, loc="upper left")
ax.set_title(r"(c) NEW: $T_{IPR\ changepoint}$ vs $T_{drift\ peak}$", fontsize=10, loc="left", fontweight="bold")

# ── (d) Scatter: IPR threshold vs drift peak (for comparison) ──
ax = fig.add_subplot(gs[1, 1])
ax.scatter(t_ipr_thresh, t_drift_peaks, c=colors, s=40, alpha=0.7, edgecolors="k", linewidths=0.3, zorder=3)
lims2 = [0, max(t_ipr_thresh.max(), t_drift_peaks.max()) * 1.1]
ax.plot(lims2, lims2, "k--", alpha=0.4, lw=1, label="Simultaneous")
ax.fill_between(lims2, lims2, [lims2[1]]*2, alpha=0.05, color="blue")
ax.fill_between(lims2, [0]*2, lims2, alpha=0.05, color="red")
ax.set_xlabel(r"$T_{IPR\ onset}$ 2.5× threshold (steps)")
ax.set_ylabel(r"$T_{drift\ peak}$ (steps)")
ax.set_xlim(lims2); ax.set_ylim(lims2)
r_th, p_th = stats.pearsonr(t_ipr_thresh, t_drift_peaks)
lag_th = t_drift_peaks - t_ipr_thresh
n_after_th = np.sum(lag_th > 0)
ax.text(0.97, 0.03,
        f"r = {r_th:.3f}\nDrift after: {n_after_th}/{len(lag_th)}",
        transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
ax.legend(fontsize=7, loc="upper left")
ax.set_title(r"(d) OLD: $T_{IPR\ 2.5\times}$ vs $T_{drift\ peak}$", fontsize=10, loc="left", fontweight="bold")

# ── (e) Lag histogram: changepoint ──
ax = fig.add_subplot(gs[2, 0])
ax.hist(lag_cp, bins=25, color="#EF5350", edgecolor="white", alpha=0.8)
ax.axvline(0, color="black", lw=1.5, label="Simultaneous")
med_cp = np.median(lag_cp)
ax.axvline(med_cp, color="red", ls="--", lw=2, label=f"Median = {med_cp:.0f}")
if len(lag_cp) > 5:
    _, w_p = stats.wilcoxon(lag_cp)
    ax.text(0.97, 0.97, f"Wilcoxon p={w_p:.2e}\n{n_after}/{len(lag_cp)} drift peak after",
            transform=ax.transAxes, va="top", ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
ax.set_xlabel(r"$T_{drift\ peak} - T_{IPR\ changepoint}$ (steps)")
ax.set_ylabel("Count")
ax.legend(fontsize=8)
ax.set_title("(e) Lag: drift peak − IPR changepoint (PELT)", fontsize=10, loc="left", fontweight="bold")

# ── (f) Lag histogram: threshold ──
ax = fig.add_subplot(gs[2, 1])
ax.hist(lag_th, bins=25, color="#FF8A65", edgecolor="white", alpha=0.8)
ax.axvline(0, color="black", lw=1.5, label="Simultaneous")
med_th = np.median(lag_th)
ax.axvline(med_th, color="red", ls="--", lw=2, label=f"Median = {med_th:.0f}")
if len(lag_th) > 5:
    _, w_p_th = stats.wilcoxon(lag_th)
    ax.text(0.97, 0.97, f"Wilcoxon p={w_p_th:.2e}\n{n_after_th}/{len(lag_th)} drift peak after",
            transform=ax.transAxes, va="top", ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
ax.set_xlabel(r"$T_{drift\ peak} - T_{IPR\ onset}$ 2.5× (steps)")
ax.set_ylabel("Count")
ax.legend(fontsize=8)
ax.set_title("(f) Lag: drift peak − IPR 2.5× threshold", fontsize=10, loc="left", fontweight="bold")

# Alpha legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker="o", color="w", markerfacecolor=alpha_cmap[a],
                          markersize=8, label=f"α={a}") for a in alphas_unique]
fig.legend(handles=legend_elements, loc="lower center", ncol=len(alphas_unique),
           fontsize=9, frameon=True, title="Dirichlet α")

plt.savefig("paper/figures/changepoint_vs_threshold.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Summary table ──
print(f"\n{'Method':<40} {'Median lag':>10} {'r':>8} {'Wilcoxon p':>12} {'After IPR':>10}")
print("-" * 85)
for name, lag, x in [
    ("PELT changepoint", lag_cp, t_ipr_cp),
    ("2.5× threshold", lag_th, t_ipr_thresh),
]:
    r, _ = stats.pearsonr(x, t_drift_peaks)
    _, p = stats.wilcoxon(lag)
    na = np.sum(lag > 0)
    print(f"{name:<40} {np.median(lag):>8.0f}   {r:>7.3f}   {p:>10.2e}   {na}/{len(lag)}")
