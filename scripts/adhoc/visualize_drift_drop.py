"""Compare T_drift_drop (50% threshold) vs T_drift_peak as temporal landmarks."""

import json, os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def smooth(arr, window=100):
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")

def load_json(path):
    with open(path) as f:
        return json.load(f)

def compute_t_grok(steps, test_accs, threshold=0.95):
    for s, a in zip(steps, test_accs):
        if a >= threshold:
            return s
    return float("inf")

EXP3A_DIR = "results/exp3_heterogeneity/exp3a"

# ── Load all grokking runs ──
alphas = [0.25, 0.3, 0.35, 0.5]
dir_alphas = [0.01, 0.1, 0.5, 1.0, 10.0, 1000.0]
seeds = [42, 123, 456]
runs = []
for alpha in alphas:
    for dir_alpha in dir_alphas:
        for seed in seeds:
            pattern = f"*_a{alpha}_K10_*_dir{dir_alpha}_s{seed}.json"
            matches = sorted(glob.glob(os.path.join(EXP3A_DIR, pattern)))
            if not matches:
                continue
            h = load_json(matches[0])
            steps = np.array(h["total_steps"])
            test_accs = np.array(h["test_acc"])
            t_grok = compute_t_grok(steps.tolist(), test_accs.tolist())
            if t_grok == float("inf"):
                continue
            runs.append({"h": h, "alpha": alpha, "steps": steps, "t_grok": t_grok})

# ── Compute both metrics for each run ──
t_ipr_onsets = []
t_drift_drops = []   # old: 50% threshold
t_drift_peaks = []   # new: just the peak
run_alphas = []

for run in runs:
    h = run["h"]
    steps = run["steps"]
    ipr = np.array(h["ipr"])
    drift = smooth(np.array(h["mean_client_drift"]), window=100)

    # IPR onset
    baseline_end = max(10, len(ipr) // 10)
    ipr_baseline = np.mean(ipr[:baseline_end])
    ipr_threshold = ipr_baseline * 2.5
    ipr_onset_idx = None
    for i in range(baseline_end, len(ipr)):
        if ipr[i] > ipr_threshold:
            ipr_onset_idx = i
            break

    # Drift peak (new — simple, no threshold)
    warmup = max(10, len(drift) // 20)
    peak_idx = warmup + np.argmax(drift[warmup:])

    # Drift drop (old — 50% threshold)
    drift_peak_val = drift[peak_idx]
    drift_threshold = drift_peak_val * 0.5
    drift_drop_idx = None
    for i in range(peak_idx, len(drift)):
        if drift[i] < drift_threshold:
            drift_drop_idx = i
            break

    if ipr_onset_idx is not None and drift_drop_idx is not None:
        t_ipr_onsets.append(float(steps[ipr_onset_idx]))
        t_drift_peaks.append(float(steps[peak_idx]))
        t_drift_drops.append(float(steps[drift_drop_idx]))
        run_alphas.append(run["alpha"])

t_ipr_onsets = np.array(t_ipr_onsets)
t_drift_peaks = np.array(t_drift_peaks)
t_drift_drops = np.array(t_drift_drops)

alphas_unique = sorted(set(run_alphas))
alpha_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas_unique)))
alpha_cmap = {a: c for a, c in zip(alphas_unique, alpha_colors)}
colors = [alpha_cmap[a] for a in run_alphas]

# ── Figure: side-by-side comparison ──
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Drift peak vs Drift drop (50% threshold) as temporal landmarks",
             fontsize=14, fontweight="bold")

# Helper for scatter panels
def scatter_panel(ax, x, y, ylabel, title):
    ax.scatter(x, y, c=colors, s=40, alpha=0.7, edgecolors="k", linewidths=0.3, zorder=3)
    lims = [0, max(x.max(), y.max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.4, lw=1, label="Simultaneous")
    ax.fill_between(lims, lims, [lims[1]]*2, alpha=0.05, color="blue")
    ax.fill_between(lims, [0]*2, lims, alpha=0.05, color="red")
    ax.set_xlabel(r"$T_{IPR\ onset}$ (steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    ax.set_xlim(lims); ax.set_ylim(lims)
    lag = y - x
    n_before = np.sum(lag < 0)
    n_after = np.sum(lag > 0)
    ax.text(0.97, 0.03,
            f"Before IPR: {n_before}\nAfter IPR: {n_after}",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
    ax.legend(fontsize=7, loc="upper left")
    return lag

# Helper for histogram panels
def hist_panel(ax, lag, xlabel, title, color):
    ax.hist(lag, bins=25, color=color, edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", lw=1.5, label="Simultaneous")
    median_val = np.median(lag)
    ax.axvline(median_val, color="red", ls="--", lw=2,
               label=f"Median = {median_val:.0f} steps")
    if len(lag) > 5:
        w_stat, w_pval = stats.wilcoxon(lag)
        n_consistent = np.sum(lag > 0)
        ax.text(0.97, 0.97,
                f"Wilcoxon p={w_pval:.2e}\n{n_consistent}/{len(lag)} after IPR",
                transform=ax.transAxes, va="top", ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    ax.legend(fontsize=8)

# Top row: NEW — drift peak
lag_peak = scatter_panel(axes[0, 0], t_ipr_onsets, t_drift_peaks,
                         r"$T_{drift\ peak}$ (steps)",
                         r"(a) NEW: $T_{drift\ peak}$ vs $T_{IPR\ onset}$")
hist_panel(axes[0, 1], lag_peak,
           r"$T_{drift\ peak} - T_{IPR\ onset}$ (steps)",
           r"(b) NEW: Lag distribution (peak)",
           "#43A047")

# Bottom row: OLD — drift drop (50% threshold)
lag_drop = scatter_panel(axes[1, 0], t_ipr_onsets, t_drift_drops,
                         r"$T_{drift\ drop}$ (steps)",
                         r"(c) OLD: $T_{drift\ drop}$ vs $T_{IPR\ onset}$")
hist_panel(axes[1, 1], lag_drop,
           r"$T_{drift\ drop} - T_{IPR\ onset}$ (steps)",
           r"(d) OLD: Lag distribution (50% drop)",
           "#7E57C2")

# Alpha legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker="o", color="w", markerfacecolor=alpha_cmap[a],
                          markersize=8, label=f"α={a}") for a in alphas_unique]
fig.legend(handles=legend_elements, loc="lower center", ncol=len(alphas_unique),
           fontsize=9, frameon=True, title="Dirichlet α")

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("paper/figures/drift_peak_vs_drop_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Print summary stats ──
print(f"\n{'Metric':<20} {'Median lag':>12} {'All after IPR?':>15} {'Wilcoxon p':>12}")
print("-" * 62)
for name, lag in [("Drift peak", lag_peak), ("Drift drop (50%)", lag_drop)]:
    _, p = stats.wilcoxon(lag)
    n_after = np.sum(lag > 0)
    print(f"{name:<20} {np.median(lag):>10.0f}   {n_after}/{len(lag):>12}   {p:>10.2e}")
