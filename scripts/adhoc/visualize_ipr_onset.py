"""Compare IPR onset detection methods: threshold-based vs max-gradient."""

import json, os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

# ── Compute both IPR onset methods + drift peak for all runs ──
t_ipr_threshold = []  # old: 2.5x baseline
t_ipr_maxgrad = []    # new: max gradient
t_drift_peaks = []
run_alphas = []

for run in runs:
    h = run["h"]
    steps = run["steps"]
    ipr = np.array(h["ipr"])
    drift = smooth(np.array(h["mean_client_drift"]), window=100)
    ipr_smooth = smooth(ipr, window=100)

    # Method 1: Threshold-based (old)
    baseline_end = max(10, len(ipr) // 10)
    ipr_baseline = np.mean(ipr[:baseline_end])
    ipr_thresh = ipr_baseline * 2.5
    ipr_onset_idx = None
    for i in range(baseline_end, len(ipr)):
        if ipr[i] > ipr_thresh:
            ipr_onset_idx = i
            break

    # Method 2: Max gradient of smoothed IPR
    ipr_grad = np.gradient(ipr_smooth)
    # Skip early warmup noise
    warmup = max(10, len(ipr_grad) // 20)
    maxgrad_idx = warmup + np.argmax(ipr_grad[warmup:])

    # Drift peak
    drift_warmup = max(10, len(drift) // 20)
    peak_idx = drift_warmup + np.argmax(drift[drift_warmup:])

    if ipr_onset_idx is not None:
        t_ipr_threshold.append(float(steps[ipr_onset_idx]))
        t_ipr_maxgrad.append(float(steps[maxgrad_idx]))
        t_drift_peaks.append(float(steps[peak_idx]))
        run_alphas.append(run["alpha"])

t_ipr_threshold = np.array(t_ipr_threshold)
t_ipr_maxgrad = np.array(t_ipr_maxgrad)
t_drift_peaks = np.array(t_drift_peaks)

alphas_unique = sorted(set(run_alphas))
alpha_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas_unique)))
alpha_cmap = dict(zip(alphas_unique, alpha_colors))
colors = [alpha_cmap[a] for a in run_alphas]

# ── Single-run explainer ──
# Pick a healthy grok run for visual explanation
demo_h = load_json("results/exp3_heterogeneity/exp3a/history_fed_addition_gd_p97_N256_a0.5_K10_le5_ft1.0_dirichlet_dir1000.0_s42.json")
demo_steps = np.array(demo_h["total_steps"])
demo_ipr = np.array(demo_h["ipr"])
demo_ipr_smooth = smooth(demo_ipr, window=100)
demo_drift = smooth(np.array(demo_h["mean_client_drift"]), window=100)

demo_be = max(10, len(demo_ipr) // 10)
demo_baseline = np.mean(demo_ipr[:demo_be])
demo_thresh = demo_baseline * 2.5
demo_thresh_idx = None
for i in range(demo_be, len(demo_ipr)):
    if demo_ipr[i] > demo_thresh:
        demo_thresh_idx = i
        break

demo_grad = np.gradient(demo_ipr_smooth)
demo_wu = max(10, len(demo_grad) // 20)
demo_maxgrad_idx = demo_wu + np.argmax(demo_grad[demo_wu:])

demo_dwu = max(10, len(demo_drift) // 20)
demo_peak_idx = demo_dwu + np.argmax(demo_drift[demo_dwu:])

# ── Figure ──
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
fig.suptitle("Threshold-free landmarks: max IPR gradient + drift peak",
             fontsize=14, fontweight="bold")

# ── Top left: Single-run IPR with both methods ──
ax = fig.add_subplot(gs[0, 0])
ax.plot(demo_steps, demo_ipr, color="#ccc", lw=0.5, alpha=0.6, label="Raw IPR")
ax.plot(demo_steps, demo_ipr_smooth, color="#7B1FA2", lw=1.8, label="Smoothed IPR")
ax.axhline(demo_thresh, ls="--", color="#FF6F00", lw=1.2,
           label=f"2.5× baseline = {demo_thresh:.4f}")
if demo_thresh_idx is not None:
    ax.axvline(demo_steps[demo_thresh_idx], ls="--", color="#FF6F00", lw=1.5, alpha=0.7,
               label=f"Threshold onset = step {int(demo_steps[demo_thresh_idx])}")
ax.axvline(demo_steps[demo_maxgrad_idx], ls="-", color="#2E7D32", lw=2, alpha=0.8,
           label=f"Max gradient = step {int(demo_steps[demo_maxgrad_idx])}")
ax.set_ylabel("IPR"); ax.legend(fontsize=7.5, loc="center right")
ax.set_title("(a) IPR: two onset methods on one run", fontsize=10, loc="left", fontweight="bold")

# ── Top right: Single-run IPR gradient ──
ax = fig.add_subplot(gs[0, 1])
ax.plot(demo_steps, demo_grad, color="#7B1FA2", lw=1.2)
ax.axvline(demo_steps[demo_maxgrad_idx], ls="-", color="#2E7D32", lw=2, alpha=0.8,
           label=f"Max gradient = step {int(demo_steps[demo_maxgrad_idx])}")
ax.axvline(demo_steps[demo_peak_idx], ls="-", color="#1565C0", lw=2, alpha=0.8,
           label=f"Drift peak = step {int(demo_steps[demo_peak_idx])}")
ax.set_ylabel("d(IPR)/dt"); ax.legend(fontsize=8)
ax.set_xlabel("Steps")
ax.set_title("(b) IPR gradient — peak = steepest rise", fontsize=10, loc="left", fontweight="bold")

# ── Middle row: scatter comparisons against drift peak ──
def scatter_panel(ax, x, y, xlabel, ylabel, title):
    ax.scatter(x, y, c=colors, s=40, alpha=0.7, edgecolors="k", linewidths=0.3, zorder=3)
    lims = [0, max(x.max(), y.max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.4, lw=1, label="Simultaneous")
    ax.fill_between(lims, lims, [lims[1]]*2, alpha=0.05, color="blue")
    ax.fill_between(lims, [0]*2, lims, alpha=0.05, color="red")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    # Correlation
    r, p = stats.pearsonr(x, y)
    ax.text(0.97, 0.03, f"r = {r:.3f}\np = {p:.1e}",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

ax = fig.add_subplot(gs[1, 0])
scatter_panel(ax, t_ipr_threshold, t_drift_peaks,
              r"$T_{IPR\ onset}$ (2.5× threshold)", r"$T_{drift\ peak}$",
              r"(c) OLD IPR onset vs drift peak")

ax = fig.add_subplot(gs[1, 1])
scatter_panel(ax, t_ipr_maxgrad, t_drift_peaks,
              r"$T_{IPR\ max\ grad}$", r"$T_{drift\ peak}$",
              r"(d) NEW IPR max-grad vs drift peak")

# ── Bottom row: lag histograms ──
lag_old = t_drift_peaks - t_ipr_threshold
lag_new = t_drift_peaks - t_ipr_maxgrad

def hist_panel(ax, lag, xlabel, title, color):
    ax.hist(lag, bins=30, color=color, edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", lw=1.5, label="Simultaneous")
    med = np.median(lag)
    ax.axvline(med, color="red", ls="--", lw=2, label=f"Median = {med:.0f}")
    if len(lag) > 5:
        _, p = stats.wilcoxon(lag)
        n_pos = np.sum(lag > 0)
        ax.text(0.97, 0.97,
                f"Wilcoxon p={p:.2e}\n{n_pos}/{len(lag)} drift peak after",
                transform=ax.transAxes, va="top", ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    ax.set_xlabel(xlabel); ax.set_ylabel("Count")
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 0])
hist_panel(ax, lag_old,
           r"$T_{drift\ peak} - T_{IPR\ onset}$ (steps)",
           "(e) Lag: drift peak − OLD IPR onset", "#FF8A65")

ax = fig.add_subplot(gs[2, 1])
hist_panel(ax, lag_new,
           r"$T_{drift\ peak} - T_{IPR\ max\ grad}$ (steps)",
           "(f) Lag: drift peak − NEW IPR max-grad", "#43A047")

# Alpha legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker="o", color="w", markerfacecolor=alpha_cmap[a],
                          markersize=8, label=f"α={a}") for a in alphas_unique]
fig.legend(handles=legend_elements, loc="lower center", ncol=len(alphas_unique),
           fontsize=9, frameon=True, title="Dirichlet α")

plt.savefig("paper/figures/ipr_onset_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Summary ──
print(f"\n{'Method pair':<45} {'Median lag':>10} {'Pearson r':>10} {'Wilcoxon p':>12}")
print("-" * 80)
for name, lag, x, y in [
    ("Drift peak − IPR threshold (old)", lag_old, t_ipr_threshold, t_drift_peaks),
    ("Drift peak − IPR max-grad (new)", lag_new, t_ipr_maxgrad, t_drift_peaks),
]:
    r, _ = stats.pearsonr(x, y)
    _, p = stats.wilcoxon(lag)
    print(f"{name:<45} {np.median(lag):>8.0f}   {r:>9.3f}   {p:>10.2e}")
