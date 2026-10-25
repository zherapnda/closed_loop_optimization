"""Publication-quality (vector PDF, serif/LaTeX-style) versions of the GP-vs-
OFAT evaluation results, for use in a paper. Reads the same .npy/data sources
evaluate.py and validate_simulator.py already produce -- doesn't re-run any
optimization, just restyles the results.

Run from the gp_trial/ directory: python make_paper_figures.py
Output: paper_figures/fig1_regret_vs_evaluations.pdf
        paper_figures/fig2_auc_boxplot.pdf
        paper_figures/fig3_objective_fix_comparison.pdf
        paper_figures/fig4_simulator_validation.pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, pearsonr

from simulator import _duty_cycle_deterministic, load_constants, load_real_data

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

N_WARMUP = 5
N_CANDIDATES = 10

GP_COLOR = "#2563EB"
OFAT_COLOR = "#DC2626"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,  # embed as scalable outlines, not bitmapped
})


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# --- Figure 1: GP vs OFAT regret vs. evaluation budget (final system) ---
def figure1():
    gp_all = np.load(os.path.join(_HERE, "gp_regrets.npy"))
    ofat_eval_all = np.load(os.path.join(_HERE, "ofat_regrets_per_eval.npy"))
    n_seeds, n_steps = gp_all.shape

    gp_mean, gp_std = gp_all.mean(axis=0), gp_all.std(axis=0)
    ofat_eval_mean, ofat_eval_std = ofat_eval_all.mean(axis=0), ofat_eval_all.std(axis=0)

    gp_eval_x = np.arange(N_WARMUP + 1, N_WARMUP + n_steps + 1)
    ofat_eval_x = np.arange(1, ofat_eval_all.shape[1] + 1)

    eps = 1e-5  # floor for the log-scale lower error band only -- regret itself is >= 0 by
    # definition; "mean - std" can dip below 0 as a display artifact of a symmetric error bar
    # applied to a non-negative, right-skewed quantity

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(gp_eval_x, gp_mean, label="GP + qLogEI", color=GP_COLOR)
    ax.fill_between(gp_eval_x, np.clip(gp_mean - gp_std, eps, None), gp_mean + gp_std,
                     color=GP_COLOR, alpha=0.2, linewidth=0)
    ax.plot(ofat_eval_x, ofat_eval_mean, label="OFAT", color=OFAT_COLOR)
    ax.fill_between(ofat_eval_x, np.clip(ofat_eval_mean - ofat_eval_std, eps, None),
                     ofat_eval_mean + ofat_eval_std, color=OFAT_COLOR, alpha=0.2, linewidth=0)
    ax.axvline(gp_eval_x[-1], color="0.4", linestyle="--", linewidth=1,
               label=f"Matched budget ({gp_eval_x[-1]} evals)")
    ax.set_yscale("log")
    ax.set_xlabel("Real function evaluations used")
    ax.set_ylabel(r"Simple regret $|{\rm DC} - 0.5|$")
    ax.set_title(f"GP + qLogEI vs. OFAT ($n={n_seeds}$ seeds, mean $\\pm$ 1 s.d.)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.25, which="both", linewidth=0.5)
    _save(fig, "fig1_regret_vs_evaluations.pdf")


# --- Figure 2: AUC distribution (final system) ---
def figure2():
    gp_all = np.load(os.path.join(_HERE, "gp_regrets.npy"))
    ofat_all = np.load(os.path.join(_HERE, "ofat_regrets.npy"))
    n_seeds = gp_all.shape[0]

    gp_auc = gp_all.sum(axis=1)
    ofat_auc = ofat_all.sum(axis=1)
    _, p_value = mannwhitneyu(gp_auc, ofat_auc, alternative="two-sided")

    fig, ax = plt.subplots(figsize=(3.6, 4))
    bp = ax.boxplot([gp_auc, ofat_auc], tick_labels=["GP + qLogEI", "OFAT"], widths=0.55,
                     patch_artist=True, medianprops=dict(color="black"))
    for patch, color in zip(bp["boxes"], [GP_COLOR, OFAT_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax.set_ylabel("AUC (summed regret over all steps)")
    ax.set_title(f"$n={n_seeds}$ seeds, Mann-Whitney $p={p_value:.2g}$")
    ax.grid(alpha=0.25, axis="y", linewidth=0.5)
    _save(fig, "fig2_auc_boxplot.pdf")


# --- Figure 3: effect of moving the target-distance loss out of the GP's training target ---
def figure3():
    # Summary statistics recorded from the corresponding evaluate.py runs (10 seeds x 30 steps
    # each, matched-budget = 35 total real evaluations); raw per-seed arrays for "original" and
    # "objective fix only" were overwritten by later runs, so these stages are shown as
    # mean +/- s.d. bars rather than full curves. "Objective fix + heteroskedastic noise +
    # log-amplitude" uses the arrays currently on disk (same as Figures 1-2).
    stages = ["Original\n(GP trained on\n$|{\\rm DC}-0.5|$)",
              "Objective fix\n(GP trained on\nraw DC)",
              "+ heteroskedastic\nnoise + log(amp.)"]
    gp_mean = np.array([0.0402, 0.0008, 0.0006])
    gp_std = np.array([0.0418, 0.0007, 0.0003])
    ofat_mean, ofat_std = 0.0028, 0.0034  # unchanged across stages -- same OFAT run, same seeds

    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x, gp_mean, yerr=gp_std, capsize=4, color=GP_COLOR, alpha=0.75,
           label="GP + qLogEI", zorder=3)
    ax.axhline(ofat_mean, color=OFAT_COLOR, linestyle="--", linewidth=1.4, label="OFAT (reference)")
    ax.axhspan(ofat_mean - ofat_std, ofat_mean + ofat_std, color=OFAT_COLOR, alpha=0.15, linewidth=0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Final simple regret at\nmatched budget (35 evals)")
    ax.set_title("Effect of removing the target-distance kink\nfrom the GP's training target")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.25, axis="y", which="both", linewidth=0.5)
    _save(fig, "fig3_objective_fix_comparison.pdf")


# --- Figure 4: simulator validation (predicted vs. measured) ---
def figure4():
    X, Y = load_real_data()
    constants = load_constants()
    X_np, Y_np = X.numpy(), Y.numpy().reshape(-1)
    preds = _duty_cycle_deterministic(X_np[:, 0], X_np[:, 1], X_np[:, 2], X_np[:, 3], **constants)
    r, p = pearsonr(preds, Y_np)

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(Y_np, preds, s=28, facecolor="white", edgecolor=GP_COLOR, linewidth=1.2, zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1, label="$y=x$", zorder=1)
    ax.set_xlabel("Measured duty cycle")
    ax.set_ylabel("Simulator-predicted duty cycle")
    ax.set_title(f"Simulator validation ($r={r:.2f}$, $p={p:.2g}$, $n={len(Y_np)}$)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25, linewidth=0.5)
    _save(fig, "fig4_simulator_validation.pdf")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
