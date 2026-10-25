"""Checks the calibrated simulator against the real chip data it was fit on
(or any other real (X, Y) you pass in) -- an in-sample fit-quality check, not
held-out validation, since evaluate.py's whole comparison rests on the
simulator being a reasonable stand-in for the real process."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from simulator import _duty_cycle_deterministic, load_constants, load_real_data

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_HERE, "simulator_validation.png")


def validate(real_X, real_Y, simulator_constants, out_path=OUT_PATH):
    """real_X: (n, 4) tensor/array of [amplitude, rise, dwell, fall].
    real_Y: (n, 1) (or (n,)) tensor/array of MEASURED duty cycle, fraction
    (0-1) units. Prints the Pearson correlation and saves a predicted-vs-
    measured scatter plot with a y=x reference line. Returns (r, p_value)."""
    X = np.asarray(real_X, dtype=float)
    Y = np.asarray(real_Y, dtype=float).reshape(-1)
    preds = _duty_cycle_deterministic(X[:, 0], X[:, 1], X[:, 2], X[:, 3], **simulator_constants)

    r, p = pearsonr(preds, Y)
    print(f"Pearson correlation (predicted vs measured): r={r:.4f} (p={p:.4g}), n={len(Y)}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(Y, preds, alpha=0.7, edgecolor="k")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="y = x")
    ax.set_xlabel("Measured duty cycle")
    ax.set_ylabel("Simulator-predicted duty cycle")
    ax.set_title(f"Simulator validation (r={r:.3f}, n={len(Y)})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")

    return r, p


if __name__ == "__main__":
    X, Y = load_real_data()
    constants = load_constants()
    validate(X, Y, constants)
