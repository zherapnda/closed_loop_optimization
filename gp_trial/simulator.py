"""Merz/Avrami-style physics simulator for the poling process, standing in
for a real chip-and-microscope measurement so the BO loop can be evaluated
against many more trials than physical fabrication allows.

    t_eff      = rise/2 + dwell + fall/2
    tau        = tau_0 * exp(Ea / amplitude)
    duty_cycle = 1 - exp(-(t_eff / tau) ^ n)

tau_0, Ea, n are fit from real chip measurements (calibrate_simulator), not
assumed -- this is a phenomenological domain-switching-kinetics form (Avrami
growth law with a Merz-style field-dependent switching time tau), not a
first-principles model, so its constants only mean something once anchored
to real data.
"""
import csv
import json
import os

import numpy as np
import torch
from scipy.optimize import least_squares

_HERE = os.path.dirname(os.path.abspath(__file__))
CONSTANTS_PATH = os.path.join(_HERE, "simulator_constants.json")
DEFAULT_DATA_PATH = os.path.join(_HERE, "warm_start_data.csv")

NOISE_STD = 0.01  # fraction units (duty cycle is 0-1) -- stands in for real
# chip-to-chip measurement noise; also the "std" term the evaluation's GP
# objective uses (see evaluate.py), since the simulator's noise is known
# exactly rather than estimated from repeated measurements.


def _duty_cycle_deterministic(amplitude, rise, dwell, fall, tau_0, Ea, n):
    """Vectorized (numpy-broadcastable) core equation, no noise. amplitude is
    floored at a small epsilon -- the search bounds (config.A_bound) allow 0
    at the edge, which would otherwise divide by zero; physically this just
    means "field too weak to switch anything," which the equation already
    predicts smoothly (duty_cycle -> 0) once amplitude is small but nonzero."""
    amplitude = np.maximum(np.asarray(amplitude, dtype=float), 1e-6)
    t_eff = rise / 2.0 + dwell + fall / 2.0
    # near-zero amplitude drives Ea/amplitude, and then exp() of it, to overflow -- the overflow
    # itself is harmless (tau -> inf -> t_eff/tau -> 0 -> duty_cycle -> 0, the physically correct
    # answer for "field too weak to switch anything"), so the warning is suppressed rather than
    # the result being wrong.
    with np.errstate(over="ignore"):
        tau = tau_0 * np.exp(Ea / amplitude)
        return 1.0 - np.exp(-((t_eff / tau) ** n))


def simulate(amplitude, rise, dwell, fall, tau_0, Ea, n, noise_std=NOISE_STD, rng=None):
    """One simulated experiment: the deterministic physics prediction plus
    injected Gaussian measurement noise (std=noise_std), clipped to [0, 1]
    since duty cycle can't physically leave that range. rng lets callers pass
    a seeded np.random.Generator for reproducibility; defaults to the global
    numpy RNG."""
    dc = _duty_cycle_deterministic(amplitude, rise, dwell, fall, tau_0, Ea, n)
    rng = rng if rng is not None else np.random
    dc_noisy = float(dc) + rng.normal(0.0, noise_std)
    return float(np.clip(dc_noisy, 0.0, 1.0))


def calibrate_simulator(X, Y):
    """Fit tau_0, Ea, n by nonlinear least squares against real measurements.

    X: (n, 4) tensor/array of [amplitude, rise, dwell, fall].
    Y: (n, 1) (or (n,)) tensor/array of MEASURED duty cycle as a FRACTION
       (0-1) -- not percent, and not the BO objective.

    Returns {"tau_0": ..., "Ea": ..., "n": ...}."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    amplitude, rise, dwell, fall = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    def residuals(params):
        tau_0, Ea, n = params
        pred = _duty_cycle_deterministic(amplitude, rise, dwell, fall, tau_0, Ea, n)
        return pred - Y

    # x0/bounds keep tau_0, Ea, n positive (required for the equation to be
    # physically meaningful -- tau and the exponent must stay real and
    # positive) without pinning them anywhere near a specific expected
    # value; least_squares' trust-region-reflective method (the default once
    # bounds are given) handles the resulting nonlinear fit.
    x0 = [1.0, 500.0, 1.5]
    result = least_squares(residuals, x0=x0, bounds=([1e-6, 1e-6, 0.1], [1e6, 1e6, 10.0]))
    tau_0, Ea, n = (float(v) for v in result.x)
    return {"tau_0": tau_0, "Ea": Ea, "n": n}


def save_constants(constants, path=CONSTANTS_PATH):
    with open(path, "w") as f:
        json.dump(constants, f, indent=2)


def load_constants(path=CONSTANTS_PATH):
    with open(path) as f:
        return json.load(f)


def load_real_data(csv_path=DEFAULT_DATA_PATH):
    """Real (amplitude, rise, dwell, fall) -> measured duty cycle pairs, read
    from the SAME warm_start_data.csv the website (streamlit_app.py) reads
    and the duty-cycle batch GUI writes to -- calibration/validation always
    reflects the latest real chip measurements on disk, not a separate or
    stale copy. Y is converted from percent (the CSV's own units) to the
    fraction (0-1) units the physics equation and simulate() use."""
    X_list, Y_list = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X_list.append([float(row["amplitude"]), float(row["rise"]),
                           float(row["dwell"]), float(row["fall"])])
            Y_list.append([float(row["mean"]) / 100.0])
    return torch.tensor(X_list, dtype=torch.double), torch.tensor(Y_list, dtype=torch.double)


if __name__ == "__main__":
    X, Y = load_real_data()
    print(f"Calibrating simulator on {len(X)} real measurements from {DEFAULT_DATA_PATH}...")
    constants = calibrate_simulator(X, Y)
    print("Fitted constants:", constants)
    save_constants(constants)
    print(f"Saved to {CONSTANTS_PATH}")

    X_np, Y_np = X.numpy(), Y.numpy().reshape(-1)
    preds = _duty_cycle_deterministic(X_np[:, 0], X_np[:, 1], X_np[:, 2], X_np[:, 3], **constants)
    rms = float(np.sqrt(np.mean((preds - Y_np) ** 2)))
    print(f"In-sample RMS residual: {rms:.4f} (fraction units)")
