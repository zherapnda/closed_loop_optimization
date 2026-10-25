"""Evaluates the GP+EI optimizer against the OFAT baseline on the calibrated
physics simulator: many more trials than real fabrication allows, so this is
where sample-efficiency claims about the BO loop get checked before trusting
it on real (slow, expensive) chip runs.

Usage:
    python evaluate.py                       # full spec: 100 seeds x 50 steps
    python evaluate.py --seeds 5 --steps 10  # quick smoke test
"""
import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mannwhitneyu, qmc

import config
from bo_session import build_bounds
from model import fit_gp
from optimizer import suggest_next
from ofat import ofat_search
from simulator import simulate, load_constants, CONSTANTS_PATH, load_real_data, calibrate_simulator, \
    save_constants, NOISE_STD

_HERE = os.path.dirname(os.path.abspath(__file__))

N_WARMUP = 5  # fresh Sobol warm-start points per seed, per the spec


def ensure_constants():
    """calibrate_simulator's whole point is to anchor the physics equation to
    real chip data -- if nobody's run simulator.py yet, do it here instead of
    just failing, using the same real (website) data it would have used."""
    if os.path.exists(CONSTANTS_PATH):
        return load_constants()
    print(f"No {CONSTANTS_PATH} found -- calibrating from real data first...")
    X, Y = load_real_data()
    constants = calibrate_simulator(X, Y)
    save_constants(constants)
    print("Calibrated:", constants)
    return constants


def make_simulate_fn(constants, rng):
    def simulate_fn(x):
        return simulate(x[0], x[1], x[2], x[3], **constants, rng=rng)
    return simulate_fn


def sobol_points(bounds_np, n_points, seed):
    sampler = qmc.Sobol(d=bounds_np.shape[1], scramble=True, seed=seed)
    unit = sampler.random(n_points)
    return qmc.scale(unit, bounds_np[0], bounds_np[1])


def run_gp(bounds, n_steps, x_init, dc_init, rng):
    """x_init: (N_WARMUP, 4) numpy warm-start points. dc_init: their already-
    evaluated raw duty cycles (so the warm start is IDENTICAL to what OFAT
    sees -- see run_one_seed). The GP is fit directly on these raw duty-cycle
    values (smooth in x); the |dc - target| loss used to rank candidates
    lives entirely inside suggest_next's acquisition function (see
    optimizer.py), not in the GP's own training target -- mirrors the same
    fix applied to the live bo_session.py. The simulator's noise is known
    exactly (NOISE_STD, flat across all x -- the simulator itself doesn't
    model heteroskedastic noise), so it's passed to the GP as known
    observation variance rather than estimated, mirroring the real
    bo_session.py path where per-point variance instead comes from each
    experiment's own measured std. Returns a length-n_steps list of simple
    regret (best raw duty-cycle-vs-target seen so far, evaluated AFTER the
    warm start, one value per search step)."""
    X = torch.tensor(x_init, dtype=torch.double)
    Y = torch.tensor([[dc] for dc in dc_init], dtype=torch.double)
    Yvar = torch.full_like(Y, NOISE_STD ** 2)
    simulate_fn = rng["simulate_fn"]

    best_regret = min(abs(dc - config.target_dc) for dc in dc_init)
    regrets = []
    rand = rng.get("py_rng") or np.random.default_rng()
    n_fallback = 0
    for step in range(n_steps):
        # a handful of accumulated points can occasionally leave the kernel matrix numerically
        # singular (confirmed on real data: fit_gpytorch_mll escalating jitter 1e-8 -> 1e-3 before
        # raising ModelFittingError) -- rare, but with 100+ independent seeds it WILL happen
        # eventually, and one step's numerical hiccup shouldn't discard an entire seed's progress.
        # Falling back to a random point for just that step keeps the run going; suggest_next's own
        # acqf optimization failing is caught the same way for the same reason.
        try:
            model = fit_gp(X, Y, config.n_restart, bounds, Yvar=Yvar)
            candidate = suggest_next(model, bounds, Y)
            x_new = candidate.squeeze(0).numpy()
        except Exception as e:
            n_fallback += 1
            print(f"    [step {step}] GP fit/acquisition failed ({type(e).__name__}), "
                  f"falling back to a random point for this step", flush=True)
            x_new = bounds[0].numpy() + rand.random(bounds.shape[1]) * (bounds[1] - bounds[0]).numpy()
            candidate = torch.tensor([x_new], dtype=torch.double)

        dc = simulate_fn(x_new)
        X = torch.cat([X, candidate.to(torch.double)], dim=0)
        Y = torch.cat([Y, torch.tensor([[dc]], dtype=torch.double)], dim=0)
        Yvar = torch.cat([Yvar, torch.tensor([[NOISE_STD ** 2]], dtype=torch.double)], dim=0)
        best_regret = min(best_regret, abs(dc - config.target_dc))
        regrets.append(best_regret)
    if n_fallback:
        print(f"    ({n_fallback}/{n_steps} steps used a random fallback due to GP fitting failures)",
              flush=True)
    return regrets


def run_one_seed(seed, bounds_t, bounds_np, constants, n_steps, n_candidates=10):
    rng = np.random.default_rng(seed)
    simulate_fn = make_simulate_fn(constants, rng)

    x_init = sobol_points(bounds_np, N_WARMUP, seed=seed)
    dc_init = [simulate_fn(x) for x in x_init]

    gp_regrets = run_gp(bounds_t, n_steps, x_init, dc_init,
                         {"constants": constants, "simulate_fn": simulate_fn, "py_rng": rng})

    # same starting point as the GP run: the first of its warm-start points,
    # so both methods begin the search from literally the same location.
    # track_per_eval=True gets the finer-grained curve (see evaluate_fair.py
    # / the evaluation-count plot below) -- the original per-STEP values
    # (one per 10-candidate sweep, matching the literal spec) are recovered
    # from it by slicing rather than calling ofat_search twice.
    ofat_per_eval = ofat_search(simulate_fn, bounds_np, n_steps, target=config.target_dc,
                                 start=x_init[0].copy(), rng=rng, n_candidates=n_candidates,
                                 track_per_eval=True)
    ofat_per_step = [ofat_per_eval[(k + 1) * n_candidates] for k in range(n_steps)]
    return gp_regrets, ofat_per_step, ofat_per_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    constants = ensure_constants()
    bounds_t = build_bounds()
    bounds_np = bounds_t.numpy()

    n_candidates = 10
    gp_all = np.zeros((args.seeds, args.steps))
    ofat_all = np.zeros((args.seeds, args.steps))
    ofat_eval_all = np.zeros((args.seeds, 1 + args.steps * n_candidates))

    t0 = time.time()
    for seed in range(args.seeds):
        gp_regrets, ofat_regrets, ofat_per_eval = run_one_seed(
            seed, bounds_t, bounds_np, constants, args.steps, n_candidates)
        gp_all[seed] = gp_regrets
        ofat_all[seed] = ofat_regrets
        ofat_eval_all[seed] = ofat_per_eval
        elapsed = time.time() - t0
        per_seed = elapsed / (seed + 1)
        remaining = per_seed * (args.seeds - seed - 1)
        print(f"seed {seed + 1}/{args.seeds} done  "
              f"(GP final regret={gp_regrets[-1]:.4f}, OFAT final regret={ofat_regrets[-1]:.4f})  "
              f"elapsed={elapsed:.0f}s  est. remaining={remaining:.0f}s", flush=True)

    np.save(os.path.join(_HERE, "gp_regrets.npy"), gp_all)
    np.save(os.path.join(_HERE, "ofat_regrets.npy"), ofat_all)
    np.save(os.path.join(_HERE, "ofat_regrets_per_eval.npy"), ofat_eval_all)

    gp_mean, gp_std = gp_all.mean(axis=0), gp_all.std(axis=0)
    ofat_mean, ofat_std = ofat_all.mean(axis=0), ofat_all.std(axis=0)

    gp_auc = gp_all.sum(axis=1)
    ofat_auc = ofat_all.sum(axis=1)

    stat, p_value = mannwhitneyu(gp_auc, ofat_auc, alternative="two-sided")
    print()
    print(f"GP   AUC: mean={gp_auc.mean():.4f}  std={gp_auc.std():.4f}")
    print(f"OFAT AUC: mean={ofat_auc.mean():.4f}  std={ofat_auc.std():.4f}")
    print(f"Mann-Whitney U p-value (two-sided): {p_value:.4g}")

    # --- regret curve: mean +/- 1 std band per method
    steps = np.arange(1, args.steps + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, gp_mean, label="GP + EI", color="#2563EB")
    ax.fill_between(steps, gp_mean - gp_std, gp_mean + gp_std, color="#2563EB", alpha=0.2)
    ax.plot(steps, ofat_mean, label="OFAT", color="#DC2626")
    ax.fill_between(steps, ofat_mean - ofat_std, ofat_mean + ofat_std, color="#DC2626", alpha=0.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Simple regret |duty cycle - 0.5|")
    ax.set_title(f"GP vs OFAT regret ({args.seeds} seeds, mean ± 1 std)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_HERE, "regret_curve.png"), dpi=150)
    plt.close(fig)

    # --- AUC box plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot([gp_auc, ofat_auc], tick_labels=["GP + EI", "OFAT"])
    ax.set_ylabel("AUC (summed regret over all steps)")
    ax.set_title(f"AUC distribution across {args.seeds} seeds\n(Mann-Whitney p={p_value:.3g})")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(_HERE, "auc_boxplot.png"), dpi=150)
    plt.close(fig)

    # --- fairness-corrected regret curve: same x-axis (real function evaluations, not "steps"),
    # since OFAT gets n_candidates=10 real evaluations per step where GP gets exactly 1 -- plotting
    # both against "step" silently gave OFAT a ~10x evaluation-budget advantage. GP's x starts at
    # N_WARMUP+1 (its own warm-start points are evaluations 1..N_WARMUP, already reflected in its
    # regret's starting value); OFAT's x starts at 1 (its "start" point is the same physical
    # location as GP's first warm-start point, but only counts as ONE evaluation here, not N_WARMUP).
    gp_eval_x = np.arange(N_WARMUP + 1, N_WARMUP + args.steps + 1)
    ofat_eval_x = np.arange(1, ofat_eval_all.shape[1] + 1)
    ofat_eval_mean, ofat_eval_std = ofat_eval_all.mean(axis=0), ofat_eval_all.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gp_eval_x, gp_mean, label="GP + EI", color="#2563EB")
    ax.fill_between(gp_eval_x, gp_mean - gp_std, gp_mean + gp_std, color="#2563EB", alpha=0.2)
    ax.plot(ofat_eval_x, ofat_eval_mean, label="OFAT", color="#DC2626")
    ax.fill_between(ofat_eval_x, ofat_eval_mean - ofat_eval_std, ofat_eval_mean + ofat_eval_std,
                     color="#DC2626", alpha=0.2)
    ax.axvline(gp_eval_x[-1], color="gray", linestyle="--", linewidth=1,
               label=f"GP's total budget ({gp_eval_x[-1]} evals)")
    ax.set_xlabel("Real function evaluations used")
    ax.set_ylabel("Simple regret |duty cycle - 0.5|")
    ax.set_title(f"GP vs OFAT regret vs. evaluation budget ({args.seeds} seeds, mean ± 1 std)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_HERE, "regret_curve_by_evaluations.png"), dpi=150)
    plt.close(fig)

    # the actually-fair comparison: OFAT's regret AFTER THE SAME NUMBER OF EVALUATIONS GP USED
    # TOTAL, not after the same number of "steps"
    matched_idx = gp_eval_x[-1] - 1  # ofat_eval_x is 1-indexed; this is its array index
    ofat_matched = ofat_eval_all[:, matched_idx]
    stat_m, p_value_m = mannwhitneyu(gp_auc, ofat_matched, alternative="two-sided")
    print()
    print(f"Matched-budget check (both methods given {gp_eval_x[-1]} total evaluations):")
    print(f"  GP   final regret:  mean={gp_all[:, -1].mean():.4f}  std={gp_all[:, -1].std():.4f}")
    print(f"  OFAT regret at same evaluation count: mean={ofat_matched.mean():.4f}  "
          f"std={ofat_matched.std():.4f}")
    print(f"  Mann-Whitney U p-value (GP final regret vs. OFAT at matched evaluations): {p_value_m:.4g}")

    print()
    print("Saved: gp_regrets.npy, ofat_regrets.npy, ofat_regrets_per_eval.npy, "
          "regret_curve.png, regret_curve_by_evaluations.png, auc_boxplot.png")


if __name__ == "__main__":
    main()
