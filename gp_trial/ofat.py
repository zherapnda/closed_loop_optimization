"""One-factor-at-a-time (OFAT) search -- the classical, non-Bayesian baseline
evaluate.py compares the GP optimizer against."""
import numpy as np


def ofat_search(simulate_fn, bounds, n_steps, target=0.5, start=None, n_candidates=10, rng=None,
                 track_per_eval=False):
    """Cycle through the four parameters one at a time. For whichever
    parameter is "up" this step, try n_candidates evenly spaced values across
    its own range (holding the other three fixed at the current position),
    keep whichever gets closest to `target`, and move the current position
    there. Repeat until n_steps parameter-sweeps are used up (so with 4
    parameters, n_steps=50 means ~12.5 full cycles).

    simulate_fn: callable taking a length-4 array-like [amplitude, rise,
        dwell, fall] and returning a scalar duty cycle.
    bounds: array-like of shape (2, 4): [lower_row, upper_row].
    start: optional length-4 starting point (e.g. to match a GP run's own
        warm-start point for a fair side-by-side comparison); a random point
        within bounds is drawn if omitted.
    rng: optional np.random.Generator for the random start; defaults to the
        global numpy RNG.
    track_per_eval: if False (default, matches the original spec), returns
        one regret value per STEP (i.e. per 10-candidate sweep). If True,
        returns one value per INDIVIDUAL candidate evaluation instead (plus
        the initial evaluation at `start`) -- n_steps*n_candidates + 1
        values total. OFAT gets n_candidates real function evaluations per
        step where GP gets exactly one, so comparing the two step-for-step
        silently gives OFAT a ~10x evaluation-budget advantage; per-eval
        tracking is what makes an apples-to-apples "regret vs. evaluation
        count" comparison possible (see evaluate.py).

    Returns a list of simple-regret values, |best duty cycle found so far -
    target| -- "found so far" means the best of EVERY candidate evaluated up
    to that point, not just the ones adopted into the current position,
    matching the usual "best observed value" convention for a regret
    curve."""
    bounds = np.asarray(bounds, dtype=float)
    d = bounds.shape[1]
    rng = rng if rng is not None else np.random

    if start is None:
        current = bounds[0] + rng.random(d) * (bounds[1] - bounds[0])
    else:
        current = np.array(start, dtype=float)

    best_dc = simulate_fn(current)
    per_step_regrets = []
    per_eval_regrets = [abs(best_dc - target)]

    for step in range(n_steps):
        i = step % d
        lo, hi = bounds[0, i], bounds[1, i]
        candidates = np.linspace(lo, hi, n_candidates)

        best_local_val, best_local_dc = current[i], None
        for v in candidates:
            trial = current.copy()
            trial[i] = v
            dc = simulate_fn(trial)

            if best_local_dc is None or abs(dc - target) < abs(best_local_dc - target):
                best_local_val, best_local_dc = v, dc
            if abs(dc - target) < abs(best_dc - target):
                best_dc = dc
            per_eval_regrets.append(abs(best_dc - target))

        current[i] = best_local_val
        per_step_regrets.append(abs(best_dc - target))

    return per_eval_regrets if track_per_eval else per_step_regrets


def ofat_next_candidate(history, bounds, start, n_candidates=10, target=0.5):
    """Resumable counterpart to ofat_search for an interactive/real session:
    given `history` -- a list of (params, dc) pairs already submitted for
    this arm, in submission order -- replays the same step/sweep/keep-best
    logic to derive the current position and returns the next candidate to
    try, instead of needing a live simulate_fn loop. This makes OFAT's state
    fully recoverable from a log file after a restart, the same way
    BOSession reconstructs its own state from disk.

    history[0] is special: it's the baseline evaluation of `start` itself
    (mirroring ofat_search's `best_dc = simulate_fn(current)` before the
    sweep loop begins), not part of any parameter sweep -- so with history
    empty, the next candidate IS `start`.

    Returns (candidate, info): candidate is a length-4 list; info is a dict
    with step/param_index/candidate_index (all None for the baseline case)
    for display purposes."""
    bounds = np.asarray(bounds, dtype=float)
    d = bounds.shape[1]

    if len(history) == 0:
        return list(start), {"step": None, "param_index": None, "candidate_index": None,
                              "is_baseline": True, "is_new_cycle": False}

    current = np.array(start, dtype=float)
    sweep_history = history[1:]  # history[0] is the baseline evaluation of `start`

    n_complete_steps = len(sweep_history) // n_candidates
    for step in range(n_complete_steps):
        i = step % d
        chunk = sweep_history[step * n_candidates:(step + 1) * n_candidates]
        best_params, best_dc = None, None
        for params, dc in chunk:
            if best_dc is None or abs(dc - target) < abs(best_dc - target):
                best_params, best_dc = params, dc
        current[i] = best_params[i]

    step = n_complete_steps
    i = step % d
    candidate_idx = len(sweep_history) % n_candidates
    lo, hi = bounds[0, i], bounds[1, i]
    candidates = np.linspace(lo, hi, n_candidates)

    trial = current.copy()
    trial[i] = candidates[candidate_idx]

    info = {
        "step": step, "param_index": i, "candidate_index": candidate_idx,
        "is_baseline": False, "is_new_cycle": (i == 0 and candidate_idx == 0),
    }
    return trial.tolist(), info
