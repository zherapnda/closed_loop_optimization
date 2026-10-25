import os
import sys
import torch

from model import fit_gp
from optimizer import suggest_next
from logger import init_logger, log_experiment
from warm_start import build_warm_start
from lhs_init import generate_lhs_points
import config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'duty_cycle'))
# tip_lines_check.estimate_duty_cycle replaces the older duty_cycle/algorithms/main.main --
# same (mean, std, diagnostics) contract, built on the tip-pairing/boundary-validation
# architecture (compute_domain_boundaries / duty_cycle_statistics) instead of the older
# column-sweep one. diagnostics["problem_fraction"] now means "fraction of validated periods
# whose span skipped over an invalid tip" (a locally underpoled region), not the old
# per-column ROI-width fraction -- same role in compute_output below, different measurement.
import tip_lines_check
estimate_duty_cycle = tip_lines_check.estimate_duty_cycle

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments.csv')
WARM_START_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'warm_start_data.csv')

N_STRUCTURES = 145  # real un-poled test structures available on the chip


def build_bounds():
    return torch.tensor([
        [config.A_bound[0], config.R_bound[0], config.D_bound[0], config.F_bound[0]],
        [config.A_bound[1], config.R_bound[1], config.D_bound[1], config.F_bound[1]],
    ], dtype=torch.double)


def compute_output(mean_pct, std_pct, problem_fraction=0.0):
    mean = mean_pct / 100
    std = std_pct / 100
    # problem_fraction (from duty_cycle diagnostics) is the fraction of the
    # sample that measured as overpoled/unstable rather than clean. A
    # parameter combination that produces a partially-overpoled sample gets
    # penalized here -- otherwise mean/std only reflect the clean region and
    # the output would look just as good as a genuinely uniform experiment.
    return abs(mean - config.target_dc) + config.lambda_uniformity * std + config.lambda_spatial * problem_fraction


def load_previous_experiments(log_path):
    """Returns (X, output, dc, dc_var): output is the logged
    |mean-target|+penalties score (used for the stopping criterion and
    best_result -- unchanged); dc is the raw duty-cycle fraction from the
    same row (used for GP fitting -- see the class docstring note on why
    these are kept separate); dc_var is dc's measurement variance
    (fraction**2, from the logged std), passed to the GP as known per-point
    observation noise."""
    import csv
    X_list, Y_list, DC_list, Var_list = [], [], [], []
    if not os.path.exists(log_path):
        return (torch.tensor(X_list), torch.tensor(Y_list),
                torch.tensor(DC_list), torch.tensor(Var_list))
    with open(log_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            X_list.append([float(row['amplitude']), float(row['rise']),
                           float(row['dwell']), float(row['fall'])])
            Y_list.append([float(row['output'])])
            DC_list.append([float(row['duty_cycle_mean']) / 100])
            std = float(row['duty_cycle_std']) / 100
            Var_list.append([max(std ** 2, config.min_observation_variance)])
    X = torch.tensor(X_list, dtype=torch.double)
    Y = torch.tensor(Y_list, dtype=torch.double)
    DC = torch.tensor(DC_list, dtype=torch.double)
    Var = torch.tensor(Var_list, dtype=torch.double)
    return X, Y, DC, Var


def used_structure_ids(log_path):
    """Set of every structure_id logged so far, any arm/source -- the one
    shared view of which physical structures on the chip have already been
    poled, since BOSession and OFATSession (ofat_session.py) both log to the
    same CSV but otherwise know nothing about each other."""
    import csv
    ids = set()
    if not os.path.exists(log_path):
        return ids
    with open(log_path, newline='') as f:
        for row in csv.DictReader(f):
            sid = row.get('structure_id')
            if sid:
                ids.add(sid)
    return ids


def count_logged_rows(log_path):
    import csv
    if not os.path.exists(log_path):
        return 0
    with open(log_path, newline='') as f:
        return sum(1 for _ in csv.DictReader(f))


def _cat(existing, new_row):
    return torch.cat([existing, new_row], dim=0) if len(existing) > 0 else new_row


class BOSession:
    """Holds all BO state and drives it one step at a time (get a
    recommendation, submit its result) instead of a blocking loop -- so a CLI
    can drive it with input() and a GUI can drive it with button clicks,
    sharing the exact same logic."""

    def __init__(self):
        self.bounds = build_bounds()
        init_logger(LOG_PATH)

        # dc_warm/dc_real hold the RAW duty cycle (fraction) -- what the GP is
        # actually fit on, with var_warm/var_real as its known per-point
        # observation noise (heteroskedastic). Y_real holds the
        # |mean-target|+penalties "output" score -- kept only for the
        # stopping criterion and best_result(), which care about overall
        # experiment quality, not the smooth physical response the surrogate
        # models.
        self.X_warm, self.dc_warm, self.var_warm = build_warm_start(WARM_START_PATH) \
            if os.path.exists(WARM_START_PATH) else (torch.tensor([]), torch.tensor([]), torch.tensor([]))
        self.X_real, self.Y_real, self.dc_real, self.var_real = load_previous_experiments(LOG_PATH)

        # no warm-start history and no previously logged real experiments --
        # bootstrap with Latin Hypercube points instead of jumping straight
        # into GP+EI with zero data to fit on
        self.lhs_queue = []
        if len(self.X_warm) == 0 and len(self.X_real) == 0:
            self.lhs_queue = generate_lhs_points(self.bounds, config.lhs_points).tolist()

        self.experiment_num = config.lhs_points
        self.pending_params = None
        self.pending_source = None
        self.pending_prediction = None  # (pred_mean, pred_std) of raw dc at pending_params, from
        # the GP that produced it -- None for LHS-sourced points (no model yet) or a fallback
        self.pending_gp_fallback = False
        self.last_model = None

    @property
    def X(self):
        parts = [t for t in (self.X_warm, self.X_real) if len(t) > 0]
        return torch.cat(parts, dim=0) if parts else torch.empty(0, 4, dtype=torch.double)

    @property
    def dc(self):
        """Raw duty-cycle fraction for every point seen so far (warm start +
        real), aligned with self.X -- this, not a target-distance score, is
        what the GP is trained on."""
        parts = [t for t in (self.dc_warm, self.dc_real) if len(t) > 0]
        return torch.cat(parts, dim=0) if parts else torch.empty(0, 1, dtype=torch.double)

    @property
    def dc_var(self):
        """Known per-point observation variance for every point in self.dc,
        in the same order -- passed to fit_gp as train_Yvar."""
        parts = [t for t in (self.var_warm, self.var_real) if len(t) > 0]
        return torch.cat(parts, dim=0) if parts else torch.empty(0, 1, dtype=torch.double)

    def status_line(self):
        line = f"Warm start: {len(self.X_warm)} points | Previous experiments: {len(self.X_real)} points"
        if self.lhs_queue:
            line += f" | {len(self.lhs_queue)} Latin Hypercube point(s) queued for initial exploration"
        return line

    def is_finished(self):
        """(done, reason) -- reason is None if not done."""
        if len(used_structure_ids(LOG_PATH)) >= N_STRUCTURES:
            return True, "All test structures used."
        if len(self.Y_real) > 0:
            best = self.Y_real.min().item()
            if best < config.stopping_threshold:
                return True, f"Early stopping: best output {best:.4f} < threshold {config.stopping_threshold}"
        return False, None

    def next_params(self):
        """The next parameter combination to try. Idempotent: calling again
        before submit_result() returns the same pending combination rather
        than drawing a new one. self.pending_source records where it came
        from ('lhs' or 'gp_ei') for callers that want to display it.
        self.pending_prediction holds the GP's own (pred_mean, pred_std) of
        the raw duty cycle at this point -- logged alongside the real result
        once it comes in, so predictions can be checked against reality
        later instead of only after the fact. self.pending_gp_fallback is
        True if the GP fit/acquisition step failed and a random point was
        substituted -- the same numerical-singularity failure mode seen on
        real data during evaluation (see evaluate.py's run_gp); one bad fit
        shouldn't block real data collection."""
        if self.pending_params is not None:
            return self.pending_params
        if self.lhs_queue:
            self.pending_params = self.lhs_queue.pop(0)
            self.pending_source = "lhs"
            self.pending_prediction = None
            self.pending_gp_fallback = False
        else:
            try:
                self.last_model = fit_gp(self.X, self.dc, config.n_restart, self.bounds, Yvar=self.dc_var)
                candidate = suggest_next(self.last_model, self.bounds, self.dc)
                self.pending_gp_fallback = False
            except Exception:
                self.last_model = None
                self.pending_gp_fallback = True
                candidate = (self.bounds[0] + torch.rand(4) * (self.bounds[1] - self.bounds[0])).unsqueeze(0)
            self.pending_params = candidate.squeeze().tolist()
            self.pending_source = "gp_ei"
            if self.last_model is not None:
                with torch.no_grad():
                    posterior = self.last_model.posterior(candidate)
                    self.pending_prediction = (posterior.mean.item(), posterior.variance.sqrt().item())
            else:
                self.pending_prediction = None
        return self.pending_params

    def submit_result(self, structure_id, image_path, angle=None):
        """Compute duty cycle for the current pending params from image_path,
        log it, and fold it into the real-experiment data. structure_id is
        the real physical structure/location label on the chip (the code
        has no way to know your chip's layout, so this comes from you, not
        an internal counter). angle must be the SAME deskew angle the caller
        used to produce the preview the user visually verified --
        estimate_duty_cycle re-derives mean/std from scratch here, so
        passing a different (or no) angle would silently log a different
        result than what was approved. Returns (mean, std, output,
        diagnostics), or None if no numeric duty cycle could be computed
        (caller should let the user try a different image -- pending_params
        is left unchanged so next_params() still returns the same
        combination)."""
        params = self.pending_params
        if params is None:
            raise RuntimeError("next_params() must be called before submit_result().")

        mean, std, diagnostics = estimate_duty_cycle(image_path, angle=angle)
        if mean is None:
            return None

        output = compute_output(mean, std, diagnostics["problem_fraction"])

        self.experiment_num += 1
        pred_mean, pred_std = self.pending_prediction if self.pending_prediction is not None else (None, None)
        log_experiment(LOG_PATH, self.experiment_num, structure_id, *params, mean, std, output,
                       source=self.pending_source, deskew_angle=angle, image_path=image_path,
                       gp_pred_mean=pred_mean, gp_pred_std=pred_std, gp_fallback=self.pending_gp_fallback)

        candidate_t = torch.tensor([params], dtype=torch.double)
        output_t = torch.tensor([[output]], dtype=torch.double)
        dc_t = torch.tensor([[mean / 100]], dtype=torch.double)
        var_t = torch.tensor([[max((std / 100) ** 2, config.min_observation_variance)]], dtype=torch.double)
        self.X_real = _cat(self.X_real, candidate_t)
        self.Y_real = _cat(self.Y_real, output_t)
        self.dc_real = _cat(self.dc_real, dc_t)
        self.var_real = _cat(self.var_real, var_t)

        self.pending_params = None
        self.pending_prediction = None
        self.pending_gp_fallback = False
        return mean, std, output, diagnostics

    def best_result(self):
        """(params, output) from real experiments only, or None if none run yet."""
        if len(self.Y_real) == 0:
            return None
        best_idx = self.Y_real.argmin().item()
        return self.X_real[best_idx].tolist(), self.Y_real.min().item()
