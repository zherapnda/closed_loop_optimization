"""Drives the OFAT arm of a live GP-vs-OFAT real-chip comparison, mirroring
BOSession's step-driven interface (next_params/submit_result) but with no
model to fit. Unlike BOSession, there's nothing expensive to cache: state is
just replayed from the shared log file on every call via
ofat.ofat_next_candidate, so no @st.cache_resource / in-memory persistence
is needed for correctness."""
import csv
import os
import sys

from bo_session import LOG_PATH, compute_output, count_logged_rows
from logger import log_experiment
from ofat import ofat_next_candidate
import config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'duty_cycle'))
import tip_lines_check
estimate_duty_cycle = tip_lines_check.estimate_duty_cycle


def load_ofat_history(log_path):
    """[(params, dc), ...] for every row logged with source='ofat', in the
    order they were submitted -- dc as a fraction (0-1), matching
    ofat_next_candidate's `target` units (the CSV stores it as a percent)."""
    history = []
    if not os.path.exists(log_path):
        return history
    with open(log_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('source') != 'ofat':
                continue
            params = [float(row['amplitude']), float(row['rise']), float(row['dwell']), float(row['fall'])]
            dc = float(row['duty_cycle_mean']) / 100
            history.append((params, dc))
    return history


class OFATSession:
    """start: the shared starting point (should be the GP arm's own first
    suggestion, so both arms begin the real comparison from the identical
    parameter combination -- see evaluate.py's same rationale for the
    simulated comparison)."""

    def __init__(self, bounds, start, n_candidates=4):
        self.bounds = bounds
        self.start = list(start)
        self.n_candidates = n_candidates
        self.pending_params = None
        self.pending_info = None

    def history(self):
        return load_ofat_history(LOG_PATH)

    def status_line(self):
        return f"OFAT arm: {len(self.history())} real experiment(s) logged"

    def next_params(self):
        """Not idempotent the way BOSession's is -- ofat_next_candidate is a
        pure, deterministic function of history (no fitting, no randomness),
        so recomputing it on every Streamlit rerun is cheap and always gives
        the same answer until a new result is actually submitted."""
        candidate, info = ofat_next_candidate(
            self.history(), self.bounds, self.start, n_candidates=self.n_candidates,
            target=config.target_dc)
        self.pending_params = candidate
        self.pending_info = info
        return candidate

    def submit_result(self, structure_id, image_path, angle=None):
        """Mirrors BOSession.submit_result's contract: structure_id is the
        real physical structure label (typed in, not auto-assigned); angle
        must match what the caller used for the preview the user approved."""
        params = self.pending_params
        if params is None:
            raise RuntimeError("next_params() must be called before submit_result().")

        mean, std, diagnostics = estimate_duty_cycle(image_path, angle=angle)
        if mean is None:
            return None

        output = compute_output(mean, std, diagnostics["problem_fraction"])
        experiment_num = count_logged_rows(LOG_PATH) + 1
        log_experiment(LOG_PATH, experiment_num, structure_id, *params, mean, std, output,
                       source="ofat", deskew_angle=angle, image_path=image_path)

        self.pending_params = None
        self.pending_info = None
        return mean, std, output, diagnostics
