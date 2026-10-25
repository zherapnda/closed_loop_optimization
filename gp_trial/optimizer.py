"""Suggests the next point to evaluate. The |dc - target| loss lives here,
inside the acquisition function, rather than being the GP's own training
target (see model.py / bo_session.py / evaluate.py) -- |dc - target| has a
hard kink exactly at the optimum, and fitting a smooth kernel (Matern-5/2)
directly to a kinked function is model misspecification right where accuracy
matters most. The GP instead models the raw, smooth duty-cycle response;
qLogExpectedImprovement is evaluated on posterior SAMPLES of that raw
response, run through the target-distance loss via a GenericMCObjective, so
"improvement" is judged in loss space without the surrogate itself ever
being trained on a non-smooth target."""
import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

from config import n_restart, raw_samples, target_dc


def _distance_objective(samples, X=None):
    """samples: raw duty-cycle draws from the GP posterior, shape
    (..., q, 1). Negated so qLogEI's "maximize" convention corresponds to
    minimizing distance to target_dc."""
    return -(samples[..., 0] - target_dc).abs()


def suggest_next(model, bounds, train_dc):
    """train_dc: (n, 1) tensor of the RAW duty-cycle values the model was fit
    on (not a transformed objective) -- used to compute best_f in the same
    units _distance_objective operates in."""
    objective = GenericMCObjective(_distance_objective)
    best_f = (-(train_dc - target_dc).abs()).max()

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]), seed=0)
    acqf = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler, objective=objective)

    candidate, _ = optimize_acqf(
        acqf,
        bounds=bounds,
        q=1,
        num_restarts=n_restart,
        raw_samples=raw_samples,
    )

    return candidate
