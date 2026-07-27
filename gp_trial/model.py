from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import torch


def fit_gp(X, Y, n_restarts):
    
    best_model = None
    best_score = float("-inf")

    for _ in range(n_restarts):
        model = SingleTaskGP(
            train_X=X,
            train_Y=Y,
            input_transform=Normalize(d=4),
            outcome_transform=Standardize(m=1),
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        with torch.no_grad():
            score = mll(model(*model.train_inputs), model.train_targets).item()

        if score > best_score:
            best_score = score
            best_model = model

    return best_model