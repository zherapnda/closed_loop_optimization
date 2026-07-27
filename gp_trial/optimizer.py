from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim import optimize_acqf
from config import n_restart 


def suggest_next(model, bounds):    
    EI = LogExpectedImprovement(model=model, best_f=model.train_targets.min())

    candidate, _ = optimize_acqf(
        EI,
        bounds=bounds,
        q=1,
        num_restarts=n_restart,
        raw_samples=n_restart * 10,
    )


    return candidate