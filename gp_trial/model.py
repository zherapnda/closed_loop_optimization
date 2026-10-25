from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize, ChainedInputTransform, ReversibleInputTransform, subset_transform
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import torch

AMPLITUDE_INDEX = 0
AMPLITUDE_LOG_EPS = 1e-3  # amplitude=0 is a valid search-bound edge but log(0) is undefined; this
# floor only affects points within 1e-3 of zero (physically indistinguishable from "field too weak
# to switch anything" either way), and matches the floor used to build the log-space Normalize
# bounds in build_input_transform below.


class Logit(ReversibleInputTransform):
    r"""Maps (0, 1) to the reals: logit(x) = log(x / (1 - x)). Chained after
    Normalize (which maps the real search bounds to [0, 1]) so the GP models
    the space unconstrained instead of on a bounded [0, 1] box -- the GP
    otherwise treats the hard edges of the search space (e.g. amplitude=0 or
    amplitude=1000) as ordinary interior points, which understates how much a
    prediction should change near a real physical boundary."""

    def __init__(self, indices, eps=1e-6, transform_on_train=True, transform_on_eval=True,
                 transform_on_fantasize=True, reverse=False):
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.eps = eps
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.reverse = reverse

    @subset_transform
    def _transform(self, X):
        X = X.clamp(self.eps, 1 - self.eps)
        return torch.log(X) - torch.log1p(-X)

    @subset_transform
    def _untransform(self, X):
        return torch.sigmoid(X)


class LogTransform(ReversibleInputTransform):
    r"""Applies log(x) to the given indices only, identity elsewhere.
    Amplitude enters the physics equation through exp(Ea / amplitude), so its
    effect on duty cycle is far more evenly spread out on a log scale than a
    linear one -- without this, the GP's single per-dimension lengthscale for
    amplitude has to compromise between the steep low-amplitude region and
    the comparatively flat high-amplitude saturation region."""

    def __init__(self, indices, eps=AMPLITUDE_LOG_EPS, transform_on_train=True, transform_on_eval=True,
                 transform_on_fantasize=True, reverse=False):
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.eps = eps
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.reverse = reverse

    @subset_transform
    def _transform(self, X):
        return torch.log(X.clamp_min(self.eps))

    @subset_transform
    def _untransform(self, X):
        return torch.exp(X)


def build_input_transform(bounds):
    d = bounds.shape[-1]
    # Normalize needs the amplitude column's bounds in LOG space too, since LogTransform runs
    # before it in the chain -- everything else keeps its original linear bounds.
    log_bounds = bounds.clone()
    log_bounds[:, AMPLITUDE_INDEX] = torch.log(bounds[:, AMPLITUDE_INDEX].clamp_min(AMPLITUDE_LOG_EPS))
    return ChainedInputTransform(
        log_amplitude=LogTransform(indices=[AMPLITUDE_INDEX]),
        normalize=Normalize(d=d, bounds=log_bounds),
        logit=Logit(indices=list(range(d))),
    )


def build_covar_module(d):
    # Matern 5/2 (the botorch/SingleTaskGP default smoothness) with automatic
    # relevance determination (a separate lengthscale per input dimension) --
    # made explicit here rather than relying on SingleTaskGP's undocumented
    # default, so it's visible and adjustable in one place.
    return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))


def fit_gp(X, Y, n_restarts, bounds, Yvar=None):
    """Yvar: optional (n, 1) known per-point observation variance (e.g. the
    measured std**2 from repeated/spatial duty-cycle sampling). When given,
    the GP treats noisy regions as noisier and clean regions as more
    trustworthy instead of assuming one global noise level."""
    d = X.shape[-1]
    best_model = None
    best_score = float("-inf")

    for _ in range(n_restarts):
        model = SingleTaskGP(
            train_X=X,
            train_Y=Y,
            train_Yvar=Yvar,
            covar_module=build_covar_module(d),
            input_transform=build_input_transform(bounds),
            outcome_transform=Standardize(m=1),
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        model.train()
        with torch.no_grad():
            score = mll(model(*model.train_inputs), model.train_targets).item()
        model.eval()

        if score > best_score:
            best_score = score
            best_model = model

    return best_model
