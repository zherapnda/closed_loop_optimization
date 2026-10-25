import torch
from scipy.stats import qmc


def generate_lhs_points(bounds, n_points, seed=None):
    """Latin Hypercube samples spread across bounds (a 2 x d tensor of
    [lower; upper]), for bootstrapping a fresh optimization that has no
    warm-start history to start from. LHS spreads points evenly across each
    dimension's range (no two points share a coordinate bin along any single
    dimension), giving better initial coverage of the search space than
    uniform random sampling would with the same number of points."""
    d = bounds.shape[-1]
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit_samples = sampler.random(n=n_points)  # n_points x d, each column in [0, 1)
    lower = bounds[0].numpy()
    upper = bounds[1].numpy()
    scaled = qmc.scale(unit_samples, lower, upper)
    return torch.tensor(scaled, dtype=torch.double)
