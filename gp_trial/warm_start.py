import csv
import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import min_observation_variance

def build_warm_start(csv_path):
    """Returns (X, Y, Yvar): Y is the RAW duty cycle (fraction, 0-1), not the
    |mean - target| distance -- the GP is fit on the smooth physical
    response; the target-distance loss lives in the acquisition function
    (optimizer.py) instead. Yvar is that row's own measurement variance
    (fraction**2, from its std column), passed to the GP as known
    per-point observation noise instead of assuming the same noise level
    everywhere."""
    X_list, Y_list, Yvar_list = [], [], []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mean = float(row['mean']) / 100  # logged as a percentage
            std = float(row['std']) / 100
            var = max(std ** 2, min_observation_variance)

            X_list.append([
                float(row['amplitude']),
                float(row['rise']),
                float(row['dwell']),
                float(row['fall'])
            ])
            Y_list.append([mean])
            Yvar_list.append([var])

    X = torch.tensor(X_list, dtype=torch.float64)
    Y = torch.tensor(Y_list, dtype=torch.float64)
    Yvar = torch.tensor(Yvar_list, dtype=torch.float64)
    return X, Y, Yvar