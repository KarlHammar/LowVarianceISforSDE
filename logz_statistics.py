"""Bootstrap uncertainty of the log-mean-weight estimator at fixed observations."""
import numpy as np


def bootstrap_std(log_weights, replicates):
    weights = np.asarray(log_weights, dtype=float)
    if weights.size < 2 or not np.isfinite(weights).all():
        raise ValueError('Need at least two finite log weights (no NaN or +/-inf)')
    indices = np.random.randint(weights.size, size=(replicates, weights.size))
    values = np.logaddexp.reduce(weights[indices], axis=1) - np.log(weights.size)
    return float(np.std(values, ddof=1))

