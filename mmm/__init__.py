import numpy as np
from .data import MMMData
from .model import BayesianMMM


# Optional: keep all numpy-only utilities here.
def geometric_adstock_np(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        y[t] = x[t] + (alpha * y[t - 1] if t > 0 else 0.0)
    return y


def hill_np(x: np.ndarray, half_sat: float, slope: float) -> np.ndarray:
    # clip to non-negative domain
    x = np.clip(x, 0, None)
    xs = x ** slope
    ks = half_sat ** slope
    return xs / (xs + ks + 1e-12)
