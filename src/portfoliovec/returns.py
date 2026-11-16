"""Vector core: return matrix and portfolio returns. Pure functions, no I/O."""
import numpy as np


def return_matrix(P: np.ndarray) -> np.ndarray:
    """P is T x n prices. Returns (T-1) x n matrix of daily returns."""
    return P[1:] / P[:-1] - 1.0


def portfolio_returns(R: np.ndarray, w: np.ndarray) -> np.ndarray:
    """rho_t = r_t^T w for every day  ->  R @ w."""
    return R @ w


def value_path(rho: np.ndarray, v0: float) -> np.ndarray:
    """V_{t+1} = V_t (1 + rho_t). Returns array of length len(rho)+1."""
    return v0 * np.concatenate(([1.0], np.cumprod(1.0 + rho)))
