"""Vector core: value, trade, weights, leverage. Pure functions, no I/O."""
import numpy as np


def portfolio_value(p: np.ndarray, s: np.ndarray) -> float:
    """Value = p^T s  (price-quantity inner product)."""
    return float(p @ s)


def apply_trade(s: np.ndarray, b: np.ndarray) -> np.ndarray:
    """New holdings = s + b  (vector addition)."""
    return s + b


def trade_cash(p: np.ndarray, b: np.ndarray) -> float:
    """Net cash spent on the trade = p^T b."""
    return float(p @ b)


def dollar_holdings(p: np.ndarray, s: np.ndarray) -> np.ndarray:
    """h_i = p_i * s_i (element-wise)."""
    return p * s


def weights_from_holdings(p: np.ndarray, s: np.ndarray) -> np.ndarray:
    """w = h / (1^T h). Entries sum to 1; short positions are negative."""
    h = dollar_holdings(p, s)
    total = h.sum()
    if total == 0:
        raise ValueError("Total portfolio value is zero; weights undefined.")
    return h / total


def leverage(w: np.ndarray) -> float:
    """L = |w1| + ... + |wn|."""
    return float(np.abs(w).sum())
