"""Summary statistics for one allocation."""
import numpy as np


def summarize(rho, path, w) -> dict:
    return {
        "avg_daily_return": float(rho.mean()),
        "best_day": float(rho.max()),
        "worst_day": float(rho.min()),
        "final_value": float(path[-1]),
        "total_return": float(path[-1] / path[0] - 1),
        "leverage": float(np.abs(w).sum()),
        "daily_risk_std": float(rho.std()),   # optional extension
    }
