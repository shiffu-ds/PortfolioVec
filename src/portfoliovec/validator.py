"""Stage 2: validate prices and config. Fail fast with a clear message."""
import numpy as np

TOL = 1e-8


def check(P: np.ndarray, cfg: dict) -> None:
    n = len(cfg["assets"])

    if P.ndim != 2 or P.shape[1] != n:
        raise ValueError(f"Price matrix must be T x {n}, got shape {P.shape}.")
    if P.shape[0] < 2:
        raise ValueError("Need at least 2 days of prices to compute returns.")
    if np.isnan(P).any():
        raise ValueError("Price matrix contains missing values (NaN).")
    if (P <= 0).any():
        raise ValueError("All prices must be positive (returns divide by prices).")

    for key in ("holdings", "trade"):
        if len(cfg[key]) != n:
            raise ValueError(f"len({key}) = {len(cfg[key])}, expected {n}.")

    for key, alloc in cfg["allocations"].items():
        w = alloc["weights"]
        if isinstance(w, str):
            if w != "derive_from_holdings":
                raise ValueError(f"Allocation {key}: unknown weights option {w!r}.")
            continue
        if len(w) != n:
            raise ValueError(f"Allocation {key}: {len(w)} weights, expected {n}.")
        total = float(np.sum(w))
        if abs(total - 1.0) > TOL:
            raise ValueError(f"Allocation {key}: weights sum to {total}, not 1.")
