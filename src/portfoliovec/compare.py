"""Run several allocations over the same return matrix."""
from .metrics import summarize
from .returns import portfolio_returns, value_path


def compare_allocations(R, allocations: dict, v0: float) -> dict:
    results = {}
    for key, alloc in allocations.items():
        rho = portfolio_returns(R, alloc["weights"])
        path = value_path(rho, v0)
        results[key] = {
            "name": alloc["name"],
            "rho": rho,
            "path": path,
            "summary": summarize(rho, path, alloc["weights"]),
        }
    return results


def winner(results: dict, metric: str = "final_value") -> str:
    """Key of the allocation with the highest value of `metric`."""
    return max(results, key=lambda k: results[k]["summary"][metric])
