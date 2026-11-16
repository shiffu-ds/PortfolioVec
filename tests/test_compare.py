import numpy as np
import pytest

from src.portfoliovec import compare, validator

R = np.array([[0.02, -0.01],
              [-0.01, 0.03],
              [0.01, 0.01]])
ALLOCS = {
    "A": {"name": "All in asset 1", "weights": np.array([1.0, 0.0])},
    "B": {"name": "Long/short", "weights": np.array([1.5, -0.5])},
}


def test_compare_structure_and_values():
    res = compare.compare_allocations(R, ALLOCS, 1000)
    assert set(res) == {"A", "B"}
    a = res["A"]
    assert a["rho"] == pytest.approx([0.02, -0.01, 0.01])
    assert a["summary"]["final_value"] == pytest.approx(1000 * 1.02 * 0.99 * 1.01)
    assert a["summary"]["best_day"] == pytest.approx(0.02)
    assert a["summary"]["worst_day"] == pytest.approx(-0.01)
    assert a["summary"]["leverage"] == pytest.approx(1.0)
    assert res["B"]["summary"]["leverage"] == pytest.approx(2.0)


def test_winner():
    res = compare.compare_allocations(R, ALLOCS, 1000)
    expected = max(res, key=lambda k: res[k]["path"][-1])
    assert compare.winner(res) == expected


def _cfg(**over):
    cfg = {
        "assets": ["X", "Y"],
        "holdings": [1, 2],
        "trade": [0, 0],
        "allocations": {
            "A": {"name": "a", "weights": "derive_from_holdings"},
            "B": {"name": "b", "weights": [0.5, 0.5]},
        },
    }
    cfg.update(over)
    return cfg


def test_validator_accepts_good_input():
    validator.check(np.array([[1.0, 2.0], [1.1, 2.1]]), _cfg())


@pytest.mark.parametrize("P, cfg", [
    (np.array([[1.0, 2.0], [np.nan, 2.1]]), _cfg()),            # missing value
    (np.array([[1.0, 2.0], [0.0, 2.1]]), _cfg()),               # non-positive price
    (np.array([[1.0, 2.0]]), _cfg()),                           # one day only
    (np.array([[1.0, 2.0], [1.1, 2.1]]), _cfg(holdings=[1])),   # wrong length
    (np.array([[1.0, 2.0], [1.1, 2.1]]),
     _cfg(allocations={"B": {"name": "b", "weights": [0.5, 0.6]}})),  # sum != 1
])
def test_validator_rejects_bad_input(P, cfg):
    with pytest.raises(ValueError):
        validator.check(P, cfg)
