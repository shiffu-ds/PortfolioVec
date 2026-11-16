import numpy as np
import pytest

from src.portfoliovec import returns


def test_return_matrix_shape():
    P = np.random.default_rng(1).uniform(50, 150, (10, 4))
    assert returns.return_matrix(P).shape == (9, 4)


def test_known_return():
    R = returns.return_matrix(np.array([[100.0], [102.0]]))
    assert R[0, 0] == pytest.approx(0.02)


def test_portfolio_return_manual():
    R = np.array([[0.01, -0.02, 0.03],
                  [0.00, 0.01, -0.01]])
    w = np.array([0.5, 0.8, -0.3])
    manual = [0.01 * 0.5 + -0.02 * 0.8 + 0.03 * -0.3,
              0.00 * 0.5 + 0.01 * 0.8 + -0.01 * -0.3]
    assert returns.portfolio_returns(R, w) == pytest.approx(manual)


def test_short_gains_when_asset_falls():
    R = np.array([[0.0, -0.05]])
    w = np.array([1.2, -0.2])
    assert returns.portfolio_returns(R, w)[0] > 0


def test_value_path_end():
    rho = np.array([0.01, -0.02, 0.03])
    path = returns.value_path(rho, 10000)
    assert len(path) == 4
    assert path[0] == 10000
    assert path[-1] == pytest.approx(10000 * 1.01 * 0.98 * 1.03)
