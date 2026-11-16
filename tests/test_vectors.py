import numpy as np
import pytest

from src.portfoliovec import vectors


def test_value():
    assert vectors.portfolio_value(np.array([10.0, 20.0]), np.array([5.0, 3.0])) == 110


def test_worked_example_trade():
    p, s, b = np.array([10.0, 20.0]), np.array([5.0, 3.0]), np.array([2.0, -1.0])
    s_new = vectors.apply_trade(s, b)
    assert s_new.tolist() == [7, 2]
    assert vectors.trade_cash(p, b) == 0
    assert vectors.portfolio_value(p, s_new) == 110


def test_trade_consistency():
    rng = np.random.default_rng(0)
    p = rng.uniform(10, 500, 5)
    s = rng.integers(-50, 100, 5).astype(float)
    b = rng.integers(-20, 20, 5).astype(float)
    v_before = vectors.portfolio_value(p, s)
    v_after = vectors.portfolio_value(p, vectors.apply_trade(s, b))
    assert abs((v_after - v_before) - vectors.trade_cash(p, b)) < 1e-9


def test_short_position():
    p = np.array([150.0, 200.0])
    with_short = vectors.portfolio_value(p, np.array([100.0, -30.0]))
    without = vectors.portfolio_value(p, np.array([100.0, 0.0]))
    assert with_short < without


def test_weights_sum():
    p = np.array([150.0, 130.0, 300.0, 120.0, 200.0])
    s = np.array([100.0, 50.0, 80.0, 40.0, -30.0])
    assert vectors.weights_from_holdings(p, s).sum() == pytest.approx(1.0)


def test_short_negative_weight():
    p = np.array([150.0, 130.0, 300.0, 120.0, 200.0])
    s = np.array([100.0, 50.0, 80.0, 40.0, -30.0])
    w = vectors.weights_from_holdings(p, s)
    assert w[4] < 0
    assert (w[:4] > 0).all()


def test_zero_total_value():
    with pytest.raises(ValueError):
        vectors.weights_from_holdings(np.array([10.0, 10.0]), np.array([5.0, -5.0]))


def test_leverage():
    assert vectors.leverage(np.array([0.2] * 5)) == pytest.approx(1.0)
    assert vectors.leverage(np.array([0.7, 0.6, -0.3])) == pytest.approx(1.6)
