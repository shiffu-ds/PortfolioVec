"""PortfolioVec pipeline runner:  python main.py"""
import numpy as np
import yaml

from src.portfoliovec import (compare, loader, plots, report, returns,
                              validator, vectors)


def resolve_weights(alloc_cfg, p, s) -> np.ndarray:
    w = alloc_cfg["weights"]
    if isinstance(w, str):  # "derive_from_holdings"
        return vectors.weights_from_holdings(p, s)
    return np.array(w, dtype=float)


def main(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 1-2 Load & validate
    P, dates, assets = loader.load_prices(cfg)
    validator.check(P, cfg)
    loader.save_processed(P, dates, assets)
    print(f"Loaded {P.shape[0]} days x {P.shape[1]} assets "
          f"({dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d})")

    # 3 Vectors
    s = np.array(cfg["holdings"], dtype=float)
    b = np.array(cfg["trade"], dtype=float)
    p = P[-1]  # latest prices

    # 4 Value & trade (fixed shares)
    v_before = vectors.portfolio_value(p, s)
    s_new = vectors.apply_trade(s, b)
    v_after = vectors.portfolio_value(p, s_new)
    cash = vectors.trade_cash(p, b)
    assert abs((v_after - v_before) - cash) < 1e-6, "trade consistency check failed"
    report.trade_table(assets, s, b, s_new, p, v_before, v_after, cash)

    # 5-6 Returns (constant weights = daily re-balancing)
    R = returns.return_matrix(P)
    allocs = {
        key: {"name": a["name"], "weights": resolve_weights(a, p, s)}
        for key, a in cfg["allocations"].items()
    }
    report.weights_table(assets, allocs)

    # 7 Compare
    results = compare.compare_allocations(R, allocs, cfg["initial_capital"])

    # 8 Report
    report.daily_returns_table(dates[1:], assets, R, results)
    report.value_paths_table(dates, results)
    report.comparison_table(results)
    report.conclusion(results, compare.winner(results), assets, R, allocs)

    figs = [
        plots.value_over_time(dates, results),
        plots.daily_returns(dates[1:], results),
        plots.weights_bar(assets, allocs),
    ]
    print("\nFigures: " + ", ".join(str(f) for f in figs))
    print("Tables : outputs/tables/")
    return results


if __name__ == "__main__":
    main()
