"""Output layer: console tables and CSV exports."""
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path("outputs/tables")

METRIC_LABELS = [
    ("avg_daily_return", "Avg daily return", "{:+.3%}"),
    ("best_day", "Best day", "{:+.2%}"),
    ("worst_day", "Worst day", "{:+.2%}"),
    ("final_value", "Final value", "${:,.2f}"),
    ("total_return", "Total return", "{:+.2%}"),
    ("leverage", "Leverage", "{:.2f}"),
    ("daily_risk_std", "Daily risk (std)", "{:.3%}"),
]


def _save(df: pd.DataFrame, name: str, index=True) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / name
    df.to_csv(out, index=index)
    return out


def trade_table(assets, s, b, s_new, p, v_before, v_after, cash) -> pd.DataFrame:
    df = pd.DataFrame({
        "Asset": assets,
        "Before": s,
        "Trade": b,
        "After": s_new,
        "Price": p,
        "Value before": p * s,
        "Value after": p * s_new,
    })
    print("\n=== TRADE SUMMARY ===")
    print(df.to_string(index=False, formatters={
        "Before": "{:,.0f}".format,
        "Trade": "{:+,.0f}".format,
        "After": "{:,.0f}".format,
        "Price": "{:,.2f}".format,
        "Value before": "{:,.2f}".format,
        "Value after": "{:,.2f}".format,
    }))
    print(f"Portfolio value before: {v_before:,.2f}")
    print(f"Portfolio value after : {v_after:,.2f}")
    print(f"Net cash of trade (p^T b): {cash:,.2f}   "
          f"(check: after - before = {v_after - v_before:,.2f})")
    _save(df, "trade_summary.csv", index=False)
    return df


def weights_table(assets, allocs) -> pd.DataFrame:
    df = pd.DataFrame({k: a["weights"] for k, a in allocs.items()}, index=assets)
    df.index.name = "Asset"
    print("\n=== WEIGHTS ===")
    print(df.to_string(float_format="{:+.4f}".format))
    print("Sum".ljust(6) + "".join(f"{df[k].sum():+10.4f}" for k in df.columns))
    _save(df, "weights.csv")
    return df


def daily_returns_table(dates, assets, R, results) -> pd.DataFrame:
    df = pd.DataFrame(R, index=pd.Index(dates, name="Date"), columns=assets)
    for k, res in results.items():
        df[f"Portfolio {k}"] = res["rho"]
    print("\n=== DAILY RETURNS (last 5 days) ===")
    print(df.tail().to_string(float_format="{:+.2%}".format))
    _save(df, "daily_returns.csv")
    return df


def value_paths_table(dates, results) -> pd.DataFrame:
    df = pd.DataFrame({f"Value {k}": res["path"] for k, res in results.items()},
                      index=pd.Index(dates, name="Date"))
    _save(df, "value_paths.csv")
    return df


def comparison_table(results) -> pd.DataFrame:
    raw = pd.DataFrame({k: res["summary"] for k, res in results.items()})
    _save(raw.rename_axis("metric"), "comparison.csv")

    cols = {k: f"{k} ({res['name']})" for k, res in results.items()}
    shown = pd.DataFrame(
        {cols[k]: [fmt.format(raw.loc[key, k]) for key, _, fmt in METRIC_LABELS]
         for k in results},
        index=[label for _, label, _ in METRIC_LABELS],
    )
    shown.index.name = "Metric"
    print("\n=== ALLOCATION COMPARISON ===")
    print(shown.to_string())
    return raw


def conclusion(results, best_key, assets, R, allocs) -> str:
    """One-paragraph verdict, including what each short position contributed."""
    best = results[best_key]
    lines = [f"\n=== CONCLUSION ===",
             f"{best_key} ({best['name']}) finished higher: "
             f"${best['summary']['final_value']:,.2f} "
             f"({best['summary']['total_return']:+.2%})."]
    for k, alloc in allocs.items():
        w = alloc["weights"]
        for i in np.where(w < 0)[0]:
            contrib = R[:, i] * w[i]
            down = R[:, i] < 0
            lines.append(
                f"Short {assets[i]} in {k} (w = {w[i]:+.3f}): asset fell on "
                f"{int(down.sum())} of {len(down)} days; the short added "
                f"{contrib.sum():+.2%} in total (sum of daily r*w)."
            )
    text = "\n".join(lines)
    print(text)
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "conclusion.txt").write_text(text.strip() + "\n")
    return text
