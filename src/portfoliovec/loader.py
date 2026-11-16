"""Stage 1: load daily closing prices into a T x n matrix P."""
from pathlib import Path

import numpy as np
import pandas as pd


def download_prices(assets, start, end) -> pd.DataFrame:
    """Download daily closing prices from Yahoo Finance (needs yfinance)."""
    import yfinance as yf

    # yfinance treats `end` as exclusive, so add one day to include it.
    end_incl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(list(assets), start=start, end=end_incl,
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no data.")
    df = raw["Close"][list(assets)]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def load_prices(cfg: dict):
    """Return (P, dates, assets).

    P is a T x n float array: rows = days, columns = assets in the same
    order as cfg["assets"].
    """
    assets = list(cfg["assets"])
    data = cfg["data"]

    if data["source"] == "yfinance":
        df = download_prices(assets, data["start"], data["end"])
    elif data["source"] == "csv":
        path = Path(data["path"])
        if not path.exists():
            raise FileNotFoundError(
                f"Price file not found: {path}. "
                "Run `python scripts/download_prices.py` first."
            )
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unknown data source: {data['source']!r}")

    missing = [a for a in assets if a not in df.columns]
    if missing:
        raise ValueError(f"Assets missing from price data: {missing}")

    df = df[assets].sort_index()
    df = df.loc[data["start"]:data["end"]]

    P = df.to_numpy(dtype=float)
    dates = list(df.index)
    return P, dates, assets


def save_processed(P: np.ndarray, dates, assets, path="data/processed/prices_clean.csv"):
    """Write the cleaned price matrix so the run is reproducible."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(P, index=pd.Index(dates, name="Date"), columns=assets).to_csv(out)
    return out
