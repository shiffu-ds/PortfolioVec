"""Download real closing prices from Yahoo Finance into data/raw/prices.csv.

Run from the project root:  python scripts/download_prices.py
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.portfoliovec.loader import download_prices  # noqa: E402

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
df = download_prices(cfg["assets"], cfg["data"]["start"], cfg["data"]["end"])

out = ROOT / cfg["data"]["path"]
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out)
print(f"Saved {df.shape[0]} days x {df.shape[1]} assets to {out}")
print(df.tail(3))
