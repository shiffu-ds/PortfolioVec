"""Output layer: charts saved to outputs/figures/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, no window needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path("outputs/figures")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _finish(fig, ax, name):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def value_over_time(dates, results):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (k, res), c in zip(results.items(), COLORS):
        ax.plot(dates, res["path"], color=c, lw=2, label=f"{k}: {res['name']}")
    ax.set_title("Portfolio value over time")
    ax.set_ylabel("Value ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(frameon=False)
    return _finish(fig, ax, "value_over_time.png")


def daily_returns(dates, results):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(dates))
    width = 0.8 / len(results)
    for j, ((k, res), c) in enumerate(zip(results.items(), COLORS)):
        ax.bar(x + (j - (len(results) - 1) / 2) * width, res["rho"] * 100,
               width, color=c, label=f"{k}: {res['name']}")
    ax.axhline(0, color="black", lw=0.8)
    step = max(1, len(dates) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([d.strftime("%b %d") for d in dates[::step]])
    ax.set_title("Daily portfolio returns")
    ax.set_ylabel("Return (%)")
    ax.legend(frameon=False)
    return _finish(fig, ax, "daily_returns.png")


def weights_bar(assets, allocs):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(assets))
    width = 0.8 / len(allocs)
    for j, ((k, alloc), c) in enumerate(zip(allocs.items(), COLORS)):
        ax.bar(x + (j - (len(allocs) - 1) / 2) * width, alloc["weights"],
               width, color=c, label=f"{k}: {alloc['name']}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(assets)
    ax.set_title("Asset weights (negative = short position)")
    ax.set_ylabel("Weight")
    ax.legend(frameon=False)
    return _finish(fig, ax, "weights.png")
