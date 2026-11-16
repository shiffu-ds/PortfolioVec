"""PortfolioVec dashboard:  streamlit run app.py

Prices are downloaded live from Yahoo Finance. Nothing here is simulated.
"""
import datetime as dt
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.portfoliovec import compare, loader, returns, vectors

ROOT = Path(__file__).parent
COLOR_A, COLOR_B = "#2a78d6", "#eb6834"

st.set_page_config(page_title="PortfolioVec", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- data ----
@st.cache_data(ttl=3600, show_spinner="Downloading prices from Yahoo Finance...")
def fetch_prices(assets: tuple, start: str, end: str) -> pd.DataFrame:
    return loader.download_prices(list(assets), start, end)


def load_saved_prices(assets, start, end):
    """Previously downloaded real prices, used only if the live download fails."""
    path = ROOT / "data/raw/prices.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if not set(assets) <= set(df.columns):
        return None
    df = df.loc[start:end, list(assets)]
    return df if len(df) >= 2 else None


cfg = yaml.safe_load(open(ROOT / "config.yaml"))
cfg_assets = cfg["assets"]

# ------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.header("Scenario")
    tickers_text = st.text_input("Tickers (comma separated)", ", ".join(cfg_assets))
    start = st.date_input("Start", dt.date.fromisoformat(cfg["data"]["start"]))
    end = st.date_input("End", dt.date.fromisoformat(cfg["data"]["end"]),
                        max_value=dt.date.today())
    v0 = st.number_input("Initial capital ($)", min_value=100.0,
                         value=float(cfg["initial_capital"]), step=1000.0)
    if st.button("Refresh prices"):
        fetch_prices.clear()
    st.caption("Source: Yahoo Finance daily closing prices (adjusted), "
               "downloaded live and cached for one hour.")

assets = list(dict.fromkeys(t.strip().upper() for t in tickers_text.split(",") if t.strip()))

st.title("PortfolioVec")
st.caption("Portfolio value, trades and returns using only vectors and inner "
           "products (Boyd & Vandenberghe, Chapter 1).")

if len(assets) < 2:
    st.error("Enter at least two tickers.")
    st.stop()
if start >= end:
    st.error("Start date must be before end date.")
    st.stop()

source = "Yahoo Finance (live download)"
try:
    prices = fetch_prices(tuple(assets), start.isoformat(), end.isoformat())
except Exception as exc:  # network down, bad ticker, ...
    prices = load_saved_prices(assets, start.isoformat(), end.isoformat())
    if prices is None:
        st.error(f"Could not download prices from Yahoo Finance: {exc}")
        st.stop()
    source = "saved Yahoo Finance download (data/raw/prices.csv)"
    st.warning("Live download failed, so this uses the real prices saved "
               f"earlier in data/raw/prices.csv. Error: {exc}")

no_data = [a for a in assets if a not in prices.columns or prices[a].isna().all()]
if no_data:
    st.error(f"No price data for: {', '.join(no_data)}. Check the ticker symbols.")
    st.stop()

before = len(prices)
prices = prices[assets].dropna()
if len(prices) < 2:
    st.error("Fewer than two days of prices in this date range.")
    st.stop()
if len(prices) < before:
    st.info(f"Dropped {before - len(prices)} day(s) with missing prices.")

P = prices.to_numpy(dtype=float)
dates = prices.index
n = len(assets)
p = P[-1]

# ---------------------------------------------------- positions editor ----
defaults = pd.DataFrame({
    "Asset": assets,
    "Holdings (shares)": [cfg["holdings"][cfg_assets.index(a)] if a in cfg_assets else 10
                          for a in assets],
    "Trade (shares)": [cfg["trade"][cfg_assets.index(a)] if a in cfg_assets else 0
                       for a in assets],
    "Weight B": [round(1 / n, 4)] * n,
})
defaults.loc[n - 1, "Weight B"] = round(1 - defaults["Weight B"][:-1].sum(), 4)

st.subheader("Positions")
st.caption("Edit the table. Negative holdings = short position. "
           "Trade: buy (+) or sell (−). Weight B must sum to 1.")
edited = st.data_editor(
    defaults, hide_index=True, disabled=["Asset"], key="pos_" + "_".join(assets),
    column_config={
        "Holdings (shares)": st.column_config.NumberColumn(step=1),
        "Trade (shares)": st.column_config.NumberColumn(step=1),
        "Weight B": st.column_config.NumberColumn(format="%.4f", step=0.01),
    },
)
if edited.isna().any().any():
    st.error("Fill in every cell of the positions table.")
    st.stop()

s = edited["Holdings (shares)"].to_numpy(dtype=float)
b = edited["Trade (shares)"].to_numpy(dtype=float)
w_B = edited["Weight B"].to_numpy(dtype=float)

# ------------------------------------------------------ value and trade ----
v_before = vectors.portfolio_value(p, s)
s_new = vectors.apply_trade(s, b)
v_after = vectors.portfolio_value(p, s_new)
cash = vectors.trade_cash(p, b)

st.subheader(f"Value and trade at latest prices ({dates[-1]:%d %b %Y})")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Value before  (pᵀs)", f"${v_before:,.2f}")
c2.metric("Value after  (pᵀ(s+b))", f"${v_after:,.2f}", f"{v_after - v_before:+,.2f}")
c3.metric("Net cash of trade  (pᵀb)", f"${cash:,.2f}")
c4.metric("Check: after − before = pᵀb",
          "✓ holds" if abs((v_after - v_before) - cash) < 1e-6 else "✗ fails")

trade_df = pd.DataFrame({
    "Asset": assets, "Before": s, "Trade": b, "After": s_new, "Price": p,
    "Value before": p * s, "Value after": p * s_new,
})
st.dataframe(trade_df, hide_index=True, column_config={
    "Trade": st.column_config.NumberColumn(format="%+d"),
    "Price": st.column_config.NumberColumn(format="dollar"),
    "Value before": st.column_config.NumberColumn(format="dollar"),
    "Value after": st.column_config.NumberColumn(format="dollar"),
})

# ------------------------------------------------ allocation comparison ----
st.subheader("Allocation comparison")

if abs(w_B.sum() - 1) > 1e-6:
    st.error(f"Weight B sums to {w_B.sum():.4f}; it must sum to 1.")
    st.stop()
try:
    w_A = vectors.weights_from_holdings(p, s)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

name_A, name_B = "A: Current holdings", "B: Custom weights"
allocs = {"A": {"name": name_A, "weights": w_A},
          "B": {"name": name_B, "weights": w_B}}
R = returns.return_matrix(P)
results = compare.compare_allocations(R, allocs, v0)
best = compare.winner(results)
sA, sB = results["A"]["summary"], results["B"]["summary"]

st.caption(f"{len(R)} daily returns, {dates[0]:%d %b %Y} to {dates[-1]:%d %b %Y}. "
           "A uses the weights implied by your holdings at the latest prices; "
           "both are re-balanced to constant weights daily (ρ = R w).")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"A final value", f"${sA['final_value']:,.2f}", f"{sA['total_return']:+.2%}")
c2.metric(f"B final value", f"${sB['final_value']:,.2f}", f"{sB['total_return']:+.2%}")
c3.metric("Leverage A / B", f"{sA['leverage']:.2f} / {sB['leverage']:.2f}")
c4.metric("Better allocation", allocs[best]["name"].split(":")[0])

scale = alt.Scale(domain=[name_A, name_B], range=[COLOR_A, COLOR_B])
legend = alt.Legend(title=None, orient="top")

left, right = st.columns([3, 2])
with left:
    paths = pd.DataFrame({name_A: results["A"]["path"], name_B: results["B"]["path"]},
                         index=dates).rename_axis("Date").reset_index()
    long = paths.melt("Date", var_name="Allocation", value_name="Value")
    st.altair_chart(
        alt.Chart(long, title="Portfolio value over time", height=320).mark_line(strokeWidth=2).encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Value:Q", title="Value ($)", scale=alt.Scale(zero=False)),
            color=alt.Color("Allocation:N", scale=scale, legend=legend),
            tooltip=["Date:T", "Allocation:N", alt.Tooltip("Value:Q", format="$,.2f")],
        ), width="stretch")
with right:
    wdf = pd.DataFrame({"Asset": assets, name_A: w_A, name_B: w_B}).melt(
        "Asset", var_name="Allocation", value_name="Weight")
    st.altair_chart(
        alt.Chart(wdf, title="Weights (negative = short)", height=320).mark_bar().encode(
            x=alt.X("Asset:N", sort=assets, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Allocation:N", sort=[name_A, name_B]),
            y=alt.Y("Weight:Q", axis=alt.Axis(format="%")),
            color=alt.Color("Allocation:N", scale=scale, legend=legend),
            tooltip=["Asset:N", "Allocation:N", alt.Tooltip("Weight:Q", format="+.2%")],
        ), width="stretch")

rho = pd.DataFrame({name_A: results["A"]["rho"], name_B: results["B"]["rho"]},
                   index=dates[1:]).rename_axis("Date").reset_index()
rho_long = rho.melt("Date", var_name="Allocation", value_name="Return")
st.altair_chart(
    alt.Chart(rho_long, title="Daily portfolio returns", height=300).mark_bar().encode(
        x=alt.X("yearmonthdate(Date):O", title=None,
                axis=alt.Axis(format="%b %d", labelAngle=0, labelOverlap=True)),
        xOffset=alt.XOffset("Allocation:N", sort=[name_A, name_B]),
        y=alt.Y("Return:Q", axis=alt.Axis(format="%")),
        color=alt.Color("Allocation:N", scale=scale, legend=legend),
        tooltip=["Date:T", "Allocation:N", alt.Tooltip("Return:Q", format="+.2%")],
    ), width="stretch")

labels = [("avg_daily_return", "Avg daily return", "{:+.3%}"),
          ("best_day", "Best day", "{:+.2%}"),
          ("worst_day", "Worst day", "{:+.2%}"),
          ("final_value", "Final value", "${:,.2f}"),
          ("total_return", "Total return", "{:+.2%}"),
          ("leverage", "Leverage", "{:.2f}"),
          ("daily_risk_std", "Daily risk (std)", "{:.3%}")]
st.dataframe(pd.DataFrame({
    "Metric": [lab for _, lab, _ in labels],
    name_A: [f.format(sA[k]) for k, _, f in labels],
    name_B: [f.format(sB[k]) for k, _, f in labels],
}), hide_index=True)

# What did each short position contribute?
for key, alloc in allocs.items():
    for i in np.where(alloc["weights"] < 0)[0]:
        contrib = (R[:, i] * alloc["weights"][i]).sum()
        st.info(f"Short **{assets[i]}** in {key} (weight {alloc['weights'][i]:+.3f}): "
                f"the asset fell on {(R[:, i] < 0).sum()} of {len(R)} days, and the "
                f"short contributed **{contrib:+.2%}** in total (sum of daily r·w).")

# ------------------------------------------------------------ raw data ----
with st.expander(f"Price data: {source}"):
    st.caption(f"{len(prices)} trading days x {n} assets. Last close: {dates[-1]:%d %b %Y}.")
    norm = (prices / prices.iloc[0] * 100).rename_axis("Date").reset_index().melt(
        "Date", var_name="Asset", value_name="Indexed price")
    st.altair_chart(
        alt.Chart(norm, title="Prices indexed to 100 at the start", height=280).mark_line().encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Indexed price:Q", scale=alt.Scale(zero=False), title=None),
            color=alt.Color("Asset:N", sort=assets, legend=legend),
            tooltip=["Date:T", "Asset:N", alt.Tooltip("Indexed price:Q", format=".1f")],
        ), width="stretch")
    st.dataframe(prices.sort_index(ascending=False).style.format("{:,.2f}"))
    st.download_button("Download prices as CSV", prices.to_csv(), "prices.csv", "text/csv")
