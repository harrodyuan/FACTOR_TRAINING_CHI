# %% [markdown]
#
# # Part 1: Predicting Returns from Historical Data
# ## Feature Engineering with Price and Fundamental Ratios (US Stocks)
#
# Financial Data Science II
# Updated March 10, 2026
#
# This lecture uses US equity data from Yahoo Finance (via yfinance).
# We build a simple feature set (returns, volatility, momentum, and
# fundamental ratios like P/E and P/B when available) and predict
# future returns.
#
# Notes:
# - Fundamental ratios are available only at low frequency (quarterly).
# - We forward-fill fundamentals between report dates.
# - This is a teaching example; production pipelines need point-in-time
#   data and survivorship-bias controls.
#

# %%
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# %% [markdown]
# ## 1) Download US stock data
#
# Choose a ticker and a date range.
#

# %%
TICKER = "AAPL"
START = "2016-01-01"
END = "2026-03-01"

px = yf.download(TICKER, start=START, end=END, auto_adjust=False, progress=False)
px = px.rename(columns={"Adj Close": "adj_close"})
px = px.dropna()
px.head()


# %% [markdown]
# ## 2) Build price-based features
#
# Returns, momentum, volatility, and drawdown features.
#

# %%
feat = pd.DataFrame(index=px.index)
feat["ret_1d"] = px["adj_close"].pct_change()
feat["log_ret_1d"] = np.log(px["adj_close"] / px["adj_close"].shift(1))

for w in [5, 21, 63, 126]:
    feat[f"mom_{w}d"] = px["adj_close"].pct_change(w)
    feat[f"vol_{w}d"] = feat["ret_1d"].rolling(w).std()

rolling_max = px["adj_close"].cummax()
feat["drawdown"] = px["adj_close"] / rolling_max - 1

feat = feat.dropna()
feat.head()


# %% [markdown]
# ## 3) Add fundamental ratios (P/E and P/B)
#
# We compute quarterly EPS and book value per share from financial statements,
# then forward-fill between report dates. If statements are missing, the
# features will be NaN and can be dropped.
#

# %%

def _get_quarterly_series(ticker: yf.Ticker, attr: str) -> pd.DataFrame:
    obj = getattr(ticker, attr, None)
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj
    return pd.DataFrame()


tk = yf.Ticker(TICKER)

income_q = _get_quarterly_series(tk, "quarterly_income_stmt")
if income_q.empty:
    income_q = _get_quarterly_series(tk, "income_stmt")

balance_q = _get_quarterly_series(tk, "quarterly_balance_sheet")
if balance_q.empty:
    balance_q = _get_quarterly_series(tk, "balance_sheet")

shares = None
try:
    shares = tk.get_shares_full()
except Exception:
    try:
        shares = tk.get_shares()
    except Exception:
        shares = None

# Build EPS and book value per share series
fund = pd.DataFrame(index=px.index)

if not income_q.empty and shares is not None and len(shares) > 0:
    # Use Net Income (quarterly) and shares outstanding
    income_q = income_q.T
    income_q.index = pd.to_datetime(income_q.index)
    net_income_col = None
    for c in income_q.columns:
        if "Net Income" in c:
            net_income_col = c
            break
    if net_income_col:
        eps_q = income_q[net_income_col] / shares.reindex(income_q.index, method="nearest")
        eps_q = eps_q.replace([np.inf, -np.inf], np.nan)
        fund["eps_q"] = eps_q.reindex(fund.index, method="ffill")

if not balance_q.empty and shares is not None and len(shares) > 0:
    balance_q = balance_q.T
    balance_q.index = pd.to_datetime(balance_q.index)
    equity_col = None
    for c in balance_q.columns:
        if "Total Stockholder Equity" in c or "Total Equity" in c:
            equity_col = c
            break
    if equity_col:
        bvps_q = balance_q[equity_col] / shares.reindex(balance_q.index, method="nearest")
        bvps_q = bvps_q.replace([np.inf, -np.inf], np.nan)
        fund["bvps_q"] = bvps_q.reindex(fund.index, method="ffill")

# Ratios
fund["pe"] = px["adj_close"].reindex(fund.index) / fund["eps_q"]
fund["pb"] = px["adj_close"].reindex(fund.index) / fund["bvps_q"]

# Merge fundamentals into feature set
features = feat.join(fund[["pe", "pb"]], how="left")
features = features.dropna()
features.head()


# %% [markdown]
# ## 4) Define prediction target
#
# Predict 5-day forward return.
#

# %%
TARGET_H = 5
features["target_fwd_ret"] = px["adj_close"].reindex(features.index).shift(-TARGET_H) / px["adj_close"].reindex(features.index) - 1
features = features.dropna()

X = features.drop(columns=["target_fwd_ret"])
y = features["target_fwd_ret"]


# %% [markdown]
# ## 5) Walk-forward evaluation (time series split)
#
# We use a Random Forest as a baseline model.
#

# %%
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=5,
)

splitter = TimeSeriesSplit(n_splits=5)
mses = []

for train_idx, test_idx in splitter.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mses.append(mean_squared_error(y_test, pred))

print("MSE mean:", np.mean(mses))
print("MSE std:", np.std(mses))


# %% [markdown]
# ## 6) Summary
#
# - Price-based features are always available and stable.
# - Fundamental ratios are lower frequency and must be aligned carefully.
# - Walk-forward validation is required to avoid data leakage.
# - This baseline can be extended with macro or alternative data.
#
