import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from data_utils import download_data_in_batches
from markowitz import classical_markowitz
from robust import RobustMarkowitz
from factor import FactorModelMarkowitz
from resample import resampled_efficient_frontier
from cvar import CvarRobustMarkowitz
import os
import numpy as np
from curl_cffi import requests
  
session = requests.Session(impersonate="chrome")

from requests.cookies import create_cookie
import yfinance.data as _data

def _wrap_cookie(cookie, session):
    """
    If cookie is just a str (cookie name), look up its value
    in session.cookies and wrap it into a real Cookie object.
    """
    if isinstance(cookie, str):
        value = session.cookies.get(cookie)
        return create_cookie(name=cookie, value=value)
    return cookie

def patch_yfdata_cookie_basic():
    """
    Monkey-patch YfData._get_cookie_basic so that
    it always returns a proper Cookie object,
    even when response.cookies is a simple dict.
    """
    original = _data.YfData._get_cookie_basic

    def _patched(self, proxy=None, timeout=30):
        cookie = original(self, proxy, timeout)
        return _wrap_cookie(cookie, self._session)

    _data.YfData._get_cookie_basic = _patched

patch_yfdata_cookie_basic()

FULL_SMP_CACHE = "pickles/sp500_full_2000_2025.pkl"

if os.path.exists(FULL_SMP_CACHE):
    print("✅ Using cached full S&P 500 data...")
    sp500_full = pd.read_pickle(FULL_SMP_CACHE)
else:
    print("⬇️ Downloading full S&P 500 data...")
    sp500_full = yf.download('^GSPC', start="2000-01-01", end="2025-01-01", session=session)['Close']
    sp500_full = sp500_full.pct_change().dropna()
    pd.to_pickle(sp500_full, FULL_SMP_CACHE)
    print(f"✅ Cached to {FULL_SMP_CACHE}")
# -----------------------------
# SETTINGS
# -----------------------------
start_date = "1985-01-01"
end_date = "2015-01-01"
lookback_years = 20
strategy = "cvarResample"  # Options: "classic", "robust", "factor". "resampled", "cvar", "cvarResample"
macro_beta_path = "stock_macro_betas_90.csv"
macro_indicator_path = "macro_indicators_yearly.csv"

# -----------------------------
# LOAD NYSE TICKERS (Top 1000)
# -----------------------------
tickers_df = pd.read_csv("nyse-listed.csv")
tickers = tickers_df["ACT Symbol"].dropna().unique().tolist()

# -----------------------------
# DOWNLOAD HISTORICAL DATA
# -----------------------------
CACHE_PATH = "pickles/cached_price_data_90.pkl"

if os.path.exists(CACHE_PATH):
    print("✅ Using cached price data...")
    price_data = pd.read_pickle(CACHE_PATH)
else:
    print("⬇️ Downloading fresh data...")
    price_data = download_data_in_batches(tickers, start_date=start_date, end_date=end_date, batch_size=50)
    price_data.to_pickle(CACHE_PATH)
    print(f"✅ Data cached to {CACHE_PATH}")

price_data = price_data.dropna(axis=1, how='any')
returns = price_data.pct_change(fill_method=None).dropna()

# -----------------------------
# ROLLING BACKTEST
# -----------------------------
years = sorted(set(returns.index.year))
years = [y for y in years if y >= 1990 + lookback_years and y < 2015]

cumulative_returns = pd.Series(index=returns.index, dtype=float)
portfolio_value = 1.0
value_tracker = []
weight_history = {}

for year in years:
    train_start = f"{year - lookback_years}-01-01"
    train_end = f"{year - 1}-12-31"
    test_start = f"{year}-01-01"
    test_end = f"{year}-12-31"

    train_returns = returns.loc[train_start:train_end]
    test_returns = returns.loc[test_start:test_end]
    mu = (1 + test_returns).prod() - 1
    avg = mu.median()
    high = mu.max()
    low = mu.min()

    print(f"\n🧪 {year} - Return Stats:")
    print(f"  Median: {avg:.4f}, Max: {high:.4f}, Min: {low:.4f}")
    if test_returns.empty:
        continue

    try:
        if strategy == "classic":
            weights, _ = classical_markowitz(train_returns, target_return=0.05)
        elif strategy == "robust":
            rm = RobustMarkowitz(train_returns)
            weights, _ = rm.solve_box_uncertainty(target_return=0.15, delta_fraction=0.3)
        elif strategy == "factor":
            fm = FactorModelMarkowitz(
                train_returns,
                betas_path=macro_beta_path,
                macro_path=macro_indicator_path,
                current_year=year,
                gamma=.003,
                top_n=200
            )
            weights, _ = fm.solve_box_uncertainty(target_return=0.15, delta_fraction=0.3)
        elif strategy == "resampled":
            weights, _ = resampled_efficient_frontier(
            RobustMarkowitz,  
            train_returns,
            n_samples=75,
            target_return=0.15,
            delta_fraction=0.1,
            top_n=200
            )
        elif strategy == "cvar":
            rm = CvarRobustMarkowitz(train_returns)
            weights, _ = rm.solve_box_uncertainty(alpha=0.90, delta_fraction=0.1, target_return=0.20)
        elif strategy == "cvarResample":
            weights, _ = resampled_efficient_frontier(
            CvarRobustMarkowitz,  
            train_returns,
            n_samples=25,
            target_return=0.20,
            delta_fraction=0.05,
            top_n=200
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        weight_history[year] = weights
        test_subset = test_returns[weights.index]
        period_returns = test_subset @ weights

    except Exception as e:
        print(f"Optimization failed for year {year}: {e}")
        continue

    period_cum = (1 + period_returns).cumprod() * portfolio_value
    portfolio_value = period_cum.iloc[-1]
    value_tracker.append(period_cum)

# -----------------------------
# BUILD CUMULATIVE RETURN CURVE
# -----------------------------
backtest_result = pd.concat(value_tracker)
backtest_result.name = 'Optimized Portfolio'

# -----------------------------
# LOAD S&P 500 FOR BENCHMARK
# -----------------------------
sp500 = sp500_full.loc[backtest_result.index.min():backtest_result.index.max()]
sp500_cum = (1 + sp500).cumprod()
sp500_cum *= backtest_result.iloc[0]
sp500_cum.name = 'S&P 500'

# -----------------------------
# PLOT PERFORMANCE
# -----------------------------
comparison_df = pd.concat([backtest_result, sp500_cum], axis=1)
comparison_df.plot(title="20-Year Rolling Portfolio vs S&P 500", ylabel="Cumulative Return")
plt.grid(True)
plt.show()

# -----------------------------
# SAVE YEARLY PORTFOLIO WEIGHTS
# -----------------------------
weights_df = pd.DataFrame(weight_history).T
weights_df.to_csv("helpers/portfolio_weights_by_year.csv")
