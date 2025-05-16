# estimate_betas.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# -----------------------------
# SETTINGS
# -----------------------------
PICKLE_PATH = "pickles/cached_price_data_90.pkl"
MACRO_PATH = "macro_indicators_yearly.csv"
OUTPUT_CSV = "stock_macro_betas_90.csv"
lookback_years = 20

# -----------------------------
# LOAD PRICE DATA
# -----------------------------
if not os.path.exists(PICKLE_PATH):
    raise FileNotFoundError(f"Pickle file not found at {PICKLE_PATH}")
price_data = pd.read_pickle(PICKLE_PATH).dropna(axis=1, how='any')

# -----------------------------
# COMPUTE DAILY RETURNS
# -----------------------------
returns = price_data.pct_change().dropna()
years = sorted(set(returns.index.year))
years = [y for y in years if y >= 1990 + lookback_years and y < 2015]

# -----------------------------
# COMPUTE YEARLY CUMULATIVE RETURNS
# -----------------------------
stock_returns = []
for year in years:
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    yearly_returns = returns.loc[start:end]
    if yearly_returns.empty:
        continue
    cumulative = (1 + yearly_returns).prod() - 1
    cumulative.name = year
    stock_returns.append(cumulative)

stock_returns_df = pd.DataFrame(stock_returns)
stock_returns_df.index.name = "Year"

# -----------------------------
# LOAD MACRO DATA
# -----------------------------
macro_df = pd.read_csv(MACRO_PATH)
macro_df = macro_df.set_index("Year")
macro_df.index = macro_df.index.astype(int)

# Shift macro data by one year to simulate known-at-decision-time data
macro_df = macro_df.shift(1)

# -----------------------------
# ALIGN AND REGRESS
# -----------------------------
intersecting_years = stock_returns_df.index.intersection(macro_df.index)
X = macro_df.loc[intersecting_years]
Y = stock_returns_df.loc[intersecting_years]

betas = pd.DataFrame(index=stock_returns_df.columns, columns=macro_df.columns)

for stock in Y.columns:
    y = Y[stock]
    if y.isna().sum() > 0:
        continue
    model = LinearRegression()
    model.fit(X, y)
    betas.loc[stock] = model.coef_

# -----------------------------
# SAVE BETAS
# -----------------------------
betas.to_csv(OUTPUT_CSV)
print(f"✅ Saved beta estimates to {OUTPUT_CSV}")
