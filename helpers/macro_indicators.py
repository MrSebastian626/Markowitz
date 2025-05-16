import pandas_datareader.data as web
import pandas as pd
from datetime import datetime

def fetch_macro_data(start="1980-01-01", end="2025-01-01"):
    """
    Fetches monthly macroeconomic indicators from FRED and aggregates to yearly values.

    Returns:
        DataFrame with yearly values for selected indicators.
    """
    indicators = {
        'CPI': 'CPIAUCSL',               # Consumer Price Index (monthly)
        'Unemployment': 'UNRATE',       # Unemployment rate (monthly)
        'GDP': 'GDP',                    # GDP (quarterly)
        'Interest_Rate': 'FEDFUNDS'     # Federal Funds Rate (monthly)
    }

    all_data = {}
    for name, fred_code in indicators.items():
        try:
            print(f"⬇️ Downloading {name} ({fred_code})...")
            series = web.DataReader(fred_code, 'fred', start, end)
            all_data[name] = series
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}")

    # Combine all series into a single DataFrame
    df = pd.concat(all_data.values(), axis=1)
    df.columns = all_data.keys()
    df = df.dropna()

    # Resample to yearly: use mean for rates and unemployment, sum for GDP
    aggregation = {
        'CPI': 'mean',
        'Unemployment': 'mean',
        'GDP': 'sum',  # Quarterly GDP → annual GDP
        'Interest_Rate': 'mean'
    }

    df_yearly = df.resample('Y').agg(aggregation)
    df_yearly.index = df_yearly.index.year  # Clean up index to show just year

    return df_yearly

if __name__ == "__main__":
    df_yearly = fetch_macro_data()
    df_yearly.to_csv("macro_indicators_yearly.csv")
    print("✅ Saved to macro_indicators_yearly.csv")
