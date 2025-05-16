import yfinance as yf
import pandas as pd
import time

def get_price_data(tickers, start="2018-01-01", end="2023-01-01"):
    data = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True, progress=False, threads=False)

    # Extract adjusted close prices
    if len(tickers) == 1:
        prices = data['Close']  # Single-ticker format is flat
    else:
        prices = pd.concat([data[ticker]['Close'] for ticker in tickers], axis=1)
        prices.columns = tickers

    returns = prices.pct_change(fill_method=None).dropna()
    returns = returns.dropna(axis=1, how='any')  # Drop any asset with missing data
    return returns


def download_data_in_batches(tickers, start_date, end_date, batch_size=50):
    all_data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Downloading batch {i // batch_size + 1}")
        try:
            data = yf.download(batch, start=start_date, end=end_date, group_by='ticker', auto_adjust=True, threads=True)
            for ticker in batch:
                if ticker in data.columns.levels[0]:
                    all_data[ticker] = data[ticker]['Close']
        except Exception as e:
            print(f"Error downloading batch {i // batch_size + 1}: {e}")
        time.sleep(1)  # Pause to respect API rate limits
    return pd.DataFrame(all_data)