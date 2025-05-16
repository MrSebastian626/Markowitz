import pandas as pd

def get_top_5_stocks_per_year(csv_path="portfolio_weights_by_year.csv"):
    """
    Parses a CSV of portfolio weights by year and returns the top 5 stocks with their weights for each year.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        dict: A dictionary where keys are years and values are lists of (ticker, weight) tuples sorted by weight.
    """
    weights_df = pd.read_csv(csv_path, index_col=0)
    top_stocks_by_year = {}

    for year, row in weights_df.iterrows():
        top_5 = row.sort_values(ascending=False).head(5)
        top_stocks_by_year[int(year)] = list(top_5.items())

    return top_stocks_by_year

top_stocks = get_top_5_stocks_per_year("helpers/portfolio_weights_by_year.csv")
for year, stocks in top_stocks.items():
    print(f"{year}:")
    for ticker, weight in stocks:
        print(f"  {ticker}: {weight:.4f}")