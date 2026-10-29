"""Loading, cleaning, return construction."""
import os
import logging
import pandas as pd
import yfinance as yf
import numpy as np
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "HD", "TLT", "EFA", "GLD"]
CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" #changing where data.py is will cause this to fail, update root level
logger = logging.getLogger(__name__)

def _cache_path(ticker_symbol, cache_dir) -> str:
    ticker_file = ticker_symbol + ".parquet"
    return os.path.join(cache_dir, ticker_file)

def _fetch_full_history(ticker_symbol) -> pd.Series:
    ticker = yf.Ticker(ticker_symbol)
    historical_data = ticker.history(period="max", auto_adjust=False)
    if historical_data.empty:
        raise ValueError("[Error]: Historical Data is empty!")
    return historical_data["Adj Close"].rename(ticker_symbol)

def _load_one(ticker_symbol, cache_dir, force_refresh) -> pd.Series:
    file_path = _cache_path(ticker_symbol, cache_dir)
    if os.path.exists(file_path) and not force_refresh:
        return pd.read_parquet(file_path)[ticker_symbol]
    historical_data = _fetch_full_history(ticker_symbol)
    #convert series to DF before saving to parquet
    df = historical_data.to_frame()
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(file_path)
    return historical_data


def load_prices(tickers, start, end=None, cache_dir=CACHE_PATH, force_refresh=False) -> pd.DataFrame:
    """#Adjusted closes. One colum per ticker, DatetimeIndex ascending"""
    prices = {}
    for ticker_symbol in tickers:
        historical_data = _load_one(ticker_symbol, cache_dir, force_refresh)
        prices[ticker_symbol] = historical_data
    prices = pd.DataFrame(prices)
    prices = prices.sort_index()[start:end]
    pre_drop_len = len(prices)
    prices = prices.dropna()
    dropped_rows = pre_drop_len - len(prices)
    logger.info(f"loaded {len(prices)} x {prices.shape[1]} tickers ({prices.index[0].date()} to {prices.index[-1].date()}); dropped {dropped_rows} of {pre_drop_len} incomplete rows {100 * dropped_rows / pre_drop_len:.2f}%")
    return prices

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if not prices.index.is_monotonic_increasing:
        raise ValueError("[ERROR] Dataframe is not monotonically increasing.")
    log_prices = prices.apply(np.log).diff()
    log_prices = log_prices[1:]
    return log_prices

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    prices = load_prices(TICKERS, start="2005-01-03")
    print(f"Shape: {prices.shape}")
    print(prices.head())
    print("------------------------------------")
    print(prices.tail())
    print("\n--------------Log Returns---------------")
    log_prices = log_returns(prices)
    print(f"Log Shape: {log_prices.shape}")
    print(log_prices.head())

    
