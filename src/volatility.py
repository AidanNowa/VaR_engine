"""Rolling, EWMA, and hand-rolled GARCH(1,1) volatility."""
import math
import pandas as pd

from src.data import load_prices, log_returns
from src.portfolio import make_weights, portfolio_returns

TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "HD", "TLT", "EFA", "GLD"]


def rolling_volatility(returns, window=250, annualize=True) -> pd.DataFrame:
    '''Must be shifted (.shift(1)) when forecasting to avoid lookahead'''
    rolling = returns.rolling(window).std()
    if annualize:
        rolling = rolling * math.sqrt(252)
    return rolling


if __name__ == "__main__":
    weights = make_weights(TICKERS)
    prices = load_prices(TICKERS, start="2005-01-03")
    returns = log_returns(prices)
    print(returns)
    # port_returns = portfolio_returns(returns, weights)
    print("----------Rolling Volatility----------")
    print(rolling_volatility(returns))