"""Rolling, EWMA, and hand-rolled GARCH(1,1) volatility."""
import math
import scipy
import pandas as pd
import numpy as np

from src.data import load_prices, log_returns
from src.portfolio import make_weights, portfolio_returns

TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "HD", "TLT", "EFA", "GLD"]


def rolling_volatility(returns, window=250, annualize=True) -> pd.DataFrame:
    '''Must be shifted (.shift(1)) when forecasting to avoid lookahead'''
    rolling = returns.rolling(window).std()
    if annualize:
        rolling = rolling * math.sqrt(252)
    return rolling

def volatility_persistence(returns, window=21):
    if not isinstance(returns, pd.Series):
        raise ValueError("[ERROR] Inputted returns must be a single series!")    
    rolling_vol = rolling_volatility(returns, window=window)
    rolling_vol = rolling_vol.dropna()
    log_rolling_vol = np.log(rolling_vol)
    pairs = pd.DataFrame({
        "y": log_rolling_vol,
        "x": log_rolling_vol.shift(1)
    }).dropna()

    slope, intercept, rval, pval, stderr = scipy.stats.linregress(pairs["x"], pairs["y"])
    if slope > 0 and slope < 1:
        half_life_days = np.log(0.5) / np.log(slope)
    else:
        half_life_days = float("inf")
    long_run_vol = math.exp(intercept / (1 - slope))
    return pd.Series({
        "slope": slope, 
        "intercept": intercept, 
        "half_life_days": half_life_days,
        "r_squared": rval**2, 
        "long_run_vol": long_run_vol, 
        "n":len(pairs), 
        "window": window
    })

if __name__ == "__main__":
    weights = make_weights(TICKERS)
    prices = load_prices(TICKERS, start="2005-01-03")
    returns = log_returns(prices)
    print(returns)
    port_returns = portfolio_returns(returns, weights) 
    print("----------Rolling Volatility----------")
    print(rolling_volatility(returns))
    print("----------Volatility Persistence----------")
    print(volatility_persistence(port_returns))