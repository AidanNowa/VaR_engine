"""Weights, P&L, position value."""
import pandas as pd 
from src.data import load_prices, log_returns

TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "HD", "TLT", "EFA", "GLD"]


def make_weights(tickers, weights=None) -> pd.Series:
    if weights is None:
        weight = 1/len(tickers)
        return pd.Series(weight, index=tickers)

    weights = pd.Series(weights, index=tickers)

    if len(weights) != len(tickers):
        raise ValueError("[ERROR] Lengths of tickers and weights do not match!")

    if (weights<0).sum() > 0:
        raise ValueError("[ERROR] Negative weight entered!")
    
    weights_sum = weights.sum()
    print(weights_sum)
    if weights_sum > 0.9999 and weights_sum < 1.0001:
        return weights
    raise ValueError("[ERROR] Inputted weights do not total to 1!")

def portfolio_returns(returns, weights) -> pd.Series:
    daily_returns = returns * weights
    daily_returns = daily_returns.sum(axis=1)
    return daily_returns

def portfolio_values(prices, weights, initial_capital) -> pd.Series:
    t_o = initial_capital * weights
    share_counts = t_o / prices.iloc[0]
    position_values_per_day = prices * share_counts
    portfolio_values_per_day = position_values_per_day.sum(axis=1)
    return portfolio_values_per_day
    


if __name__ == "__main__":
    weights = make_weights(TICKERS)
    print(weights)
    prices = load_prices(TICKERS, start="2005-01-03")
    print("----------Initial Prices----------")
    print(prices)
    returns = log_returns(prices)
    print("----------Portfolio Returns----------")
    port_returns = portfolio_returns(returns, weights)
    print(port_returns)
    print("----------Returns Check----------")
    print("Max Dates")
    print(returns.idxmax())
    print("Min Dates")
    print(returns.idxmin())
    # print(pd.DataFrame({
    #     "worst":      returns.min().round(4),
    #     "worst_date": returns.idxmin().date,
    #     "best":       returns.max().round(4),
    #     "best_date":  returns.idxmax().date,
    #     "ann_vol":    (returns.std() * (252 ** 0.5)).round(3),
    # }))
    
    print("----------Portfolio Values----------")
    print(portfolio_values(prices, weights, 1000000))
