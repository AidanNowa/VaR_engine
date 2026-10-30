import logging
import pandas as pd
import numpy as np
from scipy import stats
from src.data import load_prices, log_returns
from src.portfolio import make_weights, portfolio_returns

TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "HD", "TLT", "EFA", "GLD"]

logger = logging.getLogger(__name__)

def acf(series, max_lag) -> pd.Series:
    corr_series = {}
    for k in range(1, max_lag + 1):
        corr_series[k] = series.autocorr(lag=k)
    return pd.Series(corr_series)

def normality_stats(returns) -> pd.DataFrame:
    df_stats = {}
    for ticker in returns.columns:
        jarque_bera, _ = stats.jarque_bera(returns[ticker])
        df_stats[ticker] = {
            "mean":            returns[ticker].mean(),
            "std":             returns[ticker].std(),
            "skew":            returns[ticker].skew(),
            "excess_kurtosis": returns[ticker].kurt(),
            "jarque_bera":     jarque_bera,
        }
    return pd.DataFrame.from_dict(df_stats,orient="index")

def correlation_by_period(returns, periods) -> dict[str, pd.DataFrame]:
    '''period is a dict of named date ranges'''
    period_corrs = {}
    for period, (start, end) in periods.items():
        sliced_values = returns.loc[start:end]
        if sliced_values.empty:
            raise ValueError(f"[ERROR] Slice '{period}' is empty!")
        corr = sliced_values.corr()
        period_corrs[period] = corr
        logger.info(f"Total observations in {period} period: {len(sliced_values)}")

    return period_corrs

def average_pairwise_correlation(corr) -> float :
    rows, cols = np.triu_indices(len(corr), k=1)
    return corr.values[rows, cols].mean()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    weights = make_weights(TICKERS)
    prices = load_prices(TICKERS, start="2005-01-03")
    returns = log_returns(prices)
    port_returns = portfolio_returns(returns, weights) 
    acf_vals = {
        "raw": acf(port_returns, 50), #except all in band
        "squared": acf(port_returns ** 2, 50), #expect strong slowly decaying positives
        "abs": acf(port_returns.abs(), 50) #expect same as above but cleaner
    }
    print(pd.DataFrame(acf_vals))
    print(returns)
    print(normality_stats(returns))
    periods = {
                "calm": ("2013-01-01", "2019-12-31"),
                "gfc":  ("2008-09-01", "2009-03-31"),
                "covid": ("2020-02-19", "2020-04-30"),
                "rates_2022": ("2022-01-01", "2022-12-31")
            }

    corr = correlation_by_period(returns, periods)
    print(average_pairwise_correlation(corr["calm"]))
    print(average_pairwise_correlation(corr["covid"]))


