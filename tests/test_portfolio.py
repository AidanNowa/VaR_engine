"""Tests for weights, portfolio returns and portfolio value."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import make_weights, portfolio_returns, portfolio_values

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


# --------------------------------------------------------------------------
# make_weights
# --------------------------------------------------------------------------

def test_default_is_equal_weighted():
    w = make_weights(TICKERS)
    assert w.sum() == pytest.approx(1.0)
    assert len(w) == len(TICKERS)
    assert w.tolist() == pytest.approx([0.25] * 4)


def test_weights_are_indexed_by_ticker():
    """The index must be ticker names, not positions.

    This is what makes pandas align by name downstream. A positional array
    would silently misassign every weight the first time column order
    differed from the ticker list.
    """
    w = make_weights(TICKERS)
    assert list(w.index) == TICKERS


def test_explicit_weights_are_accepted():
    w = make_weights(TICKERS, [0.4, 0.3, 0.2, 0.1])
    assert w["AAA"] == pytest.approx(0.4)
    assert w["DDD"] == pytest.approx(0.1)


def test_weights_summing_to_one_within_float_error():
    """Sevenths do not sum to exactly 1.0 in float64."""
    tickers = [f"T{i}" for i in range(7)]
    w = make_weights(tickers, [1 / 7] * 7)
    assert w.sum() == pytest.approx(1.0)


def test_wrong_length_raises():
    with pytest.raises(ValueError):
        make_weights(TICKERS, [0.5, 0.5])


def test_negative_weight_raises():
    with pytest.raises(ValueError):
        make_weights(TICKERS, [0.8, 0.4, -0.2, 0.0])


def test_weights_not_summing_to_one_raises():
    with pytest.raises(ValueError):
        make_weights(TICKERS, [0.25, 0.25, 0.25, 0.10])


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def prices():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "AAA": [100.0, 102.0, 101.0, 105.0, 104.0, 108.0],
            "BBB": [50.0, 49.0, 50.5, 51.0, 50.0, 52.0],
        },
        index=idx,
    )


@pytest.fixture
def returns(prices):
    return np.log(prices).diff().iloc[1:]


# --------------------------------------------------------------------------
# portfolio_returns
# --------------------------------------------------------------------------

def test_single_asset_portfolio_equals_that_asset(returns):
    """Weight 1.0 on one asset reproduces its return series exactly."""
    w = pd.Series({"AAA": 1.0, "BBB": 0.0})
    pr = portfolio_returns(returns, w)
    assert pr.tolist() == pytest.approx(returns["AAA"].tolist())


def test_equal_weights_are_the_average(returns):
    w = pd.Series({"AAA": 0.5, "BBB": 0.5})
    pr = portfolio_returns(returns, w)
    expected = returns.mean(axis=1)
    assert pr.tolist() == pytest.approx(expected.tolist())


def test_column_order_does_not_matter(returns):
    """Alignment is by name, so reordering columns changes nothing.

    If this fails, weights are being applied positionally and you are
    holding the wrong asset in the wrong size.
    """
    w = pd.Series({"AAA": 0.7, "BBB": 0.3})
    normal = portfolio_returns(returns, w)
    swapped = portfolio_returns(returns[["BBB", "AAA"]], w)
    assert normal.tolist() == pytest.approx(swapped.tolist())


def test_one_row_per_date(returns):
    w = make_weights(["AAA", "BBB"])
    pr = portfolio_returns(returns, w)
    assert len(pr) == len(returns)
    assert pr.index.equals(returns.index)
    assert not pr.isna().any()


# --------------------------------------------------------------------------
# portfolio_values
# --------------------------------------------------------------------------

def test_first_value_equals_initial_capital(prices):
    """Day one is a tautology: you valued the book at its purchase prices."""
    w = make_weights(["AAA", "BBB"])
    v = portfolio_values(prices, w, 1_000_000)
    assert v.iloc[0] == pytest.approx(1_000_000)


def test_value_series_shape(prices):
    w = make_weights(["AAA", "BBB"])
    v = portfolio_values(prices, w, 1_000_000)
    assert len(v) == len(prices)
    assert v.index.equals(prices.index)


def test_single_asset_value_tracks_its_price(prices):
    """One asset at weight 1.0 grows exactly like that price series."""
    w = pd.Series({"AAA": 1.0, "BBB": 0.0})
    v = portfolio_values(prices, w, 1_000.0)
    expected = 1_000.0 * prices["AAA"] / prices["AAA"].iloc[0]
    assert v.tolist() == pytest.approx(expected.tolist())


def test_doubling_capital_doubles_value(prices):
    """The function is linear in initial capital."""
    w = make_weights(["AAA", "BBB"])
    a = portfolio_values(prices, w, 1_000_000)
    b = portfolio_values(prices, w, 2_000_000)
    assert (2 * a).tolist() == pytest.approx(b.tolist())


def test_diversified_vol_is_below_components(prices):
    """Portfolio vol must not exceed the highest component vol.

    Guards the sign of the covariance terms in w'Sigma w.
    """
    returns = np.log(prices).diff().iloc[1:]
    w = make_weights(["AAA", "BBB"])
    pr = portfolio_returns(returns, w)
    assert pr.std() <= returns.std().max() + 1e-12
