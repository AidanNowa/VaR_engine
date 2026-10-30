"""Tests for rolling volatility."""

import numpy as np
import pandas as pd
import pytest

from src.volatility import rolling_volatility

ANNUALIZATION = np.sqrt(252)


def _returns(values, start="2020-01-01"):
    n = len(next(iter(values.values())))
    return pd.DataFrame(values, index=pd.date_range(start, periods=n, freq="D"))


def test_constant_returns_have_zero_volatility():
    """A series that never varies has no volatility."""
    r = _returns({"A": [0.01] * 30})
    v = rolling_volatility(r, window=10)
    assert v["A"].dropna().eq(0.0).all()


def test_leading_nans_equal_window_minus_one():
    r = _returns({"A": np.linspace(0.0, 0.05, 40)})
    v = rolling_volatility(r, window=10)
    assert v["A"].isna().sum() == 9
    assert v.index.equals(r.index)


def test_annualize_is_sqrt_252_times_raw():
    r = _returns({"A": np.random.default_rng(0).normal(0, 0.01, 60)})
    raw = rolling_volatility(r, window=20, annualize=False)
    ann = rolling_volatility(r, window=20, annualize=True)
    assert np.allclose(ann.dropna(), raw.dropna() * ANNUALIZATION)


def test_matches_numpy_on_a_known_window():
    """Independent check against numpy's sample std over the same slice."""
    vals = [0.01, -0.02, 0.015, -0.005, 0.03, -0.01, 0.02, 0.0]
    r = _returns({"A": vals})
    v = rolling_volatility(r, window=4, annualize=False)
    # the window ending at position 3 covers positions 0..3
    expected = np.std(vals[0:4], ddof=1)
    assert v["A"].iloc[3] == pytest.approx(expected)


def test_window_is_respected():
    r = _returns({"A": np.random.default_rng(1).normal(0, 0.01, 50)})
    short = rolling_volatility(r, window=5)
    long = rolling_volatility(r, window=25)
    assert short["A"].isna().sum() == 4
    assert long["A"].isna().sum() == 24


def test_no_lookahead():
    """Volatility at date T must not move when a FUTURE return changes.

    rolling(window) looks strictly backwards, so perturbing a later return
    cannot touch an earlier estimate. This is the invariant that has to hold
    before this function feeds a VaR forecast.
    """
    r = _returns({"A": list(np.random.default_rng(2).normal(0, 0.01, 40))})
    before = rolling_volatility(r, window=10)

    perturbed = r.copy()
    perturbed.iloc[30, 0] = 0.5          # change the future
    after = rolling_volatility(perturbed, window=10)

    cutoff = r.index[29]
    pd.testing.assert_frame_equal(before.loc[:cutoff], after.loc[:cutoff])
