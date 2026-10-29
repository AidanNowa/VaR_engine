"""Tests for price loading and return construction."""

import numpy as np
import pandas as pd
import pytest

from src.data import load_prices, log_returns


def _prices(values, start="2020-01-01"):
    """Price frame with a daily index; values is {column: [prices]}."""
    n = len(next(iter(values.values())))
    return pd.DataFrame(values, index=pd.date_range(start, periods=n, freq="D"))


# --------------------------------------------------------------------------
# log_returns
# --------------------------------------------------------------------------

def test_doubling_gives_ln2():
    """A price that doubles has log return ln(2)."""
    r = log_returns(_prices({"A": [100.0, 200.0]}))
    assert r["A"].iloc[0] == pytest.approx(np.log(2))


def test_known_values_by_hand():
    """Three hand-computed returns, including a decline and a flat day."""
    r = log_returns(_prices({"A": [100.0, 110.0, 99.0, 99.0]}))
    expected = [np.log(1.10), np.log(0.90), 0.0]
    assert r["A"].tolist() == pytest.approx(expected)


def test_drops_first_row():
    prices = _prices({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    r = log_returns(prices)
    assert len(r) == len(prices) - 1
    assert r.index[0] == prices.index[1]


def test_no_nans_and_columns_preserved():
    prices = _prices({"A": [10.0, 11.0, 12.0], "B": [50.0, 49.0, 51.0]})
    r = log_returns(prices)
    assert not r.isna().any().any()
    assert list(r.columns) == ["A", "B"]


def test_unsorted_index_raises():
    prices = _prices({"A": [100.0, 110.0, 120.0]}).iloc[::-1]
    with pytest.raises(ValueError):
        log_returns(prices)


def test_log_returns_are_additive_across_time():
    """Sum of daily log returns equals the total log return over the period.

    This is the property that justified choosing log returns over simple
    returns, and it fails under almost any indexing mistake.
    """
    prices = _prices({"A": [100.0, 103.0, 99.5, 107.25, 104.0]})
    r = log_returns(prices)
    total = np.log(prices["A"].iloc[-1] / prices["A"].iloc[0])
    assert r["A"].sum() == pytest.approx(total)


def test_no_lookahead():
    """A return at date T must not change when a FUTURE price changes.

    Perturb the price at row 5 and every return strictly before that date
    must be bit-identical. If this ever fails, information is leaking
    backwards in time and every downstream model will look better than it is.
    """
    prices = _prices({"A": [100.0, 101.0, 99.0, 103.0, 102.0,
                            105.0, 104.0, 108.0, 107.0, 110.0]})
    before = log_returns(prices)

    perturbed = prices.copy()
    perturbed.iloc[5, 0] *= 1.5          # change the future
    after = log_returns(perturbed)

    cutoff = prices.index[4]             # last date unaffected by row 5
    pd.testing.assert_frame_equal(before.loc[:cutoff], after.loc[:cutoff])


# --------------------------------------------------------------------------
# load_prices — no network, _fetch_full_history is stubbed
# --------------------------------------------------------------------------

@pytest.fixture
def fake_fetch(monkeypatch):
    """Replace the only network-touching function and count its calls."""
    calls = []

    def _stub(ticker_symbol):
        calls.append(ticker_symbol)
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        return pd.Series(
            np.linspace(100.0, 109.0, 10), index=idx, name=ticker_symbol
        )

    monkeypatch.setattr("src.data._fetch_full_history", _stub)
    return calls


def test_first_call_fetches_and_writes_cache(tmp_path, fake_fetch):
    df = load_prices(["AAA", "BBB"], "2020-01-01", cache_dir=tmp_path)
    assert fake_fetch == ["AAA", "BBB"]
    assert (tmp_path / "AAA.parquet").exists()
    assert (tmp_path / "BBB.parquet").exists()
    assert list(df.columns) == ["AAA", "BBB"]
    assert len(df) == 10


def test_second_call_hits_cache(tmp_path, fake_fetch):
    load_prices(["AAA"], "2020-01-01", cache_dir=tmp_path)
    load_prices(["AAA"], "2020-01-01", cache_dir=tmp_path)
    assert fake_fetch == ["AAA"]          # fetched once, not twice


def test_force_refresh_refetches(tmp_path, fake_fetch):
    load_prices(["AAA"], "2020-01-01", cache_dir=tmp_path)
    load_prices(["AAA"], "2020-01-01", cache_dir=tmp_path, force_refresh=True)
    assert fake_fetch == ["AAA", "AAA"]


def test_cache_round_trip_preserves_values(tmp_path, fake_fetch):
    fresh = load_prices(["AAA"], "2020-01-01", cache_dir=tmp_path)
    cached = load_prices(["AAA"], "2020-01-01", cache_dir=tmp_path)
    assert np.allclose(fresh, cached)
    assert fresh.index.equals(cached.index)


def test_start_and_end_are_respected(tmp_path, fake_fetch):
    df = load_prices(["AAA"], "2020-01-03", end="2020-01-06",
                     cache_dir=tmp_path)
    assert df.index[0] == pd.Timestamp("2020-01-03")
    assert df.index[-1] == pd.Timestamp("2020-01-06")
    assert len(df) == 4


def test_incomplete_rows_are_dropped(tmp_path, monkeypatch):
    """A date one ticker never printed is dropped for all tickers."""
    idx = pd.date_range("2020-01-01", periods=5, freq="D")

    def _stub(ticker_symbol):
        s = pd.Series(np.arange(100.0, 105.0), index=idx, name=ticker_symbol)
        if ticker_symbol == "BBB":
            s = s.drop(idx[2])            # BBB missing one day
        return s

    monkeypatch.setattr("src.data._fetch_full_history", _stub)
    df = load_prices(["AAA", "BBB"], "2020-01-01", cache_dir=tmp_path)
    assert len(df) == 4
    assert idx[2] not in df.index
    assert not df.isna().any().any()


def test_returned_index_is_sorted(tmp_path, fake_fetch):
    df = load_prices(["AAA", "BBB"], "2020-01-01", cache_dir=tmp_path)
    assert df.index.is_monotonic_increasing
