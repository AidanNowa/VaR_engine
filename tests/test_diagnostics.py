"""Tests for the stylized-facts diagnostics."""

import numpy as np
import pandas as pd
import pytest

from src.diagnostics import (
    acf,
    average_pairwise_correlation,
    correlation_by_period,
    leverage_effect,
    normality_stats,
)


def _index(n, start="2005-01-03"):
    return pd.date_range(start, periods=n, freq="D")


# --------------------------------------------------------------------------
# acf
# --------------------------------------------------------------------------

def test_acf_index_and_length():
    s = pd.Series(np.random.default_rng(0).normal(size=500))
    a = acf(s, 10)
    assert len(a) == 10
    assert list(a.index) == list(range(1, 11))


def test_acf_of_alternating_series_is_exactly_plus_minus_one():
    """[1,-1,1,-1,...] has autocorrelation -1 at odd lags, +1 at even lags.

    Deterministic, so any off-by-one in the lag shift flips the signs.
    """
    s = pd.Series([1.0, -1.0] * 100)
    a = acf(s, 4)
    assert a[1] == pytest.approx(-1.0)
    assert a[2] == pytest.approx(1.0)
    assert a[3] == pytest.approx(-1.0)
    assert a[4] == pytest.approx(1.0)


def test_acf_recovers_ar1_coefficient():
    """An AR(1) with coefficient phi has ACF phi^k at lag k."""
    rng = np.random.default_rng(42)
    phi, n = 0.7, 40_000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    a = acf(pd.Series(x), 4)
    for k in range(1, 5):
        assert a[k] == pytest.approx(phi ** k, abs=0.03)


def test_acf_of_white_noise_is_within_significance_band():
    n = 20_000
    s = pd.Series(np.random.default_rng(7).normal(size=n))
    a = acf(s, 20)
    band = 1.96 / np.sqrt(n)
    assert (a.abs() < band * 1.5).all()


# --------------------------------------------------------------------------
# normality_stats
# --------------------------------------------------------------------------

def test_normality_stats_shape_and_orientation():
    """Tickers are ROWS, statistics are COLUMNS."""
    rng = np.random.default_rng(0)
    r = pd.DataFrame({"AAA": rng.normal(size=500), "BBB": rng.normal(size=500)},
                     index=_index(500))
    out = normality_stats(r)
    assert list(out.index) == ["AAA", "BBB"]
    for col in ["mean", "std", "skew", "excess_kurtosis", "jarque_bera"]:
        assert col in out.columns


def test_normal_data_has_near_zero_skew_and_excess_kurtosis():
    r = pd.DataFrame({"AAA": np.random.default_rng(3).normal(size=50_000)},
                     index=_index(50_000))
    out = normality_stats(r)
    assert out.loc["AAA", "skew"] == pytest.approx(0.0, abs=0.05)
    assert out.loc["AAA", "excess_kurtosis"] == pytest.approx(0.0, abs=0.10)


def test_fat_tails_are_detected():
    """Student-t with 3 df is heavily fat-tailed; a normal is not."""
    rng = np.random.default_rng(11)
    r = pd.DataFrame(
        {"fat": rng.standard_t(3, size=20_000), "thin": rng.normal(size=20_000)},
        index=_index(20_000),
    )
    out = normality_stats(r)
    assert out.loc["fat", "excess_kurtosis"] > 2.0
    assert out.loc["fat", "jarque_bera"] > out.loc["thin", "jarque_bera"]


def test_excess_kurtosis_convention_is_fisher():
    """A normal must give ~0, not ~3. Guards against fisher=False."""
    r = pd.DataFrame({"AAA": np.random.default_rng(5).normal(size=100_000)},
                     index=_index(100_000))
    assert abs(normality_stats(r).loc["AAA", "excess_kurtosis"]) < 0.5


# --------------------------------------------------------------------------
# average_pairwise_correlation
# --------------------------------------------------------------------------

def test_identity_matrix_averages_to_zero():
    corr = pd.DataFrame(np.eye(4), columns=list("ABCD"), index=list("ABCD"))
    assert average_pairwise_correlation(corr) == pytest.approx(0.0)


def test_all_ones_matrix_averages_to_one():
    corr = pd.DataFrame(np.ones((4, 4)), columns=list("ABCD"), index=list("ABCD"))
    assert average_pairwise_correlation(corr) == pytest.approx(1.0)


def test_known_three_by_three():
    """Off-diagonals 0.1, 0.2, 0.3 -> mean exactly 0.2.

    Including the diagonal would give 0.6, so this pins the k=1 offset.
    """
    corr = pd.DataFrame(
        [[1.0, 0.1, 0.2],
         [0.1, 1.0, 0.3],
         [0.2, 0.3, 1.0]],
        columns=list("ABC"), index=list("ABC"),
    )
    assert average_pairwise_correlation(corr) == pytest.approx(0.2)


# --------------------------------------------------------------------------
# correlation_by_period
# --------------------------------------------------------------------------

@pytest.fixture
def two_regime_returns():
    """Independent over the first half, perfectly correlated over the second."""
    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(size=2 * n)
    b = np.concatenate([rng.normal(size=n), a[n:]])
    return pd.DataFrame({"AAA": a, "BBB": b}, index=_index(2 * n))


def test_recovers_correlation_per_period(two_regime_returns):
    idx = two_regime_returns.index
    periods = {
        "independent": (str(idx[0].date()), str(idx[199].date())),
        "locked":      (str(idx[200].date()), str(idx[-1].date())),
    }
    out = correlation_by_period(two_regime_returns, periods)
    assert out["independent"].loc["AAA", "BBB"] == pytest.approx(0.0, abs=0.15)
    assert out["locked"].loc["AAA", "BBB"] == pytest.approx(1.0)


def test_returns_square_matrices_keyed_by_period_name(two_regime_returns):
    idx = two_regime_returns.index
    periods = {"all": (str(idx[0].date()), str(idx[-1].date()))}
    out = correlation_by_period(two_regime_returns, periods)
    assert set(out) == {"all"}
    assert out["all"].shape == (2, 2)
    assert list(out["all"].columns) == ["AAA", "BBB"]


def test_empty_period_raises(two_regime_returns):
    """A mistyped date must fail loudly, not return an all-NaN matrix."""
    periods = {"typo": ("2099-01-01", "2099-12-31")}
    with pytest.raises(ValueError):
        correlation_by_period(two_regime_returns, periods)


# --------------------------------------------------------------------------
# leverage_effect
# --------------------------------------------------------------------------

def _asymmetric_returns():
    """Big drops followed by turbulence; big rallies followed by calm.

    Constructed so the true answer is known: ratio must exceed 1.
    """
    rng = np.random.default_rng(0)
    out = []
    for block in range(20):
        quiet = list(rng.normal(0, 0.002, 8))
        if block % 2 == 0:
            out += [-0.06] + list(rng.normal(0, 0.04, 6)) + quiet   # drop -> wild
        else:
            out += [+0.06] + list(rng.normal(0, 0.002, 6)) + quiet  # rally -> calm
    return pd.Series(out, index=_index(len(out)))


def test_detects_asymmetry_when_it_is_constructed():
    out = leverage_effect(_asymmetric_returns(), horizon=5, quantile=0.05)
    assert out["ratio"] > 1.0
    assert out["down_vol"] > out["up_vol"]


def test_output_has_expected_fields():
    out = leverage_effect(_asymmetric_returns())
    for field in ["down_vol", "up_vol", "ratio", "n_down", "n_up",
                  "mean_abs_down", "mean_abs_up"]:
        assert field in out.index


def test_groups_are_not_pooled():
    """down_vol and up_vol must differ -- pooling with | would make them equal."""
    out = leverage_effect(_asymmetric_returns())
    assert out["down_vol"] != out["up_vol"]


def test_mean_abs_fields_measure_returns_not_volatility():
    """mean_abs_down must be the mean |return|, not a copy of down_vol."""
    s = _asymmetric_returns()
    out = leverage_effect(s, horizon=5, quantile=0.05)
    expected = s[s <= s.quantile(0.05)].abs().mean()
    assert out["mean_abs_down"] == pytest.approx(expected)
    assert out["mean_abs_down"] != pytest.approx(out["down_vol"])


def test_dataframe_input_raises():
    """A 10-column frame silently yields all-NaN without this guard."""
    r = pd.DataFrame({"AAA": [0.01, -0.01] * 50, "BBB": [0.02, -0.02] * 50},
                     index=_index(100))
    with pytest.raises(ValueError):
        leverage_effect(r)


def test_group_sizes_are_positive_and_balanced():
    out = leverage_effect(_asymmetric_returns(), quantile=0.05)
    assert out["n_down"] > 0 and out["n_up"] > 0
