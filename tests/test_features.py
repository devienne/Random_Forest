from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.gps_features import compute_displacement_rate, total_horizontal_displacement
from src.features.seismic_features import _build_column_names, _column_name, smooth_features


# ── total_horizontal_displacement ─────────────────────────────────────────────

def test_displacement_3_4_5_triangle():
    """sqrt(3² + 4²) = 5.0 — basic Euclidean norm."""
    result = total_horizontal_displacement(pd.Series([3.0]), pd.Series([4.0]))
    assert pytest.approx(result.iloc[0]) == 5.0


def test_displacement_not_naive_sum():
    """Confirms the fix: sqrt(3²+4²) = 5 ≠ 3+4 = 7 (original bug)."""
    result = total_horizontal_displacement(pd.Series([3.0]), pd.Series([4.0]))
    assert result.iloc[0] != pytest.approx(7.0)


def test_displacement_zero_components():
    result = total_horizontal_displacement(pd.Series([0.0]), pd.Series([0.0]))
    assert pytest.approx(result.iloc[0]) == 0.0


def test_displacement_negative_components():
    """Magnitude is always non-negative regardless of component signs."""
    result = total_horizontal_displacement(pd.Series([-3.0]), pd.Series([-4.0]))
    assert pytest.approx(result.iloc[0]) == 5.0


def test_displacement_series_length_preserved():
    n = pd.Series([1.0, 2.0, 3.0])
    e = pd.Series([4.0, 5.0, 6.0])
    result = total_horizontal_displacement(n, e)
    assert len(result) == 3


# ── compute_displacement_rate ─────────────────────────────────────────────────

def _linear_series(slope: float, n: int, window: int) -> pd.Series:
    """Return a perfectly linear Series y = slope * i."""
    return pd.Series(
        [slope * i for i in range(n)],
        index=pd.date_range("2010-01-01", periods=n, freq="D"),
    )


def test_displacement_rate_linear_slope_2():
    """Rolling linear regression on y=2i should return slope=2.0."""
    series = _linear_series(slope=2.0, n=100, window=10)
    rates = compute_displacement_rate(series, window_days=10)
    valid = rates.dropna()
    assert len(valid) > 0
    assert pytest.approx(valid.iloc[0], abs=1e-8) == 2.0


def test_displacement_rate_returns_slope_not_intercept():
    """Slope of y=3i is 3; intercept is 0. They must not be confused.

    This directly guards the original rf_1.py bug where regression[:,1]
    (intercept) was used instead of regression[:,0] (slope).
    """
    series = _linear_series(slope=3.0, n=80, window=10)
    rates = compute_displacement_rate(series, window_days=10)
    valid = rates.dropna()
    assert pytest.approx(valid.mean(), abs=1e-6) == 3.0
    assert valid.mean() != pytest.approx(0.0)  # 0.0 would be the intercept


def test_displacement_rate_nans_before_first_full_window():
    """Rows before the first complete window must be NaN."""
    window = 15
    series = _linear_series(slope=1.0, n=50, window=window)
    rates = compute_displacement_rate(series, window_days=window)
    assert rates.iloc[: window - 1].isna().all()


def test_displacement_rate_nan_in_window_produces_nan():
    """A NaN anywhere in a window should produce NaN for that position."""
    window = 5
    values = [float(i) for i in range(20)]
    values[7] = float("nan")
    series = pd.Series(values, index=pd.date_range("2010-01-01", periods=20, freq="D"))
    rates = compute_displacement_rate(series, window_days=window)
    # Positions 7 through 7 + window - 1 should all be NaN.
    assert rates.iloc[7:7 + window].isna().all()


# ── Column naming ─────────────────────────────────────────────────────────────

def test_column_name_format():
    assert _column_name("kurtosis", "NLLB", [8, 9]) == "kurtosis_NLLB_Band8-9Hz"
    assert _column_name("variance", "ALBH", [12, 13]) == "variance_ALBH_Band12-13Hz"
    assert _column_name("skewness", "P064", [10, 11]) == "skewness_P064_Band10-11Hz"


def test_build_column_names_total_count():
    """n_stations × n_bands × n_stats columns must be produced."""

    class _FakeCfg:
        freq_bands = [[8, 9], [9, 10], [10, 11]]
        statistics = ["kurtosis", "variance", "skewness", "value_range"]

    stations = ["NLLB", "ALBH"]
    cols = _build_column_names(stations, _FakeCfg())
    expected = len(stations) * len(_FakeCfg.freq_bands) * len(_FakeCfg.statistics)
    assert len(cols) == expected  # 2 × 3 × 4 = 24


def test_build_column_names_no_duplicates():
    class _FakeCfg:
        freq_bands = [[8, 9], [9, 10]]
        statistics = ["kurtosis", "variance"]

    cols = _build_column_names(["NLLB", "ALBH"], _FakeCfg())
    assert len(cols) == len(set(cols))


def test_build_column_names_station_in_every_column():
    class _FakeCfg:
        freq_bands = [[8, 9]]
        statistics = ["kurtosis"]

    cols = _build_column_names(["NLLB"], _FakeCfg())
    assert all("NLLB" in c for c in cols)


# ── smooth_features ───────────────────────────────────────────────────────────

def test_smooth_features_nans_before_first_window():
    window = 10
    df = pd.DataFrame(
        {"a": np.ones(50), "b": np.ones(50)},
        index=pd.date_range("2010-01-01", periods=50, freq="D"),
    )
    smoothed = smooth_features(df, window)
    assert smoothed.iloc[: window - 1].isna().all().all()


def test_smooth_features_constant_series():
    """Rolling mean of a constant series equals that constant."""
    window = 5
    df = pd.DataFrame(
        {"x": np.full(30, 4.0)},
        index=pd.date_range("2010-01-01", periods=30, freq="D"),
    )
    smoothed = smooth_features(df, window).dropna()
    assert (smoothed["x"] == pytest.approx(4.0)).all()


def test_smooth_features_preserves_shape():
    df = pd.DataFrame(
        np.random.randn(100, 5),
        index=pd.date_range("2010-01-01", periods=100, freq="D"),
    )
    smoothed = smooth_features(df, window_days=10)
    assert smoothed.shape == df.shape
