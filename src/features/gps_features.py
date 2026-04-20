from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import GpsConfig

logger = logging.getLogger(__name__)

# UNAVCO "short" report CSVs have 8 metadata rows before the column header.
_GPS_CSV_HEADER_ROW = 8


# ── Public API ────────────────────────────────────────────────────────────────

def build_gps_dataframe(data_dir: str, cfg: GpsConfig) -> pd.DataFrame:
    """Read all per-station GPS CSVs and return a single wide DataFrame.

    Each CSV is expected to be in the UNAVCO "short" position format:
    8 metadata header rows, then a comma-separated table with columns
    including 'Datetime', 'delta_N' and 'delta_E' (leading whitespace
    stripped).

    Total horizontal displacement is computed as sqrt(delta_N² + delta_E²).
    This corrects the original study's naive sum (delta_N + delta_E) which
    is not a physically meaningful displacement magnitude.

    Args:
        data_dir: Directory containing per-station CSV files named {code}.csv.
        cfg: GPS configuration section from ProjectConfig.

    Returns:
        DataFrame with DatetimeIndex (study period), one column per station.
        Values are total horizontal displacement in mm. NaN where missing.
    """
    date_index = pd.date_range(cfg.start_date, cfg.end_date, freq="D")
    station_series: dict[str, pd.Series] = {}

    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if not csv_files:
        logger.warning("No GPS CSV files found in %s", data_dir)
        return pd.DataFrame(index=date_index)

    for filepath in csv_files:
        station_code = filepath.stem
        series = _load_station_csv(filepath, date_index, cfg)
        if series is not None:
            station_series[station_code] = series
            logger.info("Loaded GPS data for station %s (%d valid days)", station_code, series.notna().sum())

    df = pd.DataFrame(station_series, index=date_index)

    # Drop stations listed as excluded in config (deactivated / excessive gaps).
    cols_to_drop = [c for c in cfg.excluded_stations if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info("Dropped excluded GPS stations: %s", cols_to_drop)

    logger.info("GPS DataFrame built: %d stations, %d days", df.shape[1], df.shape[0])
    return df


def compute_displacement_rate(series: pd.Series, window_days: int) -> pd.Series:
    """Estimate fault displacement rate via a rolling linear regression.

    For each position in the series, fit a degree-1 polynomial to the
    preceding window_days values and return the slope. The slope represents
    the rate of change of horizontal GPS displacement (mm per time step),
    used as the machine-learning target variable.

    Bug fix vs. original rf_1.py: the original stored regression[:,1]
    (the intercept) instead of regression[:,0] (the slope). The slope is
    the physically meaningful quantity — it represents the displacement rate.

    Rows where fewer than window_days observations are available, or where
    the window contains NaN values, are set to NaN.

    Args:
        series: Daily GPS total horizontal displacement (mm).
        window_days: Size of the rolling regression window in days.

    Returns:
        Series of the same length and index containing displacement rates.
    """
    slopes = pd.Series(np.nan, index=series.index, name=series.name)
    x = np.arange(window_days, dtype=float)

    for i in range(window_days - 1, len(series)):
        window = series.iloc[i - window_days + 1 : i + 1]
        if window.isna().any():
            continue
        coeffs = np.polyfit(x, window.values, 1)
        slopes.iloc[i] = coeffs[0]  # slope; coeffs[1] is the intercept

    return slopes


def total_horizontal_displacement(
    delta_n: pd.Series, delta_e: pd.Series
) -> pd.Series:
    """Compute total horizontal displacement magnitude from N and E components.

    Uses the Euclidean norm: sqrt(delta_N² + delta_E²).

    Bug fix vs. original construct_the_matrix_3.py: the original used
    delta_N + delta_E (arithmetic sum), which is not a displacement magnitude
    and produces the wrong sign when components point in opposite directions.

    Args:
        delta_n: North component of displacement (mm).
        delta_e: East component of displacement (mm).

    Returns:
        Series of total horizontal displacement magnitudes (mm).
    """
    return np.sqrt(delta_n**2 + delta_e**2)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_station_csv(
    filepath: Path,
    date_index: pd.DatetimeIndex,
    cfg: GpsConfig,
) -> pd.Series | None:
    """Parse one UNAVCO position CSV and return a daily displacement Series.

    Returns None if the file cannot be parsed or contains no usable data.
    """
    try:
        df = pd.read_csv(filepath, header=_GPS_CSV_HEADER_ROW)
    except Exception as exc:
        logger.warning("Could not read %s: %s", filepath.name, exc)
        return None

    # Strip leading/trailing whitespace from all column names.
    # UNAVCO "short" format has columns like ' delta_N' — stripping gives 'delta_N'.
    df.columns = df.columns.str.strip()

    # Identify the date column (may be 'Datetime', 'Date', or 'YYMMMDD').
    date_col = _find_date_column(df)
    if date_col is None:
        logger.warning("No date column found in %s — skipping", filepath.name)
        return None

    # Identify north and east displacement columns.
    n_col = _find_column(df, ["delta_N", "dN", "N"])
    e_col = _find_column(df, ["delta_E", "dE", "E"])
    if n_col is None or e_col is None:
        logger.warning("Could not identify N/E columns in %s — skipping", filepath.name)
        return None

    try:
        df[date_col] = pd.to_datetime(df[date_col].str[:10], format="%Y-%m-%d", errors="coerce")
        df = df.dropna(subset=[date_col, n_col, e_col])
        df = df.set_index(date_col)
        df = df[~df.index.duplicated(keep="first")]

        # Filter to study period only.
        mask = (df.index >= cfg.start_date) & (df.index <= cfg.end_date)
        df = df.loc[mask]

        disp = total_horizontal_displacement(
            pd.to_numeric(df[n_col], errors="coerce"),
            pd.to_numeric(df[e_col], errors="coerce"),
        )
        # Reindex to the full study date range so all stations share the same index.
        return disp.reindex(date_index)

    except Exception as exc:
        logger.warning("Error processing %s: %s", filepath.name, exc)
        return None


def _find_date_column(df: pd.DataFrame) -> str | None:
    """Return the first column name matching known UNAVCO date column patterns."""
    candidates = ["Datetime", "Date", "YYMMMDD", "date", "datetime"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in the DataFrame."""
    for col in candidates:
        if col in df.columns:
            return col
    # Also check for columns containing the candidate as a substring.
    for col in candidates:
        matches = [c for c in df.columns if col in c]
        if matches:
            return matches[0]
    return None
