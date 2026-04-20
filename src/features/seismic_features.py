from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import read as obspy_read
from scipy.stats import kurtosis, skew

from src.utils.config import SeismicConfig

logger = logging.getLogger(__name__)

# Maps config statistic names to their computation functions.
# np.ptp is deprecated in NumPy 2.0+; value_range uses explicit max-min instead.
_STAT_FUNCS: dict[str, callable] = {
    "kurtosis": kurtosis,
    "variance": np.var,
    "value_range": lambda x: float(np.max(x) - np.min(x)),
    "skewness": skew,
}


# ── Public API ────────────────────────────────────────────────────────────────

def build_seismic_feature_matrix(
    station_list: list[str],
    data_dir: str,
    cfg: SeismicConfig,
) -> pd.DataFrame:
    """Build the seismic feature matrix from downloaded miniSEED files.

    For each daily file: detrend (linear) + demean, then bandpass-filter
    into each frequency band and compute the configured statistics.
    Result is a DataFrame indexed by date with shape (n_days, n_features),
    where n_features = n_stations × n_bands × n_statistics.

    NaN rows indicate days with no usable file for that station/band.

    Args:
        station_list: List of seismic station codes to include.
        data_dir: Directory containing .mseed files named {station}_{date}.mseed.
        cfg: Seismic configuration section from ProjectConfig.

    Returns:
        DataFrame with DatetimeIndex and one column per (stat, station, band).
    """
    columns = _build_column_names(station_list, cfg)
    date_index = pd.date_range(cfg.start_date, cfg.end_date, freq="D")
    df = pd.DataFrame(np.nan, index=date_index, columns=columns)

    mseed_files = sorted(Path(data_dir).glob("*.mseed"))
    total = len(mseed_files)
    logger.info("Processing %d miniSEED files from %s", total, data_dir)

    for n, filepath in enumerate(mseed_files, start=1):
        try:
            stream = obspy_read(str(filepath))
        except Exception as exc:
            logger.debug("Could not read %s: %s", filepath.name, exc)
            continue

        stream.detrend("linear")
        stream.detrend("demean")

        for trace in stream:
            # Bug fix: original hardcoded 'BHE' here; channel now comes from config.
            if trace.stats.channel != cfg.channel:
                continue

            station = trace.stats.station
            if station not in station_list:
                continue

            date = pd.Timestamp(str(trace.stats.starttime)[:10])
            if date not in df.index:
                continue

            band_stats = _compute_trace_band_features(trace, cfg)

            for stat in cfg.statistics:
                for band in cfg.freq_bands:
                    col = _column_name(stat, station, band)
                    key = f"{stat}_{band[0]}-{band[1]}"
                    df.loc[date, col] = band_stats.get(key, np.nan)

        if n % 1000 == 0:
            logger.info("Processed %d / %d files", n, total)

    logger.info("Feature matrix complete: shape %s", df.shape)
    return df


def smooth_features(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Apply a rolling mean to all feature columns.

    Rows where fewer than window_days observations are available are NaN
    (min_periods=window_days). This matches the 60-day rolling mean from
    the original study.

    Args:
        df: Raw feature DataFrame from build_seismic_feature_matrix.
        window_days: Rolling window size in days.

    Returns:
        Smoothed DataFrame with the same shape and index.
    """
    return df.rolling(window=window_days, min_periods=window_days).mean()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_trace_band_features(
    trace,  # obspy.Trace — typed as Any to avoid hard obspy import in type hints
    cfg: SeismicConfig,
) -> dict[str, float]:
    """Bandpass-filter a Trace into each configured band and compute statistics.

    The full obspy Trace object is required (not raw numpy data) because the
    bandpass filter needs the trace's sampling rate metadata.

    Returns:
        Flat dict keyed as "{stat}_{fmin}-{fmax}" for every (stat, band) pair.
    """
    results: dict[str, float] = {}

    for band in cfg.freq_bands:
        fmin, fmax = band
        filtered = trace.copy()
        filtered.filter(type="bandpass", freqmin=fmin, freqmax=fmax)

        key_prefix = f"{fmin}-{fmax}"
        for stat_name in cfg.statistics:
            func = _STAT_FUNCS[stat_name]
            results[f"{stat_name}_{key_prefix}"] = float(func(filtered.data))

    return results


def _build_column_names(station_list: list[str], cfg: SeismicConfig) -> list[str]:
    """Generate the full ordered list of feature column names.

    Order: station → band → statistic. Produces names like:
        kurtosis_NLLB_Band8-9Hz, variance_NLLB_Band8-9Hz, ...
    """
    cols = []
    for station in station_list:
        for band in cfg.freq_bands:
            for stat in cfg.statistics:
                cols.append(_column_name(stat, station, band))
    return cols


def _column_name(stat: str, station: str, band: list[int]) -> str:
    return f"{stat}_{station}_Band{band[0]}-{band[1]}Hz"
