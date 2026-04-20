from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.utils.config import ModelConfig, ProjectConfig, SeismicConfig, load_config

# ── Minimal valid dicts for unit-testing Pydantic validation ─────────────────

_VALID_SEISMIC = {
    "station_url": "http://example.com/station",
    "dataselect_url": "http://example.com/data",
    "center_lat": 48.9,
    "center_lon": -123.9,
    "max_radius_deg": 0.6,
    "channel": "HHE",
    "start_date": "2010-01-01",
    "end_date": "2020-09-30",
    "window_hours": 1,
    "freq_bands": [[8, 9], [9, 10]],
    "statistics": ["kurtosis", "variance"],
    "rolling_window_days": 60,
}

_VALID_MODEL = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "max_features": 1.0,
    "random_state": 42,
    "train_size": 0.8,
}


# ── Integration: load the real config ────────────────────────────────────────

def test_load_config_returns_project_config():
    cfg = load_config()
    assert isinstance(cfg, ProjectConfig)


def test_seismic_section_types():
    cfg = load_config()
    assert isinstance(cfg.seismic.center_lat, float)
    assert isinstance(cfg.seismic.center_lon, float)
    assert isinstance(cfg.seismic.freq_bands, list)
    assert all(len(b) == 2 for b in cfg.seismic.freq_bands)
    assert isinstance(cfg.seismic.rolling_window_days, int)


def test_model_section_train_size_in_range():
    cfg = load_config()
    assert 0 < cfg.model.train_size < 1


def test_model_section_n_estimators_positive():
    cfg = load_config()
    assert cfg.model.n_estimators > 0


def test_paths_are_absolute_after_load():
    cfg = load_config()
    assert Path(cfg.paths.data_dir).is_absolute()
    assert Path(cfg.paths.figures_dir).is_absolute()
    assert Path(cfg.paths.models_dir).is_absolute()


def test_default_config_has_five_freq_bands():
    cfg = load_config()
    assert len(cfg.seismic.freq_bands) == 5


def test_default_config_has_four_statistics():
    cfg = load_config()
    assert len(cfg.seismic.statistics) == 4


# ── Unit: Pydantic field validators ──────────────────────────────────────────

def test_reversed_freq_band_raises():
    bad = {**_VALID_SEISMIC, "freq_bands": [[10, 8]]}  # low >= high
    with pytest.raises(ValidationError, match="low < high"):
        SeismicConfig(**bad)


def test_equal_freq_band_bounds_raises():
    bad = {**_VALID_SEISMIC, "freq_bands": [[9, 9]]}  # low == high
    with pytest.raises(ValidationError, match="low < high"):
        SeismicConfig(**bad)


def test_unknown_statistic_raises():
    bad = {**_VALID_SEISMIC, "statistics": ["mean"]}  # not in allowed set
    with pytest.raises(ValidationError, match="Unknown statistics"):
        SeismicConfig(**bad)


def test_train_size_above_one_raises():
    bad = {**_VALID_MODEL, "train_size": 1.5}
    with pytest.raises(ValidationError):
        ModelConfig(**bad)


def test_train_size_zero_raises():
    bad = {**_VALID_MODEL, "train_size": 0.0}
    with pytest.raises(ValidationError):
        ModelConfig(**bad)


def test_n_estimators_zero_raises():
    bad = {**_VALID_MODEL, "n_estimators": 0}
    with pytest.raises(ValidationError):
        ModelConfig(**bad)


def test_rolling_window_zero_raises():
    bad = {**_VALID_SEISMIC, "rolling_window_days": 0}
    with pytest.raises(ValidationError):
        SeismicConfig(**bad)


def test_max_radius_zero_raises():
    bad = {**_VALID_SEISMIC, "max_radius_deg": 0.0}
    with pytest.raises(ValidationError):
        SeismicConfig(**bad)
