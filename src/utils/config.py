from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()

# ── Sub-models ────────────────────────────────────────────────────────────────

class SeismicConfig(BaseModel):
    station_url: str
    dataselect_url: str
    center_lat: float
    center_lon: float
    max_radius_deg: float = Field(gt=0)
    channel: str
    start_date: str
    end_date: str
    window_hours: int = Field(gt=0)
    freq_bands: List[List[int]]
    statistics: List[str]
    rolling_window_days: int = Field(gt=0)

    @field_validator("freq_bands")
    @classmethod
    def bands_must_be_pairs(cls, v: List[List[int]]) -> List[List[int]]:
        for band in v:
            if len(band) != 2 or band[0] >= band[1]:
                raise ValueError(f"Each freq_band must be [low, high] with low < high, got {band}")
        return v

    @field_validator("statistics")
    @classmethod
    def statistics_must_be_known(cls, v: List[str]) -> List[str]:
        allowed = {"kurtosis", "variance", "value_range", "skewness"}
        unknown = set(v) - allowed
        if unknown:
            raise ValueError(f"Unknown statistics: {unknown}. Allowed: {allowed}")
        return v


class GpsConfig(BaseModel):
    metadata_url: str
    position_url: str
    center_lat: float
    center_lon: float
    search_radius_deg: float = Field(gt=0)
    reference_frame: str
    analysis_center: str
    start_date: str
    end_date: str
    excluded_stations: List[str] = Field(default_factory=list)
    rolling_window_days: int = Field(gt=0)


class ModelConfig(BaseModel):
    n_estimators: int = Field(gt=0)
    max_depth: Optional[int] = None
    min_samples_split: int = Field(ge=2)
    max_features: float = Field(gt=0, le=1.0)
    random_state: int
    train_size: float = Field(gt=0, lt=1)


class PathsConfig(BaseModel):
    data_dir: str
    raw_seismic_dir: str
    raw_gps_dir: str
    interim_dir: str
    processed_dir: str
    artifacts_dir: str
    figures_dir: str
    models_dir: str

    def resolve(self, root: Path) -> "PathsConfig":
        """Return a copy with all paths resolved relative to root."""
        return PathsConfig(
            **{k: str(root / v) for k, v in self.model_dump().items()}
        )


# ── Root config ───────────────────────────────────────────────────────────────

class ProjectConfig(BaseModel):
    seismic: SeismicConfig
    gps: GpsConfig
    model: ModelConfig
    paths: PathsConfig


# ── Loader ────────────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None, resolve_paths: bool = True) -> ProjectConfig:
    """Load and validate the project configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file. Defaults to
            configs/default.yaml relative to the repository root.
        resolve_paths: If True, all paths in [paths] are resolved
            relative to the repository root (parent of this file's package).

    Returns:
        Validated ProjectConfig instance.
    """
    if config_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        config_path = str(repo_root / "configs" / "default.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Allow environment variables to override API URLs
    if url := os.getenv("IRIS_STATION_URL"):
        raw["seismic"]["station_url"] = url
    if url := os.getenv("IRIS_DATASELECT_URL"):
        raw["seismic"]["dataselect_url"] = url
    if url := os.getenv("UNAVCO_METADATA_URL"):
        raw["gps"]["metadata_url"] = url
    if url := os.getenv("UNAVCO_POSITION_URL"):
        raw["gps"]["position_url"] = url
    if data_dir := os.getenv("DATA_DIR"):
        for key in raw["paths"]:
            raw["paths"][key] = raw["paths"][key].replace("data", data_dir, 1)

    cfg = ProjectConfig(**raw)

    if resolve_paths:
        repo_root = Path(config_path).resolve().parents[1]
        cfg = ProjectConfig(
            seismic=cfg.seismic,
            gps=cfg.gps,
            model=cfg.model,
            paths=cfg.paths.resolve(repo_root),
        )

    return cfg
