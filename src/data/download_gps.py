from __future__ import annotations

import csv
import logging
from pathlib import Path

import requests

from src.utils.config import GpsConfig

logger = logging.getLogger(__name__)


# ── Station discovery ─────────────────────────────────────────────────────────

def get_gps_station_list(cfg: GpsConfig, output_dir: str) -> list[str]:
    """Query the UNAVCO metadata API for GPS stations inside the bounding box.

    The bounding box is derived symmetrically from (center_lat, center_lon)
    ± search_radius_deg, matching the original study's ±1° selection.

    Args:
        cfg: GPS configuration section from ProjectConfig.
        output_dir: Directory where GPS_Stations.csv will be written.

    Returns:
        Sorted, deduplicated list of four-character station code strings.
    """
    url = _build_metadata_url(cfg)
    logger.info("Querying UNAVCO station metadata: %s", url)

    response = requests.get(url, headers={"accept": "application/json"}, timeout=60)
    response.raise_for_status()

    data = response.json()

    # Each element in the JSON array is a dict; the station code is its first value.
    # UNAVCO metadata v1 does not guarantee a consistent key name across API versions,
    # so we fall back to reading the first value (same approach as the original script).
    station_names: list[str] = []
    for entry in data:
        values = list(entry.values())
        if values:
            station_names.append(str(values[0]))

    station_names = sorted(set(station_names))
    logger.info("Found %d GPS stations: %s", len(station_names), station_names)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "GPS_Stations.csv"
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerow(station_names)
    logger.info("GPS station list saved to %s", out_path)

    return station_names


# ── Data download ─────────────────────────────────────────────────────────────

def download_gps_data(
    station_list: list[str],
    cfg: GpsConfig,
    output_dir: str,
) -> None:
    """Download daily GPS position CSVs from UNAVCO for each station.

    Files are saved as:  {output_dir}/{station_code}.csv
    Already-downloaded files are skipped (idempotent).

    The position data uses the NAM14 reference frame (stable North America
    interior) and is preprocessed by the USGS Earthquake Hazards Program.

    Args:
        station_list: List of four-character UNAVCO station codes.
        cfg: GPS configuration section from ProjectConfig.
        output_dir: Directory where per-station CSV files will be written.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    downloaded = skipped = errors = 0

    for sta_code in station_list:
        out_path = Path(output_dir) / f"{sta_code}.csv"

        if out_path.exists():
            skipped += 1
            continue

        url = _build_position_url(sta_code, cfg)

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to download GPS data for %s: %s", sta_code, exc)
            errors += 1
            continue

        # Write the response line-by-line as CSV, preserving the original
        # UNAVCO header metadata (first 8 rows) used in feature construction.
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            for line in response.iter_lines():
                writer.writerow(line.decode("utf-8").split(","))

        downloaded += 1
        logger.info("Downloaded GPS data: %s", sta_code)

    logger.info(
        "GPS download complete — downloaded: %d, skipped: %d, errors: %d",
        downloaded,
        skipped,
        errors,
    )


# ── URL builders ──────────────────────────────────────────────────────────────

def _build_metadata_url(cfg: GpsConfig) -> str:
    """Build the UNAVCO metadata query URL from config bounding box."""
    r = cfg.search_radius_deg
    return (
        f"{cfg.metadata_url}?"
        f"minlatitude={cfg.center_lat - r}&"
        f"maxlatitude={cfg.center_lat + r}&"
        f"minlongitude={cfg.center_lon - r}&"
        f"maxlongitude={cfg.center_lon + r}&"
        f"starttime=&endtime=&summary=false"
    )


def _build_position_url(sta_code: str, cfg: GpsConfig) -> str:
    """Build the UNAVCO position data URL for one station."""
    return (
        f"{cfg.position_url}/{sta_code}/v3?"
        f"analysisCenter={cfg.analysis_center}&"
        f"referenceFrame={cfg.reference_frame}&"
        f"starttime={cfg.start_date}&"
        f"endtime={cfg.end_date}&"
        f"report=short&"
        f"dataPostProcessing=Uncleaned&"
        f"refCoordOption=from_analysis_center"
    )
