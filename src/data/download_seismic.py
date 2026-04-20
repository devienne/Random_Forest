from __future__ import annotations

import csv
import logging
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from lxml import objectify

from src.utils.config import SeismicConfig

logger = logging.getLogger(__name__)


# ── Station discovery ─────────────────────────────────────────────────────────

def get_station_list(cfg: SeismicConfig, output_dir: str) -> list[str]:
    """Query IRIS FDSN for HHE-channel stations near the target coordinates.

    Filters to stations that were active before the study start date and have
    no recorded end date (i.e. still active as of query time). Saves the
    resulting station codes to a CSV in output_dir for reproducibility.

    Args:
        cfg: Seismic configuration section from ProjectConfig.
        output_dir: Directory where Stations.csv will be written.

    Returns:
        Sorted list of station code strings.
    """
    url = (
        f"{cfg.station_url}?"
        f"cha={cfg.channel}"
        f"&starttime={cfg.start_date}"
        f"&endtime={cfg.end_date}"
        f"&level=station"
        f"&format=xml"
        f"&lat={cfg.center_lat}"
        f"&lon={cfg.center_lon}"
        f"&minradius=0.0"
        f"&maxradius={cfg.max_radius_deg}"
        f"&includecomments=true"
        f"&nodata=404"
    )

    logger.info("Querying IRIS station metadata: %s", url)
    response = requests.get(url, headers={"accept": "application/xml"}, timeout=60)
    response.raise_for_status()

    root = objectify.parse(BytesIO(response.content)).getroot()

    raw_attrs: list[dict] = []
    for network in root.getchildren():
        for station in network.getchildren():
            if station.attrib:
                raw_attrs.append(dict(station.attrib))

    raw_attrs = [a for a in raw_attrs if a]

    df = pd.DataFrame(raw_attrs)
    df = df.dropna(subset=["code"]).reset_index(drop=True)

    # Parse start dates; keep only stations active before the study start.
    df["startDate"] = pd.to_datetime(df["startDate"].str[:10])
    study_start = pd.Timestamp(cfg.start_date)

    # endDate is NaN for stations still active — keep only those.
    df = df[df["endDate"].isna()].reset_index(drop=True)
    df = df[df["startDate"] < study_start].reset_index(drop=True)

    stations = sorted(df["code"].tolist())
    logger.info("Found %d qualifying stations: %s", len(stations), stations)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "Stations.csv"
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerow(stations)
    logger.info("Station list saved to %s", out_path)

    return stations


# ── Data download ─────────────────────────────────────────────────────────────

def download_seismic_data(
    station_list: list[str],
    cfg: SeismicConfig,
    output_dir: str,
) -> None:
    """Download one-hour miniSEED files from IRIS for each station × day.

    Files are saved as:  {output_dir}/{station_code}_{YYYY-MM-DD}.mseed
    Already-downloaded files are skipped (idempotent).

    Args:
        station_list: List of IRIS station codes to download.
        cfg: Seismic configuration section from ProjectConfig.
        output_dir: Directory where .mseed files will be written.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    date_range = pd.date_range(cfg.start_date, cfg.end_date)
    total = len(station_list) * len(date_range)
    downloaded = skipped = errors = 0

    for sta_code in station_list:
        for date in date_range:
            # Bug fix: zero-pad month and day unconditionally (original only
            # zero-padded inside the if-branch, leaving day undefined otherwise).
            date_str = date.strftime("%Y-%m-%d")
            filename = f"{sta_code}_{date_str}.mseed"
            filepath = Path(output_dir) / filename

            if filepath.exists():
                skipped += 1
                continue

            url = _build_dataselect_url(sta_code, date, cfg)

            try:
                response = requests.get(url, allow_redirects=True, timeout=30)
            except requests.RequestException as exc:
                logger.warning("Request failed for %s on %s: %s", sta_code, date_str, exc)
                errors += 1
                continue

            # IRIS returns a plain-text error message (not HTTP 4xx) for missing data.
            if response.status_code != 200 or response.text[:5] == "Error":
                errors += 1
                continue

            filepath.write_bytes(response.content)
            downloaded += 1

            if downloaded % 500 == 0:
                logger.info(
                    "Progress: %d/%d — downloaded %d, skipped %d, errors %d",
                    downloaded + skipped + errors,
                    total,
                    downloaded,
                    skipped,
                    errors,
                )

    logger.info(
        "Download complete — downloaded: %d, skipped: %d, errors: %d",
        downloaded,
        skipped,
        errors,
    )


def _build_dataselect_url(sta_code: str, date: pd.Timestamp, cfg: SeismicConfig) -> str:
    """Build an IRIS FDSN dataselect URL for one station on one day.

    Bug fixes vs. original download_seismic_3.py:
    - sta_code is never mutated (original shadowed 'station' with the URL fragment).
    - endtime is derived from date + window_hours (original undefined 'endtime' variable).
    """
    date_str = date.strftime("%Y-%m-%d")
    end_date = date + pd.Timedelta(hours=cfg.window_hours)
    end_str = end_date.strftime("%Y-%m-%dT%H:%M:%S")

    return (
        f"{cfg.dataselect_url}?"
        f"sta={sta_code}"
        f"&starttime={date_str}T00:00:00"
        f"&endtime={end_str}"
        f"&format=miniseed"
        f"&nodata=404"
    )
