"""Training pipeline entry point.

Reads merged per-station CSVs from data/processed/, applies feature smoothing
and GPS target computation, trains a Random Forest per station, and writes
metrics, figures, and saved models to artifacts/.

Usage
-----
    # train all stations found in data/processed/
    python -m src.pipelines.train

    # train a single station
    python -m src.pipelines.train --station ALBH

    # use a custom config file
    python -m src.pipelines.train --config configs/custom.yaml

Prerequisite
------------
Merged per-station CSVs must exist in data/processed/.
Run the data preparation step first:
    python -m src.pipelines.prepare_data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.features.gps_features import compute_displacement_rate
from src.features.seismic_features import smooth_features
from src.models.random_forest import save_model, train
from src.utils.config import ProjectConfig, load_config
from src.utils.plotting import plot_feature_importances, plot_prediction_vs_actual

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config)
    logger.info("Config loaded from %s", args.config or "configs/default.yaml")

    processed_dir = Path(cfg.paths.processed_dir)
    if not processed_dir.exists() or not any(processed_dir.glob("*_merged.csv")):
        logger.error(
            "No merged CSVs found in %s.\n"
            "Run the data preparation pipeline first:\n"
            "    python -m src.pipelines.prepare_data",
            processed_dir,
        )
        sys.exit(1)

    stations = _discover_stations(processed_dir, args.station)
    if not stations:
        logger.error("No stations to process. Check --station argument or %s.", processed_dir)
        sys.exit(1)

    logger.info("Stations to train: %s", stations)

    all_metrics: dict[str, dict] = {}

    for station in stations:
        logger.info("── Station %s ──────────────────────────────", station)
        result = _run_station(station, processed_dir, cfg)
        if result is not None:
            all_metrics[station] = result

    _print_summary(all_metrics)


# ── Per-station workflow ──────────────────────────────────────────────────────

def _run_station(
    station: str,
    processed_dir: Path,
    cfg: ProjectConfig,
) -> dict | None:
    """Load, prepare, train, evaluate, and save results for one GPS station."""

    # 1. Load merged CSV (seismic features + GPS column).
    csv_path = processed_dir / f"{station}_merged.csv"
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    except Exception as exc:
        logger.warning("Could not load %s: %s", csv_path, exc)
        return None

    # 2. Separate feature columns from the GPS target column.
    if station not in df.columns:
        logger.warning("GPS column '%s' not found in %s — skipping.", station, csv_path.name)
        return None

    feature_cols = [c for c in df.columns if c != station]
    X_raw = df[feature_cols].copy()
    gps_raw = df[station].copy()

    # 3. Apply 60-day rolling mean to seismic features.
    X_smooth = smooth_features(X_raw, cfg.seismic.rolling_window_days)

    # 4. Compute GPS displacement rate via rolling linear regression.
    y = compute_displacement_rate(gps_raw, cfg.gps.rolling_window_days)

    # 5. Align features and target; drop any remaining NaN rows.
    combined = pd.concat([X_smooth, y.rename("target")], axis=1).dropna()
    if combined.empty:
        logger.warning("No usable rows for station %s after smoothing — skipping.", station)
        return None

    X = combined[feature_cols]
    y_clean = combined["target"]
    logger.info("Station %s: %d usable rows after smoothing.", station, len(X))

    # 6. Train Random Forest (chronological split, metrics on test set).
    pipeline, metrics = train(X, y_clean, cfg.model)
    metrics["station"] = station

    # 7. Save trained model.
    model_path = Path(cfg.paths.models_dir) / f"{station}_rf.joblib"
    save_model(pipeline, str(model_path))

    # 8. Produce and save figures.
    from sklearn.pipeline import Pipeline as SKPipeline  # local import to keep top-level clean

    n_test = metrics["n_test"]
    X_test = X.iloc[-n_test:]
    y_pred = pipeline.predict(X_test)

    # Reconstruct full observed series for the prediction plot.
    y_full = compute_displacement_rate(gps_raw, cfg.gps.rolling_window_days).dropna()

    plot_prediction_vs_actual(
        y_true=y_full,
        y_pred=y_pred,
        test_index=X_test.index,
        station_name=station,
        output_dir=cfg.paths.figures_dir,
    )

    rf_step = pipeline.named_steps["rf"]
    plot_feature_importances(
        importances=rf_step.feature_importances_,
        feature_names=feature_cols,
        station_name=station,
        output_dir=cfg.paths.figures_dir,
    )

    # 9. Persist metrics as JSON alongside the model.
    metrics_path = Path(cfg.paths.models_dir) / f"{station}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "Station %s done — R²: %.4f  MAE: %.4f  RMSE: %.4f",
        station,
        metrics["r2"],
        metrics["mae"],
        metrics["rmse"],
    )
    return metrics


# ── Helpers ───────────────────────────────────────────────────────────────────

def _discover_stations(processed_dir: Path, station_arg: str | None) -> list[str]:
    """Return the list of stations to train, from CLI arg or directory scan."""
    if station_arg:
        return [station_arg]
    return sorted(p.stem.replace("_merged", "") for p in processed_dir.glob("*_merged.csv"))


def _print_summary(all_metrics: dict[str, dict]) -> None:
    if not all_metrics:
        logger.warning("No stations were successfully trained.")
        return

    print("\n" + "=" * 60)
    print(f"{'Station':<10} {'R²':>8} {'MAE':>10} {'RMSE':>10} {'N test':>8}")
    print("-" * 60)
    for station, m in sorted(all_metrics.items()):
        print(
            f"{station:<10} {m['r2']:>8.4f} {m['mae']:>10.4f} "
            f"{m['rmse']:>10.4f} {m['n_test']:>8}"
        )
    print("=" * 60 + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest per GPS station from processed seismic/GPS data."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--station",
        default=None,
        help="Train a single GPS station by code (default: all stations in data/processed/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
