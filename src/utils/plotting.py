from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_prediction_vs_actual(
    y_true: pd.Series,
    y_pred: np.ndarray,
    test_index: pd.Index,
    station_name: str,
    output_dir: str,
) -> Path:
    """Plot Random Forest predictions against the actual GPS displacement rate.

    Reproduces the core result figure from the original study: the full
    observed displacement-rate series is drawn as a solid red line, and
    the model predictions on the held-out test set are overlaid as blue dots.

    Args:
        y_true: Full displacement-rate Series (DatetimeIndex, all dates).
        y_pred: Predicted values for the test set, aligned to test_index.
        test_index: Index of the rows used as the test set.
        station_name: GPS station code, used in the title and filename.
        output_dir: Directory where the PNG will be saved.

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(y_true.index, y_true.values, color="red", linewidth=1.0, label="Observation")
    ax.scatter(test_index, y_pred, color="blue", s=6, label="Model", zorder=3)

    ax.set_xlabel("Date")
    ax.set_ylabel(r"GPS displacement rate (mm yr$^{-1}$)")
    ax.set_title(f"Station {station_name} — Modelled vs Observed displacement rate")
    ax.legend()
    fig.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{station_name}_prediction.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    logger.info("Saved prediction plot: %s", out_path)
    return out_path


def plot_feature_importances(
    importances: np.ndarray,
    feature_names: list[str],
    station_name: str,
    output_dir: str,
    top_n: int = 20,
) -> Path:
    """Plot a horizontal bar chart of the top-N Random Forest feature importances.

    Feature importances are the mean decrease in impurity across all trees,
    as returned by sklearn's RandomForestRegressor.feature_importances_.

    Args:
        importances: 1-D array of feature importance scores (one per feature).
        feature_names: List of feature names matching the importances array.
        station_name: GPS station code, used in the title and filename.
        output_dir: Directory where the PNG will be saved.
        top_n: Number of top features to display (default 20).

    Returns:
        Path to the saved figure.
    """
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_names = [feature_names[i] for i in indices]

    # Reverse so highest importance is at the top of the horizontal bar chart.
    top_importances = top_importances[::-1]
    top_names = top_names[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))

    ax.barh(range(top_n), top_importances, align="center", color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(f"Station {station_name} — Top {top_n} feature importances")
    fig.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{station_name}_feature_importances.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    logger.info("Saved feature importance plot: %s", out_path)
    return out_path
