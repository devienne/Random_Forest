from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.utils.config import ModelConfig

logger = logging.getLogger(__name__)


# ── Split ─────────────────────────────────────────────────────────────────────

def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into train and test sets by time order.

    Rows are sorted by their index before splitting, so the first
    train_size fraction of the timeline forms the training set and the
    remaining fraction forms the test set.

    This is the methodologically correct approach for time-series data.
    The original study used sklearn's train_test_split with a random shuffle,
    which caused severe data leakage: because 60-day rolling features make
    consecutive rows nearly identical, random sampling allows the model to
    'see' future values during training. A chronological split prevents this.

    Args:
        X: Feature DataFrame with a DatetimeIndex.
        y: Target Series with the same index.
        train_size: Fraction of rows to use for training (e.g. 0.8).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = X.sort_index()
    y = y.sort_index()

    cutoff = int(len(X) * train_size)

    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]

    logger.info(
        "Chronological split: %d train rows (up to %s), %d test rows (from %s)",
        len(X_train),
        X_train.index[-1].date() if len(X_train) > 0 else "N/A",
        len(X_test),
        X_test.index[0].date() if len(X_test) > 0 else "N/A",
    )
    return X_train, X_test, y_train, y_test


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_pipeline(cfg: ModelConfig) -> Pipeline:
    """Build a sklearn Pipeline wrapping the Random Forest regressor.

    No feature scaling step is included: decision trees are invariant to
    monotonic transformations of the input features. The Pipeline wrapper
    exists so the model is serialisable as a single object and ready for
    MLflow logging and future pre-processing steps.

    Args:
        cfg: Model configuration section from ProjectConfig.

    Returns:
        Unfitted sklearn Pipeline with a single 'rf' step.
    """
    rf = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_split=cfg.min_samples_split,
        max_features=cfg.max_features,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    return Pipeline([("rf", rf)])


# ── Train ─────────────────────────────────────────────────────────────────────

def train(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: ModelConfig,
) -> tuple[Pipeline, dict]:
    """Train the Random Forest on the training split and evaluate on the test split.

    Args:
        X: Smoothed seismic feature DataFrame (DatetimeIndex, NaN-free).
        y: Displacement-rate target Series (same index).
        cfg: Model configuration section from ProjectConfig.

    Returns:
        Tuple of (fitted Pipeline, metrics dict).
        Metrics keys: r2, mae, rmse, n_train, n_test, train_start,
        train_end, test_start, test_end.
    """
    X_train, X_test, y_train, y_test = chronological_split(X, y, cfg.train_size)

    pipeline = build_pipeline(cfg)
    logger.info(
        "Training RandomForest: n_estimators=%d, max_depth=%s, train_size=%.0f%%",
        cfg.n_estimators,
        cfg.max_depth,
        cfg.train_size * 100,
    )
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)
    metrics.update(
        {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_start": str(X_train.index[0].date()),
            "train_end": str(X_train.index[-1].date()),
            "test_start": str(X_test.index[0].date()),
            "test_end": str(X_test.index[-1].date()),
        }
    )

    logger.info(
        "Evaluation on test set — R²: %.4f  MAE: %.4f  RMSE: %.4f",
        metrics["r2"],
        metrics["mae"],
        metrics["rmse"],
    )
    return pipeline, metrics


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute regression metrics on a held-out set.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_test: Test feature DataFrame.
        y_test: Test target Series.

    Returns:
        Dict with keys: r2, mae, rmse.
    """
    y_pred = pipeline.predict(X_test)
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, path: str) -> Path:
    """Serialise a fitted pipeline to disk using joblib.

    Args:
        pipeline: Fitted sklearn Pipeline to save.
        path: Full file path (should end in .joblib).

    Returns:
        Path to the saved file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out)
    logger.info("Model saved: %s", out)
    return out


def load_model(path: str) -> Pipeline:
    """Load a fitted pipeline from a joblib file.

    Args:
        path: Path to the .joblib file written by save_model.

    Returns:
        Fitted sklearn Pipeline.
    """
    return joblib.load(path)
