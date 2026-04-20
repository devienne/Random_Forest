from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.random_forest import build_pipeline, chronological_split, evaluate, train
from src.utils.config import ModelConfig

# ── Shared fixtures ───────────────────────────────────────────────────────────

_CFG = ModelConfig(
    n_estimators=10,  # small for test speed
    max_depth=3,
    min_samples_split=2,
    max_features=1.0,
    random_state=42,
    train_size=0.8,
)


def _synthetic(n: int = 200, n_features: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """Return a reproducible synthetic dataset with a DatetimeIndex."""
    rng = np.random.default_rng(0)
    index = pd.date_range("2010-01-01", periods=n, freq="D")
    X = pd.DataFrame(rng.standard_normal((n, n_features)), index=index)
    y = pd.Series(rng.standard_normal(n), index=index, name="target")
    return X, y


# ── chronological_split ───────────────────────────────────────────────────────

def test_split_train_strictly_before_test():
    """The latest training date must be strictly earlier than the earliest test date."""
    X, y = _synthetic()
    X_tr, X_te, _, _ = chronological_split(X, y, train_size=0.8)
    assert X_tr.index.max() < X_te.index.min()


def test_split_sizes_match_train_size():
    X, y = _synthetic(n=100)
    X_tr, X_te, y_tr, y_te = chronological_split(X, y, train_size=0.8)
    assert len(X_tr) == 80
    assert len(X_te) == 20
    assert len(y_tr) == 80
    assert len(y_te) == 20


def test_split_no_date_overlap():
    X, y = _synthetic()
    X_tr, X_te, _, _ = chronological_split(X, y, train_size=0.8)
    assert len(set(X_tr.index) & set(X_te.index)) == 0


def test_split_covers_all_rows():
    X, y = _synthetic(n=150)
    X_tr, X_te, _, _ = chronological_split(X, y, train_size=0.8)
    assert len(X_tr) + len(X_te) == 150


def test_split_preserves_index_order():
    """Train and test sets must each be sorted chronologically."""
    X, y = _synthetic()
    X_tr, X_te, _, _ = chronological_split(X, y, train_size=0.8)
    assert X_tr.index.is_monotonic_increasing
    assert X_te.index.is_monotonic_increasing


# ── build_pipeline ────────────────────────────────────────────────────────────

def test_pipeline_has_rf_step():
    pipeline = build_pipeline(_CFG)
    assert "rf" in pipeline.named_steps


def test_pipeline_rf_respects_n_estimators():
    pipeline = build_pipeline(_CFG)
    assert pipeline.named_steps["rf"].n_estimators == _CFG.n_estimators


def test_pipeline_rf_respects_random_state():
    pipeline = build_pipeline(_CFG)
    assert pipeline.named_steps["rf"].random_state == _CFG.random_state


def test_pipeline_rf_respects_max_depth():
    pipeline = build_pipeline(_CFG)
    assert pipeline.named_steps["rf"].max_depth == _CFG.max_depth


# ── evaluate ─────────────────────────────────────────────────────────────────

def test_evaluate_returns_required_keys():
    X, y = _synthetic(n=100)
    X_tr, X_te, y_tr, y_te = chronological_split(X, y, train_size=0.8)
    pipeline = build_pipeline(_CFG)
    pipeline.fit(X_tr, y_tr)
    metrics = evaluate(pipeline, X_te, y_te)
    assert set(metrics.keys()) == {"r2", "mae", "rmse"}


def test_evaluate_rmse_non_negative():
    X, y = _synthetic(n=100)
    X_tr, X_te, y_tr, y_te = chronological_split(X, y, train_size=0.8)
    pipeline = build_pipeline(_CFG)
    pipeline.fit(X_tr, y_tr)
    metrics = evaluate(pipeline, X_te, y_te)
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0


def test_evaluate_metrics_are_finite():
    X, y = _synthetic(n=100)
    X_tr, X_te, y_tr, y_te = chronological_split(X, y, train_size=0.8)
    pipeline = build_pipeline(_CFG)
    pipeline.fit(X_tr, y_tr)
    metrics = evaluate(pipeline, X_te, y_te)
    assert all(np.isfinite(v) for v in metrics.values())


# ── train ─────────────────────────────────────────────────────────────────────

def test_train_returns_pipeline_and_metrics():
    X, y = _synthetic(n=100)
    pipeline, metrics = train(X, y, _CFG)
    assert pipeline is not None
    assert isinstance(metrics, dict)


def test_train_metrics_contain_all_keys():
    X, y = _synthetic(n=100)
    _, metrics = train(X, y, _CFG)
    required = {"r2", "mae", "rmse", "n_train", "n_test", "train_start", "train_end",
                "test_start", "test_end"}
    assert required.issubset(metrics.keys())


def test_train_split_sizes_consistent_with_config():
    n = 100
    X, y = _synthetic(n=n)
    _, metrics = train(X, y, _CFG)
    assert metrics["n_train"] == int(n * _CFG.train_size)
    assert metrics["n_test"] == n - int(n * _CFG.train_size)


def test_train_test_end_after_train_end():
    """Confirms the split is chronological: test period follows training period."""
    X, y = _synthetic(n=200)
    _, metrics = train(X, y, _CFG)
    assert metrics["test_start"] > metrics["train_end"]


def test_trained_pipeline_can_predict():
    X, y = _synthetic(n=100)
    pipeline, _ = train(X, y, _CFG)
    n_test = int(len(X) * (1 - _CFG.train_size))
    preds = pipeline.predict(X.iloc[-n_test:])
    assert len(preds) == n_test
    assert np.isfinite(preds).all()
