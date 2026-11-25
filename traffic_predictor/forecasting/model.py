"""The forecasting model, its evaluation helpers and its saved artifact."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import STEP_MINUTES, STEPS_PER_DAY
from .features import CATEGORICAL, FEATURE_NAMES, build_features

ARTIFACT_VERSION = 2  # 2: prediction ranges scale with the forecast
HORIZONS: tuple[int, ...] = (1, 3, 6, 12)  # steps ahead: 5, 15, 30, 60 minutes
INTERVAL_LEVEL = 0.80  # central prediction interval from held-out residuals
LEVEL_PERCENTILES = (50, 85)  # below 50th: low, 50th-85th: moderate, above: heavy
LEVELS = ("low", "moderate", "heavy")

# Candidate settings compared on the validation days only.
CANDIDATES: dict[str, dict[str, Any]] = {
    "gbm_squared_error": {"loss": "squared_error"},
    "gbm_poisson": {"loss": "poisson"},
}
_BASE_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,  # fixed iterations: deterministic, no random inner split
    "categorical_features": CATEGORICAL,
    "random_state": 0,
}


def make_estimator(candidate: str) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**_BASE_PARAMS, **CANDIDATES[candidate])


def horizon_minutes(h: int) -> int:
    return h * STEP_MINUTES


def interval_scale(pred: np.ndarray | float) -> np.ndarray:
    """How wide a prediction range is, relative to the forecast.

    Count noise grows with the count, so residuals are divided by sqrt(forecast)
    before their quantiles are taken. Checked on the validation days only (ranges
    fitted on days 41-44, scored on 45-48): fixed-width ranges covered about 94%
    of quiet-time readings and about 64% of busy ones, while sqrt-scaled ranges
    covered 73-80% at every load level, with a better interval score at every
    horizon.
    """
    return np.sqrt(np.maximum(np.asarray(pred, dtype=float), 1.0))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "n": int(len(y_true)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "r2": float(1 - (err**2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "wape": float(np.abs(err).sum() / np.abs(y_true).sum()) if y_true.any() else float("nan"),
    }


def baseline_predictions(counts: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    """Naive forecasts, laid out like `build_features` rows."""
    frame = build_features(counts, horizon).X
    return {
        "persistence": frame["now"].to_numpy(),  # next value = current value
        "same_time_yesterday": frame["same_slot_yesterday"].to_numpy(),
    }


# --------------------------------------------------------------------------
# Artifact
# --------------------------------------------------------------------------

@dataclass
class Forecaster:
    junctions: list[str]
    models: dict[int, HistGradientBoostingRegressor]
    # per horizon, per junction: quantiles of residual / interval_scale(forecast)
    interval_offsets: dict[int, list[tuple[float, float]]]
    level_thresholds: list[tuple[float, float]]  # per junction (p50, p85)
    reference_quantiles: list[list[float]]  # per junction, 0th..100th percentile
    typical_day: list[list[float]]  # per junction, mean count per 5-min slot
    fit_end_step: int  # model saw targets at steps < fit_end_step
    test_start_step: int
    info: dict[str, Any] = field(default_factory=dict)
    version: int = ARTIFACT_VERSION

    @property
    def horizons(self) -> list[int]:
        return sorted(self.models)

    def predict(self, counts: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast for every origin step: returns (steps, junctions).

        Row t holds the forecast for step t + horizon made at step t.
        """
        if horizon not in self.models:
            raise KeyError(f"no model for horizon {horizon}")
        counts = np.asarray(counts, dtype=float)
        if counts.shape[1] != len(self.junctions):
            raise ValueError(
                f"expected {len(self.junctions)} junction columns, got {counts.shape[1]}"
            )
        frame = build_features(counts, horizon)
        pred = self.models[horizon].predict(frame.X[FEATURE_NAMES])
        return np.clip(pred, 0, None).reshape(counts.shape)

    def interval(self, horizon: int, junction, pred) -> tuple[np.ndarray, np.ndarray]:
        """Central INTERVAL_LEVEL range around `pred`; `junction` may be an index array."""
        pred = np.asarray(pred, dtype=float)
        offsets = np.asarray(self.interval_offsets[horizon], dtype=float)[np.asarray(junction)]
        scale = interval_scale(pred)
        lower = np.clip(pred + offsets[..., 0] * scale, 0, None)
        upper = np.clip(pred + offsets[..., 1] * scale, 0, None)
        return lower, upper

    def level_for(self, junction: int, value: float) -> str:
        p50, p85 = self.level_thresholds[junction]
        if value < p50:
            return LEVELS[0]
        if value < p85:
            return LEVELS[1]
        return LEVELS[2]

    def percentile_for(self, junction: int, value: float) -> float:
        """Share (0-100) of historical readings at this junction below `value`."""
        q = np.asarray(self.reference_quantiles[junction])
        return float(np.interp(value, q, np.linspace(0, 100, len(q))))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str | Path) -> "Forecaster":
        # The artifact is a pickle: only ever load files this project produced.
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UserWarning)  # surface version mismatches
            try:
                obj = joblib.load(path)
            except UserWarning as exc:  # e.g. InconsistentVersionWarning
                raise RuntimeError(
                    f"{path} was saved with a different scikit-learn version; retrain it "
                    "with `python -m scripts.train`"
                ) from exc
        if not isinstance(obj, Forecaster) or obj.version != ARTIFACT_VERSION:
            raise RuntimeError(f"{path} is not a compatible forecaster artifact; retrain it")
        return obj


def typical_day_profile(counts: np.ndarray) -> np.ndarray:
    """Mean count per time-of-day slot over complete days, ignoring missing readings."""
    n_days = len(counts) // STEPS_PER_DAY
    days = counts[: n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(days, axis=0)  # (slots, junctions)


def to_frame(pred: np.ndarray, junctions: list[str]) -> pd.DataFrame:
    return pd.DataFrame(pred, columns=junctions)
