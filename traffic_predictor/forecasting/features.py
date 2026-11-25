"""Feature engineering for direct multi-horizon forecasting.

One row per (origin step t, junction j). The target is the count at junction j
at step t + h. Every feature uses only readings at steps <= t, plus the clock
position of the target step, which is known in advance. The shared code keeps
training and serving identical, and `tests/test_features.py` checks the
no-future-data rule directly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import STEPS_PER_DAY, STEPS_PER_WEEK

FEATURE_NAMES: list[str] = [
    "junction",
    "now",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "mean_15m",
    "mean_1h",
    "mean_3h",
    "std_1h",
    "delta_5m",
    "delta_15m",
    "network_total_now",
    "network_mean_1h",
    "target_slot_of_day",
    "same_slot_yesterday",
    "same_slot_yesterday_25m_mean",
    "yesterday_at_origin",
    "now_vs_yesterday",
    "same_slot_last_week",
]
CATEGORICAL = ["junction"]
MAX_HORIZON = STEPS_PER_DAY - 3  # the yesterday window must stay in the past


@dataclass(frozen=True)
class FeatureFrame:
    X: pd.DataFrame  # rows ordered by (origin, junction)
    origin: np.ndarray  # origin step of each row
    junction: np.ndarray  # junction index of each row
    target: np.ndarray  # count at origin + horizon (NaN if missing / beyond data)


def _shift(c: np.ndarray, k: int) -> np.ndarray:
    """out[t] = c[t - k]; NaN where t - k is outside the series."""
    out = np.full_like(c, np.nan)
    n = len(c)
    if k >= 0:
        if k < n:
            out[k:] = c[: n - k]
    else:
        k = -k
        if k < n:
            out[: n - k] = c[k:]
    return out


def build_features(counts: np.ndarray, horizon: int) -> FeatureFrame:
    """Build the feature matrix for every origin step and junction.

    `counts` is (steps, junctions) with NaN for missing readings.
    """
    if not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be in 1..{MAX_HORIZON}, got {horizon}")
    c = np.asarray(counts, dtype=float)
    if c.ndim != 2:
        raise ValueError("counts must be a 2-D array (steps, junctions)")
    n_steps, n_junctions = c.shape
    frame = pd.DataFrame(c)

    def rolling_mean(window: int) -> np.ndarray:
        return frame.rolling(window, min_periods=1).mean().to_numpy()

    std_1h = frame.rolling(12, min_periods=2).std().to_numpy()
    lag = {k: _shift(c, k) for k in (1, 2, 3, 6, 12)}

    # Network-wide context: NaN when any junction is missing, so it never
    # silently mixes a partial sum with full ones.
    total = c.sum(axis=1, keepdims=True)
    total_mean_1h = pd.Series(total[:, 0]).rolling(12, min_periods=1).mean().to_numpy()[:, None]

    yday = STEPS_PER_DAY - horizon  # same clock slot as the target, one day earlier
    with warnings.catch_warnings():  # all-NaN windows (outages) give NaN, by design
        warnings.simplefilter("ignore", category=RuntimeWarning)
        yday_window = np.nanmean(
            np.stack([_shift(c, yday + d) for d in (-2, -1, 0, 1, 2)]), axis=0
        )

    slot = ((np.arange(n_steps) + horizon) % STEPS_PER_DAY).astype(float)[:, None]

    columns = {
        "junction": np.broadcast_to(np.arange(n_junctions, dtype=float), c.shape),
        "now": c,
        "lag_1": lag[1],
        "lag_2": lag[2],
        "lag_3": lag[3],
        "lag_6": lag[6],
        "lag_12": lag[12],
        "mean_15m": rolling_mean(3),
        "mean_1h": rolling_mean(12),
        "mean_3h": rolling_mean(36),
        "std_1h": std_1h,
        "delta_5m": c - lag[1],
        "delta_15m": c - lag[3],
        "network_total_now": np.broadcast_to(total, c.shape),
        "network_mean_1h": np.broadcast_to(total_mean_1h, c.shape),
        "target_slot_of_day": np.broadcast_to(slot, c.shape),
        "same_slot_yesterday": _shift(c, yday),
        "same_slot_yesterday_25m_mean": yday_window,
        "yesterday_at_origin": _shift(c, STEPS_PER_DAY),
        "now_vs_yesterday": c - _shift(c, STEPS_PER_DAY),
        "same_slot_last_week": _shift(c, STEPS_PER_WEEK - horizon),
    }
    assert list(columns) == FEATURE_NAMES
    X = pd.DataFrame({name: np.ascontiguousarray(v).reshape(-1) for name, v in columns.items()})
    return FeatureFrame(
        X=X,
        origin=np.repeat(np.arange(n_steps), n_junctions),
        junction=np.tile(np.arange(n_junctions), n_steps),
        target=_shift(c, -horizon).reshape(-1),
    )
