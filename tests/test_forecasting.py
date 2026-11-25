"""Data cleaning, feature construction and the saved model."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from traffic_predictor.forecasting.data import (
    STEPS_PER_DAY,
    find_outages,
    load_dataset,
    load_raw,
    step_clock,
    step_label,
)
from traffic_predictor.forecasting.features import FEATURE_NAMES, MAX_HORIZON, build_features
from traffic_predictor.forecasting import model as model_module
from traffic_predictor.forecasting.model import HORIZONS, LEVELS, Forecaster, regression_metrics

from .conftest import JUNCTIONS, N_DAYS, make_counts


# ----------------------------------------------------------------- data
def test_outages_need_a_long_all_zero_run():
    raw = pd.DataFrame({"a": [5.0] * 40, "b": [5.0] * 40})
    raw.iloc[2:7] = 0.0  # 5 rows: too short to be an outage
    raw.iloc[10:25] = 0.0  # 15 rows: an outage
    raw.iloc[30:35, 0] = 0.0  # only one junction at zero: never an outage
    outage = find_outages(raw, min_run=12)
    assert outage[10:25].all()
    assert not outage[:10].any() and not outage[25:].any()


def test_load_raw_strips_headers_and_rejects_bad_values(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text("Cross 1, Cross 2\n1,2\n3,4\n", encoding="utf-8")
    assert list(load_raw(good).columns) == ["Cross 1", "Cross 2"]

    negative = tmp_path / "negative.csv"
    negative.write_text("a,b\n1,-2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="negative"):
        load_raw(negative)

    text = tmp_path / "text.csv"
    text.write_text("a,b\n1,x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_raw(text)


def test_load_dataset_marks_outages_missing(trained):
    ds = load_dataset(trained["data"])
    assert ds.junctions == JUNCTIONS
    assert ds.n_steps == N_DAYS * STEPS_PER_DAY
    assert ds.outage.sum() == 30
    assert ds.counts[ds.outage].isna().all().all()
    assert not ds.counts[~ds.outage].isna().any().any()


def test_step_clock():
    assert step_clock(0) == "00:00"
    assert step_clock(101) == "08:25"
    assert step_clock(STEPS_PER_DAY + 12) == "01:00"
    assert step_label(STEPS_PER_DAY * 48) == "Day 49 · 00:00"


# ------------------------------------------------------------- features
@pytest.mark.parametrize("horizon", [1, 3, 12, MAX_HORIZON])
def test_features_never_use_future_readings(horizon):
    counts = make_counts(n_days=10).to_numpy(copy=True)
    counts[300:320] = np.nan  # gaps must not break the rule either
    rng = np.random.default_rng(1)
    base = build_features(counts, horizon).X
    n_j = counts.shape[1]
    for origin in rng.integers(0, len(counts) - 1, size=25):
        changed = counts.copy()
        changed[origin + 1 :] = rng.uniform(0, 500, size=changed[origin + 1 :].shape)
        other = build_features(changed, horizon).X
        rows = slice(origin * n_j, (origin + 1) * n_j)
        pd.testing.assert_frame_equal(base.iloc[rows], other.iloc[rows])


def test_feature_rows_and_targets_line_up():
    counts = make_counts(n_days=3).to_numpy()
    h = 6
    frame = build_features(counts, h)
    assert list(frame.X.columns) == FEATURE_NAMES
    n_j = counts.shape[1]
    t, j = 400, 2
    row = t * n_j + j
    assert frame.origin[row] == t and frame.junction[row] == j
    assert frame.X["now"].iloc[row] == counts[t, j]
    assert frame.target[row] == counts[t + h, j]
    assert frame.X["target_slot_of_day"].iloc[row] == (t + h) % STEPS_PER_DAY
    assert np.isnan(frame.target[(len(counts) - 1) * n_j + j])  # beyond the data


def test_horizon_bounds():
    counts = make_counts(n_days=2).to_numpy()
    for bad in (0, MAX_HORIZON + 1):
        with pytest.raises(ValueError):
            build_features(counts, bad)


# ---------------------------------------------------------------- model
def test_regression_metrics():
    m = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert m["n"] == 3
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(4 / 3))
    assert m["wape"] == pytest.approx(2 / 6)


def test_trained_artifact(trained):
    fc = Forecaster.load(trained["model"])
    assert fc.junctions == JUNCTIONS
    assert fc.horizons == sorted(HORIZONS)
    assert fc.test_start_step == 48 * STEPS_PER_DAY  # days 49-56 never fitted
    counts = load_dataset(trained["data"]).counts.to_numpy()
    pred = fc.predict(counts, 3)
    assert pred.shape == counts.shape
    assert (pred >= 0).all()
    low, high = fc.level_thresholds[0]
    assert fc.level_for(0, low - 1) == LEVELS[0]
    assert fc.level_for(0, (low + high) / 2) == LEVELS[1]
    assert fc.level_for(0, high + 1) == LEVELS[2]
    assert 0 <= fc.percentile_for(0, low) <= 100
    with pytest.raises(ValueError):
        fc.predict(counts[:, :2], 3)


def test_model_beats_persistence_on_validation(trained):
    metrics = json.loads(trained["metrics"].read_text(encoding="utf-8"))
    assert metrics["test_evaluated"] is False and metrics["test"] is None
    for result in metrics["validation"].values():
        assert result["model"]["mae"] < result["persistence"]["mae"]


def test_load_rejects_foreign_pickles(tmp_path):
    import joblib

    path = tmp_path / "other.joblib"
    joblib.dump({"not": "a forecaster"}, path)
    with pytest.raises(RuntimeError):
        Forecaster.load(path)


def test_retest_needs_explicit_flag(trained, tmp_path):
    from scripts import train

    out = tmp_path / "out"
    out.mkdir()
    (out / "metrics.json").write_text(json.dumps({"test_evaluation_count": 1}), encoding="utf-8")
    with pytest.raises(SystemExit):
        train.main(["--data", str(trained["data"]), "--out", str(out), "--evaluate-test"])


def test_prediction_ranges_widen_with_the_forecast(trained):
    fc = Forecaster.load(trained["model"])
    pred = np.array([5.0, 50.0, 200.0])
    lower, upper = fc.interval(3, np.array([0, 0, 0]), pred)
    assert (lower <= pred).all() and (pred <= upper).all() and (lower >= 0).all()
    width = upper - lower
    assert width[0] < width[1] < width[2]
    lo1, hi1 = fc.interval(3, 0, 50.0)  # scalar form, as the service uses it
    assert float(lo1) == pytest.approx(lower[1]) and float(hi1) == pytest.approx(upper[1])


def test_evaluate_test_scores_held_out_days(trained, tmp_path):
    from scripts import train

    saved = dict(model_module._BASE_PARAMS)
    model_module._BASE_PARAMS.update(max_iter=30, learning_rate=0.3)
    try:
        assert train.main(["--data", str(trained["data"]), "--out", str(tmp_path), "--evaluate-test"]) == 0
    finally:
        model_module._BASE_PARAMS.clear()
        model_module._BASE_PARAMS.update(saved)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test_evaluated"] is True and metrics["test_evaluation_count"] == 1
    assert metrics["split_days"]["test"] == [49, 56]
    for result in metrics["test"].values():
        coverage = result["interval"]["observed_coverage"]
        assert 0.6 < coverage < 0.95
        assert set(result["interval"]["coverage_by_forecast_load"]) == {"quiet", "middle", "busy"}
