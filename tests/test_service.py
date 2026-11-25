"""The replay service behind the dashboard."""

from __future__ import annotations

import csv
import io
from datetime import datetime

import numpy as np
import pytest

from traffic_predictor.forecasting.data import STEPS_PER_DAY
from traffic_predictor.service import build_service, parse_clock

from .conftest import JUNCTIONS


@pytest.fixture(scope="module")
def service(trained):
    return build_service(trained["data"], trained["model"], trained["metrics"])


def test_replay_covers_only_held_out_days(service):
    assert (service.replay.first_day, service.replay.last_day) == (49, 56)
    assert service.replay.first_step == service.forecaster.fit_end_step


def test_parse_clock():
    assert parse_clock("00:00") == 0
    assert parse_clock("08:32") == 102  # floored to the 5-minute step
    for bad in ("24:00", "8", "ab:cd", "12:60"):
        with pytest.raises(ValueError):
            parse_clock(bad)


def test_step_mapping(service):
    assert service.step_for(49, 0) == 48 * STEPS_PER_DAY
    assert service.step_for(56, STEPS_PER_DAY - 1) == 56 * STEPS_PER_DAY - 1
    for day in (48, 57):
        with pytest.raises(ValueError):
            service.step_for(day, 0)
    now = datetime(2026, 11, 25, 8, 37)
    step = service.live_step(now)
    assert step % STEPS_PER_DAY == (8 * 60 + 35) // 5
    assert service.replay.first_step <= step < service.replay.end_step


def test_snapshot_matches_the_model(service):
    step = service.step_for(50, 100)
    snap = service.snapshot(step, focus=1)
    assert snap["day"] == 50 and snap["clock"] == "08:20"
    assert [j["name"] for j in snap["junctions"]] == JUNCTIONS
    j = snap["junctions"][1]
    assert j["now"] == round(float(service.counts[step, 1]), 1)
    for f, h in zip(j["forecasts"], service.horizons):
        expected = float(service.forecaster.predict(service.counts, h)[step, 1])
        assert f["value"] == pytest.approx(expected, abs=0.051)
        assert f["lower"] <= f["value"] <= f["upper"]
        assert "actual" not in f  # the future stays hidden by default
    assert snap["network"]["now"] == round(float(service.counts[step].sum()))


def test_snapshot_outcomes_on_request(service):
    step = service.step_for(50, 100)
    j = service.snapshot(step, include_outcomes=True)["junctions"][0]
    for f, h in zip(j["forecasts"], service.horizons):
        assert f["actual"] == round(float(service.counts[step + h, 0]), 1)


def test_focus_series_are_out_of_sample(service):
    step = service.step_for(49, 30)  # early on the first replay day
    focus = service.snapshot(step)["focus"]
    longest = max(service.horizons)
    n = len(focus["labels"])
    assert n == 72 + longest + 1
    now = focus["now_index"]
    first_replay_index = now - 30
    # "forecast made earlier" only for targets inside the held-out days, never in the future
    assert all(v is None for v in focus["earlier_forecast"][:first_replay_index])
    assert all(v is not None for v in focus["earlier_forecast"][first_replay_index : now + 1])
    assert all(v is None for v in focus["earlier_forecast"][now + 1 :])
    assert all(v is None for v in focus["recorded"][now + 1 :])
    assert sum(v is not None for v in focus["ahead"]) == len(service.horizons) + 1
    assert focus["today_points"] == 31 and focus["today_mae"] is not None


def test_snapshot_rejects_bad_input(service):
    with pytest.raises(ValueError):
        service.snapshot(service.replay.first_step - 1)
    with pytest.raises(ValueError):
        service.snapshot(service.replay.first_step, focus=len(JUNCTIONS))


def test_day_csv(service):
    rows = list(csv.reader(io.StringIO(service.day_csv(52))))
    header, data = rows[0], rows[1:]
    assert header[:4] == ["day", "time", "junction", "reading"]
    assert len(data) == STEPS_PER_DAY * len(JUNCTIONS)
    assert data[0][:3] == ["52", "00:00", JUNCTIONS[0]]
    values = np.array([float(r[4]) for r in data])
    assert (values >= 0).all()
