"""Serving: the trained forecaster, replayed over the held-out test days.

The dataset has no timestamps and there is no live sensor feed, so the
dashboard replays the recorded test days (never used for fitting or for any
modelling choice) as if they were happening now. Every number it shows is
either a recorded reading or an out-of-sample forecast computed only from
readings up to that moment.
"""

from __future__ import annotations

import csv
import io
import json
import math
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, current_app
from werkzeug.exceptions import ServiceUnavailable

from .forecasting.data import STEP_MINUTES, STEPS_PER_DAY, Dataset, load_dataset, step_clock, step_day
from .forecasting.model import LEVELS, Forecaster, horizon_minutes

EXTENSION_KEY = "forecast_service"
CHART_PAST_STEPS = 72  # 6 h of history on the focus chart


def _num(value: float | np.floating | None, digits: int = 1) -> float | None:
    """JSON-safe rounded number; None for missing readings."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else round(value, digits)


def parse_clock(text: str) -> int:
    """'HH:MM' -> slot of day (floored to the 5-minute step)."""
    try:
        hours, minutes = (int(part) for part in text.split(":"))
    except ValueError:
        raise ValueError("time must look like HH:MM") from None
    if not (0 <= hours < 24 and 0 <= minutes < 60):
        raise ValueError("time must be between 00:00 and 23:59")
    return (hours * 60 + minutes) // STEP_MINUTES


@dataclass(frozen=True)
class ReplayRange:
    first_step: int
    days: int  # complete held-out days

    @property
    def first_day(self) -> int:
        return step_day(self.first_step)

    @property
    def last_day(self) -> int:
        return self.first_day + self.days - 1

    @property
    def end_step(self) -> int:  # exclusive
        return self.first_step + self.days * STEPS_PER_DAY


class ForecastService:
    def __init__(self, dataset: Dataset, forecaster: Forecaster, metrics: dict[str, Any] | None = None):
        if list(dataset.junctions) != list(forecaster.junctions):
            raise ValueError(
                f"dataset junctions {dataset.junctions} do not match the model's {forecaster.junctions}"
            )
        self.forecaster = forecaster
        self.metrics = metrics
        self.junctions: list[str] = list(forecaster.junctions)
        self.counts = dataset.counts.to_numpy()
        self.horizons = forecaster.horizons
        start = forecaster.test_start_step
        days = (len(self.counts) - start) // STEPS_PER_DAY
        if start % STEPS_PER_DAY or days < 1:
            raise ValueError("the model leaves no complete held-out day to replay")
        self.replay = ReplayRange(first_step=start, days=days)
        # Row t of forecasts[h] is the forecast for step t + h made at step t.
        self.forecasts = {h: forecaster.predict(self.counts, h) for h in self.horizons}
        self.typical = np.asarray(forecaster.typical_day, dtype=float)  # (junctions, slots)

    # ------------------------------------------------------------------ steps
    def step_for(self, day: int, slot: int) -> int:
        if not self.replay.first_day <= day <= self.replay.last_day:
            raise ValueError(
                f"day must be between {self.replay.first_day} and {self.replay.last_day}"
            )
        if not 0 <= slot < STEPS_PER_DAY:
            raise ValueError("time is outside the day")
        return self.replay.first_step + (day - self.replay.first_day) * STEPS_PER_DAY + slot

    def live_step(self, now: datetime) -> int:
        """Map the wall clock onto the replay: same time of day, held-out days in turn."""
        day = self.replay.first_day + now.toordinal() % self.replay.days
        return self.step_for(day, (now.hour * 60 + now.minute) // STEP_MINUTES)

    # --------------------------------------------------------------- forecasts
    def _interval(self, h: int, j: int, value: float) -> tuple[float, float]:
        lower, upper = self.forecaster.interval(h, j, value)
        return float(lower), float(upper)

    def _forecast(self, origin: int, h: int, j: int) -> dict[str, Any]:
        value = float(self.forecasts[h][origin, j])
        lower, upper = self._interval(h, j, value)
        target = origin + h
        actual = self.counts[target, j] if target < len(self.counts) else math.nan
        return {
            "minutes": horizon_minutes(h),
            "target_clock": step_clock(target),
            "value": _num(value),
            "lower": _num(lower),
            "upper": _num(upper),
            "level": self.forecaster.level_for(j, value),
        } | ({"actual": _num(actual)} if target < self.replay.end_step else {})

    def _out_of_sample(self, target: int) -> bool:
        return self.replay.first_step <= target < self.replay.end_step

    def snapshot(self, step: int, focus: int = 0, *, include_outcomes: bool = False) -> dict[str, Any]:
        """Everything the dashboard shows at one replay step."""
        if not self.replay.first_step <= step < self.replay.end_step:
            raise ValueError("step is outside the replay range")
        if not 0 <= focus < len(self.junctions):
            raise ValueError(f"junction must be between 0 and {len(self.junctions) - 1}")
        slot = step % STEPS_PER_DAY
        longest = max(self.horizons)

        junctions = []
        for j, name in enumerate(self.junctions):
            now = self.counts[step, j]
            forecasts = [self._forecast(step, h, j) for h in self.horizons]
            if not include_outcomes:
                for f in forecasts:
                    f.pop("actual", None)  # the future stays hidden unless asked for
            junctions.append(
                {
                    "index": j,
                    "name": name,
                    "now": _num(now),
                    "level": None if math.isnan(now) else self.forecaster.level_for(j, now),
                    "percentile": None if math.isnan(now) else _num(self.forecaster.percentile_for(j, now), 0),
                    "typical_now": _num(self.typical[j, slot]),
                    "forecasts": forecasts,
                }
            )

        row = self.counts[step]
        network_now = float(row.sum()) if not np.isnan(row).any() else None
        network = {
            "now": _num(network_now, 0),
            "typical_now": _num(self.typical[:, slot].sum(), 0),
            "forecast": {
                str(horizon_minutes(h)): _num(self.forecasts[h][step].sum(), 0) for h in self.horizons
            },
            "heavy_now": sum(1 for j in junctions if j["level"] == LEVELS[2]),
            "heavy_in": {
                str(horizon_minutes(h)): sum(
                    1 for j in junctions if j["forecasts"][i]["level"] == LEVELS[2]
                )
                for i, h in enumerate(self.horizons)
            },
            "busiest": max(
                (j for j in junctions if j["now"] is not None), key=lambda j: j["now"], default=None
            ),
        }
        if network["busiest"] is not None:
            network["busiest"] = {"index": network["busiest"]["index"], "name": network["busiest"]["name"]}

        return {
            "step": step,
            "day": step_day(step),
            "clock": step_clock(step),
            "replay": {"first_day": self.replay.first_day, "last_day": self.replay.last_day},
            "horizons_minutes": [horizon_minutes(h) for h in self.horizons],
            "network": network,
            "junctions": junctions,
            "focus": self._focus(step, focus, longest),
        }

    def _focus(self, step: int, j: int, longest: int) -> dict[str, Any]:
        """Chart series for one junction: 6 h back, the longest horizon ahead."""
        start = max(0, step - CHART_PAST_STEPS)
        end = step + longest  # inclusive
        steps = range(start, end + 1)
        recorded = [_num(self.counts[s, j]) if s <= step and s < len(self.counts) else None for s in steps]
        earlier = [
            _num(self.forecasts[longest][s - longest, j])
            if s <= step and s - longest >= 0 and self._out_of_sample(s)
            else None
            for s in steps
        ]
        ahead: list[float | None] = [None] * len(steps)
        lower: list[float | None] = [None] * len(steps)
        upper: list[float | None] = [None] * len(steps)
        ahead[step - start] = recorded[step - start]  # join the forecast to the last reading
        for h in self.horizons:
            value = float(self.forecasts[h][step, j])
            lo, hi = self._interval(h, j, value)
            i = step + h - start
            ahead[i], lower[i], upper[i] = _num(value), _num(lo), _num(hi)

        # Accuracy of the longest-horizon forecasts so far on this replay day.
        day_start = step - step % STEPS_PER_DAY
        targets = np.arange(max(day_start, self.replay.first_step), step + 1)
        targets = targets[targets - longest >= 0]
        actual = self.counts[targets, j]
        predicted = self.forecasts[longest][targets - longest, j]
        ok = ~np.isnan(actual)
        mae = float(np.abs(predicted[ok] - actual[ok]).mean()) if ok.any() else None

        return {
            "index": j,
            "name": self.junctions[j],
            "labels": [step_clock(s) for s in steps],
            "now_index": step - start,
            "recorded": recorded,
            "typical": [_num(self.typical[j, s % STEPS_PER_DAY]) for s in steps],
            "earlier_forecast": earlier,
            "earlier_minutes": horizon_minutes(longest),
            "ahead": ahead,
            "ahead_lower": lower,
            "ahead_upper": upper,
            "today_mae": _num(mae),
            "today_points": int(ok.sum()),
            "thresholds": [_num(v) for v in self.forecaster.level_thresholds[j]],
        }

    # ------------------------------------------------------------------ export
    def day_csv(self, day: int) -> str:
        """Every forecast made on one replay day, with the readings that followed."""
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        header = ["day", "time", "junction", "reading"]
        for h in self.horizons:
            m = horizon_minutes(h)
            header += [f"forecast_{m}min", f"lower_{m}min", f"upper_{m}min", f"actual_{m}min"]
        writer.writerow(header)
        first = self.step_for(day, 0)
        for step in range(first, first + STEPS_PER_DAY):
            for j, name in enumerate(self.junctions):
                row: list[Any] = [day, step_clock(step), name, _num(self.counts[step, j])]
                for h in self.horizons:
                    f = self._forecast(step, h, j)
                    row += [f["value"], f["lower"], f["upper"], f.get("actual")]
                writer.writerow(["" if v is None else v for v in row])
        return out.getvalue()

    def meta(self) -> dict[str, Any]:
        return {
            "junctions": self.junctions,
            "horizons_minutes": [horizon_minutes(h) for h in self.horizons],
            "step_minutes": STEP_MINUTES,
            "replay": {"first_day": self.replay.first_day, "last_day": self.replay.last_day},
            "levels": list(LEVELS),
            "level_thresholds": {
                name: [_num(v) for v in self.forecaster.level_thresholds[j]]
                for j, name in enumerate(self.junctions)
            },
            "model": {
                "estimator": self.forecaster.info.get("estimator"),
                "created_at": self.forecaster.info.get("created_at"),
                "fit_days": self.forecaster.info.get("fit_days"),
            },
        }


# --------------------------------------------------------------------------
# App integration: loaded once per process, shared by every request thread.
# --------------------------------------------------------------------------

_lock = threading.Lock()


def build_service(data_path: str | Path, model_path: str | Path, metrics_path: str | Path | None) -> ForecastService:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found: train the model with `python -m scripts.train`")
    metrics = None
    if metrics_path and Path(metrics_path).exists():
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    return ForecastService(load_dataset(data_path), Forecaster.load(model_path), metrics)


def get_service(app: Flask | None = None) -> ForecastService:
    app = app or current_app._get_current_object()  # type: ignore[attr-defined]
    service = app.extensions.get(EXTENSION_KEY)
    if service is not None:
        return service
    with _lock:
        service = app.extensions.get(EXTENSION_KEY)
        if service is None:
            cfg = app.config
            try:
                service = build_service(cfg["DATA_PATH"], cfg["MODEL_PATH"], cfg["METRICS_PATH"])
            except (OSError, RuntimeError, ValueError) as exc:
                app.logger.error("forecasting unavailable: %s", exc)
                raise ServiceUnavailable(
                    description="Forecasts are unavailable because the model has not been trained "
                    "on this server. An administrator can run `python -m scripts.train`."
                ) from exc
            app.extensions[EXTENSION_KEY] = service
            app.logger.info(
                "forecaster loaded: %d junctions, replay days %d-%d",
                len(service.junctions), service.replay.first_day, service.replay.last_day,
            )
    return service


def init_service(app: Flask) -> None:
    if app.config.get("PRELOAD_MODEL"):
        try:
            get_service(app)
        except ServiceUnavailable:
            pass  # logged; pages that need forecasts explain the problem
