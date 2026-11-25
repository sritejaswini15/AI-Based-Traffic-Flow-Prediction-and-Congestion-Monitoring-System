"""JSON API used by the dashboard (signed-in users only, read-only).

GET /api/v1/meta                       junctions, horizons, replay range
GET /api/v1/snapshot                   the replay step that matches the clock now
GET /api/v1/snapshot?day=50&time=08:30 a chosen replay moment
    &junction=2                        which junction the chart focuses on
    &outcomes=1                        also return what actually happened next
GET /api/v1/forecasts/<day>.csv        every forecast of one replay day
"""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, Response, abort, jsonify, request

from .auth import login_required
from .service import get_service, parse_clock

bp = Blueprint("api", __name__, url_prefix="/api/v1")


@bp.before_request
@login_required
def _require_login():
    return None


def _int_arg(name: str, default: int | None = None) -> int | None:
    raw = request.args.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        abort(400, description=f"{name} must be a whole number")


@bp.get("/meta")
def meta():
    return jsonify(get_service().meta())


@bp.get("/snapshot")
def snapshot():
    service = get_service()
    day = _int_arg("day")
    clock = request.args.get("time")
    focus = _int_arg("junction", 0)
    outcomes = request.args.get("outcomes") in {"1", "true", "yes"}
    try:
        if day is None and clock is None:
            step, mode = service.live_step(datetime.now()), "live"
        elif day is None or clock is None:
            abort(400, description="give both day and time, or neither to follow the clock")
        else:
            step, mode = service.step_for(day, parse_clock(clock)), "fixed"
        data = service.snapshot(step, focus, include_outcomes=outcomes)
    except ValueError as exc:
        abort(400, description=str(exc))
    data["mode"] = mode
    return jsonify(data)


@bp.get("/forecasts/<int:day>.csv")
def forecasts_csv(day: int):
    service = get_service()
    try:
        body = service.day_csv(day)
    except ValueError as exc:
        abort(404, description=str(exc))
    return Response(
        body,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="forecasts-day-{day}.csv"'},
    )
