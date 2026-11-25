"""HTML pages: landing, dashboard, model report, health check."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template
from werkzeug.exceptions import ServiceUnavailable

from . import __version__
from .auth import login_required
from .service import get_service

bp = Blueprint("views", __name__)


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/dashboard")
@login_required
def dashboard():
    service = get_service()
    return render_template("dashboard.html", meta=service.meta())


@bp.get("/model")
def model_report():
    """How the model was built and how well it does: public, no account needed."""
    service = get_service()
    return render_template("model.html", metrics=service.metrics, meta=service.meta())


@bp.get("/healthz")
def health():
    """Liveness and readiness for a load balancer or uptime monitor."""
    try:
        get_service()
    except ServiceUnavailable:
        return jsonify(status="degraded", model_loaded=False, version=__version__), 503
    return jsonify(status="ok", model_loaded=True, version=__version__)
