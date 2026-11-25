"""Traffic Predictor: short-term vehicle-count forecasts for six road junctions.

Development:  flask --app traffic_predictor run --debug
Production:   waitress-serve --call traffic_predictor:create_app
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.exceptions import HTTPException, InternalServerError

from .config import default_config, load_secret_key

__version__ = "2.0.0"


def create_app(test_config: Mapping[str, Any] | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    instance_path = Path(app.instance_path)
    app.config.from_mapping(default_config(instance_path))
    if test_config:
        app.config.from_mapping(test_config)
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = load_secret_key(instance_path)

    _configure_logging(app)

    from . import api, auth, cli, db, security, service, views

    db.init_db(app)
    security.init_security(app)
    app.register_blueprint(auth.bp)
    app.register_blueprint(views.bp)
    app.register_blueprint(api.bp)
    cli.init_cli(app)
    _register_error_handlers(app)
    _register_template_helpers(app)
    service.init_service(app)
    return app


def _configure_logging(app: Flask) -> None:
    level = getattr(logging, str(app.config.get("LOG_LEVEL", "INFO")), logging.INFO)
    app.logger.setLevel(level)
    if not app.debug and not any(isinstance(h, logging.StreamHandler) for h in app.logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        app.logger.addHandler(handler)


def _wants_json() -> bool:
    return request.path.startswith("/api/") or request.path == "/healthz"


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        if _wants_json():
            return jsonify(error=exc.description, status=exc.code), exc.code
        return render_template("error.html", error=exc), exc.code

    @app.errorhandler(Exception)
    def unexpected_error(exc: Exception):
        if isinstance(exc, HTTPException):  # pragma: no cover - handled above
            return http_error(exc)
        app.logger.exception("unhandled error on %s %s", request.method, request.path)
        return http_error(InternalServerError())


def _register_template_helpers(app: Flask) -> None:
    versions: dict[str, str] = {}
    static_root = Path(app.static_folder or "")

    def asset(filename: str) -> str:
        """URL of a static file with a content hash, so cached copies never go stale."""
        if filename not in versions or app.debug:
            try:
                digest = hashlib.sha256((static_root / filename).read_bytes()).hexdigest()[:10]
            except OSError:
                digest = __version__
            versions[filename] = digest
        return url_for("static", filename=filename, v=versions[filename])

    app.jinja_env.globals.update(asset=asset, app_version=__version__)
