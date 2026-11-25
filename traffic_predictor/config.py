"""Application settings.

Every setting can be overridden with an environment variable prefixed ``TP_``
(for example ``TP_SECRET_KEY``), or with a mapping passed to ``create_app``.
"""

from __future__ import annotations

import os
import secrets
from datetime import timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def default_config(instance_path: Path) -> dict[str, Any]:
    return {
        "DATABASE": os.environ.get("TP_DATABASE", str(instance_path / "traffic_predictor.sqlite3")),
        "DATA_PATH": os.environ.get(
            "TP_DATA_PATH", str(PROJECT_ROOT / "data" / "traffic-prediction-dataset.csv")
        ),
        "MODEL_PATH": os.environ.get("TP_MODEL_PATH", str(PROJECT_ROOT / "models" / "forecaster.joblib")),
        "METRICS_PATH": os.environ.get("TP_METRICS_PATH", str(PROJECT_ROOT / "models" / "metrics.json")),
        # Sessions. Set TP_SECURE_COOKIES=1 whenever the app is served over HTTPS.
        "SESSION_COOKIE_NAME": "tp_session",
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "SESSION_COOKIE_SECURE": _env_bool("TP_SECURE_COOKIES", False),
        "PERMANENT_SESSION_LIFETIME": timedelta(hours=_env_int("TP_SESSION_HOURS", 12)),
        "MAX_CONTENT_LENGTH": 64 * 1024,  # forms only; nothing large is ever uploaded
        # Accounts.
        "REGISTRATION_OPEN": _env_bool("TP_REGISTRATION_OPEN", True),
        "LOGIN_MAX_FAILURES": _env_int("TP_LOGIN_MAX_FAILURES", 8),
        "LOGIN_WINDOW_SECONDS": _env_int("TP_LOGIN_WINDOW_SECONDS", 15 * 60),
        # Set to the number of reverse proxies in front of the app (e.g. 1 behind nginx)
        # so client IPs used for rate limiting come from X-Forwarded-For.
        "TRUSTED_PROXIES": _env_int("TP_TRUSTED_PROXIES", 0),
        # Load the model at start-up so the first dashboard request is not slow.
        "PRELOAD_MODEL": _env_bool("TP_PRELOAD_MODEL", True),
        # Static files are referenced with a content version (see `asset()`), so
        # browsers may keep them; CSS-referenced files refresh within a day.
        "SEND_FILE_MAX_AGE_DEFAULT": _env_int("TP_STATIC_MAX_AGE", 24 * 3600),
        "LOG_LEVEL": os.environ.get("TP_LOG_LEVEL", "INFO").upper(),
    }


def load_secret_key(instance_path: Path) -> str:
    """TP_SECRET_KEY if set; otherwise a random key kept in the instance folder.

    The file keeps sessions valid across restarts on a single host. For several
    hosts or containers, set TP_SECRET_KEY to the same value everywhere.
    """
    from_env = os.environ.get("TP_SECRET_KEY")
    if from_env:
        return from_env
    key_file = instance_path / "secret_key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    instance_path.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32)
    key_file.write_text(key, encoding="utf-8")
    try:
        key_file.chmod(0o600)
    except OSError:  # not supported on every filesystem
        pass
    return key
