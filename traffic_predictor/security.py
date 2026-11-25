"""CSRF protection, security headers and sign-in rate limiting."""

from __future__ import annotations

import hmac
import secrets
import time

from flask import Flask, abort, current_app, request, session

from .db import get_db

CSRF_SESSION_KEY = "_csrf"
CSRF_FORM_FIELD = "csrf_token"
CSRF_HEADER = "X-CSRF-Token"
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "font-src 'self'",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ]
)


def csrf_token() -> str:
    """The session's CSRF token, created on first use."""
    token = session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return token


def _check_csrf() -> None:
    if request.method not in UNSAFE_METHODS:
        return
    expected = session.get(CSRF_SESSION_KEY)
    sent = request.form.get(CSRF_FORM_FIELD) or request.headers.get(CSRF_HEADER)
    if not expected or not sent or not hmac.compare_digest(expected, sent):
        abort(400, description="Your form expired or was sent from another site. Reload the page and try again.")


def _set_headers(response):
    headers = response.headers
    headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
    headers.setdefault("X-Content-Type-Options", "nosniff")
    headers.setdefault("X-Frame-Options", "DENY")
    headers.setdefault("Referrer-Policy", "same-origin")
    headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()")
    if current_app.config.get("SESSION_COOKIE_SECURE"):
        headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    if request.endpoint != "static" and "Cache-Control" not in headers:
        # Pages show per-user content; never let a shared cache keep them.
        headers["Cache-Control"] = "no-store"
    return response


def init_security(app: Flask) -> None:
    app.before_request(_check_csrf)
    app.after_request(_set_headers)
    app.jinja_env.globals["csrf_token"] = csrf_token
    if app.config.get("TRUSTED_PROXIES"):
        from werkzeug.middleware.proxy_fix import ProxyFix

        n = int(app.config["TRUSTED_PROXIES"])
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=n, x_proto=n, x_host=n)


# --------------------------------------------------------------------------
# Sign-in rate limiting
# --------------------------------------------------------------------------
# Failures are counted per client address and per account name, so one client
# cannot guess many passwords and many clients cannot guess one account's. The
# counts live in SQLite, so every worker process sees the same numbers.

def _buckets(identifier: str) -> list[str]:
    ip = request.remote_addr or "unknown"
    return [f"ip:{ip}", f"account:{identifier.strip().lower()}"]


def login_blocked(identifier: str) -> bool:
    cfg = current_app.config
    since = time.time() - cfg["LOGIN_WINDOW_SECONDS"]
    db = get_db()
    db.execute("DELETE FROM login_failure WHERE attempted_at < ?", (since,))
    db.commit()
    for bucket in _buckets(identifier):
        (count,) = db.execute(
            "SELECT COUNT(*) FROM login_failure WHERE bucket = ? AND attempted_at >= ?",
            (bucket, since),
        ).fetchone()
        if count >= cfg["LOGIN_MAX_FAILURES"]:
            return True
    return False


def record_login_failure(identifier: str) -> None:
    db = get_db()
    now = time.time()
    db.executemany(
        "INSERT INTO login_failure (bucket, attempted_at) VALUES (?, ?)",
        [(b, now) for b in _buckets(identifier)],
    )
    db.commit()


def clear_login_failures(identifier: str) -> None:
    db = get_db()
    db.execute(
        "DELETE FROM login_failure WHERE bucket = ?", (_buckets(identifier)[1],)
    )
    db.commit()
