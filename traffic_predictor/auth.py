"""Accounts: sign up, sign in, sign out, change password, delete account.

Passwords are stored only as salted scrypt hashes (werkzeug). The session
cookie is signed and HttpOnly; it carries the user id and a per-user nonce
that is rotated on password change, which signs out every other session.
"""

from __future__ import annotations

import functools
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlsplit

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from .db import get_db
from .security import clear_login_failures, login_blocked, record_login_failure

bp = Blueprint("auth", __name__, url_prefix="/auth")

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")
PASSWORD_MIN, PASSWORD_MAX = 8, 128

# A real hash of a random password: checked when the account does not exist, so
# a failed sign-in takes the same time whether or not the name is registered.
_DUMMY_HASH = generate_password_hash(secrets.token_hex(16))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _start_session(user: sqlite3.Row) -> None:
    session.clear()  # drop anything from before sign-in (fixation)
    session.permanent = True
    session["user_id"] = user["id"]
    session["nonce"] = user["session_nonce"]


def _safe_next(target: str | None) -> str | None:
    """Only same-site relative paths are followed after sign-in."""
    if not target:
        return None
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or not target.startswith("/") or target.startswith("//"):
        return None
    return target


def password_problems(password: str, confirm: str) -> list[str]:
    problems = []
    if len(password) < PASSWORD_MIN:
        problems.append(f"Password must be at least {PASSWORD_MIN} characters.")
    elif len(password) > PASSWORD_MAX:
        problems.append(f"Password must be at most {PASSWORD_MAX} characters.")
    if password != confirm:
        problems.append("The two passwords do not match.")
    return problems


@bp.before_app_request
def load_logged_in_user() -> None:
    g.user = None
    user_id = session.get("user_id")
    if user_id is None or request.endpoint == "static":
        return
    user = get_db().execute(
        "SELECT id, username, email, session_nonce, created_at, last_login_at FROM user WHERE id = ?",
        (user_id,),
    ).fetchone()
    if user is None or not secrets.compare_digest(str(session.get("nonce", "")), user["session_nonce"]):
        session.clear()  # account deleted or password changed elsewhere
        return
    g.user = user


def login_required(view: Callable) -> Callable:
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            if request.path.startswith("/api/"):
                return jsonify(error="authentication required"), 401
            return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))
        return view(*args, **kwargs)

    return wrapped


@bp.route("/register", methods=["GET", "POST"])
def register():
    if g.user is not None:
        return redirect(url_for("views.dashboard"))
    if not current_app.config["REGISTRATION_OPEN"]:
        abort(403, description="New accounts are not being accepted on this server.")

    form = {"username": "", "email": ""}
    errors: list[str] = []
    if request.method == "POST":
        form["username"] = request.form.get("username", "").strip()
        form["email"] = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not USERNAME_RE.match(form["username"]):
            errors.append("Username must be 3–32 characters: letters, digits, '.', '_' or '-'.")
        if len(form["email"]) > 254 or not EMAIL_RE.match(form["email"]):
            errors.append("Enter a valid email address.")
        errors += password_problems(password, confirm)

        if not errors:
            db = get_db()
            try:
                cur = db.execute(
                    "INSERT INTO user (username, email, password_hash, session_nonce, last_login_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        form["username"],
                        form["email"].lower(),
                        generate_password_hash(password),
                        secrets.token_hex(16),
                        _now(),
                    ),
                )
                db.commit()
            except sqlite3.IntegrityError:
                errors.append("That username or email is already registered.")
            else:
                user = db.execute("SELECT * FROM user WHERE id = ?", (cur.lastrowid,)).fetchone()
                _start_session(user)
                current_app.logger.info("account created: id=%s", user["id"])
                flash(f"Welcome, {user['username']}. Your account is ready.", "success")
                return redirect(url_for("views.dashboard"))

    status = 400 if errors else 200
    return render_template("auth/register.html", form=form, errors=errors), status


@bp.route("/login", methods=["GET", "POST"])
def login():
    next_url = _safe_next(request.values.get("next"))
    if g.user is not None:
        return redirect(next_url or url_for("views.dashboard"))

    identifier = ""
    error = None
    status = 200
    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "")
        if login_blocked(identifier):
            error = "Too many failed attempts. Wait a few minutes, then try again."
            status = 429
        else:
            user = get_db().execute(
                "SELECT * FROM user WHERE username = ? OR email = ?",
                (identifier, identifier.lower()),
            ).fetchone()
            ok = check_password_hash(user["password_hash"] if user else _DUMMY_HASH, password)
            if user is not None and ok:
                clear_login_failures(identifier)
                db = get_db()
                db.execute("UPDATE user SET last_login_at = ? WHERE id = ?", (_now(), user["id"]))
                db.commit()
                _start_session(user)
                return redirect(next_url or url_for("views.dashboard"))
            record_login_failure(identifier)
            current_app.logger.warning("failed sign-in from %s", request.remote_addr)
            error = "Incorrect username/email or password."
            status = 401

    return (
        render_template("auth/login.html", identifier=identifier, error=error, next_url=next_url),
        status,
    )


@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("views.index"))


@bp.route("/account", methods=["GET", "POST"])
@login_required
def account():
    errors: list[str] = []
    status = 200
    if request.method == "POST":
        action = request.form.get("action")
        current = request.form.get("current_password", "")
        db = get_db()
        stored = db.execute("SELECT password_hash FROM user WHERE id = ?", (g.user["id"],)).fetchone()
        if not check_password_hash(stored["password_hash"], current):
            errors.append("Your current password is incorrect.")
        elif action == "change_password":
            new, confirm = request.form.get("new_password", ""), request.form.get("confirm", "")
            errors += password_problems(new, confirm)
            if not errors:
                nonce = secrets.token_hex(16)
                db.execute(
                    "UPDATE user SET password_hash = ?, session_nonce = ? WHERE id = ?",
                    (generate_password_hash(new), nonce, g.user["id"]),
                )
                db.commit()
                session["nonce"] = nonce  # keep this session; others are signed out
                flash("Password changed. Other devices have been signed out.", "success")
                return redirect(url_for("auth.account"))
        elif action == "delete_account" and request.form.get("confirm_delete") != "yes":
            errors.append("Tick the box to confirm that you want to delete your account.")
        elif action == "delete_account":
            db.execute("DELETE FROM user WHERE id = ?", (g.user["id"],))
            db.commit()
            current_app.logger.info("account deleted: id=%s", g.user["id"])
            session.clear()
            flash("Your account has been deleted.", "info")
            return redirect(url_for("views.index"))
        else:
            abort(400)
        status = 400
    return render_template("auth/account.html", errors=errors), status
