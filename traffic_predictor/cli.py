"""Admin commands: `flask --app traffic_predictor <command>`."""

from __future__ import annotations

import secrets
import sqlite3

import click
from flask import Flask
from werkzeug.security import generate_password_hash

from .auth import EMAIL_RE, USERNAME_RE, password_problems
from .db import get_db


def init_cli(app: Flask) -> None:
    @app.cli.command("create-user")
    @click.argument("username")
    @click.argument("email")
    @click.password_option(help="Prompted for (hidden) when not given.")
    def create_user(username: str, email: str, password: str) -> None:
        """Create an account, e.g. while TP_REGISTRATION_OPEN=0."""
        problems = []
        if not USERNAME_RE.match(username):
            problems.append("Username must be 3-32 characters: letters, digits, '.', '_' or '-'.")
        if len(email) > 254 or not EMAIL_RE.match(email):
            problems.append("Enter a valid email address.")
        problems += password_problems(password, password)
        if problems:
            raise click.UsageError(" ".join(problems))
        db = get_db()
        try:
            db.execute(
                "INSERT INTO user (username, email, password_hash, session_nonce) VALUES (?, ?, ?, ?)",
                (username, email.lower(), generate_password_hash(password), secrets.token_hex(16)),
            )
            db.commit()
        except sqlite3.IntegrityError:
            raise click.ClickException("That username or email is already registered.") from None
        click.echo(f"Created account {username}.")
