"""SQLite access: one connection per request, schema applied at start-up."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from flask import Flask, current_app, g


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = _connect(current_app.config["DATABASE"])
    return g.db


def close_db(_exc: BaseException | None = None) -> None:
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db(app: Flask) -> None:
    path = app.config["DATABASE"]
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    schema = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    conn = _connect(path)
    try:
        if path != ":memory:":
            # WAL lets readers proceed while a sign-up or sign-in writes.
            conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(schema)
        conn.commit()
    finally:
        conn.close()
    app.teardown_appcontext(close_db)
