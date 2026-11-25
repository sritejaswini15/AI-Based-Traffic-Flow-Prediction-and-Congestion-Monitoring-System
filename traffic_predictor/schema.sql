-- Applied on every start; every statement is idempotent.

CREATE TABLE IF NOT EXISTS user (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    email         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT    NOT NULL,
    -- Rotated on password change: sessions carrying an older value are rejected.
    session_nonce TEXT    NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    last_login_at TEXT
);

-- Failed sign-in attempts, for rate limiting (pruned as they expire).
CREATE TABLE IF NOT EXISTS login_failure (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    bucket       TEXT    NOT NULL,
    attempted_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_login_failure_bucket ON login_failure (bucket, attempted_at);
