"""Shared fixtures: a synthetic dataset, a quickly trained model, an app."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_predictor.forecasting import model as model_module
from traffic_predictor.forecasting.data import STEPS_PER_DAY

N_DAYS = 56
JUNCTIONS = ["Cross 1", "Cross 2", "Cross 3"]
PASSWORD = "correct horse battery"


def make_counts(n_days: int = N_DAYS, seed: int = 0) -> pd.DataFrame:
    """Daily-cycle counts with noise and one all-junction outage on day 3."""
    rng = np.random.default_rng(seed)
    steps = np.arange(n_days * STEPS_PER_DAY)
    slot = steps % STEPS_PER_DAY
    daily = 1 + np.sin(2 * np.pi * (slot - 72) / STEPS_PER_DAY)  # low near 00:00, high near 12:00
    columns = {}
    for j, name in enumerate(JUNCTIONS):
        mean = 10 + (30 + 15 * j) * daily
        columns[name] = rng.poisson(mean).astype(float)
    df = pd.DataFrame(columns)
    outage = slice(2 * STEPS_PER_DAY + 100, 2 * STEPS_PER_DAY + 130)  # 30 steps of zeros
    df.iloc[outage, :] = 0.0
    return df


@pytest.fixture(scope="session")
def trained(tmp_path_factory) -> dict[str, Path]:
    """Run the real training script on synthetic data, with a small model."""
    from scripts import train

    root = tmp_path_factory.mktemp("trained")
    data_path = root / "counts.csv"
    df = make_counts()
    df.columns = [JUNCTIONS[0]] + [" " + c for c in JUNCTIONS[1:]]  # like the real file's headers
    df.to_csv(data_path, index=False)

    saved = dict(model_module._BASE_PARAMS)
    model_module._BASE_PARAMS.update(max_iter=30, learning_rate=0.3)  # small but not underfit
    try:
        assert train.main(["--data", str(data_path), "--out", str(root)]) == 0
    finally:
        model_module._BASE_PARAMS.clear()
        model_module._BASE_PARAMS.update(saved)
    return {"data": data_path, "model": root / "forecaster.joblib", "metrics": root / "metrics.json", "root": root}


@pytest.fixture()
def app(trained, tmp_path):
    from traffic_predictor import create_app

    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "DATABASE": str(tmp_path / "test.sqlite3"),
            "DATA_PATH": str(trained["data"]),
            "MODEL_PATH": str(trained["model"]),
            "METRICS_PATH": str(trained["metrics"]),
            "PRELOAD_MODEL": False,
            "LOGIN_MAX_FAILURES": 3,
        }
    )
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


_CSRF_RE = re.compile(r'name="csrf_token" value="([^"]+)"')


def csrf_from(client, path: str = "/auth/login") -> str:
    page = client.get(path).get_data(as_text=True)
    match = _CSRF_RE.search(page)
    assert match, f"no CSRF token on {path}"
    return match.group(1)


def register(client, username: str = "alice", email: str = "alice@example.com", password: str = PASSWORD):
    token = csrf_from(client, "/auth/register")
    return client.post(
        "/auth/register",
        data={"csrf_token": token, "username": username, "email": email, "password": password, "confirm": password},
    )


def login(client, identifier: str = "alice", password: str = PASSWORD):
    token = csrf_from(client, "/auth/login")
    return client.post("/auth/login", data={"csrf_token": token, "identifier": identifier, "password": password})


@pytest.fixture()
def auth_client(client):
    response = register(client)
    assert response.status_code == 302
    return client
