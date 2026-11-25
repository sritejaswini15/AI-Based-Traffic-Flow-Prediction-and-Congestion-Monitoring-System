"""Pages, accounts, security and the JSON API, through the Flask test client."""

from __future__ import annotations

from traffic_predictor import create_app

from .conftest import PASSWORD, csrf_from, login, register


# ------------------------------------------------------------------ pages
def test_public_pages(client):
    assert client.get("/").status_code == 200
    model = client.get("/model")
    assert model.status_code == 200
    assert "Validation days" in model.get_data(as_text=True)
    assert client.get("/auth/login").status_code == 200
    assert client.get("/auth/register").status_code == 200


def test_dashboard_requires_sign_in(client):
    response = client.get("/dashboard")
    assert response.status_code == 302
    assert "/auth/login?next=/dashboard" in response.headers["Location"]


def test_dashboard_for_signed_in_user(auth_client):
    page = auth_client.get("/dashboard")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert 'data-first-day="49"' in html and "Cross 3" in html


def test_not_found_page(client):
    response = client.get("/no-such-page")
    assert response.status_code == 404
    assert "Not Found" in response.get_data(as_text=True)


def test_health(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is True


def test_missing_model_degrades_cleanly(trained, tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "x",
            "DATABASE": str(tmp_path / "db.sqlite3"),
            "DATA_PATH": str(trained["data"]),
            "MODEL_PATH": str(tmp_path / "missing.joblib"),
            "METRICS_PATH": str(tmp_path / "missing.json"),
        }
    )
    client = app.test_client()
    health = client.get("/healthz")
    assert health.status_code == 503 and health.get_json()["model_loaded"] is False
    assert client.get("/").status_code == 200  # the rest of the site still works
    register(client)
    page = client.get("/dashboard")
    assert page.status_code == 503
    assert "python -m scripts.train" in page.get_data(as_text=True)


# ---------------------------------------------------------------- security
def test_security_headers(client):
    headers = client.get("/").headers
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Cache-Control"] == "no-store"


def test_static_assets_are_versioned_and_cacheable(client):
    html = client.get("/").get_data(as_text=True)
    assert "/static/css/app.css?v=" in html
    response = client.get("/static/css/app.css")
    assert response.status_code == 200
    assert "no-store" not in response.headers.get("Cache-Control", "")


def test_posts_without_csrf_are_rejected(client):
    response = client.post("/auth/login", data={"identifier": "alice", "password": PASSWORD})
    assert response.status_code == 400
    response = client.post("/auth/register", data={"username": "bob", "email": "b@example.com",
                                                    "password": PASSWORD, "confirm": PASSWORD,
                                                    "csrf_token": "forged"})
    assert response.status_code == 400


def test_no_inline_scripts_or_styles(auth_client):
    for path in ("/", "/dashboard", "/model", "/auth/account"):
        html = auth_client.get(path).get_data(as_text=True)
        assert "<script>" not in html and "style=" not in html and "<style" not in html, path


# ---------------------------------------------------------------- accounts
def test_register_sign_out_sign_in(client):
    response = register(client)
    assert response.status_code == 302 and response.headers["Location"].endswith("/dashboard")

    token = csrf_from(client, "/auth/account")
    assert client.post("/auth/logout", data={"csrf_token": token}).status_code == 302
    assert client.get("/dashboard").status_code == 302

    assert login(client, "ALICE@example.com").status_code == 302  # email, any case
    assert client.get("/dashboard").status_code == 200


def test_logout_needs_post(auth_client):
    assert auth_client.get("/auth/logout").status_code == 405
    assert auth_client.get("/dashboard").status_code == 200


def test_registration_validation(client):
    response = register(client, username="x", email="not-an-email", password="short")
    assert response.status_code == 400
    html = response.get_data(as_text=True)
    assert "Username must be" in html and "valid email" in html and "at least 8" in html

    assert register(client).status_code == 302
    token = csrf_from(client, "/auth/account")
    client.post("/auth/logout", data={"csrf_token": token})
    duplicate = register(client, username="Alice", email="other@example.com")
    assert duplicate.status_code == 400
    assert "already registered" in duplicate.get_data(as_text=True)


def test_registration_can_be_closed(trained, tmp_path):
    app = create_app({"TESTING": True, "SECRET_KEY": "x", "DATABASE": str(tmp_path / "db.sqlite3"),
                      "REGISTRATION_OPEN": False, "PRELOAD_MODEL": False})
    assert app.test_client().get("/auth/register").status_code == 403


def test_wrong_password_and_rate_limit(client):
    register(client)
    token = csrf_from(client, "/auth/account")
    client.post("/auth/logout", data={"csrf_token": token})
    for _ in range(3):  # LOGIN_MAX_FAILURES is 3 in tests
        assert login(client, "alice", "wrong password!").status_code == 401
    blocked = login(client, "alice", PASSWORD)
    assert blocked.status_code == 429  # even the right password waits out the window


def test_unknown_user_gets_same_message(client):
    response = login(client, "nobody", "whatever123")
    assert response.status_code == 401
    assert "Incorrect username/email or password." in response.get_data(as_text=True)


def test_next_must_stay_on_site(client):
    register(client)
    token = csrf_from(client, "/auth/account")
    client.post("/auth/logout", data={"csrf_token": token})
    token = csrf_from(client, "/auth/login")
    for target in ("https://evil.example", "//evil.example", "javascript:alert(1)"):
        response = client.post("/auth/login", data={"csrf_token": token, "identifier": "alice",
                                                     "password": PASSWORD, "next": target})
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/dashboard"), target
        token = csrf_from(client, "/auth/account")
        client.post("/auth/logout", data={"csrf_token": token})
        token = csrf_from(client, "/auth/login")


def test_password_change_signs_out_other_sessions(app):
    first, second = app.test_client(), app.test_client()
    register(first)
    login(second)
    assert second.get("/dashboard").status_code == 200

    token = csrf_from(first, "/auth/account")
    new = "a brand new passphrase"
    response = first.post("/auth/account", data={"csrf_token": token, "action": "change_password",
                                                  "current_password": PASSWORD, "new_password": new, "confirm": new})
    assert response.status_code == 302
    assert first.get("/dashboard").status_code == 200  # this session stays signed in
    assert second.get("/dashboard").status_code == 302  # the other one is signed out
    assert login(second, "alice", new).status_code == 302


def test_delete_account_needs_password_and_confirmation(auth_client):
    token = csrf_from(auth_client, "/auth/account")
    wrong = auth_client.post("/auth/account", data={"csrf_token": token, "action": "delete_account",
                                                    "current_password": "nope", "confirm_delete": "yes"})
    assert wrong.status_code == 400
    unticked = auth_client.post("/auth/account", data={"csrf_token": token, "action": "delete_account",
                                                       "current_password": PASSWORD})
    assert unticked.status_code == 400
    done = auth_client.post("/auth/account", data={"csrf_token": token, "action": "delete_account",
                                                   "current_password": PASSWORD, "confirm_delete": "yes"})
    assert done.status_code == 302
    assert auth_client.get("/dashboard").status_code == 302
    assert login(auth_client).status_code == 401


def test_cli_create_user(app):
    runner = app.test_cli_runner()
    result = runner.invoke(args=["create-user", "carol", "carol@example.com", "--password", PASSWORD])
    assert result.exit_code == 0, result.output
    duplicate = runner.invoke(args=["create-user", "carol", "c2@example.com", "--password", PASSWORD])
    assert duplicate.exit_code != 0
    assert login(app.test_client(), "carol").status_code == 302


# --------------------------------------------------------------------- API
def test_api_requires_sign_in(client):
    response = client.get("/api/v1/snapshot")
    assert response.status_code == 401
    assert response.get_json()["error"] == "authentication required"


def test_api_snapshot_live_and_fixed(auth_client):
    live = auth_client.get("/api/v1/snapshot").get_json()
    assert live["mode"] == "live" and 49 <= live["day"] <= 56

    fixed = auth_client.get("/api/v1/snapshot?day=51&time=17:45&junction=2").get_json()
    assert fixed["mode"] == "fixed" and fixed["day"] == 51 and fixed["clock"] == "17:45"
    assert fixed["focus"]["name"] == "Cross 3"
    assert "actual" not in fixed["junctions"][0]["forecasts"][0]

    revealed = auth_client.get("/api/v1/snapshot?day=51&time=17:45&outcomes=1").get_json()
    assert "actual" in revealed["junctions"][0]["forecasts"][0]


def test_api_rejects_bad_parameters(auth_client):
    for query in ("day=12&time=08:00", "day=50", "day=50&time=25:00", "day=x&time=08:00",
                  "day=50&time=08:00&junction=9"):
        response = auth_client.get("/api/v1/snapshot?" + query)
        assert response.status_code == 400, query
        assert "error" in response.get_json()


def test_api_meta_and_csv(auth_client):
    meta = auth_client.get("/api/v1/meta").get_json()
    assert meta["replay"] == {"first_day": 49, "last_day": 56}
    assert meta["horizons_minutes"] == [5, 15, 30, 60]

    response = auth_client.get("/api/v1/forecasts/50.csv")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert "forecasts-day-50.csv" in response.headers["Content-Disposition"]
    assert auth_client.get("/api/v1/forecasts/3.csv").status_code == 404
