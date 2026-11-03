"""Smoke tests for the Flask dashboard (app.py)."""

import io
import os
import subprocess
import sys

import pytest

import app


@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    return app.app.test_client()


def test_index_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Churn Predictor" in r.data


def test_predict_json_valid(client):
    r = client.post(
        "/predict",
        json={
            "tenure": 3,
            "monthly_charges": 95.5,
            "total_charges": 286.5,
            "contract_type": "Month-to-month",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
            "senior_citizen": 0,
            "partner": 0,
            "dependents": 0,
            "support_calls": 4,
        },
    )
    assert r.status_code == 200
    body = r.get_json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in ("churn", "stay")


def test_predict_json_missing_field(client):
    r = client.post("/predict", json={"tenure": 3})
    assert r.status_code == 400
    assert "missing fields" in r.get_json()["error"]


def _form_data(**overrides):
    data = {
        "tenure": "3",
        "monthly_charges": "95.5",
        "total_charges": "286.5",
        "support_calls": "4",
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
    }
    data.update(overrides)
    return data


def test_form_predict_at_risk(client):
    r = client.post("/", data=_form_data())
    assert r.status_code == 200
    assert b"Likely to churn" in r.data


def test_form_predict_loyal(client):
    r = client.post(
        "/",
        data=_form_data(
            tenure="60",
            monthly_charges="45.0",
            total_charges="2700.0",
            support_calls="0",
            contract_type="Two year",
            internet_service="DSL",
            payment_method="Credit card",
            partner="on",
            dependents="on",
        ),
    )
    assert r.status_code == 200
    assert b"Likely to stay" in r.data


def test_explore_loads(client):
    r = client.get("/explore")
    assert r.status_code == 200
    assert b"Gradient boosting" in r.data


def test_batch_scores_csv(client):
    csv = (
        "customer_id,tenure,monthly_charges,total_charges,contract_type,"
        "internet_service,payment_method,senior_citizen,partner,dependents,"
        "support_calls,churn\n"
        "C1,3,95.5,286.5,Month-to-month,Fiber optic,Electronic check,0,0,0,4,1\n"
        "C2,60,45.0,2700.0,Two year,DSL,Credit card,0,1,1,0,0\n"
    )
    r = client.post(
        "/batch",
        data={"csv": (io.BytesIO(csv.encode()), "two.csv")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    assert b"2" in r.data  # rows scored
    assert b"churn" in r.data and b"stay" in r.data  # per-row verdicts


def test_batch_sample_downloads(client):
    r = client.get("/batch/sample")
    assert r.status_code == 200
    assert r.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert b"customer_id" in r.data


def test_figures_degrade_when_missing(monkeypatch, client):
    # PNGs are never pushed to git - the page must degrade to hints
    monkeypatch.setattr("app.os.path.exists", lambda p: False)
    r = client.get("/explore")
    assert r.status_code == 200
    assert b"Figure not generated yet" in r.data
    assert b"<img" not in r.data


def test_app_fails_fast_without_model(tmp_path):
    # run app.py from a directory with no models/ - must exit with a
    # clear, actionable message instead of a bare FileNotFoundError
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.py")
    proc = subprocess.run(
        [sys.executable, app_path],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "model not found" in proc.stderr
    assert "python train.py" in proc.stderr
