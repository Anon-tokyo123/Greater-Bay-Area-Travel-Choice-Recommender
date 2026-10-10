"""Smoke tests for the API's synthetic demo mode; no survey workbook is used."""

import os

os.environ["GBA_DEMO_MODE"] = "1"
os.environ["GBA_SURVEY_PATH"] = "demo-mode-must-not-read-a-survey.xlsx"

from fastapi.testclient import TestClient

from api import app, get_long_data, models_cache

client = TestClient(app)


def setup_function():
    get_long_data.cache_clear()
    models_cache.clear()


def test_root_identifies_synthetic_demo_mode():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["data_source"] == "synthetic_demo"
    assert "Synthetic demo only" in response.json()["notice"]


def test_vtts_endpoint_runs_without_private_workbook():
    response = client.get("/vtts", params={"purpose": "Work"})

    assert response.status_code == 200
    body = response.json()
    assert body["data_source"] == "synthetic_demo"
    assert set(body["vtts_hkd_per_hour"]) == {
        "Short (50km)", "Medium (100km)", "Long (150km)"
    }


def test_predict_choice_returns_binary_score_in_demo_mode():
    response = client.post(
        "/predict_choice",
        params={
            "purpose": "Work",
            "distance": "Short (50km)",
            "mode": "HSR",
        },
        json={
            "fare": 120,
            "in_vehicle_time": 45,
            "waiting_time": 15,
            "access_egress_time": 18,
            "transfer_time": 4,
            "crowding_level": 0.3,
            "customs_clearance_time": 15,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data_source"] == "synthetic_demo"
    assert body["tested_mode"] == "HSR"
    assert 0 <= body["estimated_binary_choice_probability"] <= 1
    assert "not normalized" in body["interpretation"]


def test_predict_choice_rejects_crowding_above_one():
    response = client.post(
        "/predict_choice",
        params={
            "purpose": "Work",
            "distance": "Short (50km)",
            "mode": "HSR",
        },
        json={
            "fare": 120,
            "in_vehicle_time": 45,
            "waiting_time": 15,
            "access_egress_time": 18,
            "transfer_time": 4,
            "crowding_level": 1.1,
            "customs_clearance_time": 15,
        },
    )

    assert response.status_code == 422
