from fastapi.testclient import TestClient
from src.app.main import app
import os
import pytest

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["message"] == "Wine Quality Prediction API"
    assert "mode" in json_response

def test_predict_structure():
    payload = {
        "alcohol": 13.2,
        "malic_acid": 1.78,
        "ash": 2.14,
        "alcalinity_of_ash": 11.2,
        "magnesium": 100.0,
        "total_phenols": 2.65,
        "flavanoids": 2.76,
        "nonflavanoid_phenols": 0.26,
        "proanthocyanins": 1.28,
        "color_intensity": 4.38,
        "hue": 1.05,
        "od280_od315_of_diluted_wines": 3.4,
        "proline": 1050.0
    }
    
    # This test assumes the app can run (either with local model or triton)
    # We are just checking if the endpoint responds structurally correct or handles error gracefully
    response = client.post("/predict", json=payload)
    
    # It might be 200 (success) or 500 (model load failure/triton failure)
    # But we want to ensure it doesn't 422 (validation error)
    assert response.status_code != 422
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], float)
