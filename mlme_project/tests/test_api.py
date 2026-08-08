import os
import sys

from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving.app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85, "TotalCharges": "399.25",
}


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid_payload():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)
    assert "model_version" in body


def test_predict_rejects_invalid_categorical():
    bad = dict(VALID_PAYLOAD)
    bad["gender"] = "NotAGender"
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_missing_field():
    bad = dict(VALID_PAYLOAD)
    del bad["tenure"]
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_high_risk_profile_scores_higher_than_low_risk():
    """Sanity check the model direction makes business sense:
    month-to-month + new tenure should score higher churn risk than
    a long-tenure two-year-contract customer, all else similar."""
    high_risk = dict(VALID_PAYLOAD)  # month-to-month, tenure=5
    low_risk = dict(VALID_PAYLOAD, Contract="Two year", tenure=60,
                     PaperlessBilling="No", TotalCharges="4800.0")

    high_resp = client.post("/predict", json=high_risk).json()
    low_resp = client.post("/predict", json=low_risk).json()

    assert high_resp["churn_probability"] > low_resp["churn_probability"]
