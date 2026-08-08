import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_engineering import build_features, encode_for_model


def _sample_row(**overrides):
    row = {
        "customerID": "0000-TEST", "gender": "Female", "SeniorCitizen": 0,
        "Partner": "Yes", "Dependents": "No", "tenure": 5,
        "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 79.85,
        "TotalCharges": "399.25",
    }
    row.update(overrides)
    return row


def test_build_features_adds_expected_columns():
    df = pd.DataFrame([_sample_row()])
    out = build_features(df)
    for col in ["tenure_band", "avg_monthly_spend_per_service",
                "is_new_customer", "contract_risk_score",
                "total_to_monthly_ratio"]:
        assert col in out.columns


def test_build_features_single_row_does_not_crash():
    """Serving calls this with exactly 1 row - must not require batch context."""
    df = pd.DataFrame([_sample_row()])
    out = build_features(df)
    assert len(out) == 1


def test_totalcharges_blank_handled_as_zero():
    df = pd.DataFrame([_sample_row(tenure=0, TotalCharges=" ")])
    out = build_features(df)
    assert out["TotalCharges"].iloc[0] == 0.0


def test_is_new_customer_flag():
    new_cust = pd.DataFrame([_sample_row(tenure=1)])
    old_cust = pd.DataFrame([_sample_row(tenure=50)])
    assert build_features(new_cust)["is_new_customer"].iloc[0] == 1
    assert build_features(old_cust)["is_new_customer"].iloc[0] == 0


def test_contract_risk_score_ordering():
    """Month-to-month should score higher risk than two-year contract."""
    mtm = pd.DataFrame([_sample_row(Contract="Month-to-month", PaperlessBilling="No")])
    two_yr = pd.DataFrame([_sample_row(Contract="Two year", PaperlessBilling="No")])
    mtm_score = build_features(mtm)["contract_risk_score"].iloc[0]
    two_yr_score = build_features(two_yr)["contract_risk_score"].iloc[0]
    assert mtm_score > two_yr_score


def test_missing_required_column_raises():
    df = pd.DataFrame([{"gender": "Female"}])
    with pytest.raises(ValueError):
        build_features(df)


def test_encode_for_model_train_then_serve_shape_matches():
    """The core training-serving skew guarantee: a single-row encode
    using saved encoders must produce the same columns as training."""
    train_df = pd.DataFrame([_sample_row(), _sample_row(gender="Male", Contract="Two year")])
    train_feat = build_features(train_df)
    X_train, encoders = encode_for_model(train_feat, encoders=None)

    serve_df = pd.DataFrame([_sample_row()])
    serve_feat = build_features(serve_df)
    X_serve, _ = encode_for_model(serve_feat, encoders=encoders)

    assert list(X_train.columns) == list(X_serve.columns)
    assert len(X_serve) == 1
