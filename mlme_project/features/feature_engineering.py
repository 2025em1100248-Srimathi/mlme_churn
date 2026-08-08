"""
features/feature_engineering.py

SINGLE SOURCE OF TRUTH for feature construction.

This module is imported by BOTH:
  - training/train.py (offline, batch transform over a DataFrame)
  - serving/app.py    (online, transform over a single request)

This is the project's answer to training-serving skew: instead of
maintaining two implementations (a notebook version for training and
a re-typed version in the API), every consumer calls the same
`build_features()` function. If the logic changes, it changes once,
for both paths.

Feature table:
- OFFLINE features (computed here, over historical data, cached to
  data/processed/features.parquet by the training pipeline):
    tenure_band, avg_monthly_spend_per_service, is_new_customer,
    contract_risk_score, total_to_monthly_ratio
- ONLINE-COMPATIBLE: every feature below is computable from fields
  present on a single incoming record (no joins against other users,
  no aggregation windows spanning multiple customers), so the exact
  same function runs at request time in the FastAPI service with no
  lookup table required. This is a deliberate design choice — see
  design doc, "Data and Feature Design" — a small feature store /
  lookup table would only be needed if we introduced cross-customer
  or time-windowed aggregates (e.g. "avg churn rate in this zip code
  over last 30 days"), which this v1 does not use.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

# Columns the raw source is expected to have. Used for validation.
RAW_REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLUMNS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "tenure_band", "avg_monthly_spend_per_service", "is_new_customer",
    "contract_risk_score", "total_to_monthly_ratio",
]


def _clean_totalcharges(df: pd.DataFrame) -> pd.Series:
    """TotalCharges arrives as a string with some blank entries for
    customers with tenure==0 (brand new, hasn't been billed yet).
    Documented cleaning assumption: treat blank as 0.0, not NaN-drop,
    since tenure==0 is a legitimate and informative state, not bad data.
    """
    tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    tc = tc.fillna(0.0)
    return tc


def _count_services(row: pd.Series) -> int:
    service_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    return sum(1 for c in service_cols if row.get(c) == "Yes")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic, side-effect-free feature builder.

    Input: a DataFrame with the raw columns (one row = one customer;
    also works for a single-row DataFrame built from an API request).
    Output: a DataFrame with raw columns preserved + engineered columns
    appended. No row is dropped or reordered — safe to run on 1 row.
    """
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")

    out = df.copy()
    out["TotalCharges"] = _clean_totalcharges(out)
    out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce").fillna(0).astype(int)
    out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")

    # 1. tenure_band: coarse bucket of tenure (non-trivial: ordinal encode
    #    of a domain-meaningful cut, not a raw passthrough)
    bins = [-1, 6, 12, 24, 48, np.inf]
    labels = [0, 1, 2, 3, 4]  # 0-6mo, 7-12mo, 1-2yr, 2-4yr, 4yr+
    out["tenure_band"] = pd.cut(out["tenure"], bins=bins, labels=labels).astype(int)

    # 2. avg_monthly_spend_per_service: ratio feature. Guards against
    #    div-by-zero for customers with zero active services.
    n_services = out.apply(_count_services, axis=1).clip(lower=1)
    out["avg_monthly_spend_per_service"] = out["MonthlyCharges"] / n_services

    # 3. is_new_customer: binary flag, tenure <= 3 months. Distinct
    #    signal from tenure_band — captures the specific "early churn
    #    risk window" called out in the EDA / stakeholder framing.
    out["is_new_customer"] = (out["tenure"] <= 3).astype(int)

    # 4. contract_risk_score: encodes known business prior that
    #    month-to-month contracts churn far more than annual ones,
    #    combined with paperless billing (proxy for a more
    #    "digital / low-friction-to-leave" customer).
    contract_weight = out["Contract"].map(
        {"Month-to-month": 2, "One year": 1, "Two year": 0}
    ).fillna(1)
    paperless_weight = (out["PaperlessBilling"] == "Yes").astype(int)
    out["contract_risk_score"] = contract_weight + paperless_weight

    # 5. total_to_monthly_ratio: relationship between cumulative spend
    #    and current monthly rate — a rough proxy for "effective
    #    tenure at current price point"; catches customers whose
    #    TotalCharges is low relative to MonthlyCharges (recent
    #    upgrades / plan changes), which raw tenure alone would miss.
    out["total_to_monthly_ratio"] = out["TotalCharges"] / out["MonthlyCharges"].replace(0, np.nan)
    out["total_to_monthly_ratio"] = out["total_to_monthly_ratio"].fillna(out["tenure"]).round(2)

    return out


def encode_for_model(df_features: pd.DataFrame, encoders: dict | None = None):
    """
    One-hot encode categorical columns using a fixed, saved column
    schema so training and serving produce identical column sets/order.

    - At training time: pass encoders=None. Returns (X, encoders) where
      encoders['columns'] is the full trained feature-column list —
      SAVE this alongside the model artifact.
    - At serving time: pass the saved encoders dict. Any category not
      seen in training is dropped (one-hot produces all-zero row for
      that field, matching how pd.get_dummies + reindex behaves);
      any training column missing from the single-row input is
      filled with 0. This is what keeps a single live request aligned
      to the exact matrix shape the model expects.
    """
    X = pd.get_dummies(df_features, columns=CATEGORICAL_COLUMNS, drop_first=False)
    drop_cols = [c for c in ["customerID", "Churn"] if c in X.columns]
    X = X.drop(columns=drop_cols, errors="ignore")

    if encoders is None:
        columns = sorted(X.columns.tolist())
        X = X.reindex(columns=columns, fill_value=0)
        return X, {"columns": columns}
    else:
        columns = encoders["columns"]
        X = X.reindex(columns=columns, fill_value=0)
        return X, encoders
