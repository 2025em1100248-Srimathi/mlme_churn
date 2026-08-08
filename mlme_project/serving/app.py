"""
serving/app.py

Minimal online inference API (request-response pattern).

Why online request-response and not batch, for this use case (see
design doc "Serving and Inference Pattern" for the full writeup):
  - A human (a retention agent, or a triggered email flow) needs a
    per-customer churn risk score right when that customer is being
    reviewed / has just triggered an event (e.g. logged a support
    ticket) -> someone/something is "waiting" on the answer.
  - Acceptable latency here is generous (seconds, not milliseconds) -
    this isn't a bidding/fraud-block use case - so a synchronous
    FastAPI endpoint over a scikit-learn model in memory is
    sufficient; no need for a dedicated low-latency serving stack.
  - The same model artifact could equally be run in a nightly batch
    scoring job (score entire customer base -> write to a table the
    retention team's dashboard reads); this API does not preclude
    that. Bullet-point trade-off discussion is in the design doc.

Run:
    uvicorn serving.app:app --reload --port 8000
"""

import os
import sys
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_engineering import build_features, encode_for_model
from serving.schemas import CustomerRequest, PredictResponse, HealthResponse

MODEL_PATH = os.environ.get("MODEL_PATH", "models/latest.joblib")

app = FastAPI(title="Churn Prediction Service", version="1.0")

_artifact = None


def get_artifact():
    """Lazy-load + cache the model artifact. Kept as a function (not
    module-level global at import time) so tests can monkeypatch
    MODEL_PATH before first call."""
    global _artifact
    if _artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No model artifact at {MODEL_PATH}. Run training/train.py first."
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        artifact = get_artifact()
        return HealthResponse(status="ok", model_version=artifact["model_version"])
    except FileNotFoundError:
        return HealthResponse(status="no_model_loaded", model_version=None)


@app.post("/predict", response_model=PredictResponse)
def predict(request: CustomerRequest):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()

    try:
        artifact = get_artifact()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    raw_df = pd.DataFrame([request.model_dump()])

    try:
        # SAME build_features() call used in training -> the
        # training-serving skew guard this whole project is built around.
        feat_df = build_features(raw_df)
        X, _ = encode_for_model(feat_df, encoders=artifact["encoders"])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature construction failed: {e}")

    model = artifact["model"]
    scaler = artifact.get("scaler")
    X_model = scaler.transform(X) if scaler is not None else X

    proba = float(model.predict_proba(X_model)[:, 1][0])
    latency_ms = (time.perf_counter() - start) * 1000

    # Structured log line -> this is what a log-shipper (e.g. to
    # CloudWatch/Datadog) would pick up for the latency/error-rate
    # dashboards described in the monitoring plan.
    print(
        f'{{"event":"prediction","request_id":"{request_id}",'
        f'"model_version":"{artifact["model_version"]}",'
        f'"latency_ms":{latency_ms:.2f},"churn_probability":{proba:.4f}}}'
    )

    return PredictResponse(
        churn_probability=round(proba, 4),
        churn_prediction=proba >= 0.5,
        model_version=artifact["model_version"],
        model_name=artifact["model_name"],
    )
