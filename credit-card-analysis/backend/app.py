from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import joblib
import os
import json
import base64
import hashlib
from typing import List, Optional, Dict, Any
try:
    # When running as module: uvicorn credit-card-analysis.backend.app:app
    from credit_card_analysis.backend.model import ensure_model, predict_proba_df, single_predict_proba, get_metrics, forecast_next_months  # type: ignore
except Exception:
    try:
        # When running from backend/ as script/module
        from model import ensure_model, predict_proba_df, single_predict_proba, get_metrics, forecast_next_months  # type: ignore
    except Exception:
        # Package-relative fallback
        from .model import ensure_model, predict_proba_df, single_predict_proba, get_metrics, forecast_next_months

app = FastAPI(title="Credit Card Analysis and Future Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SinglePredictPayload(BaseModel):
    LIMIT_BAL: float
    AGE: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float
    category: Optional[str] = "Other"


@app.on_event("startup")
async def startup_event():
    ensure_model()


@app.get("/")
async def root():
    return {"status": "ok", "message": "Credit Card Analysis API running"}


@app.get("/auth/signin", response_class=HTMLResponse)
async def signin_page():
    path = os.path.join(BASE_DIR, "signin.html")
    return FileResponse(path)


@app.get("/auth/signup", response_class=HTMLResponse)
async def signup_page():
    path = os.path.join(BASE_DIR, "signup.html")
    return FileResponse(path)


@app.post("/auth/signin")
async def signin_submit(username: str = Form(...), password: str = Form(...)):
    users = _load_users()
    token = users.get(username)
    if token and _check_pw(password, token):
        return RedirectResponse(url=STREAMLIT_URL, status_code=302)
    # failed: back to signin with query flag
    return RedirectResponse(url="/auth/signin?error=1", status_code=302)


@app.post("/auth/signup")
async def signup_submit(username: str = Form(...), password: str = Form(...)):
    users = _load_users()
    if username in users:
        return RedirectResponse(url="/auth/signup?exists=1", status_code=302)
    users[username] = _hash_pw(password)
    _save_users(users)
    # after signup, go to Streamlit app
    return RedirectResponse(url=STREAMLIT_URL, status_code=302)


@app.get("/metrics")
async def metrics():
    ensure_model()
    metrics = get_metrics()
    return JSONResponse(content=metrics)


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    ensure_model()
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    probs, used_features = predict_proba_df(df)
    results = df.copy()
    results["default_probability"] = probs

    # Build dataset-level spending trend from BILL_AMT1-6
    trend_input = df[[f"BILL_AMT{i}" for i in range(1, 7) if f"BILL_AMT{i}" in df.columns]].sum(axis=0)
    trend_input = trend_input.values.tolist() if hasattr(trend_input, 'values') else list(trend_input)
    future_trend = forecast_next_months(trend_input, n_months=6)

    return JSONResponse(content={
        "predictions": results[["default_probability"]].round(4).to_dict(orient="records"),
        "used_features": used_features,
        "future_trend": future_trend,
        "row_count": int(len(df))
    })


@app.post("/predict")
async def predict(payload: SinglePredictPayload):
    ensure_model()
    data = payload.dict()
    prob = single_predict_proba(data)

    # Trend based on single record's BILL_AMT1-6
    bill_seq = [data.get(f"BILL_AMT{i}", 0.0) for i in range(1, 7)]
    future_trend = forecast_next_months(bill_seq, n_months=6)

    return JSONResponse(content={
        "default_probability": round(float(prob), 4),
        "future_trend": future_trend
    })
