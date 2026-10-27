from __future__ import annotations

from io import BytesIO
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, precision_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="FX Research Lab API", version="0.1.0", description="Research prototype for next-candle direction; not a trading service.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

DATA: pd.DataFrame | None = None
LAST_MODEL = None
LAST_FEATURES: list[str] = []

class TrainRequest(BaseModel):
    model: Literal["logistic", "random_forest", "hist_gradient_boosting"] = "logistic"
    train_percent: int = Field(default=80, ge=60, le=90)


def normalize_csv(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).sort_values("timestamp")
    if df.empty:
        raise ValueError("No valid OHLC rows found.")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Invalid OHLC rows: high must be >= open, close, and low.")
    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid OHLC rows: low must be <= open, close, and high.")
    duplicates = int(df["timestamp"].duplicated().sum())
    df = df.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if len(df) < 100:
        raise ValueError(f"Only {len(df)} valid unique candles. At least 100 are required for a basic test; meaningful research needs much more.")
    df.attrs["duplicates_removed"] = duplicates
    return df


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Features known at candle t; label is the direction of candle t+1."""
    o, h, l, c = (df[k].astype(float) for k in ("open", "high", "low", "close"))
    eps = 1e-12
    rng = (h - l).replace(0, np.nan)
    body = c - o
    feat = pd.DataFrame(index=df.index)
    feat["body_pct_range"] = body / (rng + eps)
    feat["upper_wick_pct_range"] = (h - np.maximum(o, c)) / (rng + eps)
    feat["lower_wick_pct_range"] = (np.minimum(o, c) - l) / (rng + eps)
    feat["range_pct_close"] = rng / (c.abs() + eps)
    feat["return_1"] = c.pct_change(1)
    for lag in (2, 3, 5, 10, 20):
        feat[f"return_{lag}"] = c.pct_change(lag)
    for win in (5, 10, 20):
        feat[f"volatility_{win}"] = feat["return_1"].rolling(win).std()
        feat[f"mean_return_{win}"] = feat["return_1"].rolling(win).mean()
        feat[f"range_mean_{win}"] = feat["range_pct_close"].rolling(win).mean()
        feat[f"close_vs_sma_{win}"] = c / (c.rolling(win).mean() + eps) - 1
    # Binary pattern flags based only on the current completed candle and past context.
    feat["doji"] = (body.abs() / (rng + eps) < 0.1).astype(float)
    feat["bullish_engulfing"] = ((c > o) & (c.shift(1) < o.shift(1)) & (o <= c.shift(1)) & (c >= o.shift(1))).astype(float)
    feat["bearish_engulfing"] = ((c < o) & (c.shift(1) > o.shift(1)) & (o >= c.shift(1)) & (c <= o.shift(1))).astype(float)
    feat["hammer_like"] = ((feat["lower_wick_pct_range"] > 0.55) & (feat["body_pct_range"].abs() < 0.35)).astype(float)
    feat["shooting_star_like"] = ((feat["upper_wick_pct_range"] > 0.55) & (feat["body_pct_range"].abs() < 0.35)).astype(float)
    if "volume" in df.columns:
        volume = pd.to_numeric(df["volume"], errors="coerce")
        feat["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    label = (c.shift(-1) > o.shift(-1)).astype("float")
    # Last row has no next candle outcome and must not be used.
    valid_label = df.index < (len(df) - 1)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    features = list(feat.columns)
    combined = feat.loc[valid_label].copy()
    y = label.loc[valid_label].astype(int)
    return combined, y, features


def make_estimator(name: str):
    if name == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=250, min_samples_leaf=10, max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=-1))
    return make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, l2_regularization=1.0, random_state=42))

@app.get("/health")
def health():
    return {"ok": True, "data_loaded": DATA is not None, "rows": 0 if DATA is None else len(DATA), "model_trained": LAST_MODEL is not None}

@app.post("/data/upload")
async def upload_csv(file: UploadFile = File(...)):
    global DATA, LAST_MODEL, LAST_FEATURES
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a CSV file.")
    raw = await file.read()
    if len(raw) > 100 * 1024 * 1024:
        raise HTTPException(413, "CSV exceeds the 100 MB prototype limit.")
    try:
        frame = normalize_csv(pd.read_csv(BytesIO(raw)))
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    DATA, LAST_MODEL, LAST_FEATURES = frame, None, []
    span_days = (frame["timestamp"].max() - frame["timestamp"].min()).total_seconds() / 86400
    diffs = frame["timestamp"].diff().dropna()
    median_minutes = None if diffs.empty else round(diffs.median().total_seconds() / 60, 2)
    return {"ok": True, "rows": len(frame), "start": frame["timestamp"].min().isoformat(), "end": frame["timestamp"].max().isoformat(), "span_days": round(span_days, 2), "span_years": round(span_days / 365.25, 2), "median_interval_minutes": median_minutes, "duplicates_removed": int(frame.attrs.get("duplicates_removed", 0)), "note": "Coverage does not prove continuity; inspect gaps and provider-specific candle conventions."}

@app.post("/model/train")
def train(req: TrainRequest):
    global LAST_MODEL, LAST_FEATURES
    if DATA is None:
        raise HTTPException(400, "Upload OHLC data first at /data/upload.")
    X, y, feature_names = make_features(DATA)
    # Remove warm-up rows without changing chronological order.
    valid = X.notna().sum(axis=1) > 0
    X, y = X.loc[valid], y.loc[valid]
    split = int(len(X) * req.train_percent / 100)
    if split < 50 or len(X) - split < 30:
        raise HTTPException(400, "Not enough rows for the requested chronological split. Upload substantially more data.")
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if y_train.nunique() < 2:
        raise HTTPException(400, "Training data contains only one target class; check the dataset.")
    estimator = make_estimator(req.model)
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    try:
        prob = estimator.predict_proba(X_test)[:, 1]
        brier = float(brier_score_loss(y_test, prob))
    except Exception:
        prob, brier = None, None
    LAST_MODEL, LAST_FEATURES = estimator, list(X.columns)
    cm = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
    return {
        "ok": True,
        "model": req.model,
        "target": "next candle close > next candle open (direction of next candle body)",
        "features": list(X.columns),
        "train_rows": len(X_train), "test_rows": len(X_test), "train_percent": req.train_percent,
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
        "precision_up": round(float(precision_score(y_test, pred, pos_label=1, zero_division=0)), 4),
        "precision_down": round(float(precision_score(y_test, pred, pos_label=0, zero_division=0)), 4),
        "brier_score": None if brier is None else round(brier, 4),
        "confusion_matrix_labels_down_up": cm,
        "test_start": DATA.loc[X_test.index[0], "timestamp"].isoformat(),
        "test_end": DATA.loc[X_test.index[-1], "timestamp"].isoformat(),
        "warning": "Single chronological holdout is a baseline, not proof of a tradable edge. Run walk-forward tests, calibrate probabilities, account for costs, and compare against baselines."
    }

@app.get("/patterns/study")
def pattern_study():
    if DATA is None:
        raise HTTPException(400, "Upload OHLC data first.")
    df = DATA.copy()
    o, h, l, c = (df[k].astype(float) for k in ("open", "high", "low", "close"))
    rng = (h - l).replace(0, np.nan)
    body = c - o
    next_up = (c.shift(-1) > o.shift(-1)).astype(float)
    definitions = {
        "Doji": body.abs() / rng < 0.1,
        "Bullish Engulfing": (c > o) & (c.shift(1) < o.shift(1)) & (o <= c.shift(1)) & (c >= o.shift(1)),
        "Bearish Engulfing": (c < o) & (c.shift(1) > o.shift(1)) & (o >= c.shift(1)) & (c <= o.shift(1)),
        "Hammer-like": ((np.minimum(o, c) - l) / rng > 0.55) & (body.abs() / rng < 0.35),
        "Shooting-star-like": ((h - np.maximum(o, c)) / rng > 0.55) & (body.abs() / rng < 0.35),
    }
    out = []
    for name, mask in definitions.items():
        mask = mask.fillna(False) & next_up.notna()
        n = int(mask.sum())
        ups = int(((next_up == 1) & mask).sum())
        rate = ups / n if n else None
        # Wilson 95% interval for descriptive uncertainty, not multiple-testing adjusted.
        if n:
            z = 1.96; den = 1 + z*z/n; center = (rate + z*z/(2*n))/den
            half = z*np.sqrt((rate*(1-rate)/n) + z*z/(4*n*n))/den
            ci = [round(max(0, center-half), 4), round(min(1, center+half), 4)]
        else: ci = None
        out.append({"pattern": name, "occurrences": n, "next_candle_up_rate": None if rate is None else round(rate, 4), "wilson_95_ci": ci})
    return {"patterns": out, "warning": "Descriptive in-sample rates only; use a held-out period and correct for multiple comparisons before making claims."}
