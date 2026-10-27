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
TELEGRAM_SENT: set[tuple[str, str]] = set()

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

# Optional live-data signal endpoint. Requires a licensed Twelve Data API key in the server environment.
import os
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

@app.get('/live/signal')
def live_signal(symbol: str = 'EUR/USD', model: Literal['logistic','random_forest','hist_gradient_boosting'] = 'logistic', threshold: float = 0.60, notify_telegram: bool = False):
    """Generate a read-only research estimate from the latest completed 5-minute candle.

    This endpoint does not connect to Quotex or place trades. API-provider data may differ
    from a broker's own feed. The returned probability is a model estimate, not a guarantee.
    """
    allowed = {'EUR/USD','GBP/USD','USD/JPY','AUD/USD','USD/CAD','USD/CHF','NZD/USD'}
    symbol = symbol.upper().replace(' ', '')
    if symbol not in allowed:
        raise HTTPException(400, f'Supported symbols: {", ".join(sorted(allowed))}')
    if not 0.50 <= threshold <= 0.85:
        raise HTTPException(400, 'threshold must be between 0.50 and 0.85')

    api_key = os.getenv('TWELVE_DATA_API_KEY') or os.getenv('MARKET_DATA_API_KEY')
    source = 'Twelve Data'
    candles = None
    provider_errors = []

    # Prefer the configured Twelve Data provider. If it is unavailable or unconfigured,
    # try Yahoo Finance's public chart endpoint as a best-effort research fallback.
    if api_key:
        try:
            response = requests.get('https://api.twelvedata.com/time_series', params={
                'symbol': symbol, 'interval': '5min', 'outputsize': 5000,
                'timezone': 'UTC', 'order': 'ASC', 'apikey': api_key,
            }, timeout=20)
            response.raise_for_status()
            payload = response.json()
            if payload.get('status') == 'error' or not payload.get('values'):
                raise ValueError(payload.get('message', 'No candle data returned by Twelve Data.'))
            candles = normalize_live_frame(
                pd.DataFrame(payload['values']).rename(columns={'datetime': 'timestamp'})
            )
        except Exception as exc:
            provider_errors.append(f'Twelve Data: {exc}')

    if candles is None:
        yahoo_symbol = symbol.replace('/', '') + '=X'
        try:
            response = requests.get(
                f'https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}',
                params={'range': '60d', 'interval': '5m', 'events': 'history'},
                headers={'User-Agent': 'Mozilla/5.0 FXResearchLab/1.0'},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            chart = payload.get('chart', {})
            if chart.get('error') or not chart.get('result'):
                raise ValueError(str(chart.get('error') or 'No chart data returned by Yahoo Finance.'))
            item = chart['result'][0]
            quote = item.get('indicators', {}).get('quote', [{}])[0]
            timestamps = item.get('timestamp', [])
            if not timestamps or not quote:
                raise ValueError('Yahoo Finance returned no intraday candles.')
            candles = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps, unit='s', utc=True),
                'open': quote.get('open', []),
                'high': quote.get('high', []),
                'low': quote.get('low', []),
                'close': quote.get('close', []),
            })
            candles = normalize_live_frame(candles)
            source = 'Yahoo Finance (public fallback)'
        except Exception as exc:
            provider_errors.append(f'Yahoo Finance: {exc}')
            detail = ' | '.join(provider_errors)
            raise HTTPException(
                502,
                'Could not retrieve live candles from Twelve Data or Yahoo Finance. '
                'Check provider availability, symbol support, and Render logs. Details: ' + detail
            ) from exc
    if len(candles) < 500:
        raise HTTPException(422, f'Provider returned only {len(candles)} candles; at least 500 are required for this prototype.')

    # Add a dummy next row solely so feature generation includes the most recent real candle.
    # Exclude that dummy-labelled row from model fitting and use its feature row only for prediction.
    last_ts = candles['timestamp'].iloc[-1]
    dummy = pd.DataFrame([{'timestamp': last_ts + pd.Timedelta(minutes=5),
                           'open': candles['close'].iloc[-1], 'high': candles['close'].iloc[-1],
                           'low': candles['close'].iloc[-1], 'close': candles['close'].iloc[-1]}])
    extended = pd.concat([candles, dummy], ignore_index=True)
    X, y, _ = make_features(extended)
    X = X.replace([np.inf, -np.inf], np.nan)
    if len(X) < 300:
        raise HTTPException(422, 'Not enough usable feature rows after preparation.')
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    X_now = X.iloc[[-1]]
    # Time-ordered holdout diagnostic on the downloaded data (not a live-performance guarantee).
    split = max(200, int(len(X_train) * 0.8))
    split = min(split, len(X_train) - 50)
    estimator = make_estimator(model)
    estimator.fit(X_train.iloc[:split], y_train.iloc[:split])
    holdout_pred = estimator.predict(X_train.iloc[split:])
    holdout_acc = float(accuracy_score(y_train.iloc[split:], holdout_pred))
    # Refit on all known labelled rows before making the next-candle estimate.
    estimator.fit(X_train, y_train)
    probability_up = float(estimator.predict_proba(X_now)[0, 1])
    probability_down = 1.0 - probability_up
    if max(probability_up, probability_down) < threshold:
        direction = 'NO SIGNAL'
        probability = max(probability_up, probability_down)
    else:
        direction = 'CALL / UP' if probability_up > probability_down else 'PUT / DOWN'
        probability = max(probability_up, probability_down)
    # Do not claim a forecast for a candle that has already closed: identify the next 5-minute boundary.
    now = datetime.now(timezone.utc)
    next_boundary = now.replace(second=0, microsecond=0)
    next_boundary = next_boundary + timedelta(minutes=(5 - next_boundary.minute % 5))
    last = candles.iloc[-1]
    result = {
        'ok': True, 'source': source, 'symbol': symbol, 'timeframe': '5min',
        'signal': direction, 'probability': round(probability, 4),
        'probability_up': round(probability_up, 4), 'probability_down': round(probability_down, 4),
        'threshold': threshold, 'model': model, 'candles_used': int(len(candles)),
        'last_completed_candle_utc': last_ts.isoformat(),
        'next_candle_start_estimate_utc': next_boundary.isoformat(),
        'last_close': float(last['close']), 'holdout_accuracy_diagnostic': round(holdout_acc, 4),
        'telegram_notified': False,
        'warning': 'Research estimate only. Yahoo public intraday data may be delayed, incomplete, rate-limited, or unavailable. Provider prices may differ from broker quotes. Not a guaranteed prediction or financial advice; paper-test before relying on it.'
    }
    if notify_telegram and direction != 'NO SIGNAL':
        result['telegram_notified'] = send_telegram_signal(result)
    return result


def normalize_live_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], utc=True, errors='coerce')
    for col in ('open','high','low','close'):
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame = frame.dropna(subset=['timestamp','open','high','low','close']).sort_values('timestamp')
    frame = frame.drop_duplicates('timestamp', keep='last').reset_index(drop=True)
    if (frame['high'] < frame[['open','close','low']].max(axis=1)).any() or (frame['low'] > frame[['open','close','high']].min(axis=1)).any():
        raise ValueError('Provider returned invalid OHLC rows.')
    return frame


def send_telegram_message(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise HTTPException(503, "Telegram is not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your hosting environment.")
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise ValueError(payload.get("description", "Telegram rejected the message"))
    except Exception as exc:
        raise HTTPException(502, f"Telegram message failed: {exc}") from exc

def send_telegram_signal(result: dict) -> bool:
    key = (result["symbol"], result["last_completed_candle_utc"])
    if key in TELEGRAM_SENT:
        return False
    message = (
        f"FX Research Lab — research signal\n"
        f"Pair: {result['symbol']} | Timeframe: 5m\n"
        f"Estimate: {result['signal']}\n"
        f"Model probability: {result['probability']:.1%}\n"
        f"Last completed candle (UTC): {result['last_completed_candle_utc']}\n"
        f"Holdout accuracy diagnostic: {result['holdout_accuracy_diagnostic']:.1%}\n"
        "Research only; not a guarantee. Verify the instrument/feed before acting."
    )
    send_telegram_message(message)
    TELEGRAM_SENT.add(key)
    return True

@app.post("/telegram/test")
def telegram_test():
    send_telegram_message("FX Research Lab: Telegram connection test successful. This bot sends research alerts only.")
    return {"ok": True, "message": "Telegram test message sent."}

@app.get("/")
def dashboard():
    from fastapi.responses import FileResponse
    return FileResponse(Path(__file__).resolve().parent.parent / "index.html")
