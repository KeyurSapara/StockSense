"""Data loading, feature engineering and the LSTM forecaster behind the app.

Instead of predicting raw prices (which makes models look great while just
copying yesterday's price), the network predicts the *cumulative % return*
for each of the next H days. A price forecast is then rebuilt from the last
known close, and uncertainty bands come from the model's own errors on the
held-out test period.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf

FEATURES = ["ret", "ma20_gap", "ma50_gap", "rsi", "bb_pos", "vol_chg", "volatility"]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_prices(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def load_info(ticker: str) -> dict:
    """Best-effort company name and currency; never raises."""
    name, currency = ticker, ""
    try:
        t = yf.Ticker(ticker)
        currency = t.fast_info.get("currency") or ""
        info = t.info or {}
        name = info.get("shortName") or info.get("longName") or ticker
    except Exception:
        pass
    return {"name": name, "currency": currency}


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    std20 = close.rolling(20).std()
    df["BB_up"] = df["MA20"] + 2 * std20
    df["BB_low"] = df["MA20"] - 2 * std20

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # Model features (all scale-free so the same model works for any price level)
    df["ret"] = np.log(close).diff() * 100
    df["ma20_gap"] = (close / df["MA20"] - 1) * 100
    df["ma50_gap"] = (close / df["MA50"] - 1) * 100
    df["rsi"] = df["RSI"] / 100 - 0.5
    df["bb_pos"] = (close - df["MA20"]) / (2 * std20)
    df["vol_chg"] = np.log(df["Volume"].replace(0, np.nan)).diff().clip(-3, 3)
    df["volatility"] = df["ret"].rolling(20).std()
    return df.replace([np.inf, -np.inf], np.nan).dropna()


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, horizon))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


@dataclass
class ForecastResult:
    history: pd.DataFrame          # prices + indicators
    future: pd.DataFrame           # date, pred, low, high
    test: pd.DataFrame             # date, actual, predicted, naive (1-day-ahead)
    metrics: dict
    losses: list


def _windows(feats: np.ndarray, close: np.ndarray, lookback: int, horizon: int):
    X, Y, idx = [], [], []
    for t in range(lookback - 1, len(feats) - horizon):
        X.append(feats[t - lookback + 1 : t + 1])
        Y.append(np.log(close[t + 1 : t + 1 + horizon] / close[t]) * 100)
        idx.append(t)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(idx)


def train_and_forecast(
    df: pd.DataFrame,
    horizon: int = 30,
    lookback: int = 60,
    epochs: int = 40,
    test_frac: float = 0.2,
    seed: int = 42,
    progress=None,
) -> ForecastResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = add_indicators(df)
    close = data["Close"].to_numpy()
    raw = data[FEATURES].to_numpy(dtype=np.float32)

    X, Y, idx = _windows(raw, close, lookback, horizon)
    split = int(len(X) * (1 - test_frac))

    # Standardise with statistics from the training period only (no leakage)
    train_rows = raw[: idx[split - 1] + 1]
    mu, sd = train_rows.mean(0), train_rows.std(0) + 1e-6
    X = (X - mu) / sd
    y_scale = float(Y[:split].std()) + 1e-6

    Xtr, Ytr = torch.tensor(X[:split]), torch.tensor(Y[:split] / y_scale)
    Xte, Yte = torch.tensor(X[split:]), Y[split:]

    model = LSTMForecaster(len(FEATURES), horizon)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.HuberLoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(Xtr), 64):
            b = perm[i : i + 64]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), Ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        losses.append(total / len(Xtr))
        if progress:
            progress((epoch + 1) / epochs)

    model.eval()
    with torch.no_grad():
        pred_te = model(Xte).numpy() * y_scale
        last = torch.tensor(((raw[-lookback:] - mu) / sd)[None], dtype=torch.float32)
        pred_future = model(last).numpy()[0] * y_scale

    # ---- Test-period evaluation (1-day-ahead) -------------------------------
    te_idx = idx[split:]
    base = close[te_idx]
    actual = close[te_idx + 1]
    predicted = base * np.exp(pred_te[:, 0] / 100)
    naive = base  # "tomorrow = today"

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    metrics = {
        "mae": mae(actual, predicted),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "mape": float(np.mean(np.abs(actual - predicted) / actual) * 100),
        "naive_mae": mae(actual, naive),
        "direction_acc": float(np.mean(np.sign(pred_te[:, 0]) == np.sign(Yte[:, 0])) * 100),
        f"direction_acc_{horizon}d": float(np.mean(np.sign(pred_te[:, -1]) == np.sign(Yte[:, -1])) * 100),
        "n_test": int(len(te_idx)),
        "n_train": int(split),
    }

    # ---- Future forecast with error bands from test residuals ---------------
    resid_sd = (Yte - pred_te).std(0)  # per-horizon spread in % log-return
    last_close = close[-1]
    is_crypto = "-" in str(df.attrs.get("ticker", ""))
    freq = "D" if is_crypto else "B"
    dates = pd.date_range(data.index[-1], periods=horizon + 1, freq=freq)[1:]
    future = pd.DataFrame(
        {
            "pred": last_close * np.exp(pred_future / 100),
            "low": last_close * np.exp((pred_future - 1.28 * resid_sd) / 100),
            "high": last_close * np.exp((pred_future + 1.28 * resid_sd) / 100),
        },
        index=dates,
    )

    test = pd.DataFrame(
        {"actual": actual, "predicted": predicted, "naive": naive},
        index=data.index[te_idx + 1],
    )
    return ForecastResult(history=data, future=future, test=test, metrics=metrics, losses=losses)
