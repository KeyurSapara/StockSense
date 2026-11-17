# 🔮 StockSense — AI Stock Forecaster

A simple, good-looking stock prediction app. Pick a stock, choose 7/14/30 days, and get:

- 📈 **Interactive candlestick chart** with the AI forecast and an 80% "likely range" band
- 🐂/🐻 **Plain-English signal** (bullish / bearish / neutral)
- 📊 **Indicators**: 20/50-day averages, Bollinger Bands, RSI, explained in plain words
- 🧪 **Honest model check**: MAE, RMSE, direction hit-rate, compared with the naive "tomorrow = today" guess
- 💾 **CSV export** of the forecast and the backtest

Works with any Yahoo Finance symbol: `AAPL`, `TSLA`, `RELIANCE.NS`, `TCS.NS`, `BTC-USD` …

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

You don't need to train anything beforehand. The model trains on 5 years of data when you pick a stock (~15 s) and is cached after that.

## How it works

| Step | Detail |
|---|---|
| Data | 5 years of daily OHLCV from Yahoo Finance (`yfinance`) |
| Features | log return, gap from MA20/MA50, RSI, Bollinger position, volume change, 20-day volatility |
| Model | 2-layer LSTM (PyTorch), 60-day lookback, **multi-horizon output** (predicts all N future days at once) |
| Target | cumulative % return, not raw price, so it can't cheat by copying yesterday's price |
| Split | chronological 80/20, scaler fit on train only (no data leakage) |
| Uncertainty | per-day residual spread on the test set → 80% band |

### Compared with NeuroStock
- PyTorch + Plotly (interactive) instead of Keras + Matplotlib
- Trains live for **any** ticker, so there's no stale pre-trained `.h5` file
- Forecasts the future with a confidence band, not just a replay of the test set
- Shows a naive baseline so the results are honest

> ⚠️ Educational project, not financial advice.

## Files
- `app.py`: Streamlit UI
- `predictor.py`: data, indicators, LSTM model, training and evaluation
- `.streamlit/config.toml`: dark theme
