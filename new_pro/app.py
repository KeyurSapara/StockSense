"""StockSense — a friendly AI stock forecaster.  Run:  streamlit run app.py"""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import predictor

st.set_page_config(page_title="StockSense · AI Stock Forecaster", page_icon="🔮", layout="wide")

UP, DOWN, ACCENT, MUTED = "#22C55E", "#EF4444", "#7C5CFF", "#8B93A7"
REG = "#F59E0B"  # colour for the regression model
MODELS = {"lstm": ("🧠 LSTM", ACCENT), "regression": ("📐 Linear regression", REG)}
POPULAR = {
    "🍎 Apple": "AAPL", "⚡ Tesla": "TSLA", "🟩 NVIDIA": "NVDA", "🔎 Google": "GOOGL",
    "🏭 Reliance": "RELIANCE.NS", "💻 TCS": "TCS.NS", "🏦 HDFC Bank": "HDFCBANK.NS", "₿ Bitcoin": "BTC-USD",
}
CURRENCY = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥"}

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero h1 {font-size: 2.6rem; font-weight: 800; margin: 0;
  background: linear-gradient(90deg,#7C5CFF,#22D3EE); -webkit-background-clip: text; color: transparent;}
.hero p {color:#8B93A7; margin-top:.2rem; font-size:1.05rem;}
.card {background:#151A28; border:1px solid #232A3D; border-radius:16px; padding:18px 20px; height:100%;}
.card .label {color:#8B93A7; font-size:.8rem; text-transform:uppercase; letter-spacing:.06em;}
.card .value {font-size:1.7rem; font-weight:700; margin-top:4px;}
.card .sub {font-size:.9rem; margin-top:2px; color:#8B93A7;}
.signal {border-radius:16px; padding:18px 22px; margin:8px 0 18px; font-size:1.05rem;}
.disclaimer {color:#6B7280; font-size:.8rem; text-align:center; margin-top:2rem;}
div.stButton > button {border-radius:999px;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_prices(ticker):
    df = predictor.load_prices(ticker)
    df.attrs["ticker"] = ticker
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_info(ticker):
    return predictor.load_info(ticker)


@st.cache_resource(show_spinner=False)
def _forecast_store():
    # Shared {(ticker, horizon): (timestamp, result)} store. Training runs outside any
    # @st.cache_* function so it can drive a progress bar created in the page body
    # (Streamlit can't replay element calls on blocks created outside a cached function).
    return {}


def get_forecast(ticker, horizon, progress=None, ttl=3600, max_entries=20):
    store = _forecast_store()
    key = (ticker, horizon)
    hit = store.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    t0 = time.perf_counter()
    res = predictor.train_and_forecast(get_prices(ticker), horizon=horizon, progress=progress)
    res.extras["train_seconds"] = time.perf_counter() - t0
    store[key] = (time.time(), res)
    while len(store) > max_entries:
        store.pop(min(store, key=lambda k: store[k][0]))
    return res


@st.cache_resource(ttl=3600, show_spinner=False, max_entries=20)
def get_regression(ticker, horizon):
    t0 = time.perf_counter()
    res = predictor.train_regression_and_forecast(get_prices(ticker), horizon=horizon)
    res.extras["train_seconds"] = time.perf_counter() - t0
    return res


def card(label, value, sub="", color=None):
    style = f"color:{color};" if color else ""
    return (f'<div class="card"><div class="label">{label}</div>'
            f'<div class="value" style="{style}">{value}</div><div class="sub">{sub}</div></div>')


def base_layout(fig, height=480):
    fig.update_layout(
        height=height, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0), xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(gridcolor="#1E2435")
    fig.update_yaxes(gridcolor="#1E2435")
    return fig


# ---------------------------------------------------------------------------
# Header + input
# ---------------------------------------------------------------------------

st.markdown('<div class="hero"><h1>🔮 StockSense</h1>'
            '<p>Pick a stock and see where an AI model thinks it is heading.</p></div>',
            unsafe_allow_html=True)

if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"

cols = st.columns(len(POPULAR))
for col, (label, sym) in zip(cols, POPULAR.items()):
    if col.button(label, width="stretch", type="primary" if st.session_state.ticker == sym else "secondary"):
        st.session_state.ticker = sym
        st.rerun()

c1, c2, c3 = st.columns([3, 2, 2])
typed = c1.text_input("Or type any ticker", value=st.session_state.ticker,
                      help="Yahoo Finance symbols: AAPL, MSFT, INFY.NS (NSE India), ETH-USD …")
horizon = c2.select_slider("Forecast horizon", options=[7, 14, 30], value=14,
                           format_func=lambda d: f"{d} days")
model_key = c3.radio("Headline model", list(MODELS), format_func=lambda k: MODELS[k][0], horizontal=True,
                     help="Both models are always trained. This picks which one drives the cards and forecast chart. "
                          "See the ⚖️ tab for a head-to-head.")
ticker = typed.strip().upper()
st.session_state.ticker = ticker

if not ticker:
    st.stop()

with st.spinner(f"Fetching {ticker}…"):
    prices = get_prices(ticker)
if prices.empty or len(prices) < 300:
    st.error(f"Couldn't find enough data for **{ticker}**. Check the symbol (Indian stocks need `.NS`, e.g. `TCS.NS`).")
    st.stop()

info = get_info(ticker)
cur = CURRENCY.get(info["currency"], info["currency"] + " " if info["currency"] else "")
bar = st.empty()
res_lstm = get_forecast(ticker, horizon,
                        progress=lambda f: bar.progress(f, text=f"🧠 Training the neural network… {int(f * 100)}%"))
bar.empty()
res_reg = get_regression(ticker, horizon)
res = res_lstm if model_key == "lstm" else res_reg
other = res_reg if model_key == "lstm" else res_lstm
MODEL_NAME, MODEL_COLOR = MODELS[res.model]
OTHER_NAME, OTHER_COLOR = MODELS[other.model]
fmt = lambda v: f"{cur}{v:,.2f}"

# ---------------------------------------------------------------------------
# Summary cards
# ---------------------------------------------------------------------------

last = float(prices["Close"].iloc[-1])
prev = float(prices["Close"].iloc[-2])
day_chg = (last / prev - 1) * 100
target = float(res.future["pred"].iloc[-1])
lo, hi = float(res.future["low"].iloc[-1]), float(res.future["high"].iloc[-1])
exp_chg = (target / last - 1) * 100
m = res.metrics

st.markdown(f"### {info['name']} <span style='color:{MUTED};font-size:1rem'>· {ticker} · "
            f"forecast by {MODEL_NAME}</span>", unsafe_allow_html=True)

k = st.columns(4)
k[0].markdown(card("Current price", fmt(last), f"{'▲' if day_chg >= 0 else '▼'} {day_chg:+.2f}% today",
                   UP if day_chg >= 0 else DOWN), unsafe_allow_html=True)
k[1].markdown(card(f"Forecast in {horizon} days", fmt(target), f"{exp_chg:+.2f}% expected",
                   UP if exp_chg >= 0 else DOWN), unsafe_allow_html=True)
k[2].markdown(card("Likely range (80%)", f"{fmt(lo)} – {fmt(hi)}".replace(".00", ""),
                   f"±{(hi - lo) / 2 / last * 100:.1f}% wiggle room"), unsafe_allow_html=True)
k[3].markdown(card("Direction hit-rate", f"{m[f'direction_acc_{horizon}d']:.0f}%",
                   f"up/down calls correct on {m['n_test']} test days"), unsafe_allow_html=True)

band = (hi - lo) / 2 / last * 100
if abs(exp_chg) < band * 0.25:
    mood, color, msg = "😐 Neutral", MUTED, "The predicted move is small compared to the uncertainty — no clear direction."
elif exp_chg > 0:
    mood, color, msg = "🐂 Leaning bullish", UP, f"The {MODEL_NAME} model expects a rise of about {exp_chg:.1f}% over {horizon} days."
else:
    mood, color, msg = "🐻 Leaning bearish", DOWN, f"The {MODEL_NAME} model expects a drop of about {abs(exp_chg):.1f}% over {horizon} days."
st.markdown(f'<div class="signal" style="background:{color}1A;border:1px solid {color}55">'
            f'<b style="color:{color}">{mood}</b> &nbsp;—&nbsp; {msg}</div>', unsafe_allow_html=True)
other_chg = (float(other.future["pred"].iloc[-1]) / last - 1) * 100
if np.sign(other_chg) != np.sign(exp_chg):
    st.caption(f"⚠️ The {OTHER_NAME} model disagrees on direction ({other_chg:+.1f}%). See the ⚖️ tab.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_fc, tab_ind, tab_model, tab_cmp, tab_data = st.tabs(
    ["📈 Forecast", "📊 Indicators", "🧪 How good is the model?", "⚖️ LSTM vs Regression", "💾 Data"])

with tab_fc:
    r1, r2 = st.columns([3, 2])
    rng = r1.radio("Show history", ["3M", "6M", "1Y", "All"], index=1, horizontal=True, label_visibility="collapsed")
    overlay = r2.toggle(f"Overlay {OTHER_NAME} forecast", value=True)
    days = {"3M": 63, "6M": 126, "1Y": 252, "All": len(prices)}[rng]
    h = res.history.tail(days)
    f = res.future
    fig = go.Figure()
    fig.add_candlestick(x=h.index, open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"], name="Price",
                        increasing_line_color=UP, decreasing_line_color=DOWN)
    # Connect the forecast to the last real close so the line doesn't float
    fx = [h.index[-1], *f.index]
    fig.add_scatter(x=fx, y=[last, *f["high"]], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fx, y=[last, *f["low"]], fill="tonexty", fillcolor=MODEL_COLOR + "2E",
                    line=dict(width=0), name="80% range")
    fig.add_scatter(x=fx, y=[last, *f["pred"]], line=dict(color=MODEL_COLOR, width=3, dash="dot"),
                    name=f"{MODEL_NAME} forecast")
    if overlay:
        fig.add_scatter(x=fx, y=[last, *other.future["pred"]], line=dict(color=OTHER_COLOR, width=2, dash="dash"),
                        name=f"{OTHER_NAME} forecast", opacity=0.8)
    fig.add_vline(x=h.index[-1], line=dict(color=MUTED, dash="dash", width=1))
    st.plotly_chart(base_layout(fig, 520), width="stretch")
    st.caption("Shaded area = where the price landed 80% of the time, based on the model's past mistakes. "
               "It widens the further out you look — that's uncertainty growing.")

with tab_ind:
    h = res.history.tail(252)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_scatter(x=h.index, y=h["BB_up"], line=dict(width=0), showlegend=False, hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=h.index, y=h["BB_low"], fill="tonexty", fillcolor="rgba(34,211,238,0.08)",
                    line=dict(width=0), name="Bollinger band", row=1, col=1)
    fig.add_scatter(x=h.index, y=h["Close"], line=dict(color="#E6E8F0", width=2), name="Close", row=1, col=1)
    fig.add_scatter(x=h.index, y=h["MA20"], line=dict(color="#22D3EE", width=1.5), name="20-day avg", row=1, col=1)
    fig.add_scatter(x=h.index, y=h["MA50"], line=dict(color="#F59E0B", width=1.5), name="50-day avg", row=1, col=1)
    fig.add_scatter(x=h.index, y=h["RSI"], line=dict(color=ACCENT, width=1.5), name="RSI", row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=DOWN, opacity=0.08, line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor=UP, opacity=0.08, line_width=0, row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    st.plotly_chart(base_layout(fig, 560), width="stretch")

    rsi = float(res.history["RSI"].iloc[-1])
    rsi_txt = "overbought 🔥" if rsi > 70 else "oversold 🧊" if rsi < 30 else "in a normal zone"
    trend = "above" if last > res.history["MA50"].iloc[-1] else "below"
    a, b = st.columns(2)
    a.info(f"**RSI is {rsi:.0f}** — the stock looks {rsi_txt}. Above 70 often means it's run up fast; below 30 means it's been sold off hard.")
    b.info(f"**Price is {trend} its 50-day average** — a simple sign the medium-term trend is "
           f"{'up' if trend == 'above' else 'down'}.")

with tab_model:
    beats = m["mae"] < m["naive_mae"]
    st.markdown(
        f"The model was trained on the first **{m['n_train']}** days and tested on the last **{m['n_test']}** days "
        "it had never seen. Below is its next-day prediction vs what actually happened."
    )
    k = st.columns(4)
    k[0].markdown(card("Avg error (MAE)", fmt(m["mae"]), f"{m['mape']:.2f}% of price"), unsafe_allow_html=True)
    k[1].markdown(card("RMSE", fmt(m["rmse"]), "penalises big misses"), unsafe_allow_html=True)
    k[2].markdown(card("Next-day direction", f"{m['direction_acc']:.0f}%", "coin flip = 50%"), unsafe_allow_html=True)
    k[3].markdown(card("vs. “same as today”", "Beats it ✅" if beats else "Doesn't beat it",
                       f"naive error {fmt(m['naive_mae'])}", UP if beats else "#F59E0B"), unsafe_allow_html=True)

    t = res.test
    fig = go.Figure()
    fig.add_scatter(x=t.index, y=t["actual"], line=dict(color="#E6E8F0", width=2), name="Actual")
    fig.add_scatter(x=t.index, y=t["predicted"], line=dict(color=MODEL_COLOR, width=2), name="Predicted")
    st.plotly_chart(base_layout(fig, 400), width="stretch")

    with st.expander("🤓 Under the hood", expanded=res.model == "regression"):
        if res.model == "lstm":
            st.markdown(
                f"""
- **Model:** 2-layer LSTM (PyTorch, {res.extras['n_params']:,} parameters) → predicts the % change for each of the next {horizon} days at once.
- **Inputs:** last 60 days of returns, distance from 20/50-day averages, RSI, Bollinger position, volume change, volatility.
- **Why returns, not prices?** A model predicting raw prices can look perfect by just copying yesterday's price.
  Predicting *changes* is harder but honest, and the same model works for a ₹100 or a $50,000 asset.
- **Honest check:** we compare against the naive "tomorrow = today" guess. Markets are very hard to predict;
  if the AI doesn't beat it, that's a real finding, not a bug.
"""
            )
            lf = go.Figure(go.Scatter(y=res.losses, line=dict(color=ACCENT)))
            lf.update_layout(title="Training loss", xaxis_title="epoch")
            st.plotly_chart(base_layout(lf, 260), width="stretch")
        else:
            al = res.extras["alphas"]
            st.markdown(
                f"""
- **Model:** Ridge linear regression ({res.extras['n_params']:,} weights), one straight-line equation per forecast day.
- **Inputs:** the same 7 indicators, but summarised: today, yesterday and the day before, plus their 5/20/60-day averages
  (42 numbers). A linear model can't "read" a 60-day sequence like the LSTM, so we hand it the summaries.
- **Same targets & split as the LSTM**, so the comparison is fair.
- **Ridge penalty (α)** shrinks the weights towards zero to stop overfitting noise. It's tuned separately for each day ahead on
  the last 20% of the training period: day 1 → α = {al[0]:g}, day {horizon} → α = {al[-1]:g}.
  A huge α means "there's barely any pattern here, stay close to the average move".
"""
            )
            coefs = res.extras["coefs"][f"day_{horizon}"]
            top = coefs.reindex(coefs.abs().sort_values().index).tail(12)
            cf = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                                  marker_color=[UP if v > 0 else DOWN for v in top.values]))
            cf.update_layout(title=f"What drives the {horizon}-day forecast (top 12 weights)",
                             xaxis_title="effect on % return per 1 std-dev of input")
            st.plotly_chart(base_layout(cf, 380), width="stretch")
            st.caption("Green pushes the forecast up, red pushes it down. This transparency is the main thing a linear "
                       "model gives you that the LSTM can't.")

with tab_cmp:
    L, R = res_lstm, res_reg
    ml, mr = L.metrics, R.metrics
    cmpm = predictor.compare(L, R)
    hk = f"direction_acc_{horizon}d"

    st.markdown(
        "Both models see the **same stock, the same 7 indicators, the same training days and the same "
        f"{ml['n_test']} unseen test days**. The only difference is the model itself."
    )

    def winner(a, b, lower=True):
        if abs(a - b) < 1e-9:
            return "Tie"
        return MODELS["lstm"][0] if (a < b) == lower else MODELS["regression"][0]

    rows = [
        ("Next-day avg error (MAE)", fmt(ml["mae"]), fmt(mr["mae"]), winner(ml["mae"], mr["mae"])),
        ("Next-day RMSE", fmt(ml["rmse"]), fmt(mr["rmse"]), winner(ml["rmse"], mr["rmse"])),
        ("Next-day MAPE", f"{ml['mape']:.2f}%", f"{mr['mape']:.2f}%", winner(ml["mape"], mr["mape"])),
        ("Beats “same as today”?", "✅" if ml["mae"] < ml["naive_mae"] else "❌",
         "✅" if mr["mae"] < mr["naive_mae"] else "❌", f"naive MAE {fmt(ml['naive_mae'])}"),
        ("Next-day direction hit-rate", f"{ml['direction_acc']:.1f}%", f"{mr['direction_acc']:.1f}%",
         winner(ml["direction_acc"], mr["direction_acc"], lower=False)),
        (f"{horizon}-day direction hit-rate", f"{ml[hk]:.1f}%", f"{mr[hk]:.1f}%", winner(ml[hk], mr[hk], lower=False)),
        (f"Avg {horizon}-day % error", f"{ml['horizon_mae'][-1]:.2f}%", f"{mr['horizon_mae'][-1]:.2f}%",
         winner(ml["horizon_mae"][-1], mr["horizon_mae"][-1])),
        (f"Forecast {horizon} days out", fmt(L.future['pred'].iloc[-1]), fmt(R.future['pred'].iloc[-1]),
         f"{(L.future['pred'].iloc[-1] / last - 1) * 100:+.1f}% vs {(R.future['pred'].iloc[-1] / last - 1) * 100:+.1f}%"),
        (f"80% range width (day {horizon})",
         f"±{(L.future['high'].iloc[-1] - L.future['low'].iloc[-1]) / 2 / last * 100:.1f}%",
         f"±{(R.future['high'].iloc[-1] - R.future['low'].iloc[-1]) / 2 / last * 100:.1f}%", "narrower = more confident"),
        ("Parameters", f"{L.extras['n_params']:,}", f"{R.extras['n_params']:,}", ""),
        ("Training time", f"{L.extras.get('train_seconds', 0):.1f} s", f"{R.extras.get('train_seconds', 0) * 1000:.0f} ms", ""),
    ]
    table = pd.DataFrame(rows, columns=["Metric", MODELS["lstm"][0], MODELS["regression"][0], "Better / note"])
    st.dataframe(table, hide_index=True, width="stretch")

    # Plain-English verdict
    lstm_wins = sum(r[3] == MODELS["lstm"][0] for r in rows)
    reg_wins = sum(r[3] == MODELS["regression"][0] for r in rows)
    if lstm_wins > reg_wins:
        verdict = f"🧠 **The LSTM wins {lstm_wins} of {lstm_wins + reg_wins} scored metrics** on {ticker}."
    elif reg_wins > lstm_wins:
        verdict = (f"📐 **The linear regression wins {reg_wins} of {lstm_wins + reg_wins} scored metrics** on {ticker}, "
                   f"with {R.extras['n_params'] / L.extras['n_params'] * 100:.0f}% of the LSTM's parameters "
                   "and training in milliseconds.")
    else:
        verdict = f"🤝 **It's a draw on {ticker}.** The extra complexity of the LSTM doesn't clearly pay off here."
    gap = abs(ml["mae"] - mr["mae"]) / ml["naive_mae"] * 100
    st.info(f"{verdict} Their next-day errors differ by {gap:.1f}% of the naive error. "
            f"They agree on the {horizon}-day direction on **{cmpm['agree_h']:.0f}%** of test days "
            f"(correlation {cmpm['corr_h']:.2f}).")

    a, b = st.columns(2)
    with a:
        st.markdown("**Forecasts side by side**")
        h = res.history.tail(90)
        fig = go.Figure()
        fig.add_scatter(x=h.index, y=h["Close"], line=dict(color="#E6E8F0", width=2), name="Close")
        for r_ in (L, R):
            name, col = MODELS[r_.model]
            fx = [h.index[-1], *r_.future.index]
            fig.add_scatter(x=fx, y=[last, *r_.future["high"]], line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=fx, y=[last, *r_.future["low"]], fill="tonexty", fillcolor=col + "22",
                            line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=fx, y=[last, *r_.future["pred"]], line=dict(color=col, width=3, dash="dot"), name=name)
        st.plotly_chart(base_layout(fig, 380), width="stretch")
    with b:
        st.markdown("**Error vs. how far ahead** (lower is better)")
        d = list(range(1, horizon + 1))
        fig = go.Figure()
        fig.add_scatter(x=d, y=ml["horizon_mae"], line=dict(color=ACCENT, width=2.5), name=MODELS["lstm"][0])
        fig.add_scatter(x=d, y=mr["horizon_mae"], line=dict(color=REG, width=2.5), name=MODELS["regression"][0])
        fig.add_scatter(x=d, y=ml["naive_horizon_mae"], line=dict(color=MUTED, width=1.5, dash="dash"),
                        name="Naive: no change")
        fig.update_xaxes(title="days ahead")
        fig.update_yaxes(title="avg |error| in % return")
        st.plotly_chart(base_layout(fig, 380), width="stretch")

    st.markdown("**Backtest: next-day predictions on unseen days**")
    lt, rt = L.test, R.test
    fig = go.Figure()
    fig.add_scatter(x=lt.index, y=lt["actual"], line=dict(color="#E6E8F0", width=2), name="Actual")
    fig.add_scatter(x=lt.index, y=lt["predicted"], line=dict(color=ACCENT, width=1.5), name=MODELS["lstm"][0])
    fig.add_scatter(x=rt.index, y=rt["predicted"], line=dict(color=REG, width=1.5), name=MODELS["regression"][0])
    st.plotly_chart(base_layout(fig, 380), width="stretch")

    st.markdown("**What each model predicted for the move, vs. what really happened** "
                f"({horizon}-day % return, test period)")
    hcol = f"pred_ret_{horizon}d"
    real = np.log(lt["actual"].shift(-(horizon - 1)) / lt["naive"]) * 100  # realised horizon return
    fig = go.Figure()
    fig.add_scatter(x=lt.index, y=real, line=dict(color="#E6E8F0", width=1), name="Actual", opacity=0.6)
    fig.add_scatter(x=lt.index, y=lt[hcol], line=dict(color=ACCENT, width=2), name=MODELS["lstm"][0])
    fig.add_scatter(x=rt.index, y=rt[hcol], line=dict(color=REG, width=2), name=MODELS["regression"][0])
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
    st.plotly_chart(base_layout(fig, 340), width="stretch")
    st.caption("Predicted lines that hug zero mean the model has found little reliable pattern and is hedging. "
               "Ridge regression usually does this more than the LSTM; that's caution, not a bug.")

    with st.expander("🧩 How the two models differ"):
        st.markdown(
            """
| | 🧠 LSTM | 📐 Linear regression (Ridge) |
|---|---|---|
| **What it sees** | The full 60-day sequence, step by step | 42 summary numbers (last 3 days + 5/20/60-day averages) |
| **Shape of relationships** | Can learn curves, thresholds and interactions | Straight lines only: each input nudges the forecast up or down by a fixed amount |
| **Training** | Gradient descent, 40 epochs, random seed matters | Exact closed-form solution, same answer every time |
| **Overfitting control** | Dropout + weight decay | Ridge penalty α, tuned per forecast day |
| **Explainability** | Black box | Every weight is readable (see 🧪 tab with Linear regression selected) |
| **Speed** | ~10–20 s | milliseconds |

**Why compare them?** Daily stock returns are mostly noise. If a simple linear model matches the LSTM, the
"deep learning" isn't finding extra structure, and the simpler model is the better choice. If the LSTM wins
clearly and consistently across stocks, it's picking up non-linear patterns the regression can't.
"""
        )

with tab_data:
    out = pd.concat({"LSTM": res_lstm.future, "Regression": res_reg.future}, axis=1)
    out.columns = [f"{m} {c}" for m, c in out.columns]
    out = out.rename(columns=lambda c: c.replace("pred", "forecast").replace("low", "low (80%)")
                     .replace("high", "high (80%)")).round(2)
    out.index.name = "Date"
    a, b = st.columns(2)
    a.markdown("**Forecast**")
    a.dataframe(out, width="stretch", height=320)
    a.download_button("⬇️ Download forecast CSV", out.to_csv(), f"{ticker}_forecast.csv", "text/csv")
    b.markdown("**Recent prices**")
    recent = prices.tail(60).iloc[::-1].round(2)
    b.dataframe(recent, width="stretch", height=320)
    bt = res_lstm.test[["actual", "naive", "predicted"]].rename(columns={"predicted": "lstm_predicted"})
    bt["regression_predicted"] = res_reg.test["predicted"]
    b.download_button("⬇️ Download test results CSV", bt.round(2).to_csv(), f"{ticker}_backtest.csv", "text/csv")

st.markdown('<div class="disclaimer">For learning purposes only — not financial advice. '
            'Data from Yahoo Finance.</div>', unsafe_allow_html=True)
