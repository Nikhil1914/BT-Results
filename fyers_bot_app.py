import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import altair as alt

from config import TIMEFRAME_RESOLUTIONS, BENCHMARK_SYMBOL, COLOR_GREEN, COLOR_RED
from fyers_helper import map_live_symbol
from live_engine import LIVE_STATE, start_live_engine, stop_live_engine
from fyers_helper import load_access_token
from live_engine import log_msg

from fyers_helper import get_ltp


# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="Nikhil Auto Trading Bot", layout="wide")
st.title("🤖 Automated Trading Bot — MA/EMA Crossover")
st.caption("Built for Nifty/Banks/NFO Futures with WebSocket Auto Execution")


# ====================== SIDEBAR SETTINGS ======================
st.sidebar.header("Strategy Parameters")

symbol = st.sidebar.text_input("Index / Equity Symbol", value="NIFTY50")
resolution = st.sidebar.selectbox("Timeframe", TIMEFRAME_RESOLUTIONS, index=2)

ma_type = st.sidebar.selectbox("MA Type", ["SMA", "EMA"])
fast_period = st.sidebar.number_input("Fast MA Period", min_value=1, value=9)
slow_period = st.sidebar.number_input("Slow MA Period", min_value=2, value=21)

trade_mode = st.sidebar.radio("Trading Mode", ["Intraday", "Positional"], index=0)
trade_side = st.sidebar.radio("Trade Side", ["Long Only", "Short Only", "Long & Short"], index=2)

tp_type = st.sidebar.selectbox("Target Type", ["Points", "Percent"], index=0)
tp_value = st.sidebar.number_input("Target Value", min_value=0.0, value=100.0)

sl_type = st.sidebar.selectbox("Stop Loss Type", ["Points", "Percent"], index=0)
sl_value = st.sidebar.number_input("Stop Loss Value", min_value=0.0, value=50.0)

qty = st.sidebar.number_input("Qty / Lots", min_value=1, value=1)
product_type = st.sidebar.selectbox("Product Type", ["INTRADAY", "MARGIN"], index=0)

st.sidebar.markdown("---")
run_live = st.sidebar.checkbox("Enable Live Auto Trading")
start_button = st.sidebar.button("🚀 Start Live Engine")
stop_button = st.sidebar.button("🛑 Stop Live Engine")
st.sidebar.markdown("---")

# ================== SHOW LIVE ENGINE STATUS ==================
if LIVE_STATE["running"]:
    st.success(f"Live Running: {LIVE_STATE['current_symbol']}  | TF={LIVE_STATE['tf']} min")
else:
    st.warning("Live Trading is OFF")


# ====================== HANDLE START / STOP ======================
if start_button:
    LIVE_STATE["ma_type"] = ma_type
    LIVE_STATE["fast"] = fast_period
    LIVE_STATE["slow"] = slow_period
    LIVE_STATE["qty"] = qty
    LIVE_STATE["trade_mode"] = trade_mode
    LIVE_STATE["trade_side"] = trade_side
    LIVE_STATE["tp_type"] = tp_type
    LIVE_STATE["tp_value"] = tp_value
    LIVE_STATE["sl_type"] = sl_type
    LIVE_STATE["sl_value"] = sl_value
    LIVE_STATE["product"] = product_type

    start_live_engine(symbol, resolution)
    st.experimental_rerun()

if stop_button:
    stop_live_engine()
    st.experimental_rerun()


# ====================== STATUS & LOG DISPLAY ======================
st.markdown("## 📡 Live Trade Engine Logs")
log_container = st.empty()

def push_logs():
    if LIVE_STATE["log"]:
        log_container.text("\n".join(LIVE_STATE["log"][-40:]))

push_logs()


# ====================== LIVE PRICE BOX ======================
st.markdown("## 📈 Live Price Feed")
live_col1, live_col2 = st.columns(2)
with live_col1:
    mapped = map_live_symbol(symbol)
    st.write(f"Live Instrument: **{mapped}**")
with live_col2:
    if LIVE_STATE["current_symbol"]:
        st.metric("LTP", get_ltp(LIVE_STATE["current_symbol"]))


# ================== SHOW CURRENT PARAMS ==================
st.markdown("### ⚙ Live Engine State")
state_display = {
    "Position": LIVE_STATE["position"],
    "Entry Price": LIVE_STATE["entry_price"],
    "TP Price": LIVE_STATE["tp_price"],
    "SL Price": LIVE_STATE["sl_price"],
    "Trade Mode": LIVE_STATE["trade_mode"],
    "Trade Side": LIVE_STATE["trade_side"],
    "Last Signal": LIVE_STATE["last_signal"],
}
st.json(state_display)


# ================== ADD FOOTER ==================
st.markdown("---")
st.caption("Powered by Fyers WebSocket API | Fully Automated | Nikhil")
