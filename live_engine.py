import datetime as dt
import pandas as pd
import numpy as np
import threading
import time
from websocket import WebSocketApp

from config import MARKET_OPEN, MARKET_CLOSE, INTRADAY_SQUAREOFF, MAX_CANDLE_STORAGE, WEBSOCKET_RECONNECT_DELAY
from fyers_helper import get_ltp, place_market_order, close_position, map_live_symbol


# =============================== GLOBAL STATE ===============================

LIVE_STATE = {
    "running": False,
    "ws": None,
    "current_symbol": None,
    "tf": "1",                        # timeframe minutes string
    "candle_buffer": [],              # list of candle bars
    "position": "flat",               # flat / long / short
    "entry_price": None,
    "entry_time": None,
    "tp_price": None,
    "sl_price": None,
    "last_signal": None,
    "trade_mode": "Intraday",         # Intraday or Positional
    "trade_side": "Long Only",        # Long Only / Short Only / Long & Short
    "qty": 1,
    "product": "INTRADAY",
    "ma_type": "SMA",
    "fast": 9,
    "slow": 21,
    "tp_type": "Points",
    "tp_value": 100,
    "sl_type": "Points",
    "sl_value": 50,
    "log": [],
    "lock": threading.Lock()
}

def log_msg(msg):
    with LIVE_STATE["lock"]:
        LIVE_STATE["log"].append(f"{dt.datetime.now()} | {msg}")
    print(msg)  # Console print for user debug
# ============================================================================



# =============================== STRATEGY HELPERS ===========================

def calc_tp_sl(entry, tp_type, tp_value, sl_type, sl_value, direction):
    if tp_type == "Points":
        tp = entry + tp_value if direction == "long" else entry - tp_value
    else:
        tp = entry * (1 + tp_value / 100) if direction == "long" else entry * (1 - tp_value / 100)

    if sl_type == "Points":
        sl = entry - sl_value if direction == "long" else entry + sl_value
    else:
        sl = entry * (1 - sl_value / 100) if direction == "long" else entry * (1 + sl_value / 100)

    return tp, sl


def calculate_ma(df, ma_type, fast, slow):
    df["fast_ma"] = df["close"].rolling(fast).mean() if ma_type == "SMA" else df["close"].ewm(span=fast, adjust=False).mean()
    df["slow_ma"] = df["close"].rolling(slow).mean() if ma_type == "SMA" else df["close"].ewm(span=slow, adjust=False).mean()

    df["signal"] = 0
    df.loc[(df["fast_ma"] > df["slow_ma"]) & (df["fast_ma"].shift(1) <= df["slow_ma"].shift(1)), "signal"] = 1
    df.loc[(df["fast_ma"] < df["slow_ma"]) & (df["fast_ma"].shift(1) >= df["slow_ma"].shift(1)), "signal"] = -1

    return df


def get_signal_from_candles():
    df = pd.DataFrame(LIVE_STATE["candle_buffer"], columns=["time","open","high","low","close"])
    if len(df) < LIVE_STATE["slow"] + 2:
        return None

    df = calculate_ma(df, LIVE_STATE["ma_type"], LIVE_STATE["fast"], LIVE_STATE["slow"])
    last = df.iloc[-1]
    prev = df.iloc[-2]

    if last["signal"] == 1:
        return "long"
    if last["signal"] == -1:
        return "short"
    return None
# ============================================================================



# =============================== ORDER EXECUTION ============================

def enter_position(direction):
    """ Place entry order """
    LIVE_STATE["position"] = direction
    price = get_ltp(LIVE_STATE["current_symbol"])
    LIVE_STATE["entry_price"] = price
    LIVE_STATE["entry_time"] = dt.datetime.now()
    LIVE_STATE["tp_price"], LIVE_STATE["sl_price"] = calc_tp_sl(
        price, LIVE_STATE["tp_type"], LIVE_STATE["tp_value"],
        LIVE_STATE["sl_type"], LIVE_STATE["sl_value"], direction
    )

    side = "BUY" if direction == "long" else "SELL"
    resp = place_market_order(LIVE_STATE["current_symbol"], side, LIVE_STATE["qty"], LIVE_STATE["product"])
    log_msg(f"{direction.upper()} ENTRY @ {price} | order: {resp}")


def exit_position(reason):
    direction = LIVE_STATE["position"]
    resp = close_position(LIVE_STATE["current_symbol"], direction, LIVE_STATE["qty"], LIVE_STATE["product"])
    log_msg(f"EXIT {direction.upper()} | {reason} | order: {resp}")

    LIVE_STATE["position"] = "flat"
    LIVE_STATE["entry_price"] = None
    LIVE_STATE["entry_time"] = None
    LIVE_STATE["tp_price"] = None
    LIVE_STATE["sl_price"] = None
# ============================================================================



# =============================== STRATEGY ENGINE ============================

def evaluate_strategy(latest_tick=None):
    """ Core logic matching backtest rules """
    now = dt.datetime.now().time()
    market_open = dt.time(*MARKET_OPEN)
    market_close = dt.time(*MARKET_CLOSE)
    sqoff_time = dt.time(*INTRADAY_SQUAREOFF)

    # Auto Pause
    if not (market_open <= now <= market_close):
        return

    # Intraday square-off
    if LIVE_STATE["trade_mode"] == "Intraday" and now >= sqoff_time:
        if LIVE_STATE["position"] != "flat":
            exit_position("Intraday EOD square-off")
        return

    # Check TP/SL
    if LIVE_STATE["position"] != "flat":
        ltp = latest_tick or get_ltp(LIVE_STATE["current_symbol"])
        if ltp is None:
            return

        if LIVE_STATE["position"] == "long":
            if ltp >= LIVE_STATE["tp_price"]:
                exit_position("Target Hit")
            elif ltp <= LIVE_STATE["sl_price"]:
                exit_position("Stop Loss")
        else:  # short
            if ltp <= LIVE_STATE["tp_price"]:
                exit_position("Target Hit")
            elif ltp >= LIVE_STATE["sl_price"]:
                exit_position("Stop Loss")

    # Determine crossover signal
    signal = get_signal_from_candles()
    if signal is None:
        return

    if LIVE_STATE["trade_side"] == "Long Only" and signal == "short":
        return
    if LIVE_STATE["trade_side"] == "Short Only" and signal == "long":
        return

    # Flip logic
    if LIVE_STATE["position"] == "long" and signal == "short":
        exit_position("Signal Flip")
        enter_position("short")
    elif LIVE_STATE["position"] == "short" and signal == "long":
        exit_position("Signal Flip")
        enter_position("long")

    # New entry
    elif LIVE_STATE["position"] == "flat":
        enter_position(signal)

    LIVE_STATE["last_signal"] = signal
# ============================================================================



# =============================== CANDLE BUILDING ============================

def aggregate_tick(tick):
    """ Tick → timeframe candle builder """
    ts = dt.datetime.now()
    price = tick
    tf = int(LIVE_STATE["tf"])

    # If empty — first candle
    if not LIVE_STATE["candle_buffer"]:
        LIVE_STATE["candle_buffer"].append([ts, price, price, price, price])
        return

    last_time = LIVE_STATE["candle_buffer"][-1][0]
    delta = (ts - last_time).total_seconds() / 60.0

    # New candle?
    if delta >= tf:
        LIVE_STATE["candle_buffer"].append([ts, price, price, price, price])
        if len(LIVE_STATE["candle_buffer"]) > MAX_CANDLE_STORAGE:
            LIVE_STATE["candle_buffer"] = LIVE_STATE["candle_buffer"][-MAX_CANDLE_STORAGE:]
    else:
        # update existing candle
        c = LIVE_STATE["candle_buffer"][-1]
        c[2] = max(c[2], price)
        c[3] = min(c[3], price)
        c[4] = price
# ============================================================================



# =============================== WEBSOCKET HANDLER ===========================

def on_tick(ws, message):
    """ message structure: {'ltp': ... } """
    try:
        price = float(eval(message)["ltp"])
    except:
        return

    aggregate_tick(price)
    evaluate_strategy(price)


def on_open(ws):
    log_msg("WebSocket connected")
    ws.send(f'{{"symbol":"{LIVE_STATE["current_symbol"]}","dataType":"symbolUpdate"}}')


def on_close(ws, *args):
    log_msg("WebSocket closed")
    if LIVE_STATE["running"]:
        time.sleep(WEBSOCKET_RECONNECT_DELAY)
        start_websocket()  # reconnect


def on_error(ws, e):
    log_msg(f"WebSocket Error: {e}")


def start_websocket():
    ws_url = f"wss://api.fyers.in/socket/v3/data?access_token={ACCESS_TOKEN}"
    ws = WebSocketApp(
        ws_url,
        on_open=on_open,
        on_close=on_close,
        on_error=on_error,
        on_message=on_tick
    )
    LIVE_STATE["ws"] = ws
    threading.Thread(target=ws.run_forever, daemon=True).start()
# ============================================================================



# =============================== PUBLIC API CALLED BY STREAMLIT =============

def start_live_engine(symbol, timeframe):
    LIVE_STATE["current_symbol"] = map_live_symbol(symbol)
    LIVE_STATE["tf"] = timeframe
    LIVE_STATE["running"] = True

    log_msg(f"Starting LIVE engine for {LIVE_STATE['current_symbol']} | TF={timeframe}")
    start_websocket()


def stop_live_engine():
    LIVE_STATE["running"] = False
    if LIVE_STATE["ws"]:
        LIVE_STATE["ws"].close()
    log_msg("Stopped live engine")
