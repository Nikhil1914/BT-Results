import datetime as dt
import pandas as pd
import calendar
from fyers_apiv3 import fyersModel
from config import ACCESS_TOKEN_FILE

# ======================= LOAD ACCESS TOKEN =========================
def load_access_token():
    with open(ACCESS_TOKEN_FILE, "r") as f:
        return f.read().strip()

CLIENT_ID = "AWSGEWQA6R-100"  # your client ID
ACCESS_TOKEN = load_access_token()

fyers = fyersModel.FyersModel(
    client_id=CLIENT_ID,
    token=ACCESS_TOKEN,
    is_async=False
)

# ================= SYMBOL BUILDER & FUTURES HANDLING =================
def last_thursday(year, month):
    cal = calendar.monthcalendar(year, month)
    thursdays = [
        week[calendar.THURSDAY] for week in cal if week[calendar.THURSDAY] != 0
    ]
    return dt.date(year, month, thursdays[-1])


def current_month_fut_symbol(root, exchange="NFO"):
    today = dt.date.today()
    year = today.year
    month = today.month
    expiry = last_thursday(year, month)

    if today > expiry:
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        expiry = last_thursday(year, month)

    month_code = ["JAN","FEB","MAR","APR","MAY","JUN",
                  "JUL","AUG","SEP","OCT","NOV","DEC"][month - 1]
    yy = str(year)[-2:]
    return f"{exchange}:{root}{yy}{month_code}FUT"


def map_live_symbol(backtest_symbol):
    s = backtest_symbol.upper()
    if s in ["NIFTY50", "NIFTY"]:
        return current_month_fut_symbol("NIFTY")
    if s in ["BANKNIFTY", "NIFTYBANK"]:
        return current_month_fut_symbol("BANKNIFTY")
    return backtest_symbol  # equity or other instruments


# ======================= PRICE / LTP HELPERS =======================

def get_ltp(symbol):
    try:
        res = fyers.quotes({"symbols": symbol})
        d = res.get("d", [])
        if not d:
            return None
        v = d[0].get("v", {})
        return v.get("lp") or v.get("ltp") or v.get("last_price")
    except Exception:
        return None


# ========================== ORDER WRAPPER ===========================

def place_market_order(symbol, side, qty, product_type="INTRADAY"):
    """
    MARKET order wrapper:
    side        => "BUY" or "SELL"
    productType => INTRADAY / MARGIN
    """
    side_val = 1 if side == "BUY" else -1

    payload = {
        "symbol": symbol,
        "qty": int(qty),
        "type": 2,              # MARKET
        "side": side_val,
        "productType": product_type,
        "limitPrice": 0,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False
    }

    try:
        r = fyers.place_order(payload)
        return r
    except Exception as e:
        return {"s": "error", "message": str(e)}


def close_position(symbol, position_side, qty, product="INTRADAY"):
    """ Close open position by sending opposite order """
    if position_side == "long":
        return place_market_order(symbol, "SELL", qty, product)
    if position_side == "short":
        return place_market_order(symbol, "BUY", qty, product)
    return {"s": "ok", "message": "No open position"}
