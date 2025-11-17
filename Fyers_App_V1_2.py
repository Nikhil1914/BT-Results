import streamlit as st
from fyers_apiv3 import fyersModel
import pandas as pd
import numpy as np
import datetime as dt
import altair as alt
import calendar

# ========================= FYERS SETUP =========================
# access.txt must contain your access token string
with open('access.txt', 'r') as a:
    access_token = a.read().strip()

# CHANGE THIS TO YOUR CLIENT ID
client_id = 'AWSGEWQA6R-100'
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")

NIFTY_SYMBOL = "NSE:NIFTY50-INDEX"  # Benchmark


# ==================== SYMBOL BUILD FUNCTION ====================
def build_fyers_symbol(segment, exchange, symbol,
                       year=None, month=None, day=None, strike=None, opt_type=None):
    if segment == "Equity":
        return f"{exchange}:{symbol}-EQ"
    elif segment == "Index":
        return f"{exchange}:{symbol}-INDEX"
    elif segment == "Equity Futures":
        return f"{exchange}:{symbol}{str(year)[-2:]}{month.upper()}FUT"
    elif segment in ["Currency Futures", "Commodity Futures"]:
        return f"{exchange}:{symbol}{str(year)[-2:]}{month.upper()}FUT"
    elif segment in ["Equity Options (Monthly Expiry)", "Currency Options (Monthly Expiry)", "Commodity Options (Monthly Expiry)"]:
        return f"{exchange}:{symbol}{str(year)[-2:]}{month.upper()}{strike}{opt_type}"
    elif segment in ["Equity Options (Weekly Expiry)", "Currency Options (Weekly Expiry)"]:
        return f"{exchange}:{symbol}{str(year)[-2:]}{month.upper()}{day:02d}{strike}{opt_type}"
    else:
        raise ValueError("Unsupported segment type")


# ---------- Helper: current / next month futures for index live trading ----------
def last_thursday(year, month):
    cal = calendar.monthcalendar(year, month)
    thursdays = [week[calendar.THURSDAY] for week in cal if week[calendar.THURSDAY] != 0]
    return dt.date(year, month, thursdays[-1])


def current_month_future_symbol(underlying, exchange="NFO", as_of=None):
    """
    For Index backtests, live execution will happen on current-month futures:
    e.g. NFO:NIFTY24NOVFUT. If today is after expiry, move to next month.
    """
    if as_of is None:
        as_of = dt.date.today()

    year = as_of.year
    month = as_of.month
    exp = last_thursday(year, month)
    if as_of > exp:  # after expiry → next month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        exp = last_thursday(year, month)

    month_code = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                  "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"][month - 1]
    yy = str(year)[-2:]
    return f"{exchange}:{underlying}{yy}{month_code}FUT"


# =================== HISTORICAL DATA FETCH =====================
def fetch_data(symbol, start_date, end_date, resolution="60"):
    df = pd.DataFrame()

    resolution_minutes = {
        "1": 1, "2": 2, "3": 3, "5": 5, "10": 10, "15": 15,
        "20": 20, "30": 30, "60": 60, "120": 120, "240": 240
    }

    if resolution in resolution_minutes:
        chunk_days = 100       # intraday
    elif resolution in ["D", "1D"]:
        chunk_days = 366       # daily
    elif resolution in ["W", "M"]:
        chunk_days = 365 * 3   # weekly / monthly
    else:
        st.error("Resolution not supported for long-range.")
        return df

    current_start = start_date
    while current_start <= end_date:
        current_end = min(current_start + dt.timedelta(days=chunk_days - 1), end_date)

        params = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": 1,
            "range_from": current_start.strftime('%Y-%m-%d'),
            "range_to": current_end.strftime('%Y-%m-%d'),
            "cont_flag": 1,
            "oi_flag": 1
        }

        try:
            response = fyers.history(params)
            if 'candles' in response and response['candles']:
                first_len = len(response['candles'][0])
                if first_len == 7:
                    chunk = pd.DataFrame(
                        response['candles'],
                        columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']
                    )
                elif first_len == 6:
                    chunk = pd.DataFrame(
                        response['candles'],
                        columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    )
                    chunk['OI'] = None
                else:
                    st.error(f"Unexpected candle format from {current_start} to {current_end}")
                    current_start = current_end + dt.timedelta(days=1)
                    continue

                chunk['Date'] = pd.to_datetime(chunk['Date'], unit='s')
                chunk['Date'] = (
                    chunk['Date']
                    .dt.tz_localize('UTC')
                    .dt.tz_convert('Asia/Kolkata')
                    .dt.tz_localize(None)
                )
                chunk = chunk.set_index('Date')
                chunk['Symbol'] = symbol
                df = pd.concat([df, chunk])
        except Exception as e:
            st.error(f"Error from {current_start} to {current_end}: {e}")

        current_start = current_end + dt.timedelta(days=1)

    df = df[~df.index.duplicated(keep='first')]
    return df


# ===================== INDICATOR FUNCTIONS =====================
def add_moving_averages(df, ma_type, fast_period, slow_period):
    df = df.copy()
    price = df['Close']

    if ma_type == "SMA":
        df['fast_ma'] = price.rolling(fast_period).mean()
        df['slow_ma'] = price.rolling(slow_period).mean()
    else:  # EMA
        df['fast_ma'] = price.ewm(span=fast_period, adjust=False).mean()
        df['slow_ma'] = price.ewm(span=slow_period, adjust=False).mean()

    df['signal'] = 0
    df.loc[
        (df['fast_ma'] > df['slow_ma']) &
        (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)),
        'signal'
    ] = 1   # Bullish crossover

    df.loc[
        (df['fast_ma'] < df['slow_ma']) &
        (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1)),
        'signal'
    ] = -1  # Bearish crossover

    return df


def calc_level(entry_price, level_type, level_value, direction="long"):
    if level_type == "Points":
        if direction == "long":
            tp_price = entry_price + level_value
            sl_price = entry_price - level_value
        else:
            tp_price = entry_price - level_value
            sl_price = entry_price + level_value
    else:  # Percent
        if direction == "long":
            tp_price = entry_price * (1 + level_value / 100.0)
            sl_price = entry_price * (1 - level_value / 100.0)
        else:
            tp_price = entry_price * (1 - level_value / 100.0)
            sl_price = entry_price * (1 + level_value / 100.0)
    return tp_price, sl_price


def compute_pnl_points(direction, entry_price, exit_price):
    """Positive points = profitable, for both long & short."""
    if direction == "long":
        return exit_price - entry_price
    else:
        return entry_price - exit_price


# ===================== BACKTESTING LOGIC =======================
def backtest_ma_crossover(
    df,
    resolution,
    ma_type="SMA",
    fast_period=9,
    slow_period=21,
    sl_type="Points",
    sl_value=50.0,
    tp_type="Points",
    tp_value=100.0,
    trade_mode="Intraday",      # Intraday / Positional
    trade_side="Long Only",     # Long Only / Short Only / Long & Short
):
    """
    Entry: NEXT candle open after crossover.
    In Long & Short mode, flips position on opposite crossover signal.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy().sort_index()

    intraday_resolutions = {"1", "2", "3", "5", "10", "15", "20", "30", "60", "120", "240"}
    if resolution in intraday_resolutions:
        df = df.between_time("09:15", "15:30")

    if df.empty:
        return pd.DataFrame()

    df = add_moving_averages(df, ma_type, fast_period, slow_period)

    trades = []
    in_trade = False
    direction = None
    entry_price = None
    entry_time = None
    tp_price = None
    sl_price = None
    trade_date = None

    intraday_exit_time = dt.time(14, 55)  # 2:55 PM

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        ts = df.index[i]
        prev_ts = df.index[i - 1]
        signal_prev = prev_row['signal']

        # ------------------------------------------------------------------
        # 1) FLIP ON OPPOSITE SIGNAL (only in Long & Short mode)
        # ------------------------------------------------------------------
        if in_trade and trade_side == "Long & Short":
            flip_to_long = (direction == "short" and signal_prev == 1)
            flip_to_short = (direction == "long" and signal_prev == -1)

            if flip_to_long or flip_to_short:
                # exit current trade at THIS bar open
                exit_price = row['Open']
                exit_time = ts
                pnl_points = compute_pnl_points(direction, entry_price, exit_price)
                trades.append({
                    "Entry Time": entry_time,
                    "Exit Time": exit_time,
                    "Direction": direction,
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "PnL Points": pnl_points,
                    "Exit Reason": "Signal Flip",
                    "Entry Date": entry_time.date()
                })

                # immediately open new trade in opposite direction at same open
                if flip_to_long:
                    direction = "long"
                else:
                    direction = "short"

                entry_price = row['Open']
                entry_time = ts
                trade_date = ts.date()
                tp_price, sl_price = calc_level(entry_price, tp_type, tp_value, direction)
                # go to next bar
                continue

        # ------------------------------------------------------------------
        # 2) EXIT ON DAY CHANGE (Intraday only)
        # ------------------------------------------------------------------
        if in_trade and trade_mode == "Intraday" and ts.date() != trade_date:
            exit_price = prev_row['Close']
            exit_time = prev_ts
            pnl_points = compute_pnl_points(direction, entry_price, exit_price)
            trades.append({
                "Entry Time": entry_time,
                "Exit Time": exit_time,
                "Direction": direction,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "PnL Points": pnl_points,
                "Exit Reason": "EOD (day change)",
                "Entry Date": entry_time.date()
            })
            in_trade = False
            entry_price = entry_time = tp_price = sl_price = trade_date = direction = None

        # ------------------------------------------------------------------
        # 3) MANAGE OPEN TRADE: TP / SL
        # ------------------------------------------------------------------
        if in_trade:
            bar_high = row['High']
            bar_low = row['Low']

            if direction == "long":
                hit_tp = bar_high >= tp_price
                hit_sl = bar_low <= sl_price
            else:  # short
                hit_tp = bar_low <= tp_price
                hit_sl = bar_high >= sl_price

            if hit_sl and hit_tp:
                exit_price = sl_price
                exit_reason = "SL&TP same bar (SL priority)"
            elif hit_tp:
                exit_price = tp_price
                exit_reason = "Target"
            elif hit_sl:
                exit_price = sl_price
                exit_reason = "Stop Loss"
            else:
                exit_price = None
                exit_reason = None

            if exit_price is not None:
                exit_time = ts
                pnl_points = compute_pnl_points(direction, entry_price, exit_price)
                trades.append({
                    "Entry Time": entry_time,
                    "Exit Time": exit_time,
                    "Direction": direction,
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "PnL Points": pnl_points,
                    "Exit Reason": exit_reason,
                    "Entry Date": entry_time.date()
                })
                in_trade = False
                entry_price = entry_time = tp_price = sl_price = trade_date = direction = None
                continue

        # ------------------------------------------------------------------
        # 4) INTRADAY FORCED EXIT AT 14:55
        # ------------------------------------------------------------------
        if in_trade and trade_mode == "Intraday":
            if ts.time() >= intraday_exit_time and ts.date() == trade_date:
                exit_price = row['Close']
                exit_time = ts
                pnl_points = compute_pnl_points(direction, entry_price, exit_price)
                trades.append({
                    "Entry Time": entry_time,
                    "Exit Time": exit_time,
                    "Direction": direction,
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "PnL Points": pnl_points,
                    "Exit Reason": "Intraday EOD 14:55",
                    "Entry Date": entry_time.date()
                })
                in_trade = False
                entry_price = entry_time = tp_price = sl_price = trade_date = direction = None
                continue

        # ------------------------------------------------------------------
        # 5) NEW ENTRY WHEN FLAT
        # ------------------------------------------------------------------
        if not in_trade:
            # Long entries
            if trade_side in ["Long Only", "Long & Short"] and signal_prev == 1:
                direction = "long"
                entry_price = row['Open']
                entry_time = ts
                trade_date = ts.date()
                tp_price, sl_price = calc_level(entry_price, tp_type, tp_value, direction)
                in_trade = True

            # Short entries
            elif trade_side in ["Short Only", "Long & Short"] and signal_prev == -1:
                direction = "short"
                entry_price = row['Open']
                entry_time = ts
                trade_date = ts.date()
                tp_price, sl_price = calc_level(entry_price, tp_type, tp_value, direction)
                in_trade = True

    # ----------------------------------------------------------------------
    # 6) FINAL OPEN TRADE EXIT
    # ----------------------------------------------------------------------
    if in_trade:
        last_row = df.iloc[-1]
        last_ts = df.index[-1]
        exit_price = last_row['Close']
        exit_time = last_ts
        pnl_points = compute_pnl_points(direction, entry_price, exit_price)
        reason = "EOD (last bar)" if trade_mode == "Intraday" else "Backtest end"
        trades.append({
            "Entry Time": entry_time,
            "Exit Time": exit_time,
            "Direction": direction,
            "Entry Price": entry_price,
            "Exit Price": exit_price,
            "PnL Points": pnl_points,
            "Exit Reason": reason,
            "Entry Date": entry_time.date()
        })

    if len(trades) == 0:
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'])
    trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'])
    trades_df['Return %'] = (trades_df['PnL Points'] / trades_df['Entry Price']) * 100.0

    return trades_df


# ================= COSTS, CAPITAL & SUMMARY ====================
def add_costs_and_equity(trades_df, qty, slip_pts_side, broker_rs_side, initial_capital):
    df = trades_df.copy()
    df['Qty'] = qty
    df['GrossPoints'] = df['PnL Points']
    df['GrossPnlRs'] = df['GrossPoints'] * df['Qty']
    df['SlippageRs'] = slip_pts_side * 2 * df['Qty']   # in/out
    df['BrokerageRs'] = broker_rs_side * 2             # in/out
    df['NetPnlRs'] = df['GrossPnlRs'] - df['SlippageRs'] - df['BrokerageRs']
    df['CumNetPnlRs'] = df['NetPnlRs'].cumsum()
    df['Equity'] = initial_capital + df['CumNetPnlRs']
    return df


def build_daily_stats(trades_df, initial_capital):
    daily = (
        trades_df
        .groupby('Entry Date', as_index=False)
        .agg(
            GrossPoints=('GrossPoints', 'sum'),
            NetPnlRs=('NetPnlRs', 'sum'),
            Trades=('NetPnlRs', 'count')
        )
        .rename(columns={'Entry Date': 'Date'})
    )

    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.sort_values('Date').reset_index(drop=True)

    daily['CumNetPnlRs'] = daily['NetPnlRs'].cumsum()
    daily['Equity'] = initial_capital + daily['CumNetPnlRs']
    daily['PeakEquity'] = daily['Equity'].cummax()
    daily['DrawdownRs'] = daily['Equity'] - daily['PeakEquity']

    return daily


def compute_risk_stats(daily):
    if daily.empty:
        return {}

    best_day = daily.loc[daily['NetPnlRs'].idxmax()]
    worst_day = daily.loc[daily['NetPnlRs'].idxmin()]
    max_dd = daily['DrawdownRs'].min()
    avg_daily = daily['NetPnlRs'].mean()

    return {
        "best_day_date": best_day['Date'].date(),
        "best_day_pnl": best_day['NetPnlRs'],
        "worst_day_date": worst_day['Date'].date(),
        "worst_day_pnl": worst_day['NetPnlRs'],
        "max_drawdown": max_dd,
        "avg_daily_pnl": avg_daily,
    }


def compute_overall_summary(trades_df, initial_capital):
    total_trades = len(trades_df)
    wins = (trades_df['NetPnlRs'] > 0).sum()
    losses = (trades_df['NetPnlRs'] < 0).sum()
    breakeven = (trades_df['NetPnlRs'] == 0).sum()
    win_rate = (wins / total_trades) * 100.0 if total_trades > 0 else 0.0

    gross_points_total = trades_df['GrossPoints'].sum()
    net_pnl_rs_total = trades_df['NetPnlRs'].sum()
    end_equity = trades_df['Equity'].iloc[-1]
    cum_return_pct = (end_equity / initial_capital - 1.0) * 100.0
    avg_return_pct = trades_df['Return %'].mean()

    trades_df['Month'] = trades_df['Entry Time'].dt.to_period('M').astype(str)
    monthly = (
        trades_df
        .groupby('Month', as_index=False)
        .agg(
            Trades=('NetPnlRs', 'count'),
            NetPnlRs=('NetPnlRs', 'sum'),
            GrossPoints=('GrossPoints', 'sum'),
            Wins=('NetPnlRs', lambda x: (x > 0).sum()),
            Losses=('NetPnlRs', lambda x: (x < 0).sum())
        )
    )
    monthly['WinRatePct'] = (monthly['Wins'] / monthly['Trades']) * 100.0

    summary = {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate": win_rate,
        "gross_points_total": gross_points_total,
        "net_pnl_rs_total": net_pnl_rs_total,
        "avg_return_pct": avg_return_pct,
        "cum_return_pct": cum_return_pct,
        "monthly_stats": monthly,
        "end_equity": end_equity,
    }
    return summary


def side_summary(trades_df, side):
    df = trades_df[trades_df['Direction'] == side].copy()
    if df.empty:
        return None
    total_trades = len(df)
    wins = (df['NetPnlRs'] > 0).sum()
    losses = (df['NetPnlRs'] < 0).sum()
    breakeven = (df['NetPnlRs'] == 0).sum()
    win_rate = (wins / total_trades) * 100.0 if total_trades > 0 else 0.0
    net_pnl = df['NetPnlRs'].sum()
    gross_points = df['GrossPoints'].sum()

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate": win_rate,
        "net_pnl": net_pnl,
        "gross_points": gross_points,
    }


def build_calendar_df(daily):
    cal = daily.copy()
    cal['Year'] = cal['Date'].dt.year
    cal['Month'] = cal['Date'].dt.month_name().str[:3]
    cal['Day'] = cal['Date'].dt.day
    return cal


# ====================== LIVE TRADING HELPERS ===================
def latest_strategy_signal(df, ma_type, fast_period, slow_period):
    """
    Return 'long', 'short' or None based on the last crossover on df.
    """
    df_ma = add_moving_averages(df.copy(), ma_type, fast_period, slow_period)
    if len(df_ma) < 2:
        return None

    last = df_ma.iloc[-1]
    prev = df_ma.iloc[-2]

    if last['fast_ma'] > last['slow_ma'] and prev['fast_ma'] <= prev['slow_ma']:
        return "long"
    if last['fast_ma'] < last['slow_ma'] and prev['fast_ma'] >= prev['slow_ma']:
        return "short"
    return None


def place_market_order_fyers(symbol, side, qty, product_type="INTRADAY"):
    """
    Place a simple market order via Fyers.
    side: 'BUY' or 'SELL'
    product_type: 'INTRADAY', 'MARGIN', etc.
    """
    side_val = 1 if side == "BUY" else -1
    order_data = {
        "symbol": symbol,
        "qty": qty,
        "type": 2,           # 2 = MARKET
        "side": side_val,    # 1 = Buy, -1 = Sell
        "productType": product_type,
        "limitPrice": 0,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
        "stopLoss": 0,
        "takeProfit": 0
    }
    try:
        resp = fyers.place_order(order_data)
        return resp
    except Exception as e:
        return {"s": "error", "message": str(e)}


# ======================= STREAMLIT UI ==========================
st.set_page_config(page_title="Nikhil's Fyers Backtester", layout="wide")
st.title("📈 Nikhil's Fyers MA/EMA Crossover Backtester")

st.sidebar.header("Data & Strategy Settings")

# ---------- Segment / Symbol ----------
segment = st.sidebar.selectbox("Segment", [
    "Index", "Equity", "Equity Futures", "Equity Options (Monthly Expiry)",
    "Equity Options (Weekly Expiry)", "Currency Futures", "Currency Options (Monthly Expiry)",
    "Currency Options (Weekly Expiry)", "Commodity Futures", "Commodity Options (Monthly Expiry)"
])

exchange = st.sidebar.selectbox("Exchange", ["NSE", "BSE", "MCX", "CDS", "NFO"])
symbol = st.sidebar.text_input("Symbol", value="NIFTY50")

year = month = day = strike = opt_type = None
if segment not in ["Index", "Equity"]:
    years = list(range(2017, dt.date.today().year + 2))
    year = st.sidebar.selectbox("Year", years, index=len(years) - 1)

    month = st.sidebar.selectbox("Month", [
        "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
        "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
    ], index=dt.date.today().month - 1)

    if "Weekly" in segment:
        day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=dt.date.today().day)

    if "Options" in segment:
        strike = st.sidebar.text_input("Strike Price", value="")
        opt_type = st.sidebar.selectbox("Option Type", ["", "CE", "PE"])
else:
    st.sidebar.markdown("**Note:** No expiry/strike required for this segment.")

# ---------- Date & Resolution ----------
min_date = dt.date(2017, 7, 3)
max_date = dt.date.today()

start_date = st.sidebar.date_input(
    "Start Date",
    value=max_date - dt.timedelta(days=180),
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "End Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

resolution = st.sidebar.selectbox(
    "Resolution (Timeframe)",
    options=["1", "2", "3", "5", "10", "15", "20", "30", "60", "120", "240", "D"],
    index=9
)

# ---------- Strategy ----------
st.sidebar.markdown("---")
st.sidebar.subheader("Strategy: MA/EMA Crossover")

ma_type = st.sidebar.selectbox("MA Type", ["SMA", "EMA"])

period_options = [5, 8, 9, 10, 13, 20, 21, 34, 50, 100, 200]
fast_period = st.sidebar.selectbox("Fast MA Period", period_options, index=2)
slow_period = st.sidebar.selectbox("Slow MA Period", period_options, index=6)

if slow_period <= fast_period:
    st.sidebar.warning("Slow period should be greater than fast period for a proper crossover.")

# ---------- Target / SL ----------
st.sidebar.markdown("---")
st.sidebar.subheader("Target & Stop Loss")

tp_type = st.sidebar.selectbox("Target Type", ["Points", "Percent"], index=0)
tp_value = st.sidebar.number_input(f"Target ({tp_type})", min_value=0.0, value=100.0, step=1.0)

sl_type = st.sidebar.selectbox("Stop Loss Type", ["Points", "Percent"], index=0)
sl_value = st.sidebar.number_input(f"Stop Loss ({sl_type})", min_value=0.0, value=50.0, step=1.0)

# ---------- Trade mode & side ----------
st.sidebar.markdown("---")
trade_mode = st.sidebar.radio("Trade Mode", ["Intraday", "Positional"], index=0)
trade_side = st.sidebar.radio("Trade Side", ["Long Only", "Short Only", "Long & Short"], index=0)

# ---------- Capital & Costs ----------
st.sidebar.markdown("---")
st.sidebar.subheader("Capital & Costs")

initial_capital = st.sidebar.number_input("Initial Capital (₹)", min_value=0.0, value=100000.0, step=10000.0)
quantity = st.sidebar.number_input("Quantity / Lot Size (per trade)", min_value=1, value=25, step=1)
slippage_per_side = st.sidebar.number_input("Slippage per side (points)", min_value=0.0, value=0.5, step=0.05)
brokerage_per_side = st.sidebar.number_input("Brokerage per side (₹)", min_value=0.0, value=20.0, step=1.0)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Backtest")

# ========================= MAIN ================================
if start_date > end_date:
    st.error("⚠️ Start date must not be after end date.")
else:
    fyers_symbol = build_fyers_symbol(segment, exchange, symbol, year, month, day, strike, opt_type)
    st.caption(
        f"Symbol: **{fyers_symbol}**  |  TF: {resolution}  |  Mode: **{trade_mode}**  |  Side: **{trade_side}**  |  {start_date} → {end_date}"
    )

    if run_button:
        with st.spinner(f"Fetching data & running backtest for {fyers_symbol}..."):
            data = fetch_data(fyers_symbol, start_date, end_date, resolution)

            if data.empty:
                st.error("No data received. Try different dates / symbol / resolution.")
            else:
                price_df = data[['Open', 'High', 'Low', 'Close']].copy()

                base_trades_df = backtest_ma_crossover(
                    price_df,
                    resolution=resolution,
                    ma_type=ma_type,
                    fast_period=fast_period,
                    slow_period=slow_period,
                    sl_type=sl_type,
                    sl_value=sl_value,
                    tp_type=tp_type,
                    tp_value=tp_value,
                    trade_mode=trade_mode,
                    trade_side=trade_side,
                )

                if base_trades_df.empty:
                    st.warning("No trades generated with the selected parameters.")
                else:
                    trades_df = add_costs_and_equity(
                        base_trades_df, quantity, slippage_per_side, brokerage_per_side, initial_capital
                    )

                    overall = compute_overall_summary(trades_df, initial_capital)
                    daily_stats = build_daily_stats(trades_df, initial_capital)
                    risk = compute_risk_stats(daily_stats)

                    # ===== NIFTY Benchmark =====
                    nifty_data = fetch_data(NIFTY_SYMBOL, start_date, end_date, resolution)
                    nifty_first = nifty_last = nifty_ret_pct = None
                    strat_vs_nifty_pct = None

                    if not nifty_data.empty:
                        nifty_first = nifty_data['Close'].iloc[0]
                        nifty_last = nifty_data['Close'].iloc[-1]
                        nifty_ret_pct = (nifty_last / nifty_first - 1.0) * 100.0
                        strat_vs_nifty_pct = (overall['gross_points_total'] / nifty_first) * 100.0

                    # ===== KPI SECTION =====
                    st.markdown("## 📊 Overview")

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total Trades", overall["total_trades"])
                    k2.metric("Profitable Trades", overall["wins"])
                    k3.metric("Loss-making Trades", overall["losses"])
                    k4.metric("Win Rate (%)", f"{overall['win_rate']:.2f}")

                    k5, k6, k7, k8 = st.columns(4)
                    k5.metric("Gross PnL (Points)", f"{overall['gross_points_total']:.2f}")
                    k6.metric("Net PnL (₹)", f"{overall['net_pnl_rs_total']:.2f}")
                    k7.metric("Cumulative Return (%)", f"{overall['cum_return_pct']:.2f}")
                    k8.metric("Avg Return / Trade (%)", f"{overall['avg_return_pct']:.2f}")

                    # NIFTY comparison
                    b1, b2, b3, b4 = st.columns(4)
                    if nifty_ret_pct is not None:
                        b1.metric("NIFTY50 Start", f"{nifty_first:.2f}")
                        b2.metric("NIFTY50 End", f"{nifty_last:.2f}")
                        b3.metric("NIFTY50 Return (%)", f"{nifty_ret_pct:.2f}")
                        b4.metric("Strategy Gross Points as % of NIFTY Start", f"{strat_vs_nifty_pct:.2f}")
                    else:
                        b1.metric("NIFTY50 Return (%)", "N/A")
                        b2.metric("Strategy vs NIFTY", "N/A")
                        b3.metric("", "")
                        b4.metric("", "")

                    # Daily risk stats
                    r1, r2, r3, r4 = st.columns(4)
                    if risk:
                        r1.metric("Best Day PnL (₹)", f"{risk['best_day_pnl']:.2f}", str(risk['best_day_date']))
                        r2.metric("Worst Day PnL (₹)", f"{risk['worst_day_pnl']:.2f}", str(risk['worst_day_date']))
                        r3.metric("Max Drawdown (₹)", f"{risk['max_drawdown']:.2f}")
                        r4.metric("Avg Daily PnL (₹)", f"{risk['avg_daily_pnl']:.2f}")
                    else:
                        r1.metric("Best Day PnL", "–")
                        r2.metric("Worst Day PnL", "–")
                        r3.metric("Max Drawdown", "–")
                        r4.metric("Avg Daily PnL", "–")

                    # Long vs Short stats for Combined
                    if trade_side == "Long & Short":
                        long_summary = side_summary(trades_df, "long")
                        short_summary = side_summary(trades_df, "short")

                        st.markdown("### 🔁 Long vs Short Breakdown")
                        c1, c2 = st.columns(2)

                        with c1:
                            st.markdown("**Long Trades**")
                            if long_summary:
                                st.metric("Trades", long_summary["total_trades"])
                                st.metric("Win Rate (%)", f"{long_summary['win_rate']:.2f}")
                                st.metric("Net PnL (₹)", f"{long_summary['net_pnl']:.2f}")
                                st.metric("Gross Points", f"{long_summary['gross_points']:.2f}")
                            else:
                                st.write("No long trades.")

                        with c2:
                            st.markdown("**Short Trades**")
                            if short_summary:
                                st.metric("Trades", short_summary["total_trades"])
                                st.metric("Win Rate (%)", f"{short_summary['win_rate']:.2f}")
                                st.metric("Net PnL (₹)", f"{short_summary['net_pnl']:.2f}")
                                st.metric("Gross Points", f"{short_summary['gross_points']:.2f}")
                            else:
                                st.write("No short trades.")

                    st.markdown("---")

                    # ===== Calendar Heatmap =====
                    st.markdown("### 📅 Daily Net PnL Calendar Heatmap (₹)")
                    cal_df = build_calendar_df(daily_stats)
                    if not cal_df.empty:
                        cal_chart = (
                            alt.Chart(cal_df)
                            .mark_rect()
                            .encode(
                                x=alt.X('day(Date):O', title='Day'),
                                y=alt.Y('month(Date):O', title='Month'),
                                color=alt.condition(
                                    alt.datum.NetPnlRs >= 0,
                                    alt.value("#16a34a"),  # green
                                    alt.value("#dc2626"),  # red
                                ),
                                tooltip=['Date:T', 'NetPnlRs:Q', 'Trades:Q']
                            )
                            .properties(height=260)
                        )
                        st.altair_chart(cal_chart, use_container_width=True)
                    else:
                        st.info("No daily stats available.")

                    st.markdown("---")

                    # ===== Price + MA + Markers (Zoomable) =====
                    st.markdown("### 📈 Price + Moving Averages + Trade Markers (Zoomable)")

                    ma_df = add_moving_averages(price_df, ma_type, fast_period, slow_period)
                    plot_df = ma_df.copy()
                    plot_df = plot_df.reset_index()
                    if 'index' in plot_df.columns:
                        plot_df.rename(columns={'index': 'Date'}, inplace=True)
                    plot_df['Time'] = plot_df['Date']

                    markers_df = trades_df.copy()
                    markers_df['Entry Time'] = pd.to_datetime(markers_df['Entry Time'])
                    markers_df['Exit Time'] = pd.to_datetime(markers_df['Exit Time'])

                    # Base price + MAs
                    price_line = alt.Chart(plot_df).mark_line().encode(
                        x=alt.X('Time:T', title='Date'),
                        y=alt.Y('Close:Q', title='Price'),
                        tooltip=['Time:T', 'Open', 'High', 'Low', 'Close']
                    )

                    fast_ma_line = alt.Chart(plot_df).mark_line(color="#3b82f6").encode(
                        x='Time:T', y='fast_ma:Q'
                    )

                    slow_ma_line = alt.Chart(plot_df).mark_line(color="#fb923c").encode(
                        x='Time:T', y='slow_ma:Q'
                    )

                    # Crossover markers
                    bull_cross = alt.Chart(plot_df[plot_df['signal'] == 1]).mark_point(
                        color='blue', size=80, shape='circle'
                    ).encode(
                        x='Time:T', y='Close:Q',
                        tooltip=['Time:T', 'Close']
                    )

                    bear_cross = alt.Chart(plot_df[plot_df['signal'] == -1]).mark_point(
                        color='red', size=80, shape='circle'
                    ).encode(
                        x='Time:T', y='Close:Q',
                        tooltip=['Time:T', 'Close']
                    )

                    # Entry markers
                    long_entries = markers_df[markers_df['Direction'] == "long"]
                    short_entries = markers_df[markers_df['Direction'] == "short"]

                    long_marker = alt.Chart(long_entries).mark_triangle(
                        size=140, color="#16a34a"
                    ).encode(
                        x='Entry Time:T',
                        y='Entry Price:Q',
                        tooltip=['Entry Time', 'Entry Price']
                    )

                    short_marker = alt.Chart(short_entries).mark_triangle(
                        size=140, color="#dc2626", direction='down'
                    ).encode(
                        x='Entry Time:T',
                        y='Entry Price:Q',
                        tooltip=['Entry Time', 'Entry Price']
                    )

                    # Exit markers
                    exit_marker = alt.Chart(markers_df).mark_point(
                        size=120, color="black", shape="diamond"
                    ).encode(
                        x='Exit Time:T',
                        y='Exit Price:Q',
                        tooltip=['Exit Time', 'Exit Price', 'Exit Reason']
                    )

                    final_chart = (
                        price_line + fast_ma_line + slow_ma_line +
                        bull_cross + bear_cross +
                        long_marker + short_marker + exit_marker
                    ).interactive().properties(
                        width="container",
                        height=500,
                        title="Zoomable Price / MA / Trades"
                    )

                    st.altair_chart(final_chart, use_container_width=True)

                    st.markdown("---")

                    # ===== Performance Charts =====
                    st.markdown("## 📉 Performance Charts")
                    e1, e2 = st.columns(2)

                    with e1:
                        st.markdown("#### Equity Curve (₹)")
                        eq_chart = (
                            alt.Chart(daily_stats)
                            .mark_line(color="#16a34a")
                            .encode(
                                x='Date:T',
                                y=alt.Y('Equity:Q', title='Equity (₹)'),
                                tooltip=['Date:T', 'Equity:Q']
                            )
                        )
                        st.altair_chart(eq_chart, use_container_width=True)

                    with e2:
                        st.markdown("#### Drawdown Curve (₹)")
                        dd_chart = (
                            alt.Chart(daily_stats)
                            .mark_line(color="#dc2626")
                            .encode(
                                x='Date:T',
                                y=alt.Y('DrawdownRs:Q', title='Drawdown (₹)'),
                                tooltip=['Date:T', 'DrawdownRs:Q']
                            )
                        )
                        st.altair_chart(dd_chart, use_container_width=True)

                    # Strategy vs NIFTY %
                    if nifty_ret_pct is not None and not daily_stats.empty:
                        st.markdown("### 📈 Strategy Equity vs NIFTY50 (% Return)")
                        strat_pct = daily_stats[['Date', 'Equity']].copy()
                        strat_pct['StrategyPct'] = (strat_pct['Equity'] / initial_capital - 1.0) * 100.0

                        nifty_close = (
                            nifty_data[['Close']]
                            .copy()
                            .reset_index()
                            .rename(columns={'Date': 'Date'})
                        )
                        nifty_close['Date'] = pd.to_datetime(nifty_close['Date'])
                        nifty_close = nifty_close.sort_values('Date').drop_duplicates(subset='Date')
                        nf_first = nifty_close['Close'].iloc[0]
                        nifty_close['NiftyPct'] = (nifty_close['Close'] / nf_first - 1.0) * 100.0

                        merge_df = pd.merge(strat_pct[['Date', 'StrategyPct']],
                                            nifty_close[['Date', 'NiftyPct']],
                                            on='Date', how='inner')

                        line1 = alt.Chart(merge_df).mark_line(color="#16a34a").encode(
                            x='Date:T',
                            y=alt.Y('StrategyPct:Q', title='% Return'),
                            tooltip=['Date:T', 'StrategyPct:Q']
                        )
                        line2 = alt.Chart(merge_df).mark_line(color="#3b82f6").encode(
                            x='Date:T',
                            y='NiftyPct:Q',
                            tooltip=['Date:T', 'NiftyPct:Q']
                        )
                        st.altair_chart(line1 + line2, use_container_width=True)

                    st.markdown("---")

                    # Daily PnL & Trades
                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown("#### Daily Net PnL (₹)")
                        daily_pnl_chart = (
                            alt.Chart(daily_stats)
                            .mark_bar()
                            .encode(
                                x='Date:T',
                                y='NetPnlRs:Q',
                                tooltip=['Date:T', 'NetPnlRs:Q', 'Trades:Q'],
                                color=alt.condition(
                                    alt.datum.NetPnlRs >= 0,
                                    alt.value("#16a34a"),
                                    alt.value("#dc2626")
                                )
                            )
                        )
                        st.altair_chart(daily_pnl_chart, use_container_width=True)

                    with d2:
                        st.markdown("#### Daily Number of Trades")
                        daily_trades_chart = (
                            alt.Chart(daily_stats)
                            .mark_bar(color="#6b7280")
                            .encode(
                                x='Date:T',
                                y='Trades:Q',
                                tooltip=['Date:T', 'Trades:Q']
                            )
                        )
                        st.altair_chart(daily_trades_chart, use_container_width=True)

                    # Monthly stats
                    st.markdown("### 📆 Monthly Statistics")
                    monthly_stats = overall["monthly_stats"]
                    if monthly_stats is not None and not monthly_stats.empty:
                        m1, m2 = st.columns(2)
                        with m1:
                            st.markdown("#### Monthly Net PnL (₹)")
                            ms_pnl_chart = (
                                alt.Chart(monthly_stats)
                                .mark_bar()
                                .encode(
                                    x=alt.X('Month:O', sort=None),
                                    y='NetPnlRs:Q',
                                    tooltip=['Month', 'NetPnlRs', 'Trades'],
                                    color=alt.condition(
                                        alt.datum.NetPnlRs >= 0,
                                        alt.value("#16a34a"),
                                        alt.value("#dc2626")
                                    )
                                )
                            )
                            st.altair_chart(ms_pnl_chart, use_container_width=True)

                        with m2:
                            st.markdown("#### Monthly Win Rate (%)")
                            ms_wr_chart = (
                                alt.Chart(monthly_stats)
                                .mark_bar(color="#3b82f6")
                                .encode(
                                    x=alt.X('Month:O', sort=None),
                                    y='WinRatePct:Q',
                                    tooltip=['Month', 'WinRatePct', 'Trades']
                                )
                            )
                            st.altair_chart(ms_wr_chart, use_container_width=True)

                        st.dataframe(monthly_stats)
                    else:
                        st.info("No monthly stats available.")

                    st.markdown("---")

                    # Outcome distribution
                    st.markdown("### 🥧 Trade Outcome Distribution (Net PnL)")
                    outcome_counts = {
                        "Wins": (trades_df['NetPnlRs'] > 0).sum(),
                        "Losses": (trades_df['NetPnlRs'] < 0).sum(),
                        "Breakeven": (trades_df['NetPnlRs'] == 0).sum(),
                    }
                    dist_df = pd.DataFrame(
                        {"Outcome": list(outcome_counts.keys()),
                         "Count": list(outcome_counts.values())}
                    )

                    pie_chart = (
                        alt.Chart(dist_df)
                        .mark_arc(innerRadius=40)
                        .encode(
                            theta='Count:Q',
                            color=alt.Color(
                                'Outcome:N',
                                scale=alt.Scale(
                                    domain=['Wins', 'Losses', 'Breakeven'],
                                    range=['#16a34a', '#dc2626', '#6b7280']
                                )
                            ),
                            tooltip=['Outcome', 'Count']
                        )
                    )
                    st.altair_chart(pie_chart, use_container_width=False)

                    st.markdown("---")

                    # Trades table
                    st.markdown("## 📋 Trades Table (with Costs)")
                    st.dataframe(trades_df)

                    st.download_button(
                        "⬇️ Download Trades CSV",
                        data=trades_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{symbol}_MA_Backtest_Trades.csv",
                        mime="text/csv"
                    )

                    # ==================== LIVE DEPLOYMENT ====================
                    st.markdown("---")
                    st.markdown("## ⚡ Live Deployment to Fyers (Use with Caution)")
                    st.caption(
                        "Index backtests will execute on current-month futures (e.g. NFO:NIFTYxxMONFUT). "
                        "Always test with small quantity / paper first."
                    )

                    live_enable = st.checkbox("Enable live order section")

                    if live_enable:
                        live_col1, live_col2 = st.columns(2)
                        with live_col1:
                            live_mode = st.radio(
                                "Live Action",
                                ["Use Strategy Signal (latest MA crossover)", "Manual BUY", "Manual SELL"],
                                index=0
                            )
                        with live_col2:
                            live_qty = st.number_input(
                                "Live Quantity", min_value=1, value=quantity, step=1
                            )
                            live_product = st.selectbox(
                                "Product Type", ["INTRADAY", "MARGIN"], index=0
                            )

                        # Determine live trading symbol
                        if segment == "Index":
                            underlying = symbol.replace("NIFTY50", "NIFTY")
                            live_symbol = current_month_future_symbol(underlying)
                        else:
                            live_symbol = fyers_symbol

                        st.write(f"Live trading symbol will be: **{live_symbol}**")

                        if st.button("✅ Place Live Order Now"):
                            side = None
                            if live_mode == "Manual BUY":
                                side = "BUY"
                            elif live_mode == "Manual SELL":
                                side = "SELL"
                            else:
                                # Use latest MA crossover signal on recent data
                                latest_data = fetch_data(
                                    fyers_symbol,
                                    dt.date.today() - dt.timedelta(days=5),
                                    dt.date.today(),
                                    resolution
                                )
                                if latest_data.empty:
                                    st.error("Could not fetch recent data for live signal.")
                                else:
                                    last_sig = latest_strategy_signal(
                                        latest_data[['Open', 'High', 'Low', 'Close']],
                                        ma_type,
                                        fast_period,
                                        slow_period
                                    )
                                    if last_sig is None:
                                        st.warning("No fresh crossover signal on latest data. No order placed.")
                                    else:
                                        if last_sig == "long" and trade_side in ["Long Only", "Long & Short"]:
                                            side = "BUY"
                                        elif last_sig == "short" and trade_side in ["Short Only", "Long & Short"]:
                                            side = "SELL"
                                        else:
                                            st.warning(
                                                f"Latest signal is '{last_sig}', "
                                                f"but trade side is '{trade_side}'. No order placed."
                                            )

                            if side is not None:
                                st.info(f"Placing {side} market order for {live_qty} on {live_symbol} via Fyers...")
                                resp = place_market_order_fyers(live_symbol, side, live_qty, live_product)
                                st.write("Fyers response:", resp)
                            else:
                                st.info("No valid side determined. Order not sent.")
    else:
        st.info("Set parameters on the left and click **Run Backtest**.")
