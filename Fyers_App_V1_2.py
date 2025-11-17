# %%
import streamlit as st
from fyers_apiv3 import fyersModel
import pandas as pd
import numpy as np
import datetime as dt

# ========================= FYERS SETUP =========================
# Load access token (same as your current app)
with open('access.txt', 'r') as a:
    access_token = a.read().strip()

client_id = 'KB4YGO9V7J-100'
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")

# ==================== SYMBOL BUILD FUNCTION ====================
def build_fyers_symbol(segment, exchange, symbol, year=None, month=None, day=None, strike=None, opt_type=None):
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

# =================== HISTORICAL DATA FETCH =====================
def fetch_data(symbol, start_date, end_date, resolution="60"):
    df = pd.DataFrame()
    resolution_minutes = {
        "1": 1, "2": 2, "3": 3, "5": 5, "10": 10, "15": 15,
        "20": 20, "30": 30, "60": 60, "120": 120, "240": 240
    }

    if resolution in resolution_minutes:
        chunk_days = 100
    elif resolution in ["D", "1D"]:
        chunk_days = 366
    elif resolution in ["W", "M"]:
        chunk_days = 365 * 3
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
                first_candle_len = len(response['candles'][0])
                if first_candle_len == 7:
                    chunk = pd.DataFrame(response['candles'], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
                elif first_candle_len == 6:
                    chunk = pd.DataFrame(response['candles'], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    chunk['OI'] = None
                else:
                    st.error(f"Unexpected candle format from {current_start} to {current_end}")
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

# ===================== BACKTESTING LOGIC =======================

def add_moving_averages(df, ma_type, fast_period, slow_period):
    df = df.copy()
    price = df['Close']

    if ma_type == "SMA":
        df['fast_ma'] = price.rolling(fast_period).mean()
        df['slow_ma'] = price.rolling(slow_period).mean()
    else:  # EMA
        df['fast_ma'] = price.ewm(span=fast_period, adjust=False).mean()
        df['slow_ma'] = price.ewm(span=slow_period, adjust=False).mean()

    # Long-only crossover signals
    df['signal'] = 0
    df.loc[
        (df['fast_ma'] > df['slow_ma']) &
        (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)),
        'signal'
    ] = 1   # Bullish crossover (entry)

    df.loc[
        (df['fast_ma'] < df['slow_ma']) &
        (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1)),
        'signal'
    ] = -1  # Bearish crossover (can be used as exit)

    return df

def calc_level(entry_price, level_type, level_value, direction):
    if level_type == "Points":
        if direction == "long":
            tp_price = entry_price + level_value
            sl_price = entry_price - level_value
        else:
            tp_price = entry_price - level_value
            sl_price = entry_price + level_value
    else:  # Percent
        tp_factor = 1 + (level_value / 100.0)
        sl_factor = 1 - (level_value / 100.0)
        if direction == "long":
            tp_price = entry_price * tp_factor
            sl_price = entry_price * sl_factor
        else:
            tp_price = entry_price * (2 - tp_factor)  # basically -% from entry
            sl_price = entry_price * (2 - sl_factor)  # basically +% from entry
    return tp_price, sl_price

def backtest_ma_crossover(
    df,
    ma_type="SMA",
    fast_period=9,
    slow_period=21,
    sl_type="Points",
    sl_value=50.0,
    tp_type="Points",
    tp_value=100.0,
):
    if df.empty:
        return pd.DataFrame(), {}

    df = df.copy()
    df = df.sort_index()

    # Intraday filter for Indian market (09:15 to 15:30) for intraday frames
    if df.index.freq is None:  # just in case, use string check instead
        # We know resolution from outside, but simplest is time filter:
        df = df.between_time("09:15", "15:30")

    if df.empty:
        return pd.DataFrame(), {}

    # Add moving averages & signals
    df = add_moving_averages(df, ma_type, fast_period, slow_period)

    trades = []
    in_trade = False
    direction = "long"
    entry_price = None
    entry_time = None
    tp_price = None
    sl_price = None
    trade_date = None

    prev_row = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        ts = df.index[i]
        prev_ts = df.index[i - 1]
        signal_prev = prev_row['signal']

        # Exit at day change (no overnight) using previous bar close
        if in_trade and ts.date() != trade_date:
            exit_price = prev_row['Close']
            exit_time = prev_ts
            pnl_points = exit_price - entry_price
            trades.append({
                "Entry Time": entry_time,
                "Exit Time": exit_time,
                "Direction": direction,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "PnL Points": pnl_points,
                "Return %": (pnl_points / entry_price) * 100.0,
                "Entry Date": entry_time.date()
            })
            in_trade = False
            entry_price = entry_time = tp_price = sl_price = trade_date = None

        # Manage open trade: check SL / TP on current candle
        if in_trade:
            bar_high = row['High']
            bar_low = row['Low']

            if direction == "long":
                hit_tp = bar_high >= tp_price
                hit_sl = bar_low <= sl_price

                # If both hit in same bar, assume SL first (conservative)
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
                    pnl_points = exit_price - entry_price
                    trades.append({
                        "Entry Time": entry_time,
                        "Exit Time": exit_time,
                        "Direction": direction,
                        "Entry Price": entry_price,
                        "Exit Price": exit_price,
                        "PnL Points": pnl_points,
                        "Return %": (pnl_points / entry_price) * 100.0,
                        "Exit Reason": exit_reason,
                        "Entry Date": entry_time.date()
                    })
                    in_trade = False
                    entry_price = entry_time = tp_price = sl_price = trade_date = None
                    continue  # go to next candle

        # If still not in trade, look for new entry
        if (not in_trade) and (signal_prev == 1):
            # Enter at current bar open after previous bar signal
            entry_price = row['Open']
            entry_time = ts
            trade_date = ts.date()
            direction = "long"
            tp_price, sl_price = calc_level(entry_price, tp_type, tp_value, direction)
            in_trade = True

    # If trade still open at last bar, close at last close
    if in_trade:
        last_row = df.iloc[-1]
        last_ts = df.index[-1]
        exit_price = last_row['Close']
        exit_time = last_ts
        pnl_points = exit_price - entry_price
        trades.append({
            "Entry Time": entry_time,
            "Exit Time": exit_time,
            "Direction": direction,
            "Entry Price": entry_price,
            "Exit Price": exit_price,
            "PnL Points": pnl_points,
            "Return %": (pnl_points / entry_price) * 100.0,
            "Exit Reason": "EOD (last bar)",
            "Entry Date": entry_time.date()
        })

    if len(trades) == 0:
        return pd.DataFrame(), {}

    trades_df = pd.DataFrame(trades)
    trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'])
    trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'])

    # Summary stats
    total_trades = len(trades_df)
    wins = (trades_df['PnL Points'] > 0).sum()
    losses = (trades_df['PnL Points'] < 0).sum()
    breakeven = (trades_df['PnL Points'] == 0).sum()
    win_rate = (wins / total_trades) * 100.0 if total_trades > 0 else 0.0
    net_points = trades_df['PnL Points'].sum()
    avg_return = trades_df['Return %'].mean()
    cum_return = trades_df['Return %'].sum()

    monthly_stats = (
        trades_df
        .groupby('Month')
        .agg(
        Trades=('PnL Points', 'count'),
        Wins=('PnL Points', lambda x: (x > 0).sum()),
        Losses=('PnL Points', lambda x: (x < 0).sum()),
        WinRatePct=('PnL Points', lambda x: (x > 0).mean() * 100.0),
        NetPoints=('PnL Points', 'sum'),
        AvgReturnPct=('Return %', 'mean'),
        CumReturnPct=('Return %', 'sum')
    )
    .reset_index()
)


    summary = {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate": win_rate,
        "net_points": net_points,
        "avg_return": avg_return,
        "cum_return": cum_return,
        "monthly_stats": monthly_stats
    }

    return trades_df, summary

# ======================= STREAMLIT UI ==========================

st.set_page_config(page_title="Nikhil's Fyers Backtester", layout="wide")
st.title("📈 Nikhil's Fyers MA/EMA Crossover Backtester")

st.sidebar.header("Data & Strategy Settings")

# ---------- Data selection (same base as your original app) ----------
segment = st.sidebar.selectbox("Segment", [
    "Index", "Equity", "Equity Futures", "Equity Options (Monthly Expiry)",
    "Equity Options (Weekly Expiry)", "Currency Futures", "Currency Options (Monthly Expiry)",
    "Currency Options (Weekly Expiry)", "Commodity Futures", "Commodity Options (Monthly Expiry)"
])

exchange = st.sidebar.selectbox("Exchange", ["NSE", "BSE", "MCX", "CDS", "NFO"])
symbol = st.sidebar.text_input("Symbol", value="NIFTY")

year, month, day, strike, opt_type = None, None, None, None, None
if segment not in ["Index", "Equity"]:
    year = st.sidebar.selectbox("Year", list(range(2017, dt.date.today().year + 2)), index=5)
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
    st.sidebar.markdown("**Note:** No expiry/strike required for selected segment.")

min_date = dt.date(2017, 7, 3)
max_date = dt.date.today()
start_date = st.sidebar.date_input("Start Date", value=max_date - dt.timedelta(days=30),
                                   min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date,
                                 min_value=min_date, max_value=max_date)
resolution = st.sidebar.selectbox("Resolution (Timeframe)", options=[
    "1", "2", "3", "5", "10", "15", "20", "30", "60", "120", "240", "D"
], index=9)

# ---------- Strategy settings ----------
st.sidebar.markdown("---")
st.sidebar.subheader("Strategy: MA/EMA Crossover")

ma_type = st.sidebar.selectbox("MA Type", ["SMA", "EMA"])

# Dropdowns for fast & slow period
period_options = [5, 8, 9, 10, 13, 20, 21, 34, 50, 100, 200]
fast_period = st.sidebar.selectbox("Fast MA Period", period_options, index=2)
slow_period = st.sidebar.selectbox("Slow MA Period", period_options, index=6)

if slow_period <= fast_period:
    st.sidebar.warning("Slow period should be greater than fast period for meaningful crossover.")

st.sidebar.markdown("---")
st.sidebar.subheader("Target & Stop Loss")

tp_type = st.sidebar.selectbox("Target Type", ["Points", "Percent"], index=0)
tp_value = st.sidebar.number_input(f"Target ({tp_type})", min_value=0.0, value=100.0, step=1.0)

sl_type = st.sidebar.selectbox("Stop Loss Type", ["Points", "Percent"], index=0)
sl_value = st.sidebar.number_input(f"Stop Loss ({sl_type})", min_value=0.0, value=50.0, step=1.0)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Backtest")

# ========================= MAIN LAYOUT =========================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Historical Data & Signals")

with col2:
    st.subheader("Backtest Summary")

if start_date > end_date:
    st.error("⚠️ Start date must not be after end date.")
else:
    fyers_symbol = build_fyers_symbol(segment, exchange, symbol, year, month, day, strike, opt_type)

    if run_button:
        with st.spinner(f"Fetching data and running backtest for {fyers_symbol}..."):
            data = fetch_data(fyers_symbol, start_date, end_date, resolution)

            if data.empty:
                st.error("No data received. Check your inputs or try another range.")
            else:
                # Only use OHLC for the strategy
                price_df = data[['Open', 'High', 'Low', 'Close']].copy()

                trades_df, summary = backtest_ma_crossover(
                    price_df,
                    ma_type=ma_type,
                    fast_period=fast_period,
                    slow_period=slow_period,
                    sl_type=sl_type,
                    sl_value=sl_value,
                    tp_type=tp_type,
                    tp_value=tp_value,
                )

                # ----- Left: show price + MA preview -----
                with col1:
                    st.success("✅ Data fetched & backtest completed.")
                    preview_df = add_moving_averages(price_df, ma_type, fast_period, slow_period)
                    st.write("Last 50 candles with MAs:")
                    st.dataframe(preview_df[['Open', 'High', 'Low', 'Close', 'fast_ma', 'slow_ma']].tail(50))

                    st.line_chart(preview_df[['Close', 'fast_ma', 'slow_ma']].tail(500))

                # ----- Right: summary metrics -----
                with col2:
                    if trades_df.empty:
                        st.warning("No trades generated with the selected parameters.")
                    else:
                        st.metric("Total Trades", summary["total_trades"])
                        st.metric("Profitable Trades", summary["wins"])
                        st.metric("Loss-making Trades", summary["losses"])
                        st.metric("Win Rate (%)", f"{summary['win_rate']:.2f}")
                        st.metric("Net PnL (Points)", f"{summary['net_points']:.2f}")
                        st.metric("Avg Return per Trade (%)", f"{summary['avg_return']:.2f}")
                        st.metric("Cumulative Return (%)", f"{summary['cum_return']:.2f}")

                # ----- Full details below -----
                st.markdown("---")
                st.subheader("📊 Trades Details")

                if not trades_df.empty:
                    st.dataframe(trades_df)

                    st.download_button(
                        "⬇️ Download Trades CSV",
                        data=trades_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{symbol}_MA_Backtest_Trades.csv",
                        mime="text/csv"
                    )

                    st.markdown("### 📅 Monthly Statistics")
                    monthly_stats = summary.get("monthly_stats")
                    if monthly_stats is not None and not monthly_stats.empty:
                        st.dataframe(monthly_stats)
                        st.bar_chart(
                            monthly_stats.set_index("Month")[["NetPoints"]]
                        )
                    else:
                        st.info("No monthly stats (no trades).")
    else:
        st.info("Set your parameters on the left and click **Run Backtest** to start.")



