import streamlit as st
from fyers_apiv3 import fyersModel
import pandas as pd
import numpy as np
import datetime as dt
import altair as alt

# ========================= FYERS SETUP =========================
# Make sure 'access.txt' exists and contains your access token
with open('access.txt', 'r') as a:
    access_token = a.read().strip()

client_id = 'KB4YGO9V7J-100'   # change if needed
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")

# ==================== SYMBOL BUILD FUNCTION ====================
def build_fyers_symbol(segment, exchange, symbol, year=None, month=None, day=None, strike=None, opt_type=None):
    """
    Builds Fyers symbol string based on segment / exchange / expiry / strike etc.
    """
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
    """
    Fetch data from Fyers API in chunks and return a OHLCV(+OI) DataFrame indexed by Date.
    Date is converted to IST and timezone removed.
    """
    df = pd.DataFrame()

    resolution_minutes = {
        "1": 1, "2": 2, "3": 3, "5": 5, "10": 10, "15": 15,
        "20": 20, "30": 30, "60": 60, "120": 120, "240": 240
    }

    # Decide chunk size
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
                first_candle_len = len(response['candles'][0])
                if first_candle_len == 7:
                    chunk = pd.DataFrame(response['candles'],
                                         columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
                elif first_candle_len == 6:
                    chunk = pd.DataFrame(response['candles'],
                                         columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    chunk['OI'] = None
                else:
                    st.error(f"Unexpected candle format from {current_start} to {current_end}")
                    current_start = current_end + dt.timedelta(days=1)
                    continue

                # Convert timestamp to IST and drop tz
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
    """
    Adds fast_ma and slow_ma columns (SMA or EMA) and generates entry/exit 'signal'.
    signal = 1 => bullish crossover, signal = -1 => bearish crossover.
    """
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
    ] = -1  # Bearish (used for exit logic if needed)

    return df

def calc_level(entry_price, level_type, level_value, direction="long"):
    """
    Calculates TP & SL price based on points or percent.
    """
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
    trade_mode="Intraday",  # <-- NEW
):
    """
    Long-only MA/EMA crossover backtest.

    trade_mode:
        - "Intraday": exit by 14:55 each day, no overnight.
        - "Positional": keep across days, exit only at TP/SL
                        (or final bar of backtest).
    """
    if df.empty:
        return pd.DataFrame(), {}

    df = df.copy()
    df = df.sort_index()

    # Intraday filter only if intraday resolution (both modes)
    intraday_resolutions = {"1", "2", "3", "5", "10", "15", "20", "30", "60", "120", "240"}
    if resolution in intraday_resolutions:
        df = df.between_time("09:15", "15:30")

    if df.empty:
        return pd.DataFrame(), {}

    # Indicators and signals
    df = add_moving_averages(df, ma_type, fast_period, slow_period)

    trades = []
    in_trade = False
    direction = "long"
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

        # ---------- FORCED EXIT ON DAY CHANGE ----------
        # Only in INTRADAY mode (no overnight)
        if in_trade and trade_mode == "Intraday" and ts.date() != trade_date:
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
                "Exit Reason": "EOD (day change)",
                "Entry Date": entry_time.date()
            })
            in_trade = False
            entry_price = entry_time = tp_price = sl_price = trade_date = None

        # ---------- MANAGE OPEN TRADE: SL / TP ----------
        if in_trade:
            bar_high = row['High']
            bar_low = row['Low']

            if direction == "long":
                hit_tp = bar_high >= tp_price
                hit_sl = bar_low <= sl_price

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
                    continue  # move to next candle

        # ---------- INTRADAY EOD EXIT AT 14:55 ----------
        if in_trade and trade_mode == "Intraday":
            if ts.time() >= intraday_exit_time and ts.date() == trade_date:
                exit_price = row['Close']
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
                    "Exit Reason": "Intraday EOD 14:55",
                    "Entry Date": entry_time.date()
                })
                in_trade = False
                entry_price = entry_time = tp_price = sl_price = trade_date = None
                continue

        # ---------- NEW ENTRY ----------
        if (not in_trade) and (signal_prev == 1):
            entry_price = row['Open']
            entry_time = ts
            trade_date = ts.date()
            direction = "long"
            tp_price, sl_price = calc_level(entry_price, tp_type, tp_value, direction)
            in_trade = True

    # ---------- FINAL OPEN TRADE EXIT (end of data) ----------
    if in_trade:
        last_row = df.iloc[-1]
        last_ts = df.index[-1]
        exit_price = last_row['Close']
        exit_time = last_ts
        pnl_points = exit_price - entry_price
        reason = "EOD (last bar)" if trade_mode == "Intraday" else "Backtest end"
        trades.append({
            "Entry Time": entry_time,
            "Exit Time": exit_time,
            "Direction": direction,
            "Entry Price": entry_price,
            "Exit Price": exit_price,
            "PnL Points": pnl_points,
            "Return %": (pnl_points / entry_price) * 100.0,
            "Exit Reason": reason,
            "Entry Date": entry_time.date()
        })

    if len(trades) == 0:
        return pd.DataFrame(), {}

    trades_df = pd.DataFrame(trades)
    trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'])
    trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'])

    # ---------- Summary stats ----------
    total_trades = len(trades_df)
    wins = (trades_df['PnL Points'] > 0).sum()
    losses = (trades_df['PnL Points'] < 0).sum()
    breakeven = (trades_df['PnL Points'] == 0).sum()
    win_rate = (wins / total_trades) * 100.0 if total_trades > 0 else 0.0
    net_points = trades_df['PnL Points'].sum()
    avg_return = trades_df['Return %'].mean()
    cum_return = trades_df['Return %'].sum()

    # ---------- Monthly stats ----------
    trades_df['Month'] = trades_df['Entry Time'].dt.to_period('M').astype(str)
    monthly_stats = (
        trades_df
        .groupby('Month', as_index=False)
        .agg(
            Trades=('PnL Points', 'count'),
            Wins=('PnL Points', lambda x: (x > 0).sum()),
            Losses=('PnL Points', lambda x: (x < 0).sum()),
            WinRatePct=('PnL Points', lambda x: (x > 0).mean() * 100.0),
            NetPoints=('PnL Points', 'sum'),
            AvgReturnPct=('Return %', 'mean'),
            CumReturnPct=('Return %', 'sum')
        )
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

# ===================== ANALYTICS HELPERS =======================

def build_daily_stats(trades_df):
    """Aggregate to daily PnL and build equity & drawdown."""
    daily = (
        trades_df
        .groupby('Entry Date', as_index=False)
        .agg(PnLPoints=('PnL Points', 'sum'),
             Trades=('PnL Points', 'count'))
        .rename(columns={'Entry Date': 'Date'})
    )
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.sort_values('Date').reset_index(drop=True)

    daily['CumPnL'] = daily['PnLPoints'].cumsum()
    daily['Peak'] = daily['CumPnL'].cummax()
    daily['Drawdown'] = daily['CumPnL'] - daily['Peak']

    return daily

def compute_risk_stats(daily):
    """Basic risk stats like best/worst day and max DD."""
    if daily.empty:
        return {}

    best_day = daily.loc[daily['PnLPoints'].idxmax()]
    worst_day = daily.loc[daily['PnLPoints'].idxmin()]
    max_dd = daily['Drawdown'].min()  # negative
    avg_daily = daily['PnLPoints'].mean()

    return {
        "best_day_date": best_day['Date'].date(),
        "best_day_pnl": best_day['PnLPoints'],
        "worst_day_date": worst_day['Date'].date(),
        "worst_day_pnl": worst_day['PnLPoints'],
        "max_drawdown": max_dd,
        "avg_daily_pnl": avg_daily,
    }

def build_calendar_df(daily):
    """Prepare data for a calendar-like heatmap using Altair."""
    cal = daily.copy()
    cal['Year'] = cal['Date'].dt.year
    cal['Month'] = cal['Date'].dt.month_name().str[:3]
    cal['Day'] = cal['Date'].dt.day
    return cal

# ======================= STREAMLIT UI ==========================

st.set_page_config(page_title="Nikhil's Fyers Backtester", layout="wide")
st.title("📈 Nikhil's Fyers MA/EMA Crossover Backtester")

st.sidebar.header("Data & Strategy Settings")

# ---------- Segment / Symbol selection ----------
segment = st.sidebar.selectbox("Segment", [
    "Index", "Equity", "Equity Futures", "Equity Options (Monthly Expiry)",
    "Equity Options (Weekly Expiry)", "Currency Futures", "Currency Options (Monthly Expiry)",
    "Currency Options (Weekly Expiry)", "Commodity Futures", "Commodity Options (Monthly Expiry)"
])

exchange = st.sidebar.selectbox("Exchange", ["NSE", "BSE", "MCX", "CDS", "NFO"])
symbol = st.sidebar.text_input("Symbol", value="NIFTY")

year, month, day, strike, opt_type = None, None, None, None, None

if segment not in ["Index", "Equity"]:
    years = list(range(2017, dt.date.today().year + 2))
    year = st.sidebar.selectbox("Year", years, index=len(years)-1)

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

# ---------- Date range & resolution ----------
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
    index=9  # default "60"
)

# ---------- Strategy settings ----------
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

# ---------- NEW: Trade Mode ----------
st.sidebar.markdown("---")
trade_mode = st.sidebar.radio("Trade Mode", ["Intraday", "Positional"], index=0)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Backtest")

# ========================= MAIN LAYOUT =========================

# Top info area
if start_date > end_date:
    st.error("⚠️ Start date must not be after end date.")
else:
    fyers_symbol = build_fyers_symbol(segment, exchange, symbol, year, month, day, strike, opt_type)
    st.caption(
        f"Symbol: **{fyers_symbol}**  |  Date Range: {start_date} → {end_date}  |  TF: {resolution}  |  Mode: **{trade_mode}**"
    )

    if run_button:
        with st.spinner(f"Fetching data & running backtest for {fyers_symbol}..."):
            data = fetch_data(fyers_symbol, start_date, end_date, resolution)

            if data.empty:
                st.error("No data received. Try different dates / symbol / resolution.")
            else:
                price_df = data[['Open', 'High', 'Low', 'Close']].copy()

                trades_df, summary = backtest_ma_crossover(
                    price_df,
                    resolution=resolution,
                    ma_type=ma_type,
                    fast_period=fast_period,
                    slow_period=slow_period,
                    sl_type=sl_type,
                    sl_value=sl_value,
                    tp_type=tp_type,
                    tp_value=tp_value,
                    trade_mode=trade_mode,   # <-- pass mode
                )

                if trades_df.empty:
                    st.warning("No trades generated with the selected parameters.")
                else:
                    # ======= KPI ROWS =======
                    st.markdown("## 📊 Overview")

                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Total Trades", summary["total_trades"])
                    kpi2.metric("Profitable Trades", summary["wins"])
                    kpi3.metric("Loss-making Trades", summary["losses"])
                    kpi4.metric("Win Rate (%)", f"{summary['win_rate']:.2f}")

                    kpi5, kpi6, kpi7, kpi8 = st.columns(4)
                    kpi5.metric("Net PnL (Points)", f"{summary['net_points']:.2f}")
                    kpi6.metric("Avg Return / Trade (%)", f"{summary['avg_return']:.2f}")
                    kpi7.metric("Cumulative Return (%)", f"{summary['cum_return']:.2f}")
                    kpi8.metric("Breakeven Trades", summary["breakeven"])

                    # Daily stats & risk
                    daily_stats = build_daily_stats(trades_df)
                    risk = compute_risk_stats(daily_stats)

                    kpi9, kpi10, kpi11, kpi12 = st.columns(4)
                    if risk:
                        kpi9.metric("Best Day PnL", f"{risk['best_day_pnl']:.2f}", str(risk['best_day_date']))
                        kpi10.metric("Worst Day PnL", f"{risk['worst_day_pnl']:.2f}", str(risk['worst_day_date']))
                        kpi11.metric("Max Drawdown (Points)", f"{risk['max_drawdown']:.2f}")
                        kpi12.metric("Avg Daily PnL (Points)", f"{risk['avg_daily_pnl']:.2f}")
                    else:
                        kpi9.metric("Best Day PnL", "–")
                        kpi10.metric("Worst Day PnL", "–")
                        kpi11.metric("Max Drawdown", "–")
                        kpi12.metric("Avg Daily PnL", "–")

                    st.markdown("---")

                    # ======= Calendar-style daily heatmap =======
                    st.markdown("### 📅 Daily PnL Calendar Heatmap")

                    cal_df = build_calendar_df(daily_stats)
                    if not cal_df.empty:
                        cal_chart = (
                            alt.Chart(cal_df)
                            .mark_rect()
                            .encode(
                                x=alt.X('day(Date):O', title='Day of Month'),
                                y=alt.Y('month(Date):O', title='Month'),
                                color=alt.Color(
                                    'PnLPoints:Q',
                                    title='PnL (Pts)',
                                    scale=alt.Scale(scheme='redblue', domainMid=0)
                                ),
                                tooltip=['Date:T', 'PnLPoints:Q', 'Trades:Q']
                            )
                            .properties(height=250)
                        )
                        st.altair_chart(cal_chart, use_container_width=True)
                    else:
                        st.info("No daily stats available.")

                    st.markdown("---")

                    # ======= Price + MA preview =======
                    st.markdown("### 📈 Price with Moving Averages")
                    ma_df = add_moving_averages(price_df, ma_type, fast_period, slow_period)
                    st.dataframe(ma_df[['Open', 'High', 'Low', 'Close', 'fast_ma', 'slow_ma']].tail(50))
                    st.line_chart(ma_df[['Close', 'fast_ma', 'slow_ma']].tail(500))

                    st.markdown("---")

                    # ======= Performance Charts =======
                    st.markdown("## 📉 Performance Charts")

                    # Equity & Drawdown
                    col_e1, col_e2 = st.columns(2)

                    with col_e1:
                        st.markdown("#### Equity Curve (Cumulative PnL)")
                        eq_chart = (
                            alt.Chart(daily_stats)
                            .mark_line()
                            .encode(
                                x='Date:T',
                                y=alt.Y('CumPnL:Q', title='Cumulative PnL (Points)'),
                                tooltip=['Date:T', 'CumPnL:Q']
                            )
                        )
                        st.altair_chart(eq_chart, use_container_width=True)

                    with col_e2:
                        st.markdown("#### Drawdown Curve")
                        dd_chart = (
                            alt.Chart(daily_stats)
                            .mark_line()
                            .encode(
                                x='Date:T',
                                y=alt.Y('Drawdown:Q', title='Drawdown (Points)'),
                                tooltip=['Date:T', 'Drawdown:Q']
                            )
                        )
                        st.altair_chart(dd_chart, use_container_width=True)

                    # Daily PnL & Daily Trades
                    col_d1, col_d2 = st.columns(2)

                    with col_d1:
                        st.markdown("#### Daily PnL (Points)")
                        daily_pnl_chart = (
                            alt.Chart(daily_stats)
                            .mark_bar()
                            .encode(
                                x='Date:T',
                                y=alt.Y('PnLPoints:Q', title='PnL (Points)'),
                                tooltip=['Date:T', 'PnLPoints:Q', 'Trades:Q'],
                                color=alt.condition(
                                    alt.datum.PnLPoints >= 0,
                                    alt.value("green"),
                                    alt.value("red")
                                )
                            )
                        )
                        st.altair_chart(daily_pnl_chart, use_container_width=True)

                    with col_d2:
                        st.markdown("#### Daily Number of Trades")
                        daily_trades_chart = (
                            alt.Chart(daily_stats)
                            .mark_bar()
                            .encode(
                                x='Date:T',
                                y=alt.Y('Trades:Q', title='Number of Trades'),
                                tooltip=['Date:T', 'Trades:Q']
                            )
                        )
                        st.altair_chart(daily_trades_chart, use_container_width=True)

                    # Monthly PnL & Monthly WinRate
                    st.markdown("### 📆 Monthly Statistics")

                    monthly_stats = summary["monthly_stats"]
                    if monthly_stats is not None and not monthly_stats.empty:
                        col_m1, col_m2 = st.columns(2)

                        with col_m1:
                            st.markdown("#### Monthly Net PnL (Points)")
                            ms_pnl_chart = (
                                alt.Chart(monthly_stats)
                                .mark_bar()
                                .encode(
                                    x=alt.X('Month:O', sort=None),
                                    y=alt.Y('NetPoints:Q', title='Net PnL (Points)'),
                                    tooltip=['Month', 'NetPoints', 'Trades'],
                                    color=alt.condition(
                                        alt.datum.NetPoints >= 0,
                                        alt.value("green"),
                                        alt.value("red")
                                    )
                                )
                            )
                            st.altair_chart(ms_pnl_chart, use_container_width=True)

                        with col_m2:
                            st.markdown("#### Monthly Win Rate (%)")
                            ms_wr_chart = (
                                alt.Chart(monthly_stats)
                                .mark_bar()
                                .encode(
                                    x=alt.X('Month:O', sort=None),
                                    y=alt.Y('WinRatePct:Q', title='Win Rate (%)'),
                                    tooltip=['Month', 'WinRatePct', 'Trades']
                                )
                            )
                            st.altair_chart(ms_wr_chart, use_container_width=True)

                        st.dataframe(monthly_stats)
                    else:
                        st.info("No monthly stats available.")

                    st.markdown("---")

                    # ======= Distribution Pie (Win / Loss / BE) =======
                    st.markdown("### 🥧 Trade Outcome Distribution")

                    outcome_counts = {
                        "Wins": summary["wins"],
                        "Losses": summary["losses"],
                        "Breakeven": summary["breakeven"],
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
                            color='Outcome:N',
                            tooltip=['Outcome', 'Count']
                        )
                    )
                    st.altair_chart(pie_chart, use_container_width=False)

                    st.markdown("---")

                    # ======= Trades Table & Download =======
                    st.markdown("## 📋 Trades Table")
                    st.dataframe(trades_df)

                    st.download_button(
                        "⬇️ Download Trades CSV",
                        data=trades_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{symbol}_MA_Backtest_Trades.csv",
                        mime="text/csv"
                    )

    else:
        st.info("Set your parameters on the left and click **Run Backtest** to start.")
