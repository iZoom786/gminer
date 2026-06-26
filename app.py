import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================================================
# DATA CLASSES & BACKEND LOGIC (Your exact Python logic from previous steps)
# ============================================================================

@dataclass
class Signal:
    timestamp: datetime
    direction: str
    price: float
    score: float
    conditions_met: int
    stop_loss: float
    take_profit: float
    atr: float
    rsi: float
    macd: float
    bb_percent_b: float
    vol_ratio: float
    trend: str
    support: float
    resistance: float

class TechnicalIndicators:
    @staticmethod
    def ema(series, period): return series.ewm(span=period, adjust=False).mean()
    @staticmethod
    def sma(series, period): return series.rolling(window=period).mean()
    @staticmethod
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line, macd_line - signal_line
    @staticmethod
    def bollinger_bands(series, period=20, mult=2.0):
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (mult * std)
        lower = middle - (mult * std)
        percent_b = (series - lower) / (upper - lower)
        return upper, middle, lower, percent_b
    @staticmethod
    def atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

class XAUUSDSignalGenerator:
    def __init__(self, **kwargs):
        self.lookback_days = kwargs.get('lookback_days', 10)
        self.bars_per_day = kwargs.get('bars_per_day', 1440)
        self.lookback_bars = self.lookback_days * self.bars_per_day
        self.ema_fast_len = kwargs.get('ema_fast_len', 9)
        self.ema_mid_len = kwargs.get('ema_mid_len', 21)
        self.ema_slow_len = kwargs.get('ema_slow_len', 50)
        self.sma_len = kwargs.get('sma_len', 20)
        self.rsi_len = kwargs.get('rsi_len', 14)
        self.rsi_overbought = kwargs.get('rsi_overbought', 70)
        self.rsi_oversold = kwargs.get('rsi_oversold', 30)
        self.bb_len = kwargs.get('bb_len', 20)
        self.bb_mult = kwargs.get('bb_mult', 2.0)
        self.atr_len = kwargs.get('atr_len', 14)
        self.atr_mult_sl = kwargs.get('atr_mult_sl', 1.5)
        self.atr_mult_tp = kwargs.get('atr_mult_tp', 3.0)
        self.vol_len = kwargs.get('vol_len', 20)
        self.vol_mult = kwargs.get('vol_mult', 1.5)
        
        w_t = kwargs.get('weight_trend', 25)
        w_m = kwargs.get('weight_momentum', 25)
        w_v = kwargs.get('weight_volatility', 25)
        w_vol = kwargs.get('weight_volume', 25)
        total = w_t + w_m + w_v + w_vol
        self.w_trend = w_t / total * 100
        self.w_momentum = w_m / total * 100
        self.w_volatility = w_v / total * 100
        self.w_volume = w_vol / total * 100
        
        self.indicators = TechnicalIndicators()
        self.df = None
        self.signals = []

    def load_csv(self, file_buffer, sep=';'):
        df = pd.read_csv(file_buffer, sep=sep)
        df.columns = df.columns.str.strip()
        col_map = {col: col.lower() for col in df.columns if col.lower() in ['date', 'time', 'datetime', 'timestamp']}
        df = df.rename(columns=col_map)
        date_col = 'date' if 'date' in df.columns else df.columns[0]
        df['timestamp'] = pd.to_datetime(df[date_col])
        for c in ['open', 'high', 'low', 'close', 'volume']:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        self.df = df
        return df

    def calculate_all_indicators(self):
        df = self.df
        df['ema_fast'] = self.indicators.ema(df['close'], self.ema_fast_len)
        df['ema_mid'] = self.indicators.ema(df['close'], self.ema_mid_len)
        df['ema_slow'] = self.indicators.ema(df['close'], self.ema_slow_len)
        df['rsi'] = self.indicators.rsi(df['close'], self.rsi_len)
        df['macd_line'], df['signal_line'], df['histogram'] = self.indicators.macd(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_percent_b'] = self.indicators.bollinger_bands(df['close'], self.bb_len, self.bb_mult)
        df['atr'] = self.indicators.atr(df['high'], df['low'], df['close'], self.atr_len)
        df['vol_sma'] = self.indicators.sma(df['volume'], self.vol_len)
        df['vol_ratio'] = df['volume'] / df['vol_sma']
        df['lookback_high'] = df['high'].rolling(window=self.lookback_bars).max()
        df['lookback_low'] = df['low'].rolling(window=self.lookback_bars).min()
        df['support_1'] = df['lookback_low'].ffill()
        df['resistance_1'] = df['lookback_high'].ffill()
        self.df = df

    def calculate_score(self, idx):
        df = self.df
        trend_score = 0.0
        ema_bull = df['ema_fast'].iloc[idx] > df['ema_mid'].iloc[idx] > df['ema_slow'].iloc[idx]
        ema_bear = df['ema_fast'].iloc[idx] < df['ema_mid'].iloc[idx] < df['ema_slow'].iloc[idx]
        trend_score += 33 if ema_bull else (-33 if ema_bear else 0)
        trend_score += 33 if df['close'].iloc[idx] > df['ema_slow'].iloc[idx] else -33
        
        rsi_val = df['rsi'].iloc[idx] if not np.isnan(df['rsi'].iloc[idx]) else 50
        hist_val = df['histogram'].iloc[idx] if not np.isnan(df['histogram'].iloc[idx]) else 0
        momentum_score = (25 if rsi_val > 50 else -25) + (25 if hist_val > 0 else -25)
        
        bb_pb = df['bb_percent_b'].iloc[idx] if not np.isnan(df['bb_percent_b'].iloc[idx]) else 0.5
        volatility_score = 35 if bb_pb < 0.2 else (-35 if bb_pb > 0.8 else 0)
        
        vol_r = df['vol_ratio'].iloc[idx] if not np.isnan(df['vol_ratio'].iloc[idx]) else 1
        volume_score = 35 if (vol_r > 1 and df['close'].iloc[idx] > df['open'].iloc[idx]) else (-35 if (vol_r > 1 and df['close'].iloc[idx] < df['open'].iloc[idx]) else 0)

        return (trend_score * self.w_trend/100 + momentum_score * self.w_momentum/100 + 
                volatility_score * self.w_volatility/100 + volume_score * self.w_volume/100)

    def generate_signals(self, start_idx, end_idx):
        df = self.df
        signals = []
        for idx in range(start_idx, end_idx):
            if np.isnan(df['ema_slow'].iloc[idx]): continue
            score = self.calculate_score(idx)
            
            buy_conds = sum([score > 40, df['rsi'].iloc[idx] < self.rsi_overbought, df['close'].iloc[idx] > df['support_1'].iloc[idx], df['close'].iloc[idx] > df['bb_lower'].iloc[idx]])
            sell_conds = sum([score < -40, df['rsi'].iloc[idx] > self.rsi_oversold, df['close'].iloc[idx] < df['resistance_1'].iloc[idx], df['close'].iloc[idx] < df['bb_upper'].iloc[idx]])
            
            if buy_conds >= 3 and score > 30:
                atr_val = df['atr'].iloc[idx] if not np.isnan(df['atr'].iloc[idx]) else 5
                signals.append(Signal(df['timestamp'].iloc[idx], 'BUY', df['close'].iloc[idx], score, buy_conds, df['close'].iloc[idx] - (atr_val*self.atr_mult_sl), df['close'].iloc[idx] + (atr_val*self.atr_mult_tp), atr_val, df['rsi'].iloc[idx], df['macd_line'].iloc[idx], df['bb_percent_b'].iloc[idx], df['vol_ratio'].iloc[idx], 'BULL', df['support_1'].iloc[idx], df['resistance_1'].iloc[idx]))
            elif sell_conds >= 3 and score < -30:
                atr_val = df['atr'].iloc[idx] if not np.isnan(df['atr'].iloc[idx]) else 5
                signals.append(Signal(df['timestamp'].iloc[idx], 'SELL', df['close'].iloc[idx], score, sell_conds, df['close'].iloc[idx] + (atr_val*self.atr_mult_sl), df['close'].iloc[idx] - (atr_val*self.atr_mult_tp), atr_val, df['rsi'].iloc[idx], df['macd_line'].iloc[idx], df['bb_percent_b'].iloc[idx], df['vol_ratio'].iloc[idx], 'BEAR', df['support_1'].iloc[idx], df['resistance_1'].iloc[idx]))
        self.signals = signals
        return signals

    def predict_next_n_days(self, n_days=3):
        if not self.df.empty:
            last_idx = len(self.df) - 1
            curr_price = self.df['close'].iloc[-1]
            curr_atr = self.df['atr'].iloc[-1] if not np.isnan(self.df['atr'].iloc[-1]) else 5
            bars_fwd = n_days * self.bars_per_day
            exp_move = curr_atr * np.sqrt(bars_fwd) # Using square root for realistic volatility expansion
            fast_slope = (self.df['ema_fast'].iloc[-1] - self.df['ema_fast'].iloc[-10]) / 10
            proj_price = curr_price + (fast_slope * bars_fwd)
            
            return {
                'current': curr_price, 'atr': curr_atr,
                'proj_high': proj_price + exp_move, 'proj_low': proj_price - exp_move,
                'trend': 'BULLISH' if fast_slope > 0 else 'BEARISH'
            }
        return None

# ============================================================================
# KRONOS WEB APPLICATION UI (Streamlit)
# ============================================================================

st.set_page_config(page_title="Kronos XAUUSD", page_icon="⏳", layout="wide")

# Custom CSS for Dark Theme (Similar to TradingView/Kronos style)
st.markdown("""
<style>
    .main { background-color: #0e1117; color: white; }
    .sidebar { background-color: #1a1a2e; }
    div.stButton > button:first-child { background-color: #e6b422; color: black; font-weight: bold; }
    div.stButton > button:hover { background-color: #ffd700; }
    section[data-testid="stSidebar"] { background-color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

st.title("⏳ Kronos XAUUSD Trading System")
st.markdown("Advanced 10-Day Signal Generator & Predictor")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    st.subheader("Data Format")
    timeframe_map = {'1 Minute': 1440, '5 Minutes': 288, '15 Minutes': 96, '1 Hour': 24, '1 Day': 1}
    selected_tf = st.selectbox("Timeframe", options=timeframe_map.keys(), index=0)
    bars_per_day = timeframe_map[selected_tf]
    csv_sep = st.selectbox("CSV Separator", options=[';', ','], index=0)
    
    st.subheader("Core Logic")
    lookback_days = st.number_input("Lookback Days", min_value=1, max_value=50, value=10)
    
    st.subheader("Moving Averages")
    col1, col2 = st.columns(2)
    ema_fast = col1.number_input("Fast EMA", value=9)
    ema_mid = col2.number_input("Mid EMA", value=21)
    ema_slow = st.number_input("Slow EMA", value=50)
    
    st.subheader("RSI & MACD")
    rsi_ob = st.slider("RSI Overbought", 50, 100, value=70)
    rsi_os = st.slider("RSI Oversold", 0, 50, value=30)
    
    st.subheader("Risk Management (ATR)")
    atr_sl = st.number_input("Stop Loss Multiplier", min_value=0.5, value=1.5, step=0.1)
    atr_tp = st.number_input("Take Profit Multiplier", min_value=1.0, value=3.0, step=0.1)
    
    st.subheader("Prediction")
    pred_days = st.number_input("Days to Predict", min_value=1, max_value=30, value=3)

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload XAUUSD CSV", type=['csv'])
    run_btn = st.button("🚀 Run Analysis")

# --- MAIN LOGIC ---
if uploaded_file and run_btn:
    try:
        with st.spinner("Processing data & calculating indicators..."):
            # Initialize Generator with sidebar inputs
            gen = XAUUSDSignalGenerator(
                lookback_days=lookback_days, bars_per_day=bars_per_day,
                ema_fast_len=ema_fast, ema_mid_len=ema_mid, ema_slow_len=ema_slow,
                rsi_overbought=rsi_ob, rsi_oversold=rsi_os,
                atr_mult_sl=atr_sl, atr_mult_tp=atr_tp
            )
            
            # Load Data
            gen.load_csv(uploaded_file, sep=csv_sep)
            
            # Calculate
            gen.calculate_all_indicators()
            
            # Run Testing Mode (Last 7 days equivalent in bars)
            test_bars = 7 * bars_per_day
            start_idx = max(0, len(gen.df) - test_bars)
            signals = gen.generate_signals(start_idx, len(gen.df))
            
            # Predictions
            prediction = gen.predict_next_n_days(n_days=pred_days)
            
        st.success("Analysis Complete!")
        
        # --- DISPLAY PREDICTIONS METRICS ---
        if prediction:
            st.header("🔮 Future Projection")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"${prediction['current']:.2f}")
            col2.metric("Projected High", f"${prediction['proj_high']:.2f}", delta=f"+{prediction['proj_high'] - prediction['current']:.2f}")
            col3.metric("Projected Low", f"${prediction['proj_low']:.2f}", delta=f"{prediction['proj_low'] - prediction['current']:.2f}", delta_color="inverse")
            col4.metric("Trend Bias", prediction['trend'])
            
        # --- PLOT INTERACTIVE CHART ---
        st.header("📊 10-Day Chart & Signals")
        # We only plot the testing window so the browser doesn't crash on 1m data
        chart_df = gen.df.iloc[start_idx:].copy()
        
        fig = make_subplots(rows=1, cols=1)
        
        # Candlestick
        fig.add_trace(go.Candlestick(x=chart_df['timestamp'], open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name="Price"), row=1, col=1)
        
        # Indicators
        fig.add_trace(go.Scatter(x=chart_df['timestamp'], y=chart_df['ema_fast'], name='Fast EMA', line=dict(color='yellow', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df['timestamp'], y=chart_df['ema_slow'], name='Slow EMA', line=dict(color='red', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df['timestamp'], y=chart_df['bb_upper'], name='BB Upper', line=dict(color='blue', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df['timestamp'], y=chart_df['bb_lower'], name='BB Lower', line=dict(color='blue', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(0,0,255,0.05)'), row=1, col=1)
        
        # Plot Signals
        if signals:
            buy_signals = [s for s in signals if s.direction == 'BUY']
            sell_signals = [s for s in signals if s.direction == 'SELL']
            
            fig.add_trace(go.Scatter(x=[s.timestamp for s in buy_signals], y=[s.price for s in buy_signals], mode='markers', name='BUY Signal', marker=dict(color='lime', size=12, symbol='triangle-up')), row=1, col=1)
            fig.add_trace(go.Scatter(x=[s.timestamp for s in sell_signals], y=[s.price for s in sell_signals], mode='markers', name='SELL Signal', marker=dict(color='red', size=12, symbol='triangle-down')), row=1, col=1)

        fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # --- SIGNALS TABLE ---
        st.header("📋 Signal Log")
        if signals:
            df_signals = pd.DataFrame([{
                'Time': s.timestamp, 'Direction': s.direction, 'Entry': s.price,
                'Stop Loss': s.stop_loss, 'Take Profit': s.take_profit, 'Score': s.score
            } for s in signals])
            st.dataframe(df_signals.sort_values(by='Time', ascending=False), use_container_width=True)
        else:
            st.info("No strict signals triggered in this period. Try adjusting sidebar parameters.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

elif not uploaded_file:
    st.info("Please upload your XAUUSD CSV file in the sidebar to begin.")