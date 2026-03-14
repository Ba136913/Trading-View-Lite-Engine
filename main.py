from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import google.generativeai as genai
from cachetools import TTLCache
import uvicorn
import os
import math

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
cache = TTLCache(maxsize=500, ttl=300)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# 🔥 Top 50 Liquid F&O (Safe for Yahoo Finance, No Bans)
TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "ASIANPAINT.NS", "M&M.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATASTEEL.NS", "TATAMOTORS.NS", "NTPC.NS",
    "ULTRACEMCO.NS", "POWERGRID.NS", "TITAN.NS", "BAJAJFINSV.NS", "WIPRO.NS", "HCLTECH.NS", "NESTLEIND.NS", "ONGC.NS", "JSWSTEEL.NS",
    "HINDALCO.NS", "GRASIM.NS", "ADANIPORTS.NS", "ADANIENT.NS", "COALINDIA.NS", "TATACONSUM.NS", "DRREDDY.NS", "CIPLA.NS", "BAJAJ-AUTO.NS",
    "EICHERMOT.NS", "DIVISLAB.NS", "BRITANNIA.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ZOMATO.NS", "BHEL.NS",
    "SUZLON.NS", "DLF.NS", "HAL.NS", "BEL.NS"
]

# 🔥 SMART AI FINDER (Auto-detects which AI model works for your API Key)
def get_ai_prediction(prompt):
    if not GEMINI_API_KEY: return "⚠️ AI Key missing in Render."
    models_to_try = ['gemini-1.5-flash', 'gemini-1.0-pro', 'gemini-pro', 'gemini-1.5-pro']
    for m in models_to_try:
        try:
            model = genai.GenerativeModel(m)
            res = model.generate_content(prompt)
            if res and res.text: return res.text.replace("*", "")
        except Exception:
            continue
    return "⚠️ AI Error: Check API Key limits or Region."

@app.get("/api/swing-scanner")
def run_swing_scanner():
    if "scanner_data" in cache: return cache["scanner_data"]
    try:
        data = yf.download(TICKERS, period="5d", progress=False)
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        closes = data['Close'] if 'Close' in data else data
        
        all_perf = []
        for ticker in TICKERS:
            if ticker in closes.columns:
                series = closes[ticker].dropna()
                if len(series) >= 2:
                    pct = round(((series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]) * 100, 2)
                    all_perf.append({"Symbol": ticker.replace(".NS", ""), "Percent": pct, "Price": round(series.iloc[-1], 2)})

        gainers = sorted([s for s in all_perf if s['Percent'] > 0], key=lambda x: x['Percent'], reverse=True)[:15]
        losers = sorted([s for s in all_perf if s['Percent'] < 0], key=lambda x: x['Percent'])[:15]
        
        res = {"status": "success", "data": {"top_gainers": gainers, "top_losers": losers}}
        cache["scanner_data"] = res
        return res
    except Exception as e: return {"status": "error", "message": str(e)}

def safe_val(val):
    if pd.isna(val) or math.isnan(val) or val is None: return None
    return round(float(val), 2)

@app.get("/api/analyze/{symbol}/{timeframe}")
def analyze_stock(symbol: str, timeframe: str):
    yf_symbol = symbol.upper().replace(".NS", "") + ".NS"
    period = "5d" if timeframe in ['1m', '5m', '15m'] else ("1mo" if timeframe in ['60m', '1h'] else "1y")
    cache_key = f"analyze_{timeframe}_{yf_symbol}"
    if cache_key in cache: return {"status": "success", "data": cache[cache_key]}

    try:
        df = yf.download(yf_symbol, period=period, interval=timeframe, progress=False)
        df_daily = yf.download(yf_symbol, period="15d", interval="1d", progress=False)

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if isinstance(df_daily.columns, pd.MultiIndex): df_daily.columns = df_daily.columns.get_level_values(0)
        
        df = df.dropna(subset=['Close']) # Flush bad data immediately
        if df.empty: return {"status": "error", "message": f"Market Data unavailable for {symbol}."}

        # PROPER PIVOT MATH
        df_daily.index = df_daily.index.tz_localize(None)
        H, L, C = df_daily['High'], df_daily['Low'], df_daily['Close']
        df_daily['P'] = (H + L + C) / 3
        df_daily['R1'] = (2 * df_daily['P']) - L
        df_daily['S1'] = (2 * df_daily['P']) - H
        df_daily['R2'] = df_daily['P'] + (H - L)
        df_daily['S2'] = df_daily['P'] - (H - L)

        pivots = df_daily[['P', 'R1', 'S1', 'R2', 'S2']].shift(1)
        pivots.index = pivots.index.date
        
        df.index = df.index.tz_localize(None)
        df['date_only'] = df.index.date
        for col in ['P', 'R1', 'S1', 'R2', 'S2']:
            df[col] = df['date_only'].map(pivots[col])

        # ADVANCED INDICATORS
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.rsi(length=14, append=True)
        
        st3 = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
        df['st3'] = st3.iloc[:, 0] if st3 is not None and not st3.empty else None
        df['trend'] = st3.iloc[:, 1] if st3 is not None and not st3.empty else 1

        chart_data = []
        for dt, row in df.iterrows():
            unix_t = int(pd.Timestamp(dt).timestamp()) + (5.5 * 3600)
            chart_data.append({
                "time": unix_t, "open": safe_val(row['Open']), "high": safe_val(row['High']),
                "low": safe_val(row['Low']), "close": safe_val(row['Close']),
                "st3": safe_val(row['st3']), "trend": safe_val(row['trend']),
                "p": safe_val(row['P']), "r1": safe_val(row['R1']), "s1": safe_val(row['S1']),
                "ema9": safe_val(row.get('EMA_9', 0)), "ema21": safe_val(row.get('EMA_21', 0)),
                "rsi": safe_val(row.get('RSI_14', 50))
            })

        latest_price = round(float(df.iloc[-1]['Close']), 2)
        
        # Fire AI prediction
        prompt = f"Act as a pro intraday trader. Analyze {timeframe} chart for {symbol}. Price: ₹{latest_price}. RSI: {chart_data[-1]['rsi']}. Give a sharp 2-sentence momentum prediction."
        ai_commentary = get_ai_prediction(prompt)

        res = {"status": "success", "data": {"symbol": yf_symbol.replace(".NS", ""), "latest_close": latest_price, "ai_prediction": ai_commentary, "historical_chart_data": chart_data}}
        cache[cache_key] = res
        return res
    except Exception as e: return {"status": "error", "message": str(e)}

if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=10000)
