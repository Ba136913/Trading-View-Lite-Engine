from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
from groq import Groq
from cachetools import TTLCache
import uvicorn
import math

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
cache = TTLCache(maxsize=500, ttl=300)

# 🔥 SHIFTED TO GROQ (LLAMA 3.1)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

TICKERS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "ITC", "LT", "SBIN", "BHARTIARTL", "BAJFINANCE",
    "AXISBANK", "KOTAKBANK", "ASIANPAINT", "M&M", "MARUTI", "SUNPHARMA", "TATASTEEL", "TATAMOTORS", "NTPC",
    "ULTRACEMCO", "POWERGRID", "TITAN", "BAJAJFINSV", "WIPRO", "HCLTECH", "NESTLEIND", "ONGC", "JSWSTEEL",
    "HINDALCO", "GRASIM", "ADANIPORTS", "ADANIENT", "COALINDIA", "TATACONSUM", "DRREDDY", "CIPLA", "BAJAJ-AUTO",
    "EICHERMOT", "DIVISLAB", "BRITANNIA", "HEROMOTOCO", "INDUSINDBK", "HDFCLIFE", "SBILIFE", "ZOMATO", "BHEL",
    "SUZLON", "DLF", "HAL", "BEL"
]

def get_ai_prediction(prompt):
    if not groq_client: return "⚠️ Groq API Key is missing in Render Environment Variables."
    try:
        # 🔥 FIX: Updated to the newest active Groq model (Llama 3.1 8B Instant)
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional hedge fund quant. Give extremely sharp, concise, 2-sentence momentum predictions based on the provided technicals."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",  # <--- YAHAN CHANGE KIYA HAI
            temperature=0.5,
            max_tokens=100,
        )
        return chat_completion.choices[0].message.content.replace("*", "")
    except Exception as e:
        return f"⚠️ AI Engine Error: {str(e)[:150]}..."

@app.get("/api/swing-scanner")
def run_swing_scanner():
    return {"status": "success", "data": {"fno_stocks": TICKERS}}

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
        
        if df.empty or 'Close' not in df.columns or df['Close'].dropna().empty: 
            return {"status": "error", "message": f"Market Data unavailable for {symbol}. Yahoo Finance might be blocking this temporarily. Try again later."}

        df = df.dropna(subset=['Close'])

        # PIVOTS
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

        # INDICATORS
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
        
        # Groq AI Logic
        prompt = f"Analyze {timeframe} chart for {symbol} on NSE. Current Price: ₹{latest_price}. RSI is {chart_data[-1]['rsi']}. Tell me if it's bullish, bearish, or sideways and next immediate resistance/support."
        ai_commentary = get_ai_prediction(prompt)

        res = {"status": "success", "data": {"symbol": yf_symbol.replace(".NS", ""), "latest_close": latest_price, "ai_prediction": ai_commentary, "historical_chart_data": chart_data}}
        cache[cache_key] = res
        return res
    except Exception as e: return {"status": "error", "message": str(e)}

if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=10000)
