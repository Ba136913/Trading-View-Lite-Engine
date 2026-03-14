from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import google.generativeai as genai
from cachetools import TTLCache
import uvicorn
import os
import datetime
import math
import requests  # 🔥 NAYA IMPORT: Bypass ke liye

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = TTLCache(maxsize=500, ttl=300)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    ai_model = None

# ----------------------------------------------------
# 🔥 YAHOO FINANCE ANTI-BLOCK SYSTEM (Fake Browser)
# ----------------------------------------------------
yf_session = requests.Session()
yf_session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
})

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "INFY.NS", "TATAMOTORS.NS", "ZOMATO.NS", "BHEL.NS", "TATASTEEL.NS", "SUZLON.NS"]

@app.get("/api/swing-scanner")
def run_swing_scanner():
    if "scanner_data" in cache: return cache["scanner_data"]
    try:
        # session=yf_session pass karke ban se bachenge
        data = yf.download(TICKERS, period="5d", progress=False, session=yf_session)
        closes = data['Close'] if 'Close' in data else data
        
        all_performance = []
        for ticker in TICKERS:
            if ticker in closes.columns:
                series = closes[ticker].dropna()
                if len(series) >= 2:
                    pct_change = round(((series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]) * 100, 2)
                    all_performance.append({"Symbol": ticker.replace(".NS", ""), "Percent": pct_change, "Price": round(series.iloc[-1], 2)})

        gainers = sorted([s for s in all_performance if s['Percent'] > 0], key=lambda x: x['Percent'], reverse=True)[:15]
        losers = sorted([s for s in all_performance if s['Percent'] < 0], key=lambda x: x['Percent'])[:15]

        result = {"status": "success", "data": {"top_gainers": gainers, "top_losers": losers}}
        cache["scanner_data"] = result
        return result
    except Exception as e:
        return {"status": "error", "message": f"Scanner Error: {str(e)}"}

def safe_val(val):
    if pd.isna(val) or math.isnan(val): return None
    return round(float(val), 2)

@app.get("/api/analyze/{symbol}/{timeframe}")
def analyze_stock(symbol: str, timeframe: str):
    yf_symbol = symbol.upper().replace(".NS", "") + ".NS"
    
    if timeframe in ['1m', '5m', '15m']: period = "5d"
    elif timeframe in ['60m', '1h']: period = "1mo"
    else: period = "1y"

    cache_key = f"analyze_{timeframe}_{yf_symbol}"
    if cache_key in cache: return {"status": "success", "data": cache[cache_key]}

    try:
        # Dono Data me Session pass kiya hai
        df = yf.download(yf_symbol, period=period, interval=timeframe, progress=False, session=yf_session)
        df_daily = yf.download(yf_symbol, period="15d", interval="1d", progress=False, session=yf_session)

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if isinstance(df_daily.columns, pd.MultiIndex): df_daily.columns = df_daily.columns.get_level_values(0)

        if df.empty: return {"status": "error", "message": f"Data not found for {symbol}."}

        df = df.ffill().bfill()
        
        # ----------------------------------------------------
        # 🔥 TRADITIONAL PIVOTS CALCULATION
        # ----------------------------------------------------
        H, L, C = df_daily['High'], df_daily['Low'], df_daily['Close']
        df_daily['P'] = (H + L + C) / 3
        df_daily['R1'] = (2 * df_daily['P']) - L
        df_daily['S1'] = (2 * df_daily['P']) - H
        df_daily['R2'] = df_daily['P'] + (H - L)
        df_daily['S2'] = df_daily['P'] - (H - L)
        df_daily['R3'] = df_daily['R1'] + (H - L)
        df_daily['S3'] = df_daily['S1'] - (H - L)
        df_daily['R4'] = df_daily['R3'] + (H - L)
        df_daily['S4'] = df_daily['S3'] - (H - L)
        df_daily['R5'] = df_daily['R4'] + (H - L)
        df_daily['S5'] = df_daily['S4'] - (H - L)

        pivots_shifted = df_daily[['P', 'R1', 'R2', 'R3', 'R4', 'R5', 'S1', 'S2', 'S3', 'S4', 'S5']].shift(1)
        pivots_shifted.index = pivots_shifted.index.tz_localize(None).date
        
        df['date_only'] = df.index.tz_localize(None).date
        for col in ['P', 'R1', 'R2', 'R3', 'R4', 'R5', 'S1', 'S2', 'S3', 'S4', 'S5']:
            df[col] = df['date_only'].map(pivots_shifted[col])

        # ----------------------------------------------------
        # 🔥 SUPERTREND (10:3 and 10:1)
        # ----------------------------------------------------
        st3 = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
        st1 = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=1)

        df['st3'] = st3.iloc[:, 0] if st3 is not None and not st3.empty else None
        df['st1'] = st1.iloc[:, 0] if st1 is not None and not st1.empty else None
        df['st3_dir'] = st3.iloc[:, 1] if st3 is not None and not st3.empty else 1

        latest_price = round(float(df.iloc[-1]['Close']), 2)
        
        chart_data = []
        for dt, row in df.iterrows():
            unix_t = int(dt.timestamp()) + (5.5 * 3600)  # IST Offset
            
            chart_data.append({
                "time": unix_t, 
                "open": safe_val(row['Open']), "high": safe_val(row['High']),
                "low": safe_val(row['Low']), "close": safe_val(row['Close']),
                "st3": safe_val(row['st3']), "st1": safe_val(row['st1']),
                "trend": safe_val(row['st3_dir']),
                "p": safe_val(row['P']), 
                "r1": safe_val(row['R1']), "r2": safe_val(row['R2']), "r3": safe_val(row['R3']), "r4": safe_val(row['R4']), "r5": safe_val(row['R5']),
                "s1": safe_val(row['S1']), "s2": safe_val(row['S2']), "s3": safe_val(row['S3']), "s4": safe_val(row['S4']), "s5": safe_val(row['S5'])
            })

        ai_commentary = "⚠️ Add Gemini API Key in Render to activate AI."
        if ai_model:
            try:
                prompt = f"Act as a pro trader. Analyze {timeframe} chart for {symbol}. Price: ₹{latest_price}. Give a sharp 2-sentence momentum prediction."
                ai_commentary = ai_model.generate_content(prompt).text.replace("*", "")
            except: pass

        result = {
            "status": "success",
            "data": {
                "symbol": yf_symbol.replace(".NS", ""),
                "latest_close": latest_price,
                "ai_prediction": ai_commentary,
                "historical_chart_data": chart_data
            }
        }
        
        cache[cache_key] = result
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
