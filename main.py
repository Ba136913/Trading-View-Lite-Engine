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

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    "SBIN.NS", "INFY.NS", "LICI.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "ONGC.NS", "KOTAKBANK.NS", "NTPC.NS", "AXISBANK.NS",
    "POWERGRID.NS", "ADANIENT.NS", "BAJAJFINSV.NS", "ASIANPAINT.NS", "M&M.NS",
    "COALINDIA.NS", "TITAN.NS", "BAJAJ-AUTO.NS", "ULTRACEMCO.NS", "ADANIPORTS.NS",
    "JSWSTEEL.NS", "WIPRO.NS", "ZOMATO.NS", "LTIM.NS", "HAL.NS", "DLF.NS",
    "INDUSINDBK.NS", "GRASIM.NS", "NESTLEIND.NS", "TRENT.NS", "TVSMOTOR.NS",
    "CHOLAFIN.NS", "BEL.NS", "TECHM.NS", "SBILIFE.NS", "HDFCLIFE.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "CIPLA.NS"
]

@app.get("/api/swing-scanner")
def run_swing_scanner():
    if "scanner_data" in cache: return cache["scanner_data"]
    try:
        data = yf.download(TICKERS, period="5d", progress=False)
        closes = data['Close']
        all_performance = []
        date_str = datetime.date.today().strftime("%Y-%m-%d")

        for ticker in TICKERS:
            if ticker in closes.columns:
                series = closes[ticker].dropna()
                if len(series) >= 2:
                    close_today = float(series.iloc[-1])
                    close_yest = float(series.iloc[-2])
                    if close_yest > 0:
                        pct_change = round(((close_today - close_yest) / close_yest) * 100, 2)
                        symbol_name = ticker.replace(".NS", "")
                        all_performance.append({"Symbol": symbol_name, "Percent": pct_change, "Price": round(close_today, 2)})

        gainers = sorted([s for s in all_performance if s['Percent'] > 0], key=lambda x: x['Percent'], reverse=True)[:15]
        losers = sorted([s for s in all_performance if s['Percent'] < 0], key=lambda x: x['Percent'])[:15]

        result = {"status": "success", "date": date_str, "data": {"top_gainers": gainers, "top_losers": losers}}
        cache["scanner_data"] = result
        return result
    except Exception as e:
        return {"status": "error", "message": f"Scanner Error: {str(e)}"}

# 🔥 FIX 1: Safe Value Function (Prevents chart from collapsing to 0)
def safe_val(val):
    if pd.isna(val) or math.isnan(val): return None
    return round(float(val), 2)

@app.get("/api/analyze/{symbol}")
def analyze_stock(symbol: str):
    yf_symbol = symbol.upper().replace(" ", "").replace(".NS", "") + ".NS"
    cache_key = f"analyze_5m_{yf_symbol}"
    
    if cache_key in cache: return {"status": "success", "data": cache[cache_key]}

    try:
        # Fetch 5-min and Daily Data
        df_5m = yf.download(yf_symbol, period="5d", interval="5m", progress=False)
        df_daily = yf.download(yf_symbol, period="10d", interval="1d", progress=False)

        if isinstance(df_5m.columns, pd.MultiIndex): df_5m.columns = df_5m.columns.get_level_values(0)
        if isinstance(df_daily.columns, pd.MultiIndex): df_daily.columns = df_daily.columns.get_level_values(0)

        if df_5m.empty or len(df_5m) < 20: 
            return {"status": "error", "message": f"Data not found for {symbol}."}

        # ----------------------------------------------------
        # 🔥 TRADINGVIEW TRADITIONAL PIVOTS (P, R1-R5, S1-S5)
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

        # Shift daily pivots by 1 day (So 5-min candles use yesterday's data)
        pivots_shifted = df_daily[['P', 'R1', 'R2', 'R3', 'R4', 'R5', 'S1', 'S2', 'S3', 'S4', 'S5']].shift(1)
        pivots_shifted.index = pivots_shifted.index.tz_localize(None).date
        
        df_5m['date_only'] = df_5m.index.tz_localize(None).date

        for col in ['P', 'R1', 'R2', 'R3', 'R4', 'R5', 'S1', 'S2', 'S3', 'S4', 'S5']:
            df_5m[col] = df_5m['date_only'].map(pivots_shifted[col])

        # ----------------------------------------------------
        # 🔥 SAFELY CALCULATE SUPERTREND
        # ----------------------------------------------------
        st3 = ta.supertrend(df_5m['High'], df_5m['Low'], df_5m['Close'], length=10, multiplier=3)
        st1 = ta.supertrend(df_5m['High'], df_5m['Low'], df_5m['Close'], length=10, multiplier=1)

        df_5m['st3'] = st3.iloc[:, 0] if st3 is not None and not st3.empty else None
        df_5m['st1'] = st1.iloc[:, 0] if st1 is not None and not st1.empty else None

        latest_price = round(float(df_5m.iloc[-1]['Close']), 2)
        
        chart_data = []
        for dt, row in df_5m.iterrows():
            unix_t = int(dt.timestamp()) # UNIX time for perfect 5-min gaps
            
            chart_data.append({
                "time": unix_t, 
                "open": safe_val(row['Open']), "high": safe_val(row['High']),
                "low": safe_val(row['Low']), "close": safe_val(row['Close']),
                "st3": safe_val(row['st3']), "st1": safe_val(row['st1']),
                "p": safe_val(row['P']), 
                "r1": safe_val(row['R1']), "r2": safe_val(row['R2']), "r3": safe_val(row['R3']), "r4": safe_val(row['R4']), "r5": safe_val(row['R5']),
                "s1": safe_val(row['S1']), "s2": safe_val(row['S2']), "s3": safe_val(row['S3']), "s4": safe_val(row['S4']), "s5": safe_val(row['S5'])
            })

        result = {
            "symbol": yf_symbol.replace(".NS", ""),
            "latest_close": latest_price,
            "ai_prediction": "AI Analysis based on 5-Min timeframe generated.",
            "historical_chart_data": chart_data
        }
        
        cache[cache_key] = result
        return {"status": "success", "data": result}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
