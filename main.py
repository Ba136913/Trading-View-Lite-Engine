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

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "INFY.NS", "TATAMOTORS.NS", "ZOMATO.NS", "BHEL.NS", "TATASTEEL.NS", "SUZLON.NS"]

@app.get("/api/swing-scanner")
def run_swing_scanner():
    if "scanner_data" in cache: return cache["scanner_data"]
    try:
        data = yf.download(TICKERS, period="5d", progress=False)
        # 🔥 Yahan MultiIndex flatten nahi karna hai, warna Tickers ke naam gayab ho jate hain
        closes = data['Close'] 
        
        all_performance = []
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

        result = {"status": "success", "data": {"top_gainers": gainers, "top_losers": losers}}
        cache["scanner_data"] = result
        return result
    except Exception as e:
        return {"status": "error", "message": f"Scanner Error: {str(e)}"}

def safe_val(val):
    if pd.isna(val) or math.isnan(val): return None
    return round(float(val), 2)

# 🔥 FIX: Added timeframe parameter Support (5m, 15m, 60m, 1d)
@app.get("/api/analyze/{symbol}/{timeframe}")
def analyze_stock(symbol: str, timeframe: str):
    yf_symbol = symbol.upper().replace(".NS", "") + ".NS"
    
    # Dynamic period based on timeframe
    if timeframe in ['1m', '5m', '15m']: period = "5d"
    elif timeframe in ['60m', '1h']: period = "1mo"
    else: period = "1y"

    try:
        df = yf.download(yf_symbol, period=period, interval=timeframe, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        if df.empty: return {"status": "error", "message": f"Data not found for {symbol}."}

        df = df.ffill().bfill()
        
        # ----------------------------------------------------
        # 🔥 CLEAR SIGNALS (SuperTrend 10:3)
        # ----------------------------------------------------
        st3 = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
        df['st3'] = st3.iloc[:, 0] if st3 is not None and not st3.empty else None
        
        # Determine Trend Direction (1 for Bullish, -1 for Bearish)
        df['st3_dir'] = st3.iloc[:, 1] if st3 is not None and not st3.empty else 1

        latest_price = round(float(df.iloc[-1]['Close']), 2)
        
        chart_data = []
        for dt, row in df.iterrows():
            # Adjusting timezone offset correctly so Live data shows accurately
            unix_t = int(dt.timestamp()) + (5.5 * 3600) # IST Offset trick for lightweight charts
            
            chart_data.append({
                "time": unix_t, 
                "open": safe_val(row['Open']), "high": safe_val(row['High']),
                "low": safe_val(row['Low']), "close": safe_val(row['Close']),
                "st3": safe_val(row['st3']),
                "trend": safe_val(row['st3_dir'])
            })

        # Generate AI Prediction
        ai_commentary = "⚠️ Add Gemini API Key in Render environment variables to activate AI."
        if ai_model:
            try:
                prompt = f"Act as a pro trader. Analyze {timeframe} chart for {symbol}. Price: ₹{latest_price}. Give a sharp 2-sentence momentum prediction."
                ai_commentary = ai_model.generate_content(prompt).text.replace("*", "")
            except: pass

        return {
            "status": "success",
            "data": {
                "symbol": yf_symbol.replace(".NS", ""),
                "latest_close": latest_price,
                "ai_prediction": ai_commentary,
                "historical_chart_data": chart_data
            }
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
