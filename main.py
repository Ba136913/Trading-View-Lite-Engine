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

def get_ai_prediction(symbol, current_price):
    if not ai_model: return "⚠️ AI prediction disabled. API Key not found."
    prompt = f"""
    Act as a Hedge Fund Quant. Analyze 5-min Intraday chart for {symbol} (NSE). Current Price: ₹{current_price}.
    Give a sharp, 3-sentence intraday momentum prediction based on current price action.
    """
    try: return ai_model.generate_content(prompt).text.replace("*", "")
    except: return "⚠️ AI Engine currently busy."

@app.get("/api/analyze/{symbol}")
def analyze_stock(symbol: str):
    yf_symbol = symbol.upper().replace(" ", "").replace(".NS", "") + ".NS"
    cache_key = f"analyze_5m_{yf_symbol}"
    
    if cache_key in cache: return {"status": "success", "data": cache[cache_key]}

    try:
        # 🔥 FIX: 5-Minute Timeframe Data
        df = yf.download(yf_symbol, period="5d", interval="5m", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        if df.empty or len(df) < 50: 
            return {"status": "error", "message": f"Intraday Data not found for {symbol}."}
            
        df = df.ffill().bfill()
        
        # SuperTrend on 5-min
        df.ta.supertrend(length=10, multiplier=3, append=True)
        df.ta.supertrend(length=10, multiplier=1, append=True)
        
        # 🔥 FIX: Real Step-like Pivot Points (Calculated per day)
        daily_data = df.groupby(df.index.date).agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
        daily_data['P'] = (daily_data['High'].shift(1) + daily_data['Low'].shift(1) + daily_data['Close'].shift(1)) / 3
        daily_data['R1'] = (2 * daily_data['P']) - daily_data['Low'].shift(1)
        daily_data['S1'] = (2 * daily_data['P']) - daily_data['High'].shift(1)
        
        # Map pivots back to 5-min timeframe
        df['P'] = df.index.date
        df['P'] = df['P'].map(daily_data['P'])
        df['R1'] = df.index.date
        df['R1'] = df['R1'].map(daily_data['R1'])
        df['S1'] = df.index.date
        df['S1'] = df['S1'].map(daily_data['S1'])

        df = df.fillna(0)
        
        latest = df.iloc[-1]
        current_price = round(float(latest['Close']), 2)

        try: ai_commentary = get_ai_prediction(yf_symbol.replace(".NS",""), current_price)
        except: ai_commentary = "⚠️ AI Engine busy right now."

        chart_data = []
        for date, row in df.iterrows():
            # 🔥 Unix timestamp so Lightweight Charts shows 5-min intraday correctly
            unix_time = int(date.timestamp()) + 19800 # UTC to IST adjustment
            
            st3_val = row.get('SUPERT_10_3.0', 0)
            st1_val = row.get('SUPERT_10_1.0', 0)
            p_val = row.get('P', 0)
            r1_val = row.get('R1', 0)
            s1_val = row.get('S1', 0)
            
            chart_data.append({
                "time": unix_time, 
                "open": round(float(row['Open']), 2), "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2), "close": round(float(row['Close']), 2),
                "st3": round(float(st3_val), 2) if st3_val != 0 else None,
                "st1": round(float(st1_val), 2) if st1_val != 0 else None,
                "p": round(float(p_val), 2) if p_val != 0 else None,
                "r1": round(float(r1_val), 2) if r1_val != 0 else None,
                "s1": round(float(s1_val), 2) if s1_val != 0 else None
            })

        result = {
            "symbol": yf_symbol.replace(".NS", ""),
            "latest_close": current_price,
            "ai_prediction": ai_commentary,
            "historical_chart_data": chart_data
        }
        
        cache[cache_key] = result
        return {"status": "success", "data": result}
        
    except Exception as e:
        return {"status": "error", "message": f"Backend Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
