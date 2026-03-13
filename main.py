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
    if not ai_model: return "⚠️ AI prediction disabled."
    prompt = f"""Act as a Hedge Fund Quant. Analyze Intraday (5-min) chart for {symbol} (NSE). Price: ₹{current_price}. Give 2-sentence sharp momentum prediction."""
    try: return ai_model.generate_content(prompt).text.replace("*", "")
    except: return "⚠️ AI Engine currently busy."

@app.get("/api/analyze/{symbol}")
def analyze_stock(symbol: str):
    yf_symbol = symbol.upper().replace(" ", "").replace(".NS", "") + ".NS"
    cache_key = f"analyze_5m_{yf_symbol}"
    
    if cache_key in cache: return {"status": "success", "data": cache[cache_key]}

    try:
        # 🔥 FIX: 1. Fetch 5-Min Intraday Data
        df_5m = yf.download(yf_symbol, period="5d", interval="5m", progress=False)
        # 🔥 FIX: 2. Fetch Daily Data strictly for Authentic Pivot Points
        df_daily = yf.download(yf_symbol, period="10d", interval="1d", progress=False)

        if isinstance(df_5m.columns, pd.MultiIndex): df_5m.columns = df_5m.columns.get_level_values(0)
        if isinstance(df_daily.columns, pd.MultiIndex): df_daily.columns = df_daily.columns.get_level_values(0)

        if df_5m.empty or len(df_5m) < 20: 
            return {"status": "error", "message": f"Data not found for {symbol}."}

        # ----------------------------------------------------
        # 🔥 EXACT PIVOT POINTS MATH (Using previous day data)
        # ----------------------------------------------------
        df_daily['P'] = (df_daily['High'] + df_daily['Low'] + df_daily['Close']) / 3
        df_daily['R1'] = (2 * df_daily['P']) - df_daily['Low']
        df_daily['S1'] = (2 * df_daily['P']) - df_daily['High']
        
        # Shift daily pivots by 1 day so today uses yesterday's pivots
        df_daily['P'] = df_daily['P'].shift(1)
        df_daily['R1'] = df_daily['R1'].shift(1)
        df_daily['S1'] = df_daily['S1'].shift(1)

        # Remove Timezones for mapping
        df_daily.index = df_daily.index.tz_localize(None).date
        df_5m['date_only'] = df_5m.index.tz_localize(None).date

        # Map Authentic Pivots onto 5-min chart
        df_5m['P'] = df_5m['date_only'].map(df_daily['P'])
        df_5m['R1'] = df_5m['date_only'].map(df_daily['R1'])
        df_5m['S1'] = df_5m['date_only'].map(df_daily['S1'])

        # ----------------------------------------------------
        # 🔥 SAFELY CALCULATE SUPERTREND
        # ----------------------------------------------------
        st3 = ta.supertrend(df_5m['High'], df_5m['Low'], df_5m['Close'], length=10, multiplier=3)
        st1 = ta.supertrend(df_5m['High'], df_5m['Low'], df_5m['Close'], length=10, multiplier=1)

        df_5m['st3'] = st3.iloc[:, 0] if st3 is not None and not st3.empty else 0
        df_5m['st1'] = st1.iloc[:, 0] if st1 is not None and not st1.empty else 0

        df_5m = df_5m.fillna(0)
        latest_price = round(float(df_5m.iloc[-1]['Close']), 2)

        chart_data = []
        for dt, row in df_5m.iterrows():
            # 🔥 Generate True Unix Timestamp for 5-Min Graph plotting
            unix_t = int(pd.Timestamp(dt).timestamp())
            
            chart_data.append({
                "time": unix_t, 
                "open": round(float(row['Open']), 2), "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2), "close": round(float(row['Close']), 2),
                "st3": round(float(row['st3']), 2) if row['st3'] != 0 else None,
                "st1": round(float(row['st1']), 2) if row['st1'] != 0 else None,
                "p": round(float(row['P']), 2) if row['P'] != 0 else None,
                "r1": round(float(row['R1']), 2) if row['R1'] != 0 else None,
                "s1": round(float(row['S1']), 2) if row['S1'] != 0 else None
            })

        result = {
            "symbol": yf_symbol.replace(".NS", ""),
            "latest_close": latest_price,
            "ai_prediction": get_ai_prediction(yf_symbol.replace(".NS",""), latest_price),
            "historical_chart_data": chart_data
        }
        
        cache[cache_key] = result
        return {"status": "success", "data": result}
        
    except Exception as e:
        return {"status": "error", "message": f"Backend Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
