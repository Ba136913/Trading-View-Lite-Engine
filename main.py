from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import uvicorn
import os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
] # 50 tak bada lena

@app.get("/api/swing-scanner")
def run_swing_scanner():
    try:
        data = yf.download(TICKERS, period="2d", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        
        all_perf = []
        for ticker in TICKERS:
            series = data['Close'][ticker].dropna()
            if len(series) >= 2:
                pct = round(((series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]) * 100, 2)
                all_perf.append({"Symbol": ticker.replace(".NS",""), "Percent": pct, "Price": round(series.iloc[-1], 2)})
        
        return {"status": "success", "data": {"top_gainers": sorted(all_perf, key=lambda x: x['Percent'], reverse=True)[:15], "top_losers": sorted(all_perf, key=lambda x: x['Percent'])[:15]}}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.get("/api/analyze/{symbol}")
def analyze_stock(symbol: str):
    yf_symbol = symbol.upper().replace(".NS", "") + ".NS"
    try:
        df = yf.download(yf_symbol, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.ffill().bfill()

        # --- Matrix 1.0 Calculations ---
        # 1. SuperTrends
        st3 = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
        st1 = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=1)
        df['st3'] = st3['SUPERT_10_3.0']
        df['st1'] = st1['SUPERT_10_1.0']
        
        # 2. Daily Pivots
        prev = df.iloc[-2]
        p = (prev['High'] + prev['Low'] + prev['Close']) / 3
        r1, s1 = (2*p)-prev['Low'], (2*p)-prev['High']

        # 3. SMC BOS Detection
        df['hh'] = df['High'].rolling(10).max().shift(1)
        df['bos'] = (df['Close'] > df['hh']).map({True: "BOS ↑", False: ""})

        chart_data = []
        for date, row in df.tail(150).iterrows():
            chart_data.append({
                "time": str(date.date()), "open": row['Open'], "high": row['High'], "low": row['Low'], "close": row['Close'],
                "st3": row['st3'], "st1": row['st1'], "bos": row['bos']
            })

        return {
            "status": "success", 
            "data": {
                "symbol": symbol, "price": round(df['Close'].iloc[-1], 2),
                "pivots": {"p": round(p,2), "r1": round(r1,2), "s1": round(s1,2)},
                "chart": chart_data
            }
        }
    except Exception as e: return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
