from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import uvicorn
import os
import datetime
from cachetools import TTLCache

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
cache = TTLCache(maxsize=500, ttl=300)

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
    try:
        data = yf.download(TICKERS, period="5d", progress=False)
        closes = data['Close']
        all_performance = []
        for ticker in TICKERS:
            if ticker in closes.columns:
                series = closes[ticker].dropna()
                if len(series) >= 2:
                    pct = round(((series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]) * 100, 2)
                    all_performance.append({"Symbol": ticker.replace(".NS", ""), "Percent": pct, "Price": round(series.iloc[-1], 2)})
        return {"status": "success", "data": {"top_gainers": sorted(all_performance, key=lambda x: x['Percent'], reverse=True)[:15], "top_losers": sorted(all_performance, key=lambda x: x['Percent'])[:15]}}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.get("/api/analyze/{symbol}")
def analyze_stock(symbol: str):
    yf_symbol = symbol.upper().replace(".NS", "") + ".NS"
    try:
        df = yf.download(yf_symbol, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.ffill().bfill()

        # Matrix 1.0 Calculations
        st3 = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
        st1 = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=1)
        
        # Daily Pivots
        prev = df.iloc[-2]
        p = (prev['High'] + prev['Low'] + prev['Close']) / 3
        r1, s1 = (2*p)-prev['Low'], (2*p)-prev['High']

        chart_data = []
        for date, row in df.tail(150).iterrows():
            idx = df.index.get_loc(date)
            chart_data.append({
                "time": str(date.date()),
                "open": round(row['Open'], 2), "high": round(row['High'], 2),
                "low": round(row['Low'], 2), "close": round(row['Close'], 2),
                "st3": round(st3['SUPERT_10_3.0'].iloc[idx], 2),
                "st1": round(st1['SUPERT_10_1.0'].iloc[idx], 2),
                "is_st3_up": int(st3['SUPERTd_10_3.0'].iloc[idx]),
                "is_st1_up": int(st1['SUPERTd_10_1.0'].iloc[idx])
            })

        return {
            "status": "success",
            "data": {
                "symbol": symbol, "latest_close": round(df['Close'].iloc[-1], 2),
                "pivots": {"p": round(p, 2), "r1": round(r1, 2), "s1": round(s1, 2)},
                "historical_chart_data": chart_data
            }
        }
    except Exception as e: return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
