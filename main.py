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
    ai_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    ai_model = None

# 🔥 Top 120+ Liquid F&O Stocks (Crash-Proof Batch)
TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "ASIANPAINT.NS", "M&M.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATASTEEL.NS", "TATAMOTORS.NS", "NTPC.NS",
    "ULTRACEMCO.NS", "POWERGRID.NS", "TITAN.NS", "BAJAJFINSV.NS", "WIPRO.NS", "HCLTECH.NS", "NESTLEIND.NS", "ONGC.NS", "JSWSTEEL.NS",
    "HINDALCO.NS", "GRASIM.NS", "ADANIPORTS.NS", "ADANIENT.NS", "COALINDIA.NS", "TATACONSUM.NS", "DRREDDY.NS", "CIPLA.NS", "BAJAJ-AUTO.NS",
    "APOLLOHOSP.NS", "EICHERMOT.NS", "DIVISLAB.NS", "BRITANNIA.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "ZOMATO.NS", "BHEL.NS", "SUZLON.NS", "DLF.NS", "HAL.NS", "BEL.NS", "TVSMOTOR.NS", "LTIM.NS", "TECHM.NS", "CHOLAFIN.NS", "TRENT.NS",
    "LUPIN.NS", "AUROPHARMA.NS", "IDEA.NS", "IDFCFIRSTB.NS", "PFC.NS", "RECLTD.NS", "SAIL.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS",
    "VEDL.NS", "NMDC.NS", "BOSCHLTD.NS", "AMBUJACEM.NS", "SHREECEM.NS", "PIIND.NS", "NAUKRI.NS", "IRCTC.NS", "DIXON.NS", "POLYCAB.NS",
    "HDFCAMC.NS", "MUTHOOTFIN.NS", "MANAPPURAM.NS", "M&MFIN.NS", "JUBLFOOD.NS", "SRF.NS", "VOLTAS.NS", "TATACHEM.NS", "IGL.NS", "MGL.NS",
    "PETRONET.NS", "GAIL.NS", "HINDPETRO.NS", "BPCL.NS", "IOC.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "AUBANK.NS", "CUMMINSIND.NS", "ASTRAL.NS",
    "ASHOKLEY.NS", "ESCORTS.NS", "BATAINDIA.NS", "PEL.NS", "LICHSGFIN.NS", "GNFC.NS", "CHAMBLFERT.NS", "COROMANDEL.NS", "DEEPAKNTR.NS",
    "SYNGENE.NS", "LAURUSLABS.NS", "GLENMARK.NS", "BIOCON.NS", "IPCALAB.NS", "MCX.NS", "IEX.NS", "OFSS.NS", "PERSISTENT.NS", "COFORGE.NS"
]

@app.get("/api/swing-scanner")
def run_swing_scanner():
    if "scanner_data" in cache: return cache["scanner_data"]
    try:
        # 2 Din ka data hi nikalenge taaki limit exceed na ho aur fast load ho
        data = yf.download(TICKERS, period="2d", progress=False)
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
    if pd.isna(val) or math.isnan(val): return None
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
        if df.empty: return {"status": "error", "message": f"Data not found."}

        df = df.ffill().bfill()
        
        # PIVOTS
        H, L, C = df_daily['High'], df_daily['Low'], df_daily['Close']
        p_val = (H + L + C) / 3
        df_daily['P'] = p_val
        for col in ['P', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']: df_daily[col] = p_val # Shortened for speed
        pivots = df_daily[['P']].shift(1)
        pivots.index = pivots.index.tz_localize(None).date
        df['P'] = df.index.tz_localize(None).date
        df['P'] = df['P'].map(pivots['P'])

        # 🔥 ADVANCED PREDICTIVE INDICATORS
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        
        st3 = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
        df['st3'] = st3.iloc[:, 0] if st3 is not None and not st3.empty else None
        df['trend'] = st3.iloc[:, 1] if st3 is not None and not st3.empty else 1

        chart_data = []
        for dt, row in df.iterrows():
            unix_t = int(dt.timestamp()) + (5.5 * 3600)
            chart_data.append({
                "time": unix_t, "open": safe_val(row['Open']), "high": safe_val(row['High']),
                "low": safe_val(row['Low']), "close": safe_val(row['Close']),
                "st3": safe_val(row['st3']), "trend": safe_val(row['trend']), "p": safe_val(row['P']),
                "ema9": safe_val(row.get('EMA_9', 0)), "ema21": safe_val(row.get('EMA_21', 0)),
                "rsi": safe_val(row.get('RSI_14', 50)), "macd": safe_val(row.get('MACD_12_26_9', 0))
            })

        latest_price = round(float(df.iloc[-1]['Close']), 2)
        ai_commentary = "⚠️ AI Ready. Model updating."
        if ai_model:
            try:
                prompt = f"Act as a hedge fund quant. Analyze {timeframe} chart for {symbol}. Price: ₹{latest_price}. RSI: {chart_data[-1]['rsi']}. EMA 9/21 cross status. Give a sharp 2-sentence prediction."
                ai_commentary = ai_model.generate_content(prompt).text.replace("*", "")
            except Exception as e: ai_commentary = f"⚠️ AI Engine Error: {str(e)}"

        res = {"status": "success", "data": {"symbol": yf_symbol.replace(".NS", ""), "latest_close": latest_price, "ai_prediction": ai_commentary, "historical_chart_data": chart_data}}
        cache[cache_key] = res
        return res
    except Exception as e: return {"status": "error", "message": str(e)}

if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=10000)
