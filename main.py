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
    "AARTIIND.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "ADANIENT.NS", "ADANIPORTS.NS", 
    "ALKEM.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", 
    "ATUL.NS", "AUROPHARMA.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BALKRISIND.NS", 
    "BALRAMCHIN.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BATAINDIA.NS", "BEL.NS", "BERGEPAINT.NS", "BHARATFORG.NS", 
    "BHARTIARTL.NS", "BHEL.NS", "BIOCON.NS", "BOSCHLTD.NS", "BPCL.NS", "BRITANNIA.NS", "BSOFT.NS", "CANBK.NS", 
    "CANFINHOME.NS", "CHAMBLFERT.NS", "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", 
    "CONCOR.NS", "COROMANDEL.NS", "CROMPTON.NS", "CUB.NS", "CUMMINSIND.NS", "DABUR.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", 
    "DIVISLAB.NS", "DIXON.NS", "DLF.NS", "DRREDDY.NS", "EICHERMOT.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", 
    "GAIL.NS", "GLENMARK.NS", "GMRINFRA.NS", "GNFC.NS", "GODREJCP.NS", "GODREJPROP.NS", "GRANULES.NS", "GRASIM.NS", 
    "GUJGASLTD.NS", "HAL.NS", "HAVELLS.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", 
    "HINDALCO.NS", "HINDCOPPER.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", 
    "IDEA.NS", "IDFCFIRSTB.NS", "IEX.NS", "IGL.NS", "INDHOTEL.NS", "INDIACEM.NS", "INDIAMART.NS", "INDIGO.NS", 
    "INDUSINDBK.NS", "INDUSTOWER.NS", "INFY.NS", "IPCALAB.NS", "IRCTC.NS", "ITC.NS", "JINDALSTEL.NS", "JSWSTEEL.NS", 
    "JUBLFOOD.NS", "KOTAKBANK.NS", "LALPATHLAB.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", "LT.NS", "LTIM.NS", "LTTS.NS", 
    "LUPIN.NS", "M&M.NS", "M&MFIN.NS", "MANAPPURAM.NS", "MARICO.NS", "MARUTI.NS", "MCDOWELL-N.NS", "MCX.NS", 
    "METROPOLIS.NS", "MFSL.NS", "MGL.NS", "MOTHERSON.NS", "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS", 
    "NAUKRI.NS", "NAVINFLUOR.NS", "NESTLEIND.NS", "NMDC.NS", "NTPC.NS", "OBEROIRLTY.NS", "OFSS.NS", "ONGC.NS", 
    "PAGEIND.NS", "PEL.NS", "PERSISTENT.NS", "PETRONET.NS", "PFC.NS", "PIDILITIND.NS", "PIIND.NS", "PNB.NS", 
    "POLYCAB.NS", "POWERGRID.NS", "PVRINOX.NS", "RAMCOCEM.NS", "RBLBANK.NS", "RECLTD.NS", "RELIANCE.NS", "SAIL.NS", 
    "SBICARD.NS", "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS", "SIEMENS.NS", "SRF.NS", "SUNPHARMA.NS", "SUNTV.NS", 
    "SYNGENE.NS", "TATACHEM.NS", "TATACOMM.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", 
    "TCS.NS", "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "ULTRACEMCO.NS", "UPL.NS", 
    "VEDL.NS", "VOLTAS.NS", "WIPRO.NS", "ZEEL.NS", "ZYDUSLIFE.NS"
]

@app.get("/api/swing-scanner")
def run_swing_scanner():
    if "scanner_data" in cache: return cache["scanner_data"]
    try:
        # 🔥 FIX: Sirf 5 din ka data download karo aur sirf 'Close' price uthao
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

        result = {
            "status": "success", "date": date_str, 
            "data": {"top_gainers": gainers, "top_losers": losers}
        }
        cache["scanner_data"] = result
        return result
    except Exception as e:
        return {"status": "error", "message": f"Scanner Error: {str(e)}"}


def get_ai_prediction(symbol, current_price, ind):
    if not ai_model: return "⚠️ AI prediction disabled. API Key not found."
    prompt = f"""
    Act as a Hedge Fund Quant. Analyze daily chart for {symbol} (NSE). Price: ₹{current_price}.
    RSI: {ind['RSI_14']}. 20-EMA: {ind['EMA_20']}. 50-EMA: {ind['EMA_50']}. MACD: {ind['MACD']}.
    Give a sharp, 3-sentence institutional trading prediction. Focus on immediate trend, momentum, and next support/resistance.
    """
    try: return ai_model.generate_content(prompt).text.replace("*", "")
    except: return "⚠️ AI Engine currently busy."

@app.get("/api/analyze/{symbol}")
def analyze_stock(symbol: str):
    yf_symbol = symbol.upper().replace(" ", "").replace(".NS", "") + ".NS"
    cache_key = f"analyze_{yf_symbol}"
    
    if cache_key in cache: return {"status": "success", "data": cache[cache_key]}

    try:
        df = yf.download(yf_symbol, period="1y", interval="1d", progress=False)
        
        # 🔥 FIX: Agar yfinance MultiIndex bhejta hai, toh usko normal columns me convert karo
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty or len(df) < 50: 
            return {"status": "error", "message": f"Data not found for {symbol}. Try valid F&O symbol."}
            
        df = df.ffill().bfill()
        
        # Technicals Calculation
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df = df.fillna(0)
        
        latest = df.iloc[-1]
        current_price = round(float(latest['Close']), 2)
        
        ind_dict = {
            "RSI_14": round(float(latest['RSI_14']), 2),
            "EMA_20": round(float(latest['EMA_20']), 2),
            "EMA_50": round(float(latest['EMA_50']), 2),
            "MACD": round(float(latest['MACD_12_26_9']), 2)
        }

        try:
            ai_commentary = get_ai_prediction(yf_symbol.replace(".NS",""), current_price, ind_dict)
        except:
            ai_commentary = "⚠️ AI Engine busy right now. Please check technicals below."

        chart_data = []
        for date, row in df.tail(150).iterrows():
            chart_data.append({
                "time": str(date.date()), 
                "open": round(float(row['Open']), 2), "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2), "close": round(float(row['Close']), 2),
                "ema20": round(float(row['EMA_20']), 2) if row['EMA_20'] != 0 else None,
                "ema50": round(float(row['EMA_50']), 2) if row['EMA_50'] != 0 else None
            })

        result = {
            "symbol": yf_symbol.replace(".NS", ""),
            "latest_close": current_price,
            "indicators": ind_dict,
            "ai_prediction": ai_commentary,
            "historical_chart_data": chart_data
        }
        
        cache[cache_key] = result
        return {"status": "success", "data": result}
        
    except Exception as e:
        return {"status": "error", "message": f"Backend Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
