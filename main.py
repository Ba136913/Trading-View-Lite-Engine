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

# F&O STOCKS LIST
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
    if "scanner_data" in cache:
        return cache["scanner_data"]

    try:
        data = yf.download(TICKERS, period="60d", group_by='ticker', threads=True)
        reversals, breakouts, all_performance = [], [], []
        date_str = datetime.date.today().strftime("%Y-%m-%d")

        for ticker in TICKERS:
            try:
                df = data[ticker].dropna()
                if len(df) < 21: continue
                date_str = df.index[-1].strftime("%Y-%m-%d")

                close_today = float(df['Close'].iloc[-1])
                open_today = float(df['Open'].iloc[-1])
                high_today = float(df['High'].iloc[-1])
                low_today = float(df['Low'].iloc[-1])
                close_yest = float(df['Close'].iloc[-2])
                open_yest = float(df['Open'].iloc[-2])
                high_yest = float(df['High'].iloc[-2])
                low_yest = float(df['Low'].iloc[-2])
                
                pct_change = round(((close_today - close_yest) / close_yest) * 100, 2)
                symbol_name = ticker.replace(".NS", "")

                # Performance list for Gainers/Losers
                all_performance.append({"Symbol": symbol_name, "Percent": pct_change, "Price": close_today})

                # Breakout Logic
                past_20_days = df.iloc[-21:-1]
                high_20 = float(past_20_days['High'].max())
                low_20 = float(past_20_days['Low'].min())
                
                if ((high_20 - low_20) / low_20) < 0.10 and close_today > high_20:
                    breakouts.append({"Symbol": symbol_name, "Percent": pct_change, "Signal": "BULL"})
                elif ((high_20 - low_20) / low_20) < 0.10 and close_today < low_20:
                    breakouts.append({"Symbol": symbol_name, "Percent": pct_change, "Signal": "BEAR"})

                # Reversal Logic
                if close_yest < open_yest and close_today > open_today and close_today > high_yest:
                    reversals.append({"Symbol": symbol_name, "Percent": pct_change, "Signal": "BULL"})
                elif close_yest > open_yest and close_today < open_today and close_today < low_yest:
                    reversals.append({"Symbol": symbol_name, "Percent": pct_change, "Signal": "BEAR"})
            except: continue
                
        # Sorting
        reversals.sort(key=lambda x: abs(x['Percent']), reverse=True)
        breakouts.sort(key=lambda x: abs(x['Percent']), reverse=True)
        
        # Gainers & Losers Logic
        gainers = sorted([s for s in all_performance if s['Percent'] > 0], key=lambda x: x['Percent'], reverse=True)[:10]
        losers = sorted([s for s in all_performance if s['Percent'] < 0], key=lambda x: x['Percent'])[:10]

        result = {
            "status": "success", 
            "date": date_str, 
            "data": {
                "top_gainers": gainers,
                "top_losers": losers,
                "reversals": reversals, 
                "breakouts": breakouts
            }
        }
        cache["scanner_data"] = result
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_ai_prediction(symbol, current_price, ind):
    if not ai_model: return "⚠️ AI prediction disabled."
    prompt = f"""
    Act as a Hedge Fund Quant. Analyze daily chart for {symbol} (NSE). Price: ₹{current_price}.
    RSI: {ind['RSI_14']}. 20-EMA: {ind['EMA_20']}. 50-EMA: {ind['EMA_50']}. MACD: {ind['MACD']}.
    Give a sharp, 3-sentence institutional trading prediction. Focus on immediate trend, momentum, and next support/resistance.
    """
    try: return ai_model.generate_content(prompt).text.replace("*", "")
    except: return "⚠️ AI Engine currently busy."

@app.get("/api/analyze/{symbol}")
def analyze_stock(symbol: str):
    # Auto-format smart cleanup
    yf_symbol = symbol.upper().replace(" ", "").replace(".NS", "") + ".NS"

    cache_key = f"analyze_{yf_symbol}"
    if cache_key in cache: return {"status": "success", "data": cache[cache_key]}

    try:
        df = yf.download(yf_symbol, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 50: raise HTTPException(status_code=404, detail="Stock data not found.")
        df = df.ffill().bfill()

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

        ai_commentary = get_ai_prediction(yf_symbol.replace(".NS",""), current_price, ind_dict)

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
        return {"status": "error", "message": "Invalid Stock Symbol or Data Missing."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
