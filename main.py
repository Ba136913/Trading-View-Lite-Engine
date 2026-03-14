from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
from groq import Groq
from cachetools import TTLCache
import uvicorn
import math

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
cache = TTLCache(maxsize=500, ttl=300)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

TICKERS = [
    "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENT", "ADANIPORTS", "ALKEM", "AMBUJACEM",
    "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT", "ASTRAL", "ATUL", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO",
    "BAJAJFINSV", "BAJFINANCE", "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "BANKBARODA", "BATAINDIA", "BEL", "BERGEPAINT", "BHARATFORG",
    "BHARTIARTL", "BHEL", "BIOCON", "BOSCHLTD", "BPCL", "BRITANNIA", "BSOFT", "CANBK", "CANFINHOME", "CHAMBLFERT",
    "CHOLAFIN", "CIPLA", "COALINDIA", "COFORGE", "COLPAL", "CONCOR", "COROMANDEL", "CROMPTON", "CUB", "CUMMINSIND",
    "DABUR", "DALBHARAT", "DEEPAKNTR", "DIVISLAB", "DIXON", "DLF", "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND",
    "FEDERALBNK", "GAIL", "GLENMARK", "GMRINFRA", "GNFC", "GODREJCP", "GODREJPROP", "GRANULES", "GRASIM", "GUJGASLTD",
    "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDCOPPER", "HINDPETRO",
    "HINDUNILVR", "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX", "IGL", "INDHOTEL", "INDIACEM",
    "INDIAMART", "INDIGO", "INDUSINDBK", "INFY", "IPCALAB", "IRCTC", "ITC", "JINDALSTEL", "JKCEMENT", "JSWSTEEL",
    "JUBLFOOD", "KOTAKBANK", "LALPATHLAB", "LAURUSLABS", "LICHSGFIN", "LT", "LTIM", "LTTS", "LUPIN", "M&M",
    "M&MFIN", "MANAPPURAM", "MARICO", "MARUTI", "MCX", "METROPOLIS", "MFSL", "MGL", "MOTHERSON", "MPHASIS",
    "MRF", "MUTHOOTFIN", "NATIONALUM", "NAUKRI", "NAVINFLUOR", "NESTLEIND", "NMDC", "NTPC", "OBEROIRLTY", "OFSS",
    "ONGC", "PAGEIND", "PEL", "PERSISTENT", "PETRONET", "PFC", "PIDILITIND", "PIIND", "PNB", "POLYCAB",
    "POWERGRID", "PVRINOX", "RAMCOCEM", "RBLBANK", "RECLTD", "RELIANCE", "SAIL", "SBICARD", "SBILIFE", "SBIN",
    "SHREECEM", "SIEMENS", "SRF", "SUNPHARMA", "SUNTV", "SYNGENE", "TATACHEM", "TATACOMM", "TATACONSUM", "TATAMOTORS",
    "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "TITAN", "TORNTPHARM", "TRENT", "TVSMOTOR", "UBL", "ULTRACEMCO",
    "UPL", "VEDL", "VOLTAS", "WIPRO", "ZOMATO", "ZYDUSLIFE", "SUZLON", "PAYTM", "JIOFIN"
]

class ChatRequest(BaseModel):
    symbol: str = "General"
    timeframe: str = ""
    message: str
    price: float = 0.0
    rsi: float = 0.0
    is_home: bool = False

@app.post("/api/chat")
def chat_with_ai(req: ChatRequest):
    if not groq_client: return {"status": "error", "message": "Groq API Key missing."}
    try:
        # 🔥 FIX: Differentiating between Home Chat and Stock Chat, and asking for DETAILED answers
        if req.is_home:
            system_prompt = "You are a Master Hedge Fund Quant and Trading Mentor. The user is asking general market questions or about trading strategies. Provide detailed, comprehensive, and insightful answers. If asked for live real-time data like today's top gainers, politely inform them you don't have a live internet data feed."
        else:
            system_prompt = f"You are a professional trading assistant advising on {req.symbol} ({req.timeframe} chart). Current price is ₹{req.price}, RSI is {req.rsi}. You specialize in Ichimoku Cloud and Pivot Points. Provide a DETAILED, in-depth technical analysis explaining your rationale step-by-step."
        
        chat = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=800,  # 🔥 FIX: Increased from 100/200 to 800 for long, detailed answers
        )
        return {"status": "success", "reply": chat.choices[0].message.content.replace("*", "")}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/swing-scanner")
def run_swing_scanner():
    return {"status": "success", "data": {"fno_stocks": TICKERS}}

def safe_val(val):
    if pd.isna(val) or math.isnan(val) or val is None: return None
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
        
        if df.empty or 'Close' not in df.columns or df['Close'].dropna().empty: 
            return {"status": "error", "message": f"Market Data unavailable for {symbol}."}

        df = df.dropna(subset=['Close'])

        # PIVOTS
        df_daily.index = df_daily.index.tz_localize(None)
        H, L, C = df_daily['High'], df_daily['Low'], df_daily['Close']
        P_val = (H + L + C) / 3
        R1 = (2 * P_val) - L
        S1 = (2 * P_val) - H
        R2 = P_val + (H - L)
        S2 = P_val - (H - L)
        R3 = R1 + (H - L)
        S3 = S1 - (H - L)
        R4 = R3 + (H - L)
        S4 = S3 - (H - L)
        R5 = R4 + (H - L)
        S5 = S4 - (H - L)

        pivots = pd.DataFrame({'P': P_val, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3, 'R4': R4, 'S4': S4, 'R5': R5, 'S5': S5}).shift(1)
        pivots.index = pivots.index.date
        
        df.index = df.index.tz_localize(None)
        df['date_only'] = df.index.date
        for col in ['P', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3', 'R4', 'S4', 'R5', 'S5']: 
            df[col] = df['date_only'].map(pivots[col])

        # ICHIMOKU
        df.ta.rsi(length=14, append=True)
        high_9 = df['High'].rolling(9).max(); low_9 = df['Low'].rolling(9).min(); df['tenkan'] = (high_9 + low_9) / 2
        high_26 = df['High'].rolling(26).max(); low_26 = df['Low'].rolling(26).min(); df['kijun'] = (high_26 + low_26) / 2
        df['span_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
        high_52 = df['High'].rolling(52).max(); low_52 = df['Low'].rolling(52).min(); df['span_b'] = ((high_52 + low_52) / 2).shift(26)
        df['chikou'] = df['Close'].shift(-26)

        chart_data = []
        for dt, row in df.iterrows():
            unix_t = int(pd.Timestamp(dt).timestamp()) + (5.5 * 3600)
            chart_data.append({
                "time": unix_t, "open": safe_val(row['Open']), "high": safe_val(row['High']), "low": safe_val(row['Low']), "close": safe_val(row['Close']),
                "p": safe_val(row['P']), "r1": safe_val(row['R1']), "s1": safe_val(row['S1']), "r2": safe_val(row['R2']), "s2": safe_val(row['S2']), "r3": safe_val(row['R3']), "s3": safe_val(row['S3']),
                "r4": safe_val(row['R4']), "s4": safe_val(row['S4']), "r5": safe_val(row['R5']), "s5": safe_val(row['S5']),
                "rsi": safe_val(row.get('RSI_14', 50)), "tenkan": safe_val(row['tenkan']), "kijun": safe_val(row['kijun']), "span_a": safe_val(row['span_a']), "span_b": safe_val(row['span_b']), "chikou": safe_val(row['chikou'])
            })

        latest_price = round(float(df.iloc[-1]['Close']), 2)
        ai_commentary = "⚠️ AI Chat Active. Ask about Pivots or Ichimoku Cloud."
        if groq_client:
            try:
                # 🔥 FIX: Asking for detailed analysis here too
                prompt = f"Analyze {timeframe} chart for {symbol} on NSE. Price: ₹{latest_price}. Provide a detailed, multi-sentence technical breakdown."
                chat = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert technical analyst. Provide highly detailed and thorough analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.5,
                    max_tokens=600,
                )
                ai_commentary = chat.choices[0].message.content.replace("*", "")
            except Exception as e:
                ai_commentary = f"⚠️ AI Initial Error: {str(e)[:150]}"

        res = {"status": "success", "data": {"symbol": yf_symbol.replace(".NS", ""), "latest_close": latest_price, "ai_prediction": ai_commentary, "historical_chart_data": chart_data}}
        cache[cache_key] = res
        return res
    except Exception as e: return {"status": "error", "message": str(e)}

if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=10000)
