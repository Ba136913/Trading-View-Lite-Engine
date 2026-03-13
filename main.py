from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf # type: ignore
import datetime
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hum abhi testing ke liye kuch top stocks le rahe hain. (Tu isme poora Nifty 200 daal sakta hai)
TICKERS = [
    "INDHOTEL.NS", "PNBHOUSING.NS", "RELIANCE.NS", "TVSMOTOR.NS", "ICICIBANK.NS", 
    "AUROPHARMA.NS", "BHEL.NS", "GAIL.NS", "TATASTEEL.NS", "ONGC.NS", "TATAMOTORS.NS"
]

@app.get("/api/swing-scanner")
def run_swing_scanner():
    try:
        # Pichle 2 mahine ka daily data download kar rahe hain
        data = yf.download(TICKERS, period="2mo", group_by='ticker', threads=True)
        
        reversals = []
        breakouts = []
        today_date = datetime.date.today().strftime("%Y-%m-%d")

        for ticker in TICKERS:
            try:
                # Stock ka data nikalo
                df = data[ticker].dropna()
                if len(df) < 20: continue

                # Prices nikalo
                close_today = df['Close'].iloc[-1]
                open_today = df['Open'].iloc[-1]
                close_yest = df['Close'].iloc[-2]
                open_yest = df['Open'].iloc[-2]
                high_yest = df['High'].iloc[-2]
                
                # Percentage change
                pct_change = round(((close_today - close_yest) / close_yest) * 100, 2)
                symbol_name = ticker.replace(".NS", "")

                # -----------------------------------------
                # 1. CHANNEL BREAKOUT LOGIC (BO)
                # -----------------------------------------
                # Rule: Pichle 20 din ek tight range me tha, aur aaj range tod di
                past_20_days = df.iloc[-21:-1]
                high_20 = past_20_days['High'].max()
                low_20 = past_20_days['Low'].min()
                
                # Agar range 10% ke andar thi (consolidation) aur aaj current price high tod chuka hai
                if ((high_20 - low_20) / low_20) < 0.10 and close_today > high_20:
                    breakouts.append({
                        "Symbol": symbol_name, "Percent": pct_change, "Date": today_date, "Signal": "BULL"
                    })
                # Bearish Breakout (Niche ki taraf toota)
                elif ((high_20 - low_20) / low_20) < 0.10 and close_today < low_20:
                    breakouts.append({
                        "Symbol": symbol_name, "Percent": pct_change, "Date": today_date, "Signal": "BEAR"
                    })

                # -----------------------------------------
                # 2. REVERSAL RADAR LOGIC
                # -----------------------------------------
                # Rule: Pichle din gir raha tha (Red candle), par aaj achi Green candle bani jo pichla High tod de
                if close_yest < open_yest and close_today > open_today and close_today > high_yest:
                    reversals.append({
                        "Symbol": symbol_name, "Percent": pct_change, "Date": today_date, "Signal": "BULL"
                    })
                # Bearish Reversal
                elif close_yest > open_yest and close_today < open_today and close_today < df['Low'].iloc[-2]:
                    reversals.append({
                        "Symbol": symbol_name, "Percent": pct_change, "Date": today_date, "Signal": "BEAR"
                    })

            except Exception as e:
                continue
                
        # Agar market flat hai aur koi signal nahi mila, toh UI dikhane ke liye Dummy data (Sirf design check ke liye)
        if not reversals and not breakouts:
            reversals = [
                {"Symbol": "INDHOTEL", "Percent": 3.02, "Date": today_date, "Signal": "BULL"},
                {"Symbol": "PNBHOUSING", "Percent": 5.00, "Date": today_date, "Signal": "BULL"},
                {"Symbol": "ONGC", "Percent": -2.13, "Date": today_date, "Signal": "BEAR"}
            ]
            breakouts = [
                {"Symbol": "AUROPHARMA", "Percent": 1.20, "Date": today_date, "Signal": "BULL"},
                {"Symbol": "BHEL", "Percent": -5.47, "Date": today_date, "Signal": "BEAR"}
            ]

        return {
            "status": "success",
            "data": {
                "reversals": reversals,
                "breakouts": breakouts
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)