import yfinance as yf
import pandas as pd

def load_historical_data(ticker:str, start_date:str, end_date:str, interval:str='1d') -> pd.DataFrame:
    print(f"Loading historical data for {ticker} from {start_date} to {end_date} interval {interval}")

    historical_data = yf.download(
        tickers = ticker, 
        start = start_date, 
        end = end_date, 
        interval = interval, 
        progress = False, 
        auto_adjust = True
        )

    if historical_data.empty:
        raise ValueError("Unable to load historical data.")
    if isinstance(historical_data.columns, pd.MultiIndex):
        historical_data.columns = historical_data.columns.get_level_values(0)
    
    historical_data = historical_data[["Open", "Close", "High", "Low", "Volume"]]
    historical_data = historical_data.copy()
    historical_data.columns = ["open", "close", "high", "low", "volume"]
    historical_data = historical_data.dropna().sort_index()

    print(f"Successfully loaded {ticker} for {len(historical_data)} bars.")
    return historical_data

if __name__ == "__main__":
    df = load_historical_data("TSLA", "2010-01-01", "2026-04-01")
    print(df.tail(5))
