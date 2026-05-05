import pandas as pd
import numpy as np
from typing import Dict
from misc.fiddle_position_sizer import calculate_position_size

class Backtester():
    def __init__(self, initial_balance:float=1000, risk_pct:float=0.02, slippage_pct:float=0.001):
        self.initial_balance = initial_balance
        self.risk_pct = risk_pct
        self.slippage_pct = slippage_pct
        self._reset()
    
    def _reset(self):
        self.current_balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.current_direction = 0
        self.trade_open = False
        self.trades = []
        self.equity_curve = []
        self.entry_date = None
    
    def run(self, data:pd.DataFrame, strategy, verbose:bool=True) -> dict:
        self._reset()

        strategy.set_data(data)
        df = strategy.generate_signals()

        if verbose:
            print(f"Strategy: {strategy.name}")
            print(f"Initial Balance: {self.initial_balance:.2f}")
            print(f"Risk: {(self.risk_pct * 100):.2f} %")

        pending_signal = 0.0
        pending_sl = 0.0

        for date, row in df.iterrows():
            # EXIT
            # ENTRY
            pending_signal = int(row['signal'])     # Why not df['singal']?
            pending_sl = row['low'] if pending_signal == 1 else row['high']

            self.equity_curve.append({
                "Date": date,
                "Current Balance": self.current_balance
            })
        if self.trade_open:
            final_price = df.iloc[-1]['close']
            pnl = (final_price - self.entry_price) * self.position * self.current_direction
            self.current_balance += pnl
        
        #metrics = self._calculate_metrics()
        #return metrics




if __name__ == "__main__":

    from misc.fiddle_data_loader import load_historical_data
    from misc.fiddle_ma_cross import MovingAverageCrossover

    df = load_historical_data("NVDA", "2020-01-01", "2024-01-01")
    strategy = MovingAverageCrossover(10, 20)
    backtester = Backtester()
    backtester.run(df, strategy)

