import pandas as pd
import numpy as np
from typing import Dict
from misc.fiddle_position_sizer import calculate_position_size

class Backtester():
    def __init__(self, initial_balance, risk_pct, slippage_pct):
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
    
    def run(self, data:pd.DataFrame, strategy, verbose:bool=True) -> Dict:
        self._reset()
        strategy.set_data(data)
        df = strategy.generate_signals()

        if verbose:
            print(f"Account Balance: {self.current_balance:,.2f}")
            print(f"Strategy: {strategy.name}")
            print(f"Risk per trade: {self.risk_pct * 100:.1f}%")
            print(f"Slippage: {self.slippage_pct * 100:.3f}%")

        pending_signal = 0
        pending_sl = 0.0
        pending_exit = False

        has_custom_sl:bool = 'stop_loss' in df.columns

        for date, row in df.iterrows():
            # ============================================================ EXIT RULE ============================================================
            if self.trade_open:
                # ================== Is stop loss hit? ==================
                sl_hit:bool = (self.current_direction == 1 and row['low'] <= self.stop_loss) or (self.current_direction == -1 and row['high'] >= self.stop_loss)
                
                if sl_hit:
                    exit_price = self.stop_loss * (1 - self.slippage_pct) if self.current_direction == 1 else self.stop_loss * (1 + self.slippage_pct)
                elif pending_exit:
                    exit_price = row['open'] * (1 - self.slippage_pct) if self.current_direction == 1 else row['open'] * (1 + self.slippage_pct)
                else:
                    exit_price = None
                
                if exit_price is not None:
                    pnl = (exit_price - self.entry_price) * self.position * self.current_direction
                    pnl_pct = (pnl / self.current_balance) * 100
                    self.current_balance += pnl


                    self.trades.append({
                        "Entry_Date": self.entry_date,
                        "Exit_Date": date,
                        "Direction": "LONG" if self.current_direction == 1 else "SHORT",
                        "Entry_Price": self.entry_price,
                        "Exit_Price": exit_price,
                        "PnL": pnl,
                        "PnL_pct": f"{pnl_pct}%"
                    })

                    if verbose:
                        direction_label = "LONG" if self.current_direction == 1 else "SHORT"
                        exit_reason = "SL HIT" if sl_hit else "SIGNAL"
                        print(f"Balance: £{self.current_balance:,.2f} | CLOSED {direction_label} on {date.date()} | {self.position:.4f} units @ £{exit_price:.2f} | PnL: £{pnl:.2f} ({pnl_pct:.2f}%) | {exit_reason}")

                    self.trade_open = False
                    self.position = 0.0
                    self.current_direction = 0
                # ================== If Opposite Signal / Pending Exit == True ==================
                if self.trade_open:
                    opposite_signal:bool = (self.current_direction == 1 and row['signal'] == -1) or (self.current_direction == -1 and row['signal'] == 1)
                    pending_exit = opposite_signal
                else:
                    pending_exit = False
            else: 
                pending_exit = False
                    

            # ============================================================ ENTRY RULE ============================================================
            if not self.trade_open and pending_signal != 0 and self.current_balance > 0:
                '''
                Setting local variables for temporary calculation. Then will store them back to self:
                direction, entry_price, stop_loss
                '''
                direction = pending_signal

                entry_price = row['open'] * (1 + self.slippage_pct) if direction == 1 else row['open'] * (1 - self.slippage_pct)

                stop_loss = pending_sl

                # SKip if stop loss = entry
                if stop_loss == entry_price:
                    pending_signal = 0
                    self.equity_curve.append({
                        "Date": date,
                        "Balance": self.current_balance
                    })
                    continue

                # Position Sizing: we calculate position size using those local variables
                sizing = calculate_position_size(
                    account_balance = self.current_balance,
                    entry_price = entry_price,
                    stop_loss_price = stop_loss,
                    risk_pct = self.risk_pct
                )
                max_units = (self.current_balance * 20) / entry_price  # 20x leverage cap
                self.position = min(sizing['units_to_trade'], max_units)

                self.entry_price = entry_price
                self.stop_loss = stop_loss
                self.entry_date = date
                self.trade_open = True
                self.current_direction = direction

                if verbose:
                    direction_label = sizing['direction'].upper()
                    print(f"Balance: £{self.current_balance:,.2f} | OPENED {direction_label} on {date.date()} | {self.position:.4f} units @ £{entry_price:.2f} | SL: £{stop_loss:.2f}")

            # ============================================================ END ============================================================
            pending_signal = int(row['signal'])
            if pending_signal == 1:
                pending_sl = row['stop_loss'] if has_custom_sl and pd.notna(row['stop_loss']) else row['low']
            elif pending_signal == -1:
                pending_sl = row['stop_loss'] if has_custom_sl and pd.notna(row['stop_loss']) else row['high']
            
            self.equity_curve.append({
                "Date": date,
                "Balance": self.current_balance
            })
        # ========================================== Force liquidate any remaining positions at the end of data ==========================================
        if self.trade_open:
            final_price = df.iloc[-1]['close']
            pnl = (final_price - self.entry_price) * self.position * self.current_direction
            self.current_balance += pnl
            self.equity_curve.append({
                "Date": df.index[-1],
                "Balance": self.current_balance
            })
            if verbose:
                print(f"\nForce-closed open position at end of data | PnL: £{pnl:.2f}")

        # ====================================================== Metrics Calculation ======================================================
        metrics = self._calculate_metrics()
        if verbose:
            print(f"Final Balance:  ${metrics['Final_Balance']:,.2f}")
            print(f"Return:          {metrics['Total_Return_pct']:,.2f}%")
            print(f"Trade Count:     {metrics['Trade_Count']:,}")
            print(f"Win Rate:        {metrics['Win_Rate']:.2f}%")
            print(f"Sharpe Ratio:    {metrics['Sharpe']:.2f}")
            print(f"Max Drawdown:    {metrics['Max_Drawdown']:.2f}%")
            print(f"Profit Factor:   {metrics['Profit_Factor']:.2f}")
            print(f"Avg Win:        ${metrics['Avg_Win']:,.2f}")
            print(f"Avg Loss:       ${metrics['Avg_Loss']:,.2f}")
            print(f"Expectancy:     ${metrics['Expectancy']:,.2f}")
            print(f"Largest Win:    ${metrics['Largest_Win']:,.2f}")
            print(f"Largest Loss:    ${metrics['Largest_Loss']:,.2f}")
            
            


        return metrics
        
    def _calculate_metrics(self):
        
        # 1. Total Return % 
        total_return_pct = (self.current_balance - self.initial_balance) / self.initial_balance * 100

        # 2. Trade counts and win rate 
        num_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['PnL'] > 0]
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0

        # 3. Sharpe ratio
        equity_df = pd.DataFrame(self.equity_curve)
        daily_returns = equity_df['Balance'].pct_change().dropna() 
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # 4. Max drawdown
        rolling_max = equity_df['Balance'].cummax()
        drawdown = (equity_df['Balance'] - rolling_max) / rolling_max
        max_drawdown_pct = drawdown.min() * 100

        # 5. Profit factor
        gross_profit = sum(t['PnL'] for t in self.trades if t['PnL'] > 0)
        gross_loss = abs(sum(t['PnL'] for t in self.trades if t['PnL'] < 0))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')

        # 6. Average win & average loss
        losing_trades = [t for t in self.trades if t['PnL'] < 0]
        avg_win = round(gross_profit / len(winning_trades), 2) if winning_trades else 0
        avg_loss = round(gross_loss / len(losing_trades), 2) if losing_trades else 0

        # 7. Expectancy
        expectancy = round(sum(t['PnL'] for t in self.trades) / num_trades, 2) if num_trades > 0 else 0

        # 8. Largest win / lose
        largest_win = round(max((t['PnL'] for t in self.trades), default=0), 2)
        largest_loss = round(min((t['PnL'] for t in self.trades), default=0), 2)


        return{
            'Final_Balance': self.current_balance,
            'Total_Return_pct': total_return_pct,
            'Trade_Count': num_trades,
            'Win_Rate': win_rate,
            'Sharpe': sharpe,
            'Max_Drawdown': max_drawdown_pct,
            'Profit_Factor': profit_factor,
            'Avg_Win': avg_win,
            'Avg_Loss': avg_loss,
            'Expectancy': expectancy,
            'Largest_Win': largest_win,
            'Largest_Loss': largest_loss,
            'Trades': self.trades,
            'Equity_Curve': self.equity_curve
        }




        
    









































if __name__ == "__main__":

    from misc.fiddle_data_loader import load_historical_data
    from misc.fiddle_ma_cross import MovingAverageCrossover

    df = load_historical_data("NVDA", "2020-01-01", "2024-01-01")
    strategy = MovingAverageCrossover(10, 20)
    backtester = Backtester(1000, 0.02, 0.001)
    backtester.run(df, strategy)

