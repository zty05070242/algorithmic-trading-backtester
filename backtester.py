import pandas as pd
import numpy as np                          # numpy lets us do maths on arrays efficiently
from typing import Dict
from position_sizer import calculate_position_size


class Backtester:
    """
    Main backtesting engine.
    Entries on signal, exits on opposite signal.
    Position sizing via position_sizer().
    """

    def __init__(self, initial_balance: float = 6000.0, risk_pct: float = 0.02,
                 slippage_pct: float = 0.0):
        self.initial_balance = initial_balance
        self.risk_pct = risk_pct
        self.slippage_pct = slippage_pct    # combined spread + slippage cost per fill
        self._reset()                       # Use a reset method so run() can call it cleanly

    def _reset(self):
        """Reset all state — called before each backtest run."""
        self.current_balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.entry_date = None
        self.trade_open = False
        self.current_direction = 0          # 1 = long, -1 = short
        self.trades = []
        self.equity_curve = []

    def run(self, data: pd.DataFrame, strategy, verbose: bool = True) -> Dict:
        """
        Run the backtest on a dataset using a given strategy.

        Args:
            data: OHLCV DataFrame from data_loader.
            strategy: Any Strategy subclass instance.
            verbose: If True, prints trade-by-trade output.

        Returns:
            Dictionary of performance metrics and trade log.
        """
        self._reset()

        strategy.set_data(data)
        df = strategy.generate_signals()

        if verbose:
            print(f"Strategy       : {strategy.name}")
            print(f"Initial balance: £{self.initial_balance:,.2f}")
            print(f"Risk per trade : {self.risk_pct * 100:.1f}%")
            print(f"Slippage       : {self.slippage_pct * 100:.3f}%\n")

        # pending_signal holds the signal from the previous bar,
        # so we enter on the NEXT bar's open instead of the signal bar's close.
        # This removes look-ahead bias: you see the close, decide to trade,
        # and the earliest you can realistically execute is the next bar's open.
        pending_signal = 0
        pending_sl = 0.0                    # signal bar's low (long) or high (short)

        for date, row in df.iterrows():

            # === EXIT: stop loss hit or opposite signal ===
            if self.trade_open:
                sl_hit = (
                    (self.current_direction == 1 and row['low'] <= self.stop_loss) or
                    (self.current_direction == -1 and row['high'] >= self.stop_loss)
                )
                opposite_signal = (
                    (self.current_direction == 1 and row['signal'] == -1) or
                    (self.current_direction == -1 and row['signal'] == 1)
                )
                if sl_hit:
                    # Apply slippage: SL fill is worse than the SL level
                    if self.current_direction == 1:
                        exit_price = self.stop_loss * (1 - self.slippage_pct)  # long SL slips lower
                    else:
                        exit_price = self.stop_loss * (1 + self.slippage_pct)  # short SL slips higher
                elif opposite_signal:
                    # Exit at next bar's open would add complexity; for exits we use
                    # the close with slippage as a reasonable approximation — you see
                    # the signal forming and can place a market-on-close order.
                    if self.current_direction == 1:
                        exit_price = row['close'] * (1 - self.slippage_pct)  # selling: slips lower
                    else:
                        exit_price = row['close'] * (1 + self.slippage_pct)  # covering short: slips higher
                else:
                    exit_price = None

                if exit_price is not None:

                    # Multiply by direction: long profits when price rises, short when it falls
                    pnl = (exit_price - self.entry_price) * self.position * self.current_direction
                    pnl_pct = (pnl / self.current_balance) * 100   # % of balance AT entry, not initial

                    self.current_balance += pnl

                    self.trades.append({
                        'entry_date': self.entry_date,
                        'exit_date': date,
                        'direction': "long" if self.current_direction == 1 else "short",
                        'entry_price': self.entry_price,
                        'exit_price': round(exit_price, 2),
                        'position_size': self.position,
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl_pct, 2)
                    })

                    if verbose:
                        direction_label = "LONG" if self.current_direction == 1 else "SHORT"
                        exit_reason = "SL HIT" if sl_hit else "SIGNAL"
                        print(f"Balance: £{self.current_balance:,.2f} | CLOSED {direction_label} on {date.date()} | "
                              f"{self.position:.4f} units @ £{exit_price:.2f} | PnL: £{pnl:.2f} ({pnl_pct:.2f}%) | {exit_reason}")

                    self.trade_open = False
                    self.position = 0.0
                    self.current_direction = 0

            # === ENTRY: execute pending signal from previous bar at today's open ===
            if not self.trade_open and pending_signal != 0 and self.current_balance > 0:
                direction = pending_signal

                # Entry at this bar's open, with slippage making the fill worse
                if direction == 1:
                    entry_price = row['open'] * (1 + self.slippage_pct)   # buying: slips higher
                else:
                    entry_price = row['open'] * (1 - self.slippage_pct)   # shorting: slips lower

                # Stop loss: from the SIGNAL bar (t), not the entry bar (t+1)
                # We already know this level when we decide to trade
                stop_loss = pending_sl

                # Skip trade if entry price equals stop loss — no room for a valid SL
                if stop_loss == entry_price:
                    pending_signal = 0
                    self.equity_curve.append({'date': date, 'balance': self.current_balance})
                    continue

                sizing = calculate_position_size(
                    account_balance=self.current_balance,
                    risk_pct=self.risk_pct,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss
                )

                max_units = (self.current_balance * 20) / entry_price  # 20x max leverage
                self.position = min(sizing['units_to_trade'], max_units)
                self.entry_price = entry_price
                self.stop_loss = sizing['stop_loss']
                self.entry_date = date
                self.trade_open = True
                self.current_direction = direction

                if verbose:
                    direction_label = sizing['direction'].upper()
                    print(f"Balance: £{self.current_balance:,.2f} | OPENED {direction_label} on {date.date()} | "
                          f"{self.position:.4f} units @ £{entry_price:.2f} | SL: £{stop_loss:.2f}")

            # Capture this bar's signal and SL level for execution on the NEXT bar
            pending_signal = int(row['signal'])
            # SL = this bar's low for longs, high for shorts (known at close)
            if pending_signal == 1:
                pending_sl = row['low']
            elif pending_signal == -1:
                pending_sl = row['high']

            # === Record equity AFTER processing this bar ===
            self.equity_curve.append({
                'date': date,
                'balance': self.current_balance
            })

        # === Close any open position at end of data ===
        if self.trade_open:
            final_price = df.iloc[-1]['close']
            pnl = (final_price - self.entry_price) * self.position * self.current_direction
            self.current_balance += pnl
            if verbose:
                print(f"\nForce-closed open position at end of data | PnL: £{pnl:.2f}")

        # === Calculate performance metrics ===
        metrics = self._calculate_metrics()

        if verbose:
            print(f"\n{'='*40}")
            print(f"Final balance  : £{metrics['final_balance']:,.2f}")
            print(f"Total return   : {metrics['total_return_pct']:.2f}%")
            print(f"Total trades   : {metrics['num_trades']}")
            print(f"Win rate       : {metrics['win_rate_pct']:.1f}%")
            print(f"Sharpe ratio   : {metrics['sharpe_ratio']:.2f}")
            print(f"Max drawdown   : {metrics['max_drawdown_pct']:.2f}%")

        return metrics

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics from completed trades and equity curve."""

        total_return_pct = (
            (self.current_balance - self.initial_balance) / self.initial_balance
        ) * 100

        num_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]  # list of trades where pnl > 0
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0

        # Sharpe ratio: average daily return divided by std deviation of daily returns
        # Annualised by multiplying by sqrt(252) — 252 trading days in a year
        equity_df = pd.DataFrame(self.equity_curve)         # converts equity curve list into a DataFrame
        daily_returns = equity_df['balance'].pct_change().dropna()  # pct_change() calculates % change between each row

        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown: largest peak-to-trough fall in balance
        rolling_max = equity_df['balance'].cummax()         # cummax() = running maximum up to each point
        drawdown = (equity_df['balance'] - rolling_max) / rolling_max
        max_drawdown_pct = drawdown.min() * 100             # most negative value = worst drawdown

        return {
            'final_balance': self.current_balance,
            'total_return_pct': total_return_pct,
            'num_trades': num_trades,
            'win_rate_pct': round(win_rate * 100, 1),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }


if __name__ == "__main__":
    from data_loader import load_historical_data
    from strategy_folder.ma_cross import MovingAverageCrossover

    df = load_historical_data("^GSPC", "2000-01-01", "2026-04-09")
    strategy = MovingAverageCrossover(fast_period=10, slow_period=20)
    backtester = Backtester(initial_balance=10000, risk_pct=0.02, slippage_pct=0.0001)
    results = backtester.run(df, strategy, verbose=True)

    # --- Interactive Chart ---
    from chart import plot_signals
    plot_signals(strategy)