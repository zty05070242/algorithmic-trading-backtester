import numpy as np
import pandas as pd
from strategy_base_class import Strategy


class TwoB(Strategy):
    """
    2B Reversal (Victor Sperandeo):
    Finds the most recent confirmed swing low/high within the lookback window.
    - 2B Bottom: today's low strictly breaks below the prior swing low,
                 but the candle closes back above it → BUY
    - 2B Top:    today's high strictly breaks above the prior swing high,
                 but the candle closes back below it → SELL

    Parameters
    ----------
    lookback : int
        How many bars back to search for the most recent swing point.
        Use a large value (e.g. 200) so meaningful levels are never missed.
    swing_n : int
        How many bars on each side a bar must be the extreme of to qualify
        as a swing point (e.g. 5 means lower than the 5 bars before and after).
    """
    def __init__(self, lookback: int = 200, swing_n: int = 5, min_retracement: float = 0.05):
        if lookback <= 0:
            raise ValueError("lookback must be a positive integer.")
        if swing_n <= 0:
            raise ValueError("swing_n must be a positive integer.")
        if not (0 <= min_retracement < 1):
            raise ValueError("min_retracement must be between 0 and 1.")

        super().__init__(name=f"2B Reversal (lookback={lookback}, swing_n={swing_n}, min_ret={min_retracement:.0%})")
        self.lookback = lookback
        self.swing_n = swing_n
        self.min_retracement = min_retracement

    def generate_signals(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded. Call set_data() first.")

        df = self.data.copy()
        n = self.swing_n

        # A bar is a swing low if its low is the minimum of the [i-n, i+n] window.
        # center=True means the window is centred on bar i, so it looks n bars
        # forward — that look-ahead is neutralised by the shift(n) below.
        window = 2 * n + 1
        swing_low_mask  = df['low']  == df['low'].rolling(window=window, center=True).min()
        swing_high_mask = df['high'] == df['high'].rolling(window=window, center=True).max()

        # Retracement filter: the move into the swing must be >= min_retracement.
        # For a swing low:  (surrounding peak high - swing low) / surrounding peak high >= min_retracement
        # For a swing high: (swing high - surrounding trough low) / swing high >= min_retracement
        surrounding_high = df['high'].rolling(window=window, center=True).max()
        surrounding_low  = df['low'].rolling(window=window,  center=True).min()

        swing_low_mask  = swing_low_mask  & ((surrounding_high - df['low'])  / surrounding_high >= self.min_retracement)
        swing_high_mask = swing_high_mask & ((df['high'] - surrounding_low)  / df['high']        >= self.min_retracement)

        # Keep only the price value at confirmed swing bars; NaN everywhere else.
        # shift(n) removes the look-ahead: at bar i we only see swings confirmed
        # by at least n subsequent bars.
        swing_low_vals  = df['low'].where(swing_low_mask).shift(n)
        swing_high_vals = df['high'].where(swing_high_mask).shift(n)

        # Within the lookback window, pick the most recent confirmed swing level.
        def last_valid(x):
            valid = x[~np.isnan(x)]
            return valid[-1] if len(valid) > 0 else np.nan

        prior_swing_low  = swing_low_vals.rolling(window=self.lookback, min_periods=1).apply(last_valid, raw=True)
        prior_swing_high = swing_high_vals.rolling(window=self.lookback, min_periods=1).apply(last_valid, raw=True)

        # 2B Bottom: today's low strictly breaks below the prior swing low,
        #            but the candle closes back above it.
        buy  = (df['low']  < prior_swing_low)  & (df['close'] > prior_swing_low)

        # 2B Top: today's high strictly breaks above the prior swing high,
        #         but the candle closes back below it.
        sell = (df['high'] > prior_swing_high) & (df['close'] < prior_swing_high)

        df['signal'] = 0.0
        df.loc[buy,  'signal'] = 1
        df.loc[sell, 'signal'] = -1

        df = df.dropna()
        self.data = df
        self._signals_generated = True

        return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data_loader import load_historical_data

    df = load_historical_data("NVDA", "2000-01-01", "2026-04-09")

    strategy = TwoB(lookback=200, swing_n=5, min_retracement=0.05)
    strategy.set_data(df)
    signals = strategy.generate_signals()

    print(f"Strategy: {strategy.name}")
    print(f"Bars: {len(signals):,}")
    print(signals[['close', 'signal']].tail(5))
    print(f"Buy signals:  {(signals['signal'] == 1).sum()}")
    print(f"Sell signals: {(signals['signal'] == -1).sum()}")
