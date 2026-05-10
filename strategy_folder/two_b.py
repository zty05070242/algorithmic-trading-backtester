from strategy_folder._strategy_base_class import Strategy
from data_loader import load_historical_data
import pandas as pd

class TwoB(Strategy):
    def __init__(self, lookback: int = 20, confirmation_days: int = 3,
                 atr_period: int = 14, min_breakout_atr: float = 0.3,
                 volume_factor: float = 1.0):
        """
        lookback: number of bars to look back when finding swing highs / swing lows.
        confirmation_days: how many bars after the breakout the price must reverse
                          back through the prior extreme for the signal to be valid (1-3 per Sperandeo).
        atr_period: period for ATR calculation (used for noise filtering).
        min_breakout_atr: minimum breakout distance as a multiple of ATR.
                         e.g. 0.3 means price must exceed the swing extreme by at least 0.3 * ATR.
                         Set to 0 to disable.
        volume_factor: breakout bar volume must be >= volume_factor * average volume.
                       e.g. 1.0 means at least average volume. Set to 0 to disable.
        """
        if lookback < 3:
            raise ValueError("lookback must be at least 3.")
        if confirmation_days < 1 or confirmation_days > 5:
            raise ValueError("confirmation_days must be between 1 and 5.")

        super().__init__(name=f"2B Rule (lookback={lookback}, confirm={confirmation_days})")
        self.lookback = lookback
        self.confirmation_days = confirmation_days
        self.atr_period = atr_period
        self.min_breakout_atr = min_breakout_atr
        self.volume_factor = volume_factor

    def generate_signals(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded, call set_data() first.")

        df = self.data.copy()
        df['signal'] = 0.0

        # Pre-compute rolling swing highs and swing lows over the lookback window.
        # The swing high/low represents the prior extreme that price must break and then fail to hold.
        df['swing_high'] = df['high'].rolling(window=self.lookback).max()
        df['swing_low'] = df['low'].rolling(window=self.lookback).min()

        # ATR for noise filtering: a breakout must exceed the swing extreme by min_breakout_atr * ATR
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.atr_period).mean()

        # Average volume for volume filtering
        df['avg_volume'] = df['volume'].rolling(window=self.lookback).mean()

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values
        atrs = df['atr'].values
        avg_volumes = df['avg_volume'].values
        signals = df['signal'].values.copy()
        n = len(df)

        for i in range(self.lookback, n):
            atr = atrs[i]
            min_distance = self.min_breakout_atr * atr if atr > 0 else 0

            # ---- Bearish 2B (short signal) ----
            prior_swing_high = df['high'].iloc[i - self.lookback:i].max()
            breakout_distance = highs[i] - prior_swing_high
            if breakout_distance > min_distance:
                # Volume check: breakout bar must have above-average volume
                if self.volume_factor <= 0 or volumes[i] >= self.volume_factor * avg_volumes[i]:
                    end = min(i + self.confirmation_days + 1, n)
                    for j in range(i, end):
                        if closes[j] < prior_swing_high:
                            signals[j] = -1.0
                            break

            # ---- Bullish 2B (long signal) ----
            prior_swing_low = df['low'].iloc[i - self.lookback:i].min()
            breakout_distance = prior_swing_low - lows[i]
            if breakout_distance > min_distance:
                if self.volume_factor <= 0 or volumes[i] >= self.volume_factor * avg_volumes[i]:
                    end = min(i + self.confirmation_days + 1, n)
                    for j in range(i, end):
                        if closes[j] > prior_swing_low:
                            if signals[j] == 0.0:
                                signals[j] = 1.0
                            break

        df['signal'] = signals
        df = df.dropna()
        self.data = df
        self._signals_generated = True

        return df

if __name__ == "__main__":
    strategy = TwoB(lookback=20, confirmation_days=3)
    print(f"Testing strategy: {strategy.name}")
    df = load_historical_data("NVDA", "2010-01-01", "2026-04-12")
    strategy.set_data(df)
    signals = strategy.generate_signals()
    print(f"Bars: {len(signals)}")
    print(signals[['close', 'high', 'low', 'swing_high', 'swing_low', 'signal']].tail(100))
    print(f"LONG signals:  {(signals['signal'] == 1).sum()}")
    print(f"SHORT signals: {(signals['signal'] == -1).sum()}")

# ----------------------------------------------------------Strategy Logic Flow--------------------------------------------------------
# Victor Sperandeo's 2B Rule — a failed-breakout reversal pattern.
#
# BEARISH 2B (short):
#   1. Identify the swing high over the lookback window (the prior high extreme).
#   2. Price breaks ABOVE that swing high (a new high is made).
#   3. Within confirmation_days, price closes BACK BELOW the prior swing high.
#   4. → Short signal on the bar that closes below.
#
# BULLISH 2B (long):
#   1. Identify the swing low over the lookback window (the prior low extreme).
#   2. Price breaks BELOW that swing low (a new low is made).
#   3. Within confirmation_days, price closes BACK ABOVE the prior swing low.
#   4. → Long signal on the bar that closes above.
