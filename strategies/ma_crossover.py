import pandas as pd
from strategy_base_class import Strategy


class MovingAverageCrossover(Strategy):
    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("fast_period and slow_period must be positive integers.")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period.")

        super().__init__(name=f"MA Crossover {fast_period}/{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded. Call set_data() first.")

        df = self.data.copy()

        df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()

        df['signal'] = 0.0

        # BUY signal: fast MA crosses above slow MA
        df.loc[
            (df['fast_ma'] > df['slow_ma']) &
            (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)),
            'signal'
        ] = 1

        # SELL signal: fast MA crosses below slow MA
        df.loc[
            (df['fast_ma'] < df['slow_ma']) &
            (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1)),
            'signal'
        ] = -1

        df = df.dropna()
        self.data = df
        self._signals_generated = True

        return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data_loader import load_historical_data

    df = load_historical_data("NVDA", "2000-01-01", "2026-04-09")

    strategy = MovingAverageCrossover(fast_period=10, slow_period=20)
    strategy.set_data(df)
    signals = strategy.generate_signals()

    print(f"Strategy: {strategy.name}")
    print(f"Bars: {len(signals):,}")
    print(signals[['close', 'fast_ma', 'slow_ma', 'signal']].tail(5))
    print(f"Buy signals:  {(signals['signal'] == 1).sum()}")
    print(f"Sell signals: {(signals['signal'] == -1).sum()}")
