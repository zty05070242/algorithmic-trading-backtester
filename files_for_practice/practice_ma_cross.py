from practice_strategy_base_class import Strategy
from practice_data_loader import load_historical_data

class MovingAverageCrossover(Strategy):
    def __init__(self, fast_period:int=10, slow_period:int=20):
        
        #checks
        if fast_period > slow_period:
            raise ValueError("fast_period must be smaller than slow_period.")
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("they must be positive.")
        #calls parent class
        super().__init__(name = f"MA Crossover {fast_period}/{slow_period}")

        #stores the periods
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self):
        if self.data is None:
            raise ValueError("No data loaded, call set_data() first.")
        df = self.data.copy()
        
        df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()

        df['signal'] = 0.0

        df.loc[(df['fast_ma'] > df['slow_ma']) & (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)), 'signal'] = 1
        df.loc[(df['fast_ma'] < df['slow_ma']) & (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1)), 'signal'] = -1

        df = df.dropna()
        self.data = df
        self._signals_generated = True
        return df
    
if __name__ == "__main__":
    data = load_historical_data("TSLA", "2010-01-01", "2026-04-15")
    strategy = MovingAverageCrossover(10, 20)
    strategy.set_data(data)
    signals = strategy.generate_signals()
    print(f"Loaded {len(signals)} bars.")
    print(signals[['close', 'fast_ma', 'slow_ma', 'signal']].tail(10))
    print(f"BUY signals: {(signals['signal']==1).sum()}")
    print(f"SELL signals: {(signals['signal']==-1).sum()}")

    


