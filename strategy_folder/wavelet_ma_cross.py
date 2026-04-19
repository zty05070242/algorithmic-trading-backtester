import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from ._strategy_bass_class import Strategy
except ImportError:
    from _strategy_bass_class import Strategy

from data_loader import load_historical_data
from wavelet_denoiser import wavelet_denoise, rolling_wavelet_denoise
import pandas as pd


class WaveletMACrossover(Strategy):
    """
    MA crossover on wavelet-denoised close prices.

    Pipeline:
      1. Compute daily returns from the raw close. Returns are (approximately)
         stationary — the universal-threshold noise model assumes that, so we
         do NOT denoise price directly.
      2. Denoise the returns (rolling-causal by default, or global look-ahead
         if mode='global' for honesty-check comparisons).
      3. Reconstruct a cleaned close price by cumulative compounding:
             cleaned_close[t] = close[0] * prod_{k<=t} (1 + cleaned_return[k])
      4. Run the same fast/slow MA crossover logic as MovingAverageCrossover,
         but on cleaned_close instead of raw close.

    `mode='rolling'` is the realistic live version. `mode='global'` is for
    measuring how much the look-ahead version flatters the result.
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        wavelet: str = "db6",
        mode: str = "rolling",
        rolling_window: int = 252,
    ):
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("fast_period and slow_period must be positive.")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be shorter than slow_period.")
        if mode not in ("rolling", "global"):
            raise ValueError("mode must be 'rolling' or 'global'.")

        super().__init__(
            name=f"Wavelet({mode})+MA Crossover {fast_period}/{slow_period}"
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.wavelet = wavelet
        self.mode = mode
        self.rolling_window = rolling_window

    def generate_signals(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded, call set_data() first.")

        df = self.data.copy()

        # --- Step 1: returns from raw close ---
        # pct_change() gives (close[t] - close[t-1]) / close[t-1], i.e. daily return.
        returns = df['close'].pct_change()

        # --- Step 2: denoise returns ---
        # Drop the leading NaN for denoising, then re-align below.
        returns_clean_input = returns.dropna()
        if self.mode == "rolling":
            # Causal: only past data used at each t. Live-tradeable.
            denoised_returns = rolling_wavelet_denoise(
                returns_clean_input,
                window=self.rolling_window,
                wavelet=self.wavelet,
                mode="soft",
            )
        else:
            # Global: uses future data. For look-ahead comparison only.
            denoised_returns = wavelet_denoise(
                returns_clean_input,
                wavelet=self.wavelet,
                mode="soft",
            )

        # Re-align to df.index — rolling mode will leave the warm-up period as NaN.
        denoised_returns = denoised_returns.reindex(df.index)

        # --- Step 3: rebuild a cleaned close by compounding cleaned returns ---
        # Only start compounding once the rolling window is warm (first non-NaN).
        # Before that, cleaned_close is undefined and we'll drop those rows later.
        first_valid = denoised_returns.first_valid_index()
        if first_valid is None:
            raise ValueError(
                "Not enough data for wavelet denoising. "
                f"Need at least {self.rolling_window} bars."
            )

        # Use the raw close at first_valid as the anchor. From there, compound
        # (1 + cleaned_return). fillna(0) so the anchor bar contributes no change.
        anchor_close = df.loc[first_valid, 'close']
        cumulative = (1.0 + denoised_returns.loc[first_valid:].fillna(0.0)).cumprod()
        cleaned_close = anchor_close * cumulative

        df['wavelet_close'] = cleaned_close
        # Drop rows before the cleaned series starts — the crossover logic needs
        # a valid cleaned_close at every row it acts on.
        df = df.dropna(subset=['wavelet_close']).copy()

        # --- Step 4: MA crossover on cleaned_close ---
        df['fast_ma'] = df['wavelet_close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['wavelet_close'].rolling(window=self.slow_period).mean()
        df['signal'] = 0.0

        # Same cross logic as MovingAverageCrossover
        df.loc[
            (df['fast_ma'] > df['slow_ma'])
            & (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)),
            'signal'
        ] = 1
        df.loc[
            (df['fast_ma'] < df['slow_ma'])
            & (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1)),
            'signal'
        ] = -1

        df = df.dropna()
        self.data = df
        self._signals_generated = True

        return df


if __name__ == "__main__":
    strategy = WaveletMACrossover(fast_period=20, slow_period=50, mode="rolling")
    print(f"Testing strategy {strategy.name}")
    df = load_historical_data("^GSPC", "2010-01-01", "2026-04-15")
    strategy.set_data(df)
    signals = strategy.generate_signals()
    print(f"Bars: {len(signals)}")
    print(signals[['close', 'wavelet_close', 'fast_ma', 'slow_ma', 'signal']].tail(10))
    print(f"BUY  signals: {(signals['signal']==1).sum()}")
    print(f"SELL signals: {(signals['signal']==-1).sum()}")
