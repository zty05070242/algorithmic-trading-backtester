import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from ._strategy_bass_class import Strategy
except ImportError:
    from _strategy_bass_class import Strategy

from data_loader import load_historical_data
from wavelet_denoiser import wavelet_denoise, rolling_wavelet_denoise
from pykalman import KalmanFilter
import pandas as pd


class WaveletKalmanCrossover(Strategy):
    """
    Kalman crossover on wavelet-denoised close prices.

    Same denoise-returns-then-recompound trick as WaveletMACrossover, but the
    two crossover lines are Kalman filters (fast and slow) on the cleaned close,
    not moving averages.

    Reason to stack wavelet + Kalman: the wavelet step removes bursty, broadband
    high-frequency noise in the return series, and the Kalman filter then
    adaptively tracks the remaining trend. Think of it as a high-SNR front-end
    feeding a tracker — analogous to denoising audio before pitch-tracking.
    """

    def __init__(
        self,
        fast_cov: float = 0.01,
        slow_cov: float = 0.001,
        wavelet: str = "db6",
        mode: str = "rolling",
        rolling_window: int = 252,
    ):
        if fast_cov <= slow_cov:
            raise ValueError(
                "fast_cov must be larger than slow_cov "
                "(higher covariance = more responsive filter)."
            )
        if mode not in ("rolling", "global"):
            raise ValueError("mode must be 'rolling' or 'global'.")

        super().__init__(
            name=f"Wavelet({mode})+Kalman Crossover {fast_cov}/{slow_cov}"
        )
        self.fast_cov = fast_cov
        self.slow_cov = slow_cov
        self.wavelet = wavelet
        self.mode = mode
        self.rolling_window = rolling_window

    def _kalman_smooth(self, series: pd.Series, transition_covariance: float) -> pd.Series:
        """Causal 1D Kalman filter — no look-ahead (uses .filter(), not .smooth())."""
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=series.iloc[0],
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=transition_covariance,
        )
        state_means, _ = kf.filter(series.values)
        return pd.Series(state_means.flatten(), index=series.index)

    def generate_signals(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded, call set_data() first.")

        df = self.data.copy()

        # --- Step 1: returns ---
        # We denoise returns (stationary) rather than price (non-stationary).
        returns = df['close'].pct_change()
        returns_clean_input = returns.dropna()

        # --- Step 2: denoise returns ---
        if self.mode == "rolling":
            denoised_returns = rolling_wavelet_denoise(
                returns_clean_input,
                window=self.rolling_window,
                wavelet=self.wavelet,
                mode="soft",
            )
        else:
            denoised_returns = wavelet_denoise(
                returns_clean_input,
                wavelet=self.wavelet,
                mode="soft",
            )

        denoised_returns = denoised_returns.reindex(df.index)

        # --- Step 3: reconstruct cleaned close by cumulative compounding ---
        first_valid = denoised_returns.first_valid_index()
        if first_valid is None:
            raise ValueError(
                "Not enough data for wavelet denoising. "
                f"Need at least {self.rolling_window} bars."
            )

        anchor_close = df.loc[first_valid, 'close']
        cumulative = (1.0 + denoised_returns.loc[first_valid:].fillna(0.0)).cumprod()
        cleaned_close = anchor_close * cumulative

        df['wavelet_close'] = cleaned_close
        df = df.dropna(subset=['wavelet_close']).copy()

        # --- Step 4: Kalman crossover on cleaned close ---
        df['kalman_fast'] = self._kalman_smooth(df['wavelet_close'], self.fast_cov)
        df['kalman_slow'] = self._kalman_smooth(df['wavelet_close'], self.slow_cov)
        df['signal'] = 0.0

        # Same cross logic as KalmanCrossover
        df.loc[
            (df['kalman_fast'] > df['kalman_slow'])
            & (df['kalman_fast'].shift(1) <= df['kalman_slow'].shift(1)),
            'signal'
        ] = 1
        df.loc[
            (df['kalman_fast'] < df['kalman_slow'])
            & (df['kalman_fast'].shift(1) >= df['kalman_slow'].shift(1)),
            'signal'
        ] = -1

        df = df.dropna()
        self.data = df
        self._signals_generated = True

        return df


if __name__ == "__main__":
    strategy = WaveletKalmanCrossover(fast_cov=0.01, slow_cov=0.001, mode="rolling")
    print(f"Testing strategy {strategy.name}")
    df = load_historical_data("^GSPC", "2010-01-01", "2026-04-15")
    strategy.set_data(df)
    signals = strategy.generate_signals()
    print(f"Bars: {len(signals)}")
    print(signals[['close', 'wavelet_close', 'kalman_fast', 'kalman_slow', 'signal']].tail(10))
    print(f"BUY  signals: {(signals['signal']==1).sum()}")
    print(f"SELL signals: {(signals['signal']==-1).sum()}")
