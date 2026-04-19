# Signal Extraction for Trend-Following Strategies: Kalman Filtering vs Wavelet Denoising

A modular algorithmic trading backtester in Python, used as the testbed for a
2×2 comparison of signal-extraction techniques applied to simple trend-following
strategies on equity and commodity markets.

## Motivation

I'm a third-year BMus Tonmeister (audio engineering / DSP) student self-teaching
quantitative finance. Price series and audio signals sit on opposite sides of an
engineering desk but share the same problem: a latent signal of interest buried
in additive noise, and the need to recover the signal *causally* — without
peeking at future samples.

This repo transplants two standard DSP tools — **Kalman filtering** and
**wavelet denoising** — into the generation step of a trend-following
backtester, and asks whether either pre-processing step meaningfully improves
a vanilla moving-average crossover baseline on daily S&P 500 and silver data.

## Methodology

Four strategy variants are benchmarked through the same backtester, on the
same data, with the same risk sizing and slippage assumptions:

| # | Label            | Fast line              | Slow line                |
|---|------------------|------------------------|--------------------------|
| 1 | `MA/MA`          | SMA(20) of close       | SMA(50) of close         |
| 2 | `Kalman/MA`      | Kalman filter of close | SMA(50) of close         |
| 3 | `Wavelet+MA/MA`  | SMA(20) of cleaned close | SMA(50) of cleaned close |
| 4 | `Wavelet+Kalman` | Fast Kalman of cleaned close | Slow Kalman of cleaned close |

**Wavelet pre-processing:** the cleaned close is produced by denoising *returns*
(not price), then reconstructing a price path by cumulative compounding —
`cleaned_close = close[0] * (1 + cleaned_returns).cumprod()`. Returns are
approximately stationary, so the universal-threshold noise model used by the
wavelet denoiser is well-posed on them; price itself is non-stationary.

**Denoising details:** `db6` wavelet, soft thresholding, universal threshold
`sigma * sqrt(2 * log(n))` with `sigma` estimated from the finest detail band
via median absolute deviation (the Donoho-Johnstone recipe). See
[wavelet_denoiser.py](wavelet_denoiser.py).

**Trading assumptions:** £10,000 starting balance, 2% risk per trade, 0.0001
(1 bp) slippage on each fill, entries on next bar's open, exits on same-bar
close or stop-loss, 20× max leverage cap. Implemented in
[backtester.py](backtester.py).

## Causal implementations

Both smoothers use strictly past-and-current data at every step — same guarantee
as `pykalman.KalmanFilter.filter()`, which is what the Kalman strategies use
(not `.smooth()`, which looks both directions).

- **Kalman:** `KalmanFilter.filter()` in
  [strategy_folder/kalman_ma_hybrid.py](strategy_folder/kalman_ma_hybrid.py),
  [strategy_folder/kalman_cross.py](strategy_folder/kalman_cross.py),
  [strategy_folder/wavelet_kalman_cross.py](strategy_folder/wavelet_kalman_cross.py).
- **Wavelet:** `rolling_wavelet_denoise()` in
  [wavelet_denoiser.py](wavelet_denoiser.py) — at each bar `t`, denoise the
  prior `window` samples (default 252) and take only the last reconstructed
  value. No future samples ever feed into the estimate at `t`.

For sanity-checking, the wavelet strategies also expose `mode='global'` which
uses the full-series one-shot denoise (which *does* use future data). The
comparison harness runs both modes and flags any case where the look-ahead
version's Sharpe is materially higher — that would confirm the offline
version was cheating.

## Results

Produced by `python run_comparison.py`. Tickers: `^GSPC` (S&P 500) and `SI=F`
(silver futures), start 2015-01-01.

### S&P 500 (^GSPC)

| Strategy         | Sharpe | Max DD % | Total Return % | Trades | Win % | PF | Expectancy (£) |
|------------------|-------:|---------:|---------------:|-------:|------:|---:|---------------:|
| MA/MA            |  TBD   |   TBD    |      TBD       |  TBD   |  TBD  | TBD|      TBD       |
| Kalman/MA        |  TBD   |   TBD    |      TBD       |  TBD   |  TBD  | TBD|      TBD       |
| Wavelet+MA/MA    |  TBD   |   TBD    |      TBD       |  TBD   |  TBD  | TBD|      TBD       |
| Wavelet+Kalman   |  TBD   |   TBD    |      TBD       |  TBD   |  TBD  | TBD|      TBD       |

### Silver Futures (SI=F)

| Strategy         | Sharpe | Max DD % | Total Return % | Trades | Win % | PF | Expectancy (£) |
|------------------|-------:|---------:|---------------:|-------:|------:|---:|---------------:|
| MA/MA            |  TBD   |   TBD    |      TBD       |  TBD   |  TBD  | TBD|      TBD       |
| Kalman/MA        |  TBD   |   TBD    |      TBD       |  TBD   |  TBD  | TBD|      TBD       |
| Wavelet+MA/MA    |  TBD   |   TBD    |      TBD       |  TBD   |  TBD  | TBD|      TBD       |
| Wavelet+Kalman   |  TBD   |   TBD    |      TBD       |  TBD   |  TBD  | TBD|      TBD       |

### Honesty check (rolling vs global wavelet Sharpe)

Fill in after running. The closer these are, the less the global version was
flattering itself with future data.

## Limitations

- **Wavelet rolling-window cost.** `rolling_wavelet_denoise()` is O(window × N)
  — for a 252-bar window this runs in seconds on a decade of daily data, but
  an intraday version would need a faster implementation (or a different
  transform, e.g. SWT).
- **Single parameter set per method.** Every variant is run with one hand-picked
  configuration — `MA(20/50)`, Kalman `fast_cov=0.01`, `db6` wavelet with
  universal-threshold soft shrinkage. No grid search, no walk-forward
  optimisation. Results are an existence proof, not a calibrated production
  strategy.
- **Transaction costs.** Only per-fill slippage (1 bp each side) is modelled.
  No commissions, no borrow costs for shorts, no exchange fees, no overnight
  financing.
- **Single-asset backtests.** Each strategy trades one ticker at a time, with
  all-in / all-out sizing. No portfolio effects, no correlation-aware sizing.
- **Data source.** `yfinance` daily adjusted closes. Fine for a portfolio-piece
  backtest, not the quality you'd run a real strategy on.

## Repo layout

```
data_loader.py                      yfinance -> OHLCV DataFrame
position_sizer.py                   risk-based unit sizing
backtester.py                       basic (all-in/all-out) engine
backtester_scaled.py                tranche-based scaling variant (not used here)
chart.py                            plotly candle chart with auto-overlay
wavelet_denoiser.py                 pywt denoising utilities (global + rolling-causal)
compare_smoothers.py                visual smoother comparison
run_comparison.py                   main deliverable — runs the 2x2 grid
strategy_folder/
    _strategy_bass_class.py         abstract Strategy base class
    ma_cross.py                     MovingAverageCrossover
    kalman_cross.py                 KalmanCrossover
    kalman_ma_hybrid.py             KalmanMAHybrid
    wavelet_ma_cross.py             WaveletMACrossover
    wavelet_kalman_cross.py         WaveletKalmanCrossover
    two_b.py                        TwoB (Sperandeo reversal)
```

## How to run

```
pip install yfinance pandas numpy plotly pykalman PyWavelets
python run_comparison.py
```

Outputs:
- `results/comparison_YYYYMMDD.csv` — full results table.
- `results/equity_<ticker>.html` — equity curves overlaid per ticker.

## Dependencies

No `requirements.txt` is committed. The four non-standard packages used are:

- `yfinance`
- `pandas`, `numpy`
- `plotly`
- `pykalman`
- `PyWavelets` (imported as `pywt`) — **added for this experiment.**
