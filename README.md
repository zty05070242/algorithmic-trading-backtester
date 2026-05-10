# Algorithmic Trading Backtester

A modular Python backtesting framework for daily-bar trend-following strategies,
with an experiment comparing Kalman filtering and wavelet denoising as
signal pre-processing tools.

## Overview

The project has two layers:

1. **The backtesting engine** - a reusable framework for loading data, sizing
   positions, running strategies, and computing performance metrics.
2. **The signal-extraction experiment** — a comparison asking whether
   applying Kalman filtering or wavelet denoising before a crossover strategy
   improves its performance. This experiment is inspired by Stefan Jansen - "Machine Learning for Algorithmic Trading".


The motivation comes from by background as a Tonmeister. Like audio signals, prices contain noise that disrupt the generation of trading signals.
This project aims to build Kalman filter and wavelet denoising into the signal generation step of a backtester.

---

## Architecture

### Data pipeline

```
yfinance  →  data_loader.py  →  DataFrame (OHLCV, DatetimeIndex)
```

[data_loader.py](data_loader.py) — wraps `yfinance.download()`. Flattens
multi-index columns (yfinance ≥ 0.2 behaviour), renames to lowercase
`open/close/high/low/volume`, drops NaNs.

```python
df = load_historical_data("^GSPC", "2015-01-01", "2026-04-15")
```

### Strategy interface

All strategies inherit from
[strategy_folder/_strategy_bass_class.py](strategy_folder/_strategy_bass_class.py):

```python
class Strategy(ABC):
    def set_data(self, data: pd.DataFrame): ...
    def generate_signals(self) -> pd.DataFrame: ...  # must return df with 'signal' column
    def get_signals(self) -> pd.DataFrame: ...
```

`generate_signals()` must return the full DataFrame with at minimum a `signal`
column (`1` = long, `-1` = short, `0` = flat). Strategies may also add a
`stop_loss` column; if present, the backtester uses it instead of bar
low/high.

### Position sizing

[position_sizer.py](position_sizer.py) — fixed fractional risk sizing:

```
units = (account_balance × risk_pct) / |entry_price − stop_loss|
```

Inspired by the 2% rule. Takes `account_balance`, `entry_price`, `stop_loss_price`, `risk_pct`.
Returns a dict with `units_to_trade`, `position_size`, `direction`, and
supporting fields.

### Backtesting engines

Two engines share the same strategy interface and metrics output:

**[backtester.py](backtester.py) — `Backtester` (all-in / all-out)**

- One position at a time: full risk on every signal.
- Entry on the *next bar's open* after the signal bar (removes look-ahead bias).
- Exit on stop-loss hit or opposite signal (at the signal bar's close).
- Reads `stop_loss` column from strategy if present; falls back to bar
  low/high.
- 20× max leverage cap.

**[backtester_scaled.py](backtester_scaled.py) — `BacktesterScaled` (tranche scaling)**

- Up to `max_tranches` (default 3) independent positions per side.
- Each signal opens one new tranche and closes the oldest opposite tranche
  (FIFO scale-in / scale-out).
- Each tranche has its own entry price and stop loss — allows pyramiding into
  trends.
- Risk per tranche = `risk_pct / max_tranches`, so total risk exposure stays
  the same as the all-in engine.

Both engines compute the same metrics: total return, Sharpe ratio, max
drawdown, win rate, profit factor, expectancy, avg win/loss, largest win/loss.

### Charting

[chart.py](chart.py) — `plot_signals(strategy)` renders a Plotly candlestick
chart with buy/sell markers and auto-detected indicator overlays (any column
not in `{open, high, low, close, volume, signal}` is plotted as a line).
Saves to HTML.

---

## Strategies

| File | Class | Type | Description |
|------|-------|------|-------------|
| [ma_cross.py](strategy_folder/ma_cross.py) | `MovingAverageCrossover` | Trend | SMA(fast) / SMA(slow) crossover on raw close |
| [kalman_cross.py](strategy_folder/kalman_cross.py) | `KalmanCrossover` | Trend | Dual Kalman crossover — fast and slow trackers on raw close |
| [kalman_ma_hybrid.py](strategy_folder/kalman_ma_hybrid.py) | `KalmanMAHybrid` | Trend | Kalman fast line vs SMA slow line |
| [wavelet_ma_cross.py](strategy_folder/wavelet_ma_cross.py) | `WaveletMACrossover` | Trend | SMA crossover on wavelet-denoised close |
| [wavelet_kalman_cross.py](strategy_folder/wavelet_kalman_cross.py) | `WaveletKalmanCrossover` | Trend | Dual Kalman crossover on wavelet-denoised close |
| [two_b.py](strategy_folder/two_b.py) | `TwoB` | Reversal | Sperandeo 2B Rule — failed breakout reversal |
| [wavelet_kalman_calibrate.py](wavelet_kalman_calibrate.py) | — | Analysis | Offline spectral calibration: derives Kalman Q from wavelet band energies |

### The 2B Rule (Sperandeo)

Victor Sperandeo's failed-breakout reversal pattern:

- **Short signal**: price breaks above the rolling swing high, then closes back
  below it within `confirmation_days`. ATR and volume filters optional.
- **Long signal**: price breaks below the rolling swing low, then closes back
  above it within `confirmation_days`.

Parameters: `lookback`, `confirmation_days`, `min_breakout_atr`, `volume_factor`.

---

## Signal-Extraction Experiment

### Motivation

Can wavelet denoising or Kalman filtering improve a standard crossover strategy
when applied as a pre-processing step? And do they improve it for the *same*
reason, or different reasons?

### Strategies in Experiment

| # | Label | Fast line | Slow line |
|---|-------|-----------|-----------|
| 1 | `MA/MA` | SMA(20) of close | SMA(50) of close |
| 2 | `Kalman/MA` | Kalman(fast) of close | SMA(50) of close |
| 3 | `Wavelet+MA/MA` | SMA(20) of denoised close | SMA(50) of denoised close |
| 4 | `Wavelet+Kalman` | Kalman(fast) of denoised close | Kalman(slow) of denoised close |

### Wavelet denoising details (with the help from Claude Code)

Implemented in [wavelet_denoiser.py](wavelet_denoiser.py).

**Algorithm:** Donoho-Johnstone universal threshold.
1. Discrete wavelet transform (DWT) decomposes the price into a coarse
   approximation + a pyramid of detail coefficient bands.
2. Estimate noise `sigma` from the finest detail band via MAD:
   `sigma = median(|cD1|) / 0.6745`
3. Apply threshold `sigma * sqrt(2 * log(n)) * threshold_scale` to every
   detail band (soft shrinkage).
4. Inverse DWT reconstructs the cleaned price.

**Why I denoised price directly, not returns:** My original approach denoised returns and
reconstructed price via `cumprod()`. The universal threshold (~2.8% at 252
bars) zeros out almost every daily return, so the reconstructed price drifts
upward in bull markets when compounded. Within a 252-bar rolling window the
price series is approximately locally stationary, so the noise model is still
valid, and denoising price directly avoids the drift.

**`threshold_scale=0.5`:** half the Donoho-Johnstone threshold. The full
universal threshold is designed for worst-case noise recovery in stationary
signals; `0.5` preserves medium-frequency structure (weekly/monthly swings)
while still suppressing fine-scale noise.

**Causal implementation:** `rolling_wavelet_denoise()` slides a 252-bar window
and keeps only the last reconstructed value at each step — no future data ever
feeds into the estimate at time `t`. This avoids lookahead bias and is the same causality guarantee as
`pykalman.KalmanFilter.filter()` (vs `.smooth()`, which is non-causal).

**Honesty check:** wavelet strategies also expose `mode='global'` (full-series
one-shot, uses future data) for comparison. If the global Sharpe is 
higher than the rolling Sharpe, the offline version was cheating. The
comparison harness flags any gap > 0.5 Sharpe.

### Wavelet-derived stop loss

Wavelet strategies set their stop-loss at the edge of the noise band rather
than at the bar low/high:

```
threshold_price = sigma_price * sqrt(2 * log(window)) * threshold_scale
long  SL = min(wavelet_close − threshold_price, bar_low)
short SL = max(wavelet_close + threshold_price, bar_high)
```

The wider of the wavelet SL and the raw bar low/high is always taken — the SL
is never *tighter* than what the bar range alone gives. The backtester reads
the `stop_loss` column directly; non-wavelet strategies fall back to bar
low/high.

---

## Key Findings

Benchmarked on `^GSPC` (S&P 500), 2000–2026, £10,000 initial balance, 2%
risk/trade, 1 bp slippage.

### Finding 1 — Wavelet preprocessing is redundant before MA crossover

| Metric | MA/MA (baseline) | Wavelet+MA/MA |
|--------|:----------------:|:-------------:|
| Total return | **+423%** | −34% |
| Trades | 139 | 125 |
| Win rate | 18% | 17.6% |
| Avg win | **£4,758** | £724 |
| Avg loss | £672 | £188 |

**Root cause — the double low-pass problem.** A moving average and a wavelet
denoiser are both low-pass filters operating on overlapping frequency bands.
Stacking them makes the slow MA track the trend so tightly that any minor
pullback causes the fast MA to cross back below it — generating premature exits.
The baseline holds winners for an average of £4,758; the wavelet version exits
the same trends for only £724. Adjusting to a wider stop-loss didn't fix this: it is bad signal timing, not the risk management.

### Finding 2 — Wavelet preprocessing is complementary before Kalman crossover

| Metric | Kalman crossover | Wavelet+Kalman |
|--------|:----------------:|:--------------:|
| Total return | 312% | 168% |
| Sharpe ratio | 0.19 | **0.26** |
| Max drawdown | −64% | **−30%** |
| Win rate | 8% | **16%** |
| Avg win | £5,645 | £2,785 |
| Avg loss | £151 | £304 |
| Profit factor | 3.24 | 1.76 |
| Trades | 100 | 87 |

Wavelet preprocessing halves the maximum drawdown (−64% → −30%) and improves
Sharpe by 37%, at the cost of roughly half the raw return. The wavelet removes
broadband noise spikes that caused Kalman to fire false crossovers — hence
higher win rate (8% → 16%) and fewer catastrophic losing runs. Raw return falls
because the same smoothing causes slightly later entries and earlier exits.

### Finding 3 — Why the combination matters

The key difference is what the second stage does. A moving average is a
fixed-frequency low-pass filter — it adds nothing the wavelet has not already
done. A Kalman filter is an *adaptive* tracker: its effective bandwidth changes
bar-by-bar based on a variance model and responds to curvature as well as
level. The combination is partially non-redundant — wavelet handles broadband
noise removal, Kalman handles adaptive trend tracking.

**Summary:** wavelet denoising adds value as a front-end to an adaptive tracker
(Kalman), not to a fixed smoother (MA). The improvement is in risk-adjusted
terms (Sharpe, drawdown), not raw return.

### Finding 4 — Wavelet spectral calibration of Kalman Q (academic, offline)

[wavelet_kalman_calibrate.py](wavelet_kalman_calibrate.py) uses a global
(look-ahead) DWT of returns to derive data-driven Q parameters for the Kalman
crossover, rather than hand-tuning them. The method:

1. Decompose returns with a full-depth DWT (`db6`).
2. Compute per-band energy. SNR = band energy / noise floor energy (noise
   estimated from the finest detail band via MAD).
3. Find the two highest-SNR bands (at least one octave apart).
4. Convert each band's centre period `T` to Kalman Q via `Q = 4 / (T² − 1)` —
   derived from the steady-state gain `K = 2/(T+1)` of the 1D random-walk
   Kalman model with R=1.

**Spectral output (`^GSPC`, 2000–2026, db6):**

| Level | Period (bars) | SNR | → Kalman Q |
|------:|:-------------:|----:|----------:|
| 1 | 2–4 | 2.26 | 0.571 |
| 2 | 4–8 | 1.94 | 0.129 |
| 3 | 8–16 | **1.95** | **0.031** |
| 4 | 16–32 | 1.43 | 0.008 |
| 5 | 32–64 | 1.68 | 0.002 |
| 6 | 64–128 | 1.48 | 0.000 |
| 7 | 128–256 | 1.38 | 0.000 |
| 8 | 256–512 | 1.48 | 0.000 |
| 9 | 512–1024 | **4.97** | **0.000008** |

Dominant bands: level 3 (11-bar, SNR=1.95) and level 9 (724-bar, SNR=4.97).
Derived parameters: `fast_cov=0.031`, `slow_cov=0.000008`.

**Comparison vs hand-tuned (`^GSPC` 2000–2026):**

| Metric | Hand-tuned (20/63-bar) | Wavelet-derived (11/724-bar) |
|--------|:---------------------:|:---------------------------:|
| Total return | **312%** | 87% |
| Sharpe ratio | **0.19** | **0.19** |
| Max drawdown | −64% | **−32%** |
| Win rate | 8% | **15%** |
| Trades | 100 | 40 |

**Finding:** the spectral method correctly identifies the two mathematically
dominant signal frequencies — the ~2-year market cycle (level 9, SNR≈5) and a
short-term noise band (level 3). However, *dominant signal ≠ most profitable
to trade*. The 724-bar slow Kalman is so sluggish that it fires only 40 trades
in 26 years, missing most intermediate-term trends. The result is equal
Sharpe but a quarter of the raw return.

The hand-tuned parameters (20/63 bars) land in the intermediate frequency
range (levels 4–5), which the spectral analysis shows as relatively flat in
SNR (1.4–1.7) — not the mathematically dominant bands, but the most
*tradeable* ones given the signal structure.

**Academic caveat:** this calibration uses the full 2000–2026 series, so the
derived Q values carry look-ahead bias and cannot be used for live trading
as-is. For production use, re-calibrate on a rolling training window and
re-apply to the subsequent out-of-sample period.

---

## Results Table

Full multi-ticker results from `python run_comparison.py`. To be filled in after running.

### TICKER

| Strategy | Sharpe | Max DD % | Total Return % | Trades | Win % | PF | Expectancy (£) |
|----------|-------:|---------:|---------------:|-------:|------:|---:|---------------:|
| MA/MA | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Kalman/MA | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Wavelet+MA/MA | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Wavelet+Kalman | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## How to run

```bash
pip install yfinance pandas numpy plotly pykalman PyWavelets

# Run the full 4-strategy comparison (saves CSV + equity HTML per ticker)
python run_comparison.py

# Run a single strategy interactively
python backtester.py

# Visualise wavelet denoising on returns
python wavelet_denoiser.py

# Plot signals for any strategy
python strategy_folder/wavelet_kalman_cross.py
```

---

## Repo layout

```
data_loader.py              yfinance → OHLCV DataFrame
position_sizer.py           fixed-fractional risk sizing
backtester.py               all-in/all-out engine
backtester_scaled.py        tranche scaling engine (pyramid in/out)
chart.py                    Plotly candlestick + auto-overlay
wavelet_denoiser.py         pywt denoising (global + causal rolling)
run_comparison.py           2×2 experiment harness
strategy_folder/
    _strategy_bass_class.py abstract Strategy base class
    ma_cross.py             MovingAverageCrossover
    kalman_cross.py         KalmanCrossover
    kalman_ma_hybrid.py     KalmanMAHybrid (Kalman fast / MA slow)
    wavelet_ma_cross.py     WaveletMACrossover
    wavelet_kalman_cross.py WaveletKalmanCrossover
    two_b.py                TwoB (Sperandeo failed-breakout reversal)
```

---

## Limitations

- **Single parameter set.** Every variant uses one hand-picked configuration —
  no grid search, no walk-forward optimisation. Results are an existence proof,
  not a calibrated production strategy.
- **Wavelet rolling-window cost.** `rolling_wavelet_denoise()` is O(window × N).
  Runs in seconds on a decade of daily data; an intraday version would need a
  faster implementation (e.g. SWT or a fixed-level partial DWT).
- **Transaction costs.** Only per-fill slippage (1 bp each side). No
  commissions, borrow costs for shorts, exchange fees, or overnight fees.
- **Single-asset backtests.** All-in / all-out per ticker. No portfolio
  effects, correlation-aware sizing, or cross-asset allocation.
- **Data source.** `yfinance` daily adjusted closes — adequate for a
  portfolio-piece backtest, not production quality.

---

## Dependencies

```
yfinance
pandas
numpy
plotly
pykalman
PyWavelets   (imported as pywt)
```
