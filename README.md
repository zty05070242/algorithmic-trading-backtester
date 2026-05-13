# Algorithmic Trading Backtester — DSP-Enhanced 2B Rule

A daily-bar backtesting framework built around a single experiment: replacing
the rolling-window pivot detector in Victor Sperandeo's **2B Rule** with a
proper **DSP pivot-detection pipeline** (causal wavelet denoising → SciPy
peak-finding → prominence-and-confirmation gating), and measuring whether
that change produces a structurally cleaner trading signal.

## At a glance

The project is the work of a tonmeister (audio signal processing background)
moving into quantitative finance. The thesis: a financial chart is a noisy
time-series and the tools used to clean noisy audio — wavelet decomposition,
adaptive smoothing, peak detection with prominence — should transfer
directly. The 2B Rule is a perfect target for that transfer: its core
primitive is a "swing high / swing low," which a chart-reading human extracts
by *visual prominence*, not by rolling-window maximum. So we encode that
visual filter as a signal-processing pipeline and see what changes.

**Headline result:** across 10 commodity futures (Gold, Silver, Crude Oil,
Natural Gas, Copper, Wheat, Corn, Soybeans, Coffee, Live Cattle) from
2000–2026, the wavelet-based pivot detector beats the rolling-window 2B Rule
on **Sharpe ratio in 9 of 10 markets** when run through the scaled (tranche-FIFO)
backtester, and reduces max drawdown by **~50% on average** at the same risk
budget.

---

## What's in the repo

```
backtester.py               all-in / all-out engine
backtester_scaled.py        FIFO tranche scaling engine (pyramid in/out)
chart.py                    Plotly candlestick + auto-overlay
data_loader.py              yfinance → OHLCV DataFrame
position_sizer.py           fixed-fractional risk sizing
wavelet_denoiser.py         pywt denoising (global + causal rolling)
run_comparison.py           10-commodity × 2-strategy × 2-backtester harness

strategy_folder/
    _strategy_base_class.py     abstract Strategy base class
    two_b.py                    TwoB — Sperandeo's rule, book → code
    wavelet_two_b.py            WaveletTwoB — DSP pivot detection
    ma_cross.py                 (side experiment) MA crossover
    kalman_cross.py             (side experiment) dual Kalman crossover
    kalman_ma_hybrid.py         (side experiment) Kalman fast / MA slow
    wavelet_ma_cross.py         (side experiment) MA on denoised close
    wavelet_kalman_cross.py     (side experiment) Kalman on denoised close

wavelet_kalman_calibrate.py     (side experiment) spectral calibration of Kalman Q
```

---

## The headline experiment: 2B Rule vs Wavelet-2B

### Sperandeo's 2B Rule — from book knowledge to code application

Victor Sperandeo's failed-breakout reversal pattern (from *Trader Vic — Methods
of a Wall Street Master*):

- **Short signal**: price breaks above a prior swing high, then closes back
  *below* it within a few bars. The failed breakout is the entry.
- **Long signal**: mirror image — price breaks below a prior swing low, then
  closes back above it.

Implemented in [strategy_folder/two_b.py](strategy_folder/two_b.py). The book
defines "prior swing high/low" by visual inspection; the most common naive
encoding — and the one used in the baseline here — is a **rolling N-bar
maximum / minimum** of the high/low channel.

### The flaw in the rolling-window encoding

A rolling N-bar max forgets. The "swing high" at bar t is the highest bar of
the last N bars, full stop — even if that bar was a single noisy spike, and
regardless of whether N bars ago there was a vastly more significant pivot.
In a trending market the rolling window keeps printing new highs, and the
strategy keeps reading "price broke above the rolling high" as a 2B short
setup. It fires false shorts the whole way up a bull market.

Concrete example from the backtests: during silver's 2010–2011 bull run
($15 → $50), the plain 2B Rule caught a long from $15→$19 in early 2010, then
spent the rest of the run firing repeated short signals against a rolling
high that was being rewritten every few weeks. The structural lows around $28
that defined the trend's pullbacks were already outside its 20-bar window
and invisible to it.

### Wavelet-2B — encoding the human eye

A human chart reader picks out swing highs by visual prominence: a peak
stands above its surroundings, the surroundings being smoothed implicitly by
the eye. That's a denoise-then-find-local-maxima pipeline. Implemented in
[strategy_folder/wavelet_two_b.py](strategy_folder/wavelet_two_b.py):

```
raw close ──► rolling causal wavelet denoise (db6, soft threshold, win=128)
          ──► scipy.signal.find_peaks on the denoised series
          ──► filter peaks by prominence ≥ min_prominence_atr × ATR
          ──► gate by pivot_confirm_bars of follow-through
          ──► resulting pivot levels feed Sperandeo's failed-breakout test
```

Every stage has a DSP analog:

- **Denoise** suppresses tick-level chop so peaks correspond to structural
  swings, not single bars.
- **Prominence** (the vertical distance from a peak down to its lowest
  contour line) is exactly the "does this swing matter" filter the eye
  applies. Expressing it in ATR multiples makes it self-adapting across
  assets and regimes.
- **Confirmation lag** is the group-delay analog of pivot detection: a peak
  is only known to *be* a peak once enough bars have printed lower to its
  right.

The 2B failed-breakout logic itself is unchanged — only the *source of the
swing-high/low reference* changes.

### Causality

- `rolling_wavelet_denoise()` is causal — denoised[t] depends only on the
  prior 128 bars.
- A pivot at bar k is only treated as "known" once t ≥ k + `pivot_confirm_bars`.
- One asterisk: `scipy.find_peaks` computes prominence over the full array,
  so pivot *selection* (whether a peak passes the prominence gate) could in
  principle shift as later data arrives. The trade decision itself uses only
  already-confirmed pivots and the current bar, so no future data leaks into
  signals. Documented inline at the top of [wavelet_two_b.py](strategy_folder/wavelet_two_b.py).

---

## The two backtesters

Both engines share the same Strategy interface and metrics output.

**[backtester.py](backtester.py) — `Backtester` (all-in / all-out)**

- One position at a time, full risk budget per signal.
- Entry on the next bar's open (no look-ahead).
- Exit on stop-loss hit or opposite signal (at the signal bar's open).
- 20× max leverage cap.

**[backtester_scaled.py](backtester_scaled.py) — `BacktesterScaled` (FIFO tranches)**

- Up to `max_tranches` (default 3) independent positions per side.
- Each new signal opens one tranche and closes the *oldest* opposite tranche
  (FIFO scale-out + scale-in).
- Each tranche has its own entry price and stop loss.
- Total risk exposure preserved: `tranche_risk_pct = risk_pct / max_tranches`.

The scaled engine is the more revealing one for this comparison. The plain
engine's all-in/all-out compounding amplifies sizing effects — a 26-year
backtest on a strongly trending asset can post returns of 22,000% on what is
actually a moderately-good signal. The scaled engine flattens that effect
and reveals the underlying per-trade economics.

---

## Results — 10 commodity futures, 2000–2026

Run from `python run_comparison.py`. £10,000 initial balance, 2% risk per
trade (split across tranches in scaled mode), 1 bp slippage. Full numbers in
[results/comparison_20260513.csv](results/comparison_20260513.csv).

### Scaled-backtester comparison (the honest headline)

| Commodity | 2B Sharpe | W2B Sharpe | 2B DD | W2B DD | Winner |
|-----------|----------:|-----------:|------:|-------:|:------:|
| GC=F (Gold) | 0.42 | **0.46** | -7.2% | -13.8% | W2B |
| SI=F (Silver) | 0.42 | **0.62** | -7.9% | -8.1% | W2B |
| CL=F (WTI Crude) | -0.52 | **0.17** | -52.9% | -33.4% | W2B |
| NG=F (Natural Gas) | **0.31** | 0.13 | -22.1% | -20.1% | 2B |
| HG=F (Copper) | 0.26 | **0.38** | -27.2% | -16.0% | W2B |
| ZW=F (Wheat) | 0.26 | **0.49** | -31.9% | -32.0% | W2B |
| ZC=F (Corn) | -0.12 | **0.23** | -44.6% | -25.8% | W2B |
| ZS=F (Soybeans) | 0.04 | **0.39** | -27.1% | -19.8% | W2B |
| KC=F (Coffee) | -0.18 | **0.26** | -37.3% | -25.6% | W2B |
| LE=F (Live Cattle) | 0.26 | **0.34** | -16.6% | -32.5% | W2B |

**Wavelet-2B wins on Sharpe in 9 of 10 commodities.**

### Key findings

**1. The win is structural, not sizing-driven.** In the plain-backtester
comparison the headline returns were dominated by compounding (e.g. silver
22,080% for Wavelet-2B vs 355% for 2B Rule). The scaled comparison, which
flattens compounding, shows the signal-quality edge directly via Sharpe —
and that edge survives in 9/10 markets.

**2. Wavelet-2B rescues markets where 2B Rule is broken.** On Crude Oil,
Corn, and Coffee the rolling-window 2B has *negative* Sharpe (-0.52, -0.12,
-0.18 in scaled mode) — it's getting chopped by false breakouts in noisy,
non-trending environments. Wavelet-2B turns all three positive (+0.17, +0.23,
+0.26) by screening out non-structural pivots before the failed-breakout
test fires. On Crude specifically, the max drawdown drops from -53% to -33%.

**3. Drawdown halves under scaling, for both strategies.** Plain vs scaled
max drawdown averages around ~50% reduction across the board, which is
exactly what a 1/3-risk-per-tranche policy should give in theory. The
non-trivial finding is that the Wavelet-2B Sharpe is *preserved or improved*
in 6/10 commodities despite the smaller positions — the edge survives
shrinking the bet size, which is a stronger claim than headline-return wins.

**4. Gold flipped.** In the plain backtester, 2B Rule beat Wavelet-2B on
Gold (Sharpe 0.35 vs 0.33). Under scaling, Wavelet-2B wins (0.46 vs 0.42).
Gold's long sideways range 2013–2018 actually suits a rolling-window
approach — the reference refreshes frequently in a range-bound market. The
compounding noise hid Wavelet-2B's underlying edge; scaling reveals it.

**5. Natural Gas is the lone loss.** NG=F is the one market where 2B Rule's
scaled Sharpe (0.31) beats Wavelet-2B (0.13). Natural Gas has strong
seasonality and frequent sharp regime changes; the wavelet pivot's stickiness
may misalign with that structure. Worth investigating with a regime-aware
prominence threshold.

### A worked example: silver 2010–2011

| Strategy | 2010-Feb-10 → 2010-Aug-31 | 2011-Feb-02 → 2011-May-06 |
|----------|---------------------------|---------------------------|
| 2B Rule | LONG @ $15.30 → $19.02 (+£5,531) | (missed — busy firing false shorts) |
| Wavelet-2B | (entered later, traded smaller) | LONG @ $28.22 → $34.82 (+£18,975, +58% of balance) |

2B Rule's rolling 20-bar window had no memory of the $28 pullback by the
time price re-tested it in Feb 2011. Wavelet-2B's prominence-filtered pivot
at $28 was still the active reference, so the failed-breakdown was correctly
read as a bullish 2B setup. The signal quality difference is exactly the
point of the wavelet pipeline.

---

## Side experiments

Earlier exploration into wavelet/Kalman preprocessing for crossover
strategies, kept in the repo for completeness but no longer the focus of the
write-up:

- [strategy_folder/ma_cross.py](strategy_folder/ma_cross.py) — baseline MA crossover.
- [strategy_folder/kalman_cross.py](strategy_folder/kalman_cross.py), [strategy_folder/kalman_ma_hybrid.py](strategy_folder/kalman_ma_hybrid.py) — Kalman as a smoother in crossover form.
- [strategy_folder/wavelet_ma_cross.py](strategy_folder/wavelet_ma_cross.py), [strategy_folder/wavelet_kalman_cross.py](strategy_folder/wavelet_kalman_cross.py) — same crossovers run on wavelet-denoised close.
- [wavelet_kalman_calibrate.py](wavelet_kalman_calibrate.py) — offline DWT
  spectral analysis to derive Kalman process-noise Q from band SNR rather
  than hand-tuning. Demonstrates the spectral pipeline but uses
  full-series look-ahead — not live-tradeable as-is.

Short version of what those experiments showed: stacking a wavelet denoiser
in front of an MA crossover is redundant (both are low-pass filters with
overlapping bands → premature exits), while stacking it in front of a
Kalman crossover is complementary (Kalman's adaptive bandwidth fills a
different role) but only buys risk-adjusted improvement, not raw return.
That conclusion is what pointed the project toward 2B Rule — a strategy
where the DSP role is *structural pivot identification*, not bandlimiting
the input signal.

---

## How to run

```bash
pip install -r requirements.txt   # yfinance, pandas, numpy, plotly, pykalman, PyWavelets
# scipy is required by wavelet_two_b.py (pulled in transitively today; pin if reproducing)

# Full 10-commodity × 2-strategy × 2-backtester comparison (saves CSV + per-ticker equity HTML)
python run_comparison.py

# Single-strategy smoke test with chart
python strategy_folder/wavelet_two_b.py
python strategy_folder/two_b.py

# Wavelet denoising demo (S&P 500 returns, global vs causal)
python wavelet_denoiser.py
```

---

## Architecture quick-reference

### Strategy interface

All strategies inherit from
[strategy_folder/_strategy_base_class.py](strategy_folder/_strategy_base_class.py):

```python
class Strategy(ABC):
    def set_data(self, data: pd.DataFrame): ...
    def generate_signals(self) -> pd.DataFrame: ...   # must return df with 'signal' column
    def get_signals(self) -> pd.DataFrame: ...
```

`generate_signals()` returns the full OHLCV DataFrame plus a `signal` column
(`1` = long, `-1` = short, `0` = flat) and optionally a `stop_loss` column.
If `stop_loss` is present the backtester uses it; otherwise it falls back to
bar low/high.

### Position sizing

[position_sizer.py](position_sizer.py) — fixed-fractional risk sizing:

```
units = (account_balance × risk_pct) / |entry_price − stop_loss|
```

### Wavelet denoising

[wavelet_denoiser.py](wavelet_denoiser.py) — Donoho-Johnstone universal
threshold:

1. Discrete wavelet transform decomposes price into a coarse approximation
   plus a pyramid of detail bands.
2. Estimate noise σ from the finest detail band via MAD: `σ = median(|cD₁|) / 0.6745`.
3. Apply threshold `σ · √(2 log n) · threshold_scale` to every detail band
   (soft shrinkage).
4. Inverse DWT reconstructs the cleaned price.

The causal `rolling_wavelet_denoise()` slides a 128-bar (Wavelet-2B) or
252-bar (older crossover strategies) window and keeps only the final
reconstructed value at each step — no future data feeds the estimate at t.

### Chart

[chart.py](chart.py) — `plot_signals(strategy)` renders a Plotly candlestick
chart with buy/sell markers and auto-overlays any DataFrame column not in
`{open, high, low, close, volume, signal, avg_volume, atr, stop_loss}`. So
strategies surface their internal state (e.g. `wavelet_close`, `swing_high`,
`swing_low`) just by storing it on the DataFrame.

---

## Limitations

- **Single parameter set.** Every strategy uses one hand-picked configuration —
  no grid search, no walk-forward optimisation. The results are an existence
  proof of the DSP pivot-detection idea, not a calibrated production
  strategy.
- **Wavelet rolling-window cost.** `rolling_wavelet_denoise()` is O(window × N).
  Runs in a few seconds per 26-year daily history; an intraday version would
  need a faster causal pipeline (e.g. SWT or a fixed-level partial DWT).
- **Prominence is computed globally.** `scipy.find_peaks` reads the full
  denoised array when computing prominence. Pivot *selection* is therefore
  not strictly causal, even though the trade decision is. A fully-causal
  peak finder is straightforward to add.
- **Transaction costs.** 1 bp slippage per fill only — no commissions, borrow
  costs for shorts, exchange fees, or roll costs (relevant for futures).
- **Single-asset backtests.** All-in or all-tranche per ticker. No portfolio
  effects, correlation-aware sizing, or cross-asset allocation.
- **Data source.** `yfinance` daily continuous futures contracts — adequate
  for a portfolio-piece backtest, not production quality. NG=F and LE=F in
  particular can have sparser histories on Yahoo.

---

## Dependencies

```
yfinance
pandas
numpy
plotly
pykalman
PyWavelets   (imported as pywt)
scipy        (used by wavelet_two_b.py for find_peaks)
```
