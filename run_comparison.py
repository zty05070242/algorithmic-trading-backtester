"""
run_comparison.py — main deliverable for the Kalman vs Wavelet experiment.

For each ticker, runs four strategy variants through the basic Backtester and
prints a per-ticker results table. Also saves:
  - results/comparison_YYYYMMDD.csv — full table across every (ticker, strategy).
  - results/equity_<ticker>.html    — equity curves overlaid for all 4 strategies.

Extra: an "honesty check" re-runs the wavelet strategies in global (look-ahead)
mode and flags any case where the look-ahead version dramatically beats the
causal rolling version — that would mean the wavelet was cheating.

Usage: python run_comparison.py
"""

from __future__ import annotations
import os
from datetime import datetime, date
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go

from data_loader import load_historical_data
from backtester import Backtester

from strategy_folder.ma_cross import MovingAverageCrossover
from strategy_folder.kalman_ma_hybrid import KalmanMAHybrid
from strategy_folder.wavelet_ma_cross import WaveletMACrossover
from strategy_folder.wavelet_kalman_cross import WaveletKalmanCrossover


# --- Configuration ----------------------------------------------------------

DEFAULT_TICKERS = ['^GSPC', 'SI=F']              # S&P 500 and silver futures
START_DATE      = '2015-01-01'
END_DATE        = '2026-04-15'

INITIAL_BALANCE = 10000.0
RISK_PCT        = 0.02
SLIPPAGE_PCT    = 0.0001

RESULTS_DIR     = 'results'

# Strategy params. Kept in one place so the README can reference them.
MA_FAST, MA_SLOW     = 20, 50
KMA_FAST_COV         = 0.01
KMA_SLOW_PERIOD      = 50
WK_FAST_COV          = 0.01      # matches KalmanMAHybrid's fast_cov
WK_SLOW_COV          = 0.001     # matches KalmanCrossover __main__ default

# Short, stable labels — used as both the table row name and the equity plot legend.
LBL_MA         = 'MA/MA'
LBL_KMA        = 'Kalman/MA'
LBL_WMA        = 'Wavelet+MA/MA'
LBL_WKA        = 'Wavelet+Kalman'


def _build_strategies(mode: str = 'rolling') -> List[Tuple[str, object]]:
    """
    Build the four strategy variants for one comparison pass.
    `mode` only affects the two wavelet strategies; the non-wavelet ones are
    constructed identically both times.
    """
    return [
        (LBL_MA,  MovingAverageCrossover(fast_period=MA_FAST, slow_period=MA_SLOW)),
        (LBL_KMA, KalmanMAHybrid(fast_cov=KMA_FAST_COV, slow_period=KMA_SLOW_PERIOD)),
        (LBL_WMA, WaveletMACrossover(fast_period=MA_FAST, slow_period=MA_SLOW, mode=mode)),
        (LBL_WKA, WaveletKalmanCrossover(fast_cov=WK_FAST_COV, slow_cov=WK_SLOW_COV, mode=mode)),
    ]


def _run_one(df: pd.DataFrame, strategy) -> dict:
    """Run a single strategy on df and return the metrics dict from Backtester."""
    bt = Backtester(
        initial_balance=INITIAL_BALANCE,
        risk_pct=RISK_PCT,
        slippage_pct=SLIPPAGE_PCT,
    )
    return bt.run(df, strategy, verbose=False)


def _metrics_row(ticker: str, label: str, metrics: dict) -> dict:
    """Flatten the metrics dict to the subset we want in the summary table."""
    return {
        'ticker':            ticker,
        'strategy':          label,
        'sharpe_ratio':      metrics['sharpe_ratio'],
        'max_drawdown_pct':  metrics['max_drawdown_pct'],
        'total_return_pct':  round(metrics['total_return_pct'], 2),
        'num_trades':        metrics['num_trades'],
        'win_rate_pct':      metrics['win_rate_pct'],
        'profit_factor':     metrics['profit_factor'],
        'expectancy':        metrics['expectancy'],
    }


def _plot_equity_curves(ticker: str, curves: List[Tuple[str, list]], save_path: str):
    """
    Overlay equity curves for all strategies on one figure.
    `curves` is a list of (label, equity_curve_list) tuples, where each entry
    in equity_curve_list is a {'date': ..., 'balance': ...} dict.
    """
    fig = go.Figure()
    for label, eq in curves:
        if not eq:
            continue
        eq_df = pd.DataFrame(eq)
        fig.add_trace(go.Scatter(
            x=eq_df['date'], y=eq_df['balance'],
            name=label, mode='lines', line=dict(width=1.5),
        ))
    fig.update_layout(
        title=f'Equity curves — {ticker} (initial £{INITIAL_BALANCE:,.0f})',
        xaxis_title='Date', yaxis_title='Balance (£)',
        hovermode='x unified',
    )
    fig.write_html(save_path)
    print(f"  saved equity plot -> {save_path}")


def run_comparison(tickers: List[str] = None,
                   start: str = START_DATE,
                   end: str = END_DATE) -> pd.DataFrame:
    """Main entry point. Returns the assembled results DataFrame."""
    tickers = tickers or DEFAULT_TICKERS
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    honesty_notes = []  # collect messages for the final "honesty check" summary

    for ticker in tickers:
        print(f"\n=== {ticker} ===")
        df = load_historical_data(ticker, start, end)

        # --- Rolling (causal) pass: the headline numbers ---
        rolling_results = {}           # label -> metrics dict (kept for equity plots)
        for label, strat in _build_strategies(mode='rolling'):
            print(f"  running: {label}")
            metrics = _run_one(df, strat)
            rolling_results[label] = metrics
            rows.append(_metrics_row(ticker, label, metrics))

        # --- Equity plot from the rolling pass ---
        equity_curves = [(lbl, rolling_results[lbl]['equity_curve'])
                         for lbl in [LBL_MA, LBL_KMA, LBL_WMA, LBL_WKA]]
        _plot_equity_curves(
            ticker,
            equity_curves,
            os.path.join(RESULTS_DIR, f"equity_{ticker.replace('^', '').replace('=', '_')}.html"),
        )

        # --- Honesty check: global (look-ahead) vs rolling (causal) ---
        # Re-run just the wavelet strategies in global mode and compare Sharpe.
        # If global is dramatically better, the look-ahead was inflating results.
        print(f"  honesty check (global vs rolling wavelet)...")
        for label_rolling, strat_global in [
            (LBL_WMA, WaveletMACrossover(fast_period=MA_FAST, slow_period=MA_SLOW, mode='global')),
            (LBL_WKA, WaveletKalmanCrossover(fast_cov=WK_FAST_COV, slow_cov=WK_SLOW_COV, mode='global')),
        ]:
            global_metrics = _run_one(df, strat_global)
            rolling_sharpe = rolling_results[label_rolling]['sharpe_ratio']
            global_sharpe  = global_metrics['sharpe_ratio']
            diff = global_sharpe - rolling_sharpe

            # Threshold is arbitrary but reasonable: 0.5 on annualised Sharpe is
            # a material gap. If global is +0.5 Sharpe above rolling, flag it.
            verdict = "OK" if diff < 0.5 else "LOOK-AHEAD INFLATION"
            honesty_notes.append(
                f"{ticker:<6} {label_rolling:<16} rolling Sharpe={rolling_sharpe:>5.2f}  "
                f"global Sharpe={global_sharpe:>5.2f}  diff={diff:+.2f}  [{verdict}]"
            )

    results_df = pd.DataFrame(rows)

    # --- Print per-ticker tables ---
    print("\n" + "=" * 72)
    print("RESULTS (rolling, causal mode)")
    print("=" * 72)
    for ticker in tickers:
        sub = results_df[results_df['ticker'] == ticker].drop(columns='ticker')
        print(f"\n{ticker}:")
        print(sub.to_string(index=False))

    # --- Print honesty-check block ---
    print("\n" + "=" * 72)
    print("HONESTY CHECK — did wavelet look-ahead inflate results?")
    print("=" * 72)
    print("If 'global' Sharpe is much higher than 'rolling' Sharpe, the offline")
    print("wavelet was cheating with future data. Rolling is the realistic number.")
    print()
    for note in honesty_notes:
        print("  " + note)

    # --- Save CSV ---
    stamp = date.today().strftime('%Y%m%d')
    csv_path = os.path.join(RESULTS_DIR, f'comparison_{stamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    return results_df


if __name__ == "__main__":
    run_comparison()
