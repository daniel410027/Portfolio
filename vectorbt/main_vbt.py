"""
main_vbt.py
===========
Pipeline entry point — vectorbt / yfinance edition.

Usage:
    python main_vbt.py

Author: Daniel Huang
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from preprocess_vbt import VbtPreprocessConfig, VbtDataPreprocessor
from model_vbt      import WalkForwardConfig, WalkForwardTrainer


# ============================================================
#  RUN CONFIG
# ============================================================

class RunConfig:
    # ── Steps ────────────────────────────────────────────────
    RUN_PREPROCESS = False   # EDA (distribution plots + labels.csv)
    RUN_ML         = True    # Walk-forward LightGBM training

    # ── EDA range (used when RUN_PREPROCESS=True) ────────────
    START_YEAR = 2014
    END_YEAR   = 2016

    # ── Universe ─────────────────────────────────────────────
    # "twse"   → auto-fetch from opendata.twse.com.tw
    # "custom" → use CUSTOM_TICKERS below
    UNIVERSE_MODE   = "twse"
    CUSTOM_TICKERS  = []     # e.g. ["2330.TW", "2317.TW", "2454.TW"]

    # ── Market index ─────────────────────────────────────────
    MARKET_TICKER = "^TWII"

    # ── Cache ─────────────────────────────────────────────────
    CACHE_DIR = Path("database/cache")
    USE_CACHE = True         # set False to force re-download

    # ── Label params ─────────────────────────────────────────
    RETURN_CLIP     = 0.15
    TICK_THRESHOLD  = 0.01
    TICK0_THRESHOLD = 0.00
    BETA_WINDOW     = 60

    # ── ML range ─────────────────────────────────────────────
    ML_WINDOW_START = 2014
    ML_WINDOW_END   = 2025   # last window test year = ML_WINDOW_END - 1

    # ── Output ───────────────────────────────────────────────
    OUTPUT_BASE_DIR = Path("database/processed")

    # Label used for EDA distribution plots
    LABEL = "excess_return_tick"


# ============================================================
#  Output path
# ============================================================

def get_output_dir(cfg: RunConfig) -> Path:
    return cfg.OUTPUT_BASE_DIR / f"{cfg.START_YEAR}_{cfg.END_YEAR}"


# ============================================================
#  EDA helpers
# ============================================================

def save_distribution_plots(df: pd.DataFrame, output_dir: Path):
    PLOT_COLS = {
        "return":             "continuous",
        "excess_return":      "continuous",
        "return_tick":        "discrete",
        "excess_return_tick": "discrete",
    }
    for col, kind in PLOT_COLS.items():
        if col not in df.columns:
            print(f"  ⚠ Column '{col}' not found — skipped")
            continue
        series = df[col].dropna()
        fig, ax = plt.subplots(figsize=(8, 4))

        if kind == "continuous":
            ax.hist(series, bins=100, color="#2c7bb6", edgecolor="none", alpha=0.85)
            ax.axvline(series.mean(),   color="#d7191c", linewidth=1.2,
                       label=f"mean={series.mean():.4f}")
            ax.axvline(series.median(), color="#fdae61", linewidth=1.2, linestyle="--",
                       label=f"median={series.median():.4f}")
            ax.legend(fontsize=9)
        else:
            counts = series.value_counts().sort_index()
            labels = [str(int(v)) for v in counts.index]
            colors = ["#d7191c" if v == 1 else "#2c7bb6" for v in counts.index]
            bars   = ax.bar(labels, counts.values, color=colors, edgecolor="none", alpha=0.85)
            total  = counts.sum()
            for bar, val in zip(bars, counts.values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.005,
                    f"{val:,}\n({val/total:.1%})",
                    ha="center", va="bottom", fontsize=9,
                )

        ax.set_title(f"Distribution of {col}  (n={len(series):,})", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        out_path = output_dir / f"{col}_dist.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  ✓ Plot: {out_path}")


def save_label_csv(df: pd.DataFrame, output_dir: Path):
    cols    = ["ticker", "date", "return", "excess_return", "beta"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  ⚠ Missing columns {missing} — CSV skipped")
        return
    out_path = output_dir / "labels.csv"
    df[cols].to_csv(out_path, index=False)
    print(f"  ✓ CSV: {out_path}  ({len(df):,} rows)")


# ============================================================
#  Step functions
# ============================================================

def _make_preprocess_cfg(cfg: RunConfig, start_year: int, end_year: int) -> VbtPreprocessConfig:
    return VbtPreprocessConfig(
        start_year      = start_year,
        end_year        = end_year,
        market_ticker   = cfg.MARKET_TICKER,
        universe_mode   = cfg.UNIVERSE_MODE,
        custom_tickers  = list(cfg.CUSTOM_TICKERS),
        cache_dir       = cfg.CACHE_DIR,
        use_cache       = cfg.USE_CACHE,
        return_clip     = cfg.RETURN_CLIP,
        tick_threshold  = cfg.TICK_THRESHOLD,
        tick0_threshold = cfg.TICK0_THRESHOLD,
        beta_window     = cfg.BETA_WINDOW,
    )


def run_preprocess(cfg: RunConfig) -> pd.DataFrame:
    """EDA: run preprocessor, save plots + labels.csv."""
    preprocess_cfg = _make_preprocess_cfg(cfg, cfg.START_YEAR, cfg.END_YEAR)
    df = VbtDataPreprocessor(preprocess_cfg).run()

    if cfg.LABEL not in df.columns:
        print(f"\n  ✗ Label '{cfg.LABEL}' not found — check preprocessing.")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print("  DataFrame Head (5)")
    print(f"{'─'*60}")
    print(df.head(5).to_string())
    print(f"\n{'─'*60}")
    print("  DataFrame Tail (5)")
    print(f"{'─'*60}")
    print(df.tail(5).to_string())

    output_dir = get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n► Output: {output_dir}")
    save_distribution_plots(df, output_dir)
    save_label_csv(df, output_dir)
    return df


def run_ml(cfg: RunConfig, precomputed_df: pd.DataFrame | None = None):
    """Walk-forward ML training."""
    ml_config = WalkForwardConfig(
        window_start    = cfg.ML_WINDOW_START,
        window_end      = cfg.ML_WINDOW_END,
        market_ticker   = cfg.MARKET_TICKER,
        universe_mode   = cfg.UNIVERSE_MODE,
        custom_tickers  = list(cfg.CUSTOM_TICKERS),
        cache_dir       = cfg.CACHE_DIR,
        use_cache       = cfg.USE_CACHE,
        return_clip     = cfg.RETURN_CLIP,
        tick_threshold  = cfg.TICK_THRESHOLD,
        tick0_threshold = cfg.TICK0_THRESHOLD,
        beta_window     = cfg.BETA_WINDOW,
        use_vol_weight  = False,
    )
    trainer = WalkForwardTrainer(ml_config)

    # Inject EDA df as cache for first window (avoid duplicate preprocessing)
    if precomputed_df is not None:
        first_end = cfg.ML_WINDOW_START + 2
        if cfg.START_YEAR == cfg.ML_WINDOW_START and cfg.END_YEAR == first_end:
            trainer.inject_cache(cfg.ML_WINDOW_START, first_end, precomputed_df)
        else:
            print(
                f"  ⚠ EDA range ({cfg.START_YEAR}–{cfg.END_YEAR}) ≠ "
                f"first ML window ({cfg.ML_WINDOW_START}–{first_end}) — cache not injected"
            )
    trainer.run()


# ============================================================
#  MAIN
# ============================================================

def main():
    cfg = RunConfig()

    steps = [
        ("Preprocess EDA", cfg.RUN_PREPROCESS, run_preprocess),
        ("ML Training",    cfg.RUN_ML,         run_ml),
    ]
    active = [name for name, flag, _ in steps if flag]

    print(f"\n{'='*60}")
    print(f"  Pipeline Config")
    print(f"  Universe    : {cfg.UNIVERSE_MODE}")
    print(f"  EDA range   : {cfg.START_YEAR} ~ {cfg.END_YEAR}")
    print(f"  ML windows  : {cfg.ML_WINDOW_START} ~ {cfg.ML_WINDOW_END}")
    print(f"  Cache dir   : {cfg.CACHE_DIR}")
    print(f"  Steps       : {' → '.join(active) if active else '(none)'}")
    print(f"{'='*60}")

    if not active:
        print("  All steps disabled. Exiting.")
        return

    t0     = time.time()
    eda_df = None

    for name, flag, fn in steps:
        if not flag:
            continue
        print(f"\n{'─'*60}")
        print(f"  ▶ {name}")
        print(f"{'─'*60}")
        if name == "Preprocess EDA":
            eda_df = fn(cfg)
        elif name == "ML Training":
            fn(cfg, precomputed_df=eda_df)
        else:
            fn(cfg)

    print(f"\n  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
