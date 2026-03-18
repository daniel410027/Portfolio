"""
result.py
=========
Walk-forward backtest comprehensive analysis.

Fixes vs. plot_pnl_vbt.py:
    - TC is applied on DAILY TURNOVER, not flat daily fee
      (turnover = fraction of portfolio that actually changes each day)
    - Drawdown panel uses GROSS returns (net TC panel kept as reference)
    - Adds turnover statistics

Metrics printed:
    - ML classification metrics (per window + aggregate)
    - IC / ICIR analysis
    - Portfolio performance (gross / net-TC / market benchmark)
    - Turnover & transaction cost breakdown
    - Drawdown analysis (max DD, avg DD, time underwater)

Usage:
    python result.py
    python result.py --pred_dir database/experiment --market ^TWII

Author: Daniel Huang
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

TRADING_DAYS  = 252
TC_ONE_WAY    = 0.001425   # 0.1425 %


# ============================================================
#  Loading
# ============================================================

def load_predictions(pred_dir: Path) -> pd.DataFrame:
    files = sorted(pred_dir.rglob("predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No predictions.csv under {pred_dir}")
    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["date"])
        df["window"] = f.parent.name
        dfs.append(df)
    pred = pd.concat(dfs, ignore_index=True).sort_values("date").reset_index(drop=True)
    return pred


def load_metrics(pred_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(pred_dir.rglob("metrics.json")):
        with open(f, encoding="utf-8") as fp:
            m = json.load(fp)
        m.setdefault("window", f.parent.name)
        rows.append(m)
    return pd.DataFrame(rows).sort_values("test_year").reset_index(drop=True)


def load_market(start: pd.Timestamp, end: pd.Timestamp,
                ticker: str = "^TWII") -> pd.Series:
    raw = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        auto_adjust=True,
    )
    if raw.empty:
        raise RuntimeError(f"Cannot download {ticker}")
    close = raw["Close"].copy()
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close.pct_change().dropna().rename("market")


# ============================================================
#  Turnover-based TC
# ============================================================

def compute_turnover(pred: pd.DataFrame) -> pd.Series:
    """
    Daily portfolio turnover = fraction of holdings that changed.

    For each date:
        turnover = |set(held today) Δ set(held yesterday)| / max(|today|, |yesterday|)

    Returns pd.Series indexed by date.
    """
    dates   = sorted(pred["date"].unique())
    held    = {d: set(pred.loc[(pred["date"] == d) & (pred["y_pred"] == 1), "ticker"])
               for d in dates}
    rows = []
    prev_set = set()
    for d in dates:
        curr_set = held[d]
        if not prev_set and not curr_set:
            rows.append({"date": d, "turnover": 0.0})
        else:
            denom = max(len(curr_set), len(prev_set), 1)
            churn = len(curr_set.symmetric_difference(prev_set))
            rows.append({"date": d, "turnover": churn / denom})
        prev_set = curr_set

    tv = pd.DataFrame(rows).set_index("date")["turnover"]
    return tv


def apply_tc_turnover(gross_ret: pd.Series, turnover: pd.Series,
                      tc_one_way: float = TC_ONE_WAY) -> pd.Series:
    """
    Net return = gross return − turnover × 2 × tc_one_way
    (round-trip cost scales with fraction of portfolio traded)
    """
    tv = turnover.reindex(gross_ret.index).fillna(0)
    return gross_ret - tv * 2 * tc_one_way


# ============================================================
#  Portfolio construction
# ============================================================

def daily_eq_return(pred: pd.DataFrame, mask: pd.Series) -> pd.Series:
    return pred[mask].groupby("date")["return"].mean()


# ============================================================
#  Performance metrics
# ============================================================

def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan


def sortino(r: pd.Series) -> float:
    r = r.dropna()
    neg = r[r < 0]
    ds  = neg.std() * np.sqrt(TRADING_DAYS)
    return float(r.mean() * TRADING_DAYS / ds) if ds > 0 else np.nan


def ann_ret(r: pd.Series) -> float:
    return float(r.dropna().mean() * TRADING_DAYS)


def max_dd(r: pd.Series) -> float:
    v  = (1 + r.fillna(0)).cumprod()
    dd = v / v.cummax() - 1
    return float(dd.min())


def calmar(r: pd.Series) -> float:
    md = max_dd(r)
    return float(ann_ret(r) / abs(md)) if md != 0 else np.nan


def time_underwater(r: pd.Series) -> float:
    """Fraction of days where portfolio is below its previous high."""
    v  = (1 + r.fillna(0)).cumprod()
    return float((v < v.cummax()).mean())


def win_rate(r: pd.Series) -> float:
    return float((r.dropna() > 0).mean())


def compute_ic(pred: pd.DataFrame) -> tuple[float, float, float]:
    """
    Daily Spearman IC between y_prob and return.
    Returns (mean_IC, IC_std, ICIR_annualized).
    """
    rows = []
    for (win, d), g in pred.groupby(["window", "date"]):
        if len(g) < 5 or g["return"].isna().all():
            continue
        c, _ = stats.spearmanr(g["y_prob"], g["return"])
        rows.append(c)
    ic = pd.Series(rows).dropna()
    mean_ic  = float(ic.mean())
    std_ic   = float(ic.std())
    icir     = mean_ic / std_ic * np.sqrt(TRADING_DAYS) if std_ic > 0 else np.nan
    return mean_ic, std_ic, icir


def compute_per_window_ic(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for win, wg in pred.groupby("window"):
        daily_ic = []
        for d, dg in wg.groupby("date"):
            if len(dg) < 5:
                continue
            c, _ = stats.spearmanr(dg["y_prob"], dg["return"])
            daily_ic.append(c)
        ic = pd.Series(daily_ic).dropna()
        rows.append({
            "window":   win,
            "mean_ic":  ic.mean(),
            "icir":     ic.mean() / ic.std() * np.sqrt(TRADING_DAYS) if ic.std() > 0 else np.nan,
            "ic_gt0":   (ic > 0).mean(),
        })
    return pd.DataFrame(rows).sort_values("window")


# ============================================================
#  Printing helpers
# ============================================================

SEP  = "=" * 66
SEP2 = "─" * 66


def hdr(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def sub(title: str):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


def row(label: str, val: str, width: int = 24):
    print(f"  {label:<{width}}: {val}")


# ============================================================
#  Main
# ============================================================

def run(pred_dir: str = "database/experiment",
        market_ticker: str = "^TWII",
        tc_one_way: float = TC_ONE_WAY):

    pred_dir = Path(pred_dir)

    # ── Load ──────────────────────────────────────────────────────
    hdr("Loading data")
    pred    = load_predictions(pred_dir)
    metrics = load_metrics(pred_dir)
    print(f"  Predictions : {len(pred):,} rows | "
          f"{pred['ticker'].nunique()} stocks | "
          f"{pred['date'].nunique()} dates")
    print(f"  Windows     : {len(metrics)} "
          f"({metrics['test_year'].min()} ~ {metrics['test_year'].max()})")

    start, end = pred["date"].min(), pred["date"].max()
    print(f"  Downloading market index: {market_ticker} ...")
    mkt_r = load_market(start, end, market_ticker)

    # ── Portfolio returns ─────────────────────────────────────────
    long_gross = daily_eq_return(pred, pred["y_pred"] == 1)
    short_ret  = daily_eq_return(pred, pred["y_pred"] == 0)
    all_ret    = daily_eq_return(pred, pd.Series(True, index=pred.index))

    idx = long_gross.index.union(mkt_r.index)
    long_gross = long_gross.reindex(idx).fillna(0)
    short_r    = short_ret.reindex(idx).fillna(0)
    all_r      = all_ret.reindex(idx).fillna(0)
    mkt_r      = mkt_r.reindex(idx).fillna(0)

    # ── Turnover & net TC ─────────────────────────────────────────
    hdr("Turnover Analysis")
    print("  Computing daily portfolio turnover (this may take a moment) ...")
    tv = compute_turnover(pred)
    tv = tv.reindex(idx).fillna(0)
    long_net = apply_tc_turnover(long_gross, tv, tc_one_way)

    avg_tv   = tv[tv > 0].mean()
    med_tv   = tv[tv > 0].median()
    daily_tc = (tv * 2 * tc_one_way)
    ann_tc   = daily_tc.mean() * TRADING_DAYS

    row("Avg daily turnover",   f"{avg_tv:.2%}")
    row("Median daily turnover",f"{med_tv:.2%}")
    row("Avg daily TC cost",    f"{daily_tc.mean():.4%}")
    row("Annualized TC drag",   f"{ann_tc:.2%}")
    row("TC one-way assumption",f"{tc_one_way:.4%}")
    row("TC method",            "turnover-scaled (correct)")
    print(f"\n  ⚠ NOTE: flat-daily TC (old method) would assume "
          f"{2*tc_one_way*TRADING_DAYS:.1%}/yr cost regardless of turnover.\n"
          f"    Turnover-based TC reduces this to {ann_tc:.2%}/yr (actual drag).")

    # ── IC ────────────────────────────────────────────────────────
    hdr("Information Coefficient (IC)")
    mean_ic, std_ic, icir = compute_ic(pred)
    row("Mean daily IC (Spearman)", f"{mean_ic:.4f}")
    row("IC Std Dev",               f"{std_ic:.4f}")
    row("ICIR (annualized)",        f"{icir:.4f}")
    row("Interpretation", "IC > 0.02 = weak signal; ICIR > 0.5 = tradeable")

    sub("Per-Window IC")
    ic_win = compute_per_window_ic(pred)
    print(f"  {'Window':<12} {'Mean IC':>9} {'ICIR':>9} {'IC>0 %':>9}")
    print(f"  {'─'*42}")
    for _, r_ in ic_win.iterrows():
        flag = " ✓" if r_["mean_ic"] > 0 else " ✗"
        print(f"  {r_['window']:<12} {r_['mean_ic']:>9.4f} {r_['icir']:>9.4f} "
              f"{r_['ic_gt0']:>9.1%}{flag}")

    # ── ML metrics ────────────────────────────────────────────────
    hdr("ML Classification Metrics")
    for col, label in [("roc_auc","AUC"), ("f1","F1"),
                       ("precision","Precision"), ("recall","Recall")]:
        if col not in metrics.columns:
            continue
        m, s = metrics[col].mean(), metrics[col].std()
        print(f"  {label:<12}: mean={m:.4f}  std={s:.4f}  "
              f"min={metrics[col].min():.4f}  max={metrics[col].max():.4f}")

    print(f"\n  Positive rate (true)  avg: {metrics['positive_rate_true'].mean():.2%}")
    print(f"  Positive rate (pred)  avg: {metrics['positive_rate_pred'].mean():.2%}")

    sub("Per-Window Detail")
    cols = ["test_year","roc_auc","f1","precision","recall",
            "positive_rate_true","positive_rate_pred","n_test"]
    cols = [c for c in cols if c in metrics.columns]
    print(metrics[cols].to_string(index=False))

    # ── Portfolio performance ─────────────────────────────────────
    hdr("Portfolio Performance")

    strategies = {
        "Long gross":               long_gross,
        f"Long net TC ({avg_tv:.0%} avg to)": long_net,
        "Short (y_pred=0)":         short_r,
        "All stocks":               all_r,
        f"Market ({market_ticker})":mkt_r,
        "L-S spread":               long_gross - short_r,
    }

    print(f"\n  {'Strategy':<32} {'Ann.Ret':>8} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>7} {'WinRate':>8}")
    print(f"  {'─'*82}")
    for name, r_ in strategies.items():
        ar  = ann_ret(r_)
        sh  = sharpe(r_)
        so  = sortino(r_)
        md  = max_dd(r_)
        cal = calmar(r_)
        wr  = win_rate(r_)
        sh_s  = f"{sh:7.3f}" if not np.isnan(sh)  else "    N/A"
        so_s  = f"{so:8.3f}" if not np.isnan(so)  else "     N/A"
        cal_s = f"{cal:7.3f}" if not np.isnan(cal) else "    N/A"
        print(f"  {name:<32} {ar:>8.2%} {sh_s} {so_s} {md:>8.2%} {cal_s} {wr:>8.2%}")

    # ── Drawdown deep-dive ────────────────────────────────────────
    sub("Drawdown Analysis (Long gross)")
    v_g  = (1 + long_gross).cumprod()
    dd_g = v_g / v_g.cummax() - 1

    print(f"  Max Drawdown          : {dd_g.min():.2%}")
    print(f"  Avg Drawdown          : {dd_g[dd_g < 0].mean():.2%}")
    print(f"  Time underwater (frac): {time_underwater(long_gross):.2%}")

    # Longest drawdown duration
    in_dd   = dd_g < 0
    durations, cur = [], 0
    for v in in_dd:
        cur = cur + 1 if v else 0
        if not v and cur > 0:
            durations.append(cur)
    if in_dd.iloc[-1]:
        durations.append(cur)
    if durations:
        print(f"  Longest DD streak     : {max(durations)} trading days")
        print(f"  Avg DD streak         : {np.mean(durations):.1f} trading days")

    # ── Drawdown bug explanation ──────────────────────────────────
    hdr("Note: Why Long (net TC) showed -99.5% Drawdown in Previous Chart")
    print(
        "  Root cause: flat-daily TC was applied regardless of actual portfolio turnover.\n"
        f"  Old method : {2*tc_one_way:.4%} deducted EVERY trading day\n"
        f"             = {2*tc_one_way*TRADING_DAYS:.1%}/year cost (far exceeds alpha)\n"
        f"  This method: {daily_tc.mean():.4%} avg daily (turnover-scaled)\n"
        f"             = {ann_tc:.2%}/year  (realistic)\n\n"
        f"  The drawdown FORMULA was mathematically correct:\n"
        f"    DD = (V - V_max) / V_max  (equivalent form used in code)\n"
        f"  The -99.5% drawdown was REAL given the wrong TC assumption.\n"
        f"  With correct turnover-based TC, net DD is now: {max_dd(long_net):.2%}"
    )

    hdr("Done")
    print(f"  Source: {pred_dir}\n")


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward result analyzer")
    parser.add_argument("--pred_dir", default="database/experiment")
    parser.add_argument("--market",   default="^TWII")
    parser.add_argument("--tc",       type=float, default=TC_ONE_WAY)
    args = parser.parse_args()

    run(pred_dir=args.pred_dir, market_ticker=args.market, tc_one_way=args.tc)