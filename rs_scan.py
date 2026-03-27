"""
Relative Strength Scanner
==========================
Finds stocks showing relative strength during a market correction.

Metrics computed per stock:
  RS Score    — IBD-style composite: weighted % change vs SPY over
                63/126/189/252 trading days (recent periods weighted more)
  RS Line     — stock price / SPY price ratio
  RS Trend    — slope of RS line over last 20 days (rising = strengthening)
  Drawdown    — % below own 52-week high
  SPY Delta   — drawdown vs SPY drawdown (positive = holding up better)
  Above MA    — whether stock is above its 21/50/200-day MA

Output: ranked list of top RS stocks, saved as CSV + HTML

Usage:
    python rs_scan.py                    # scan NYSE + NASDAQ large-cap
    python rs_scan.py --top 50           # show top 50 (default 30)
    python rs_scan.py --min-price 10     # skip penny stocks (default $5)
    python rs_scan.py --universe nyse    # nyse | nasdaq | sp500
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import argparse
import base64
import io
import os
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import resend

# ─────────────────────────────────────────────────────────────────────────────
# Email config
# ─────────────────────────────────────────────────────────────────────────────
resend.api_key = os.environ.get("RESEND_API_KEY", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", "onboarding@resend.dev")
EMAIL_TO   = os.environ.get("EMAIL_TO", "rksh64@gmail.com")


# ─────────────────────────────────────────────────────────────────────────────
# Universe helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_nyse_universe() -> list:
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    print("  Fetching NYSE universe…")
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            raw = r.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw), sep="|")
        nyse = df[df["Exchange"] == "N"].copy()
        exclude = nyse["ETF"] == "Y"
        exclude |= nyse["ACT Symbol"].str.contains(r"[\+\$\*\^]", regex=True, na=True)
        exclude |= nyse["ACT Symbol"].str.contains(r"\.(W|R|U|V)$", regex=True, na=True)
        exclude |= nyse["ACT Symbol"].str.match(r"^(CTEST|NTEST)", na=True)
        exclude |= nyse["Security Name"].str.contains(
            r"\bETF\b|\bFund\b|\bTrust\b|\bWarrant\b|\bUnit\b"
            r"|\bPreferred\b|\bPfd\b|\bLP\b|\bRight\b",
            case=False, regex=True, na=True,
        )
        nyse = nyse[~exclude]
        tickers = (
            nyse["ACT Symbol"].dropna().str.strip()
            .str.replace(r"\.([A-Z])$", r"-\1", regex=True)
            .tolist()
        )
        tickers = sorted(set(tickers))
        print(f"  NYSE: {len(tickers)} common stocks")
        return tickers
    except Exception as e:
        print(f"  NYSE fetch failed ({e}) — using fallback")
        return _fallback_universe()


def get_nasdaq_universe() -> list:
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    print("  Fetching NASDAQ universe…")
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            raw = r.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw), sep="|")
        exclude = df["ETF"] == "Y"
        exclude |= df["Symbol"].str.contains(r"[\+\$\*\^]", regex=True, na=True)
        exclude |= df["Symbol"].str.match(r"^(CTEST|NTEST)", na=True)
        exclude |= df["Security Name"].str.contains(
            r"\bETF\b|\bFund\b|\bTrust\b|\bWarrant\b|\bUnit\b"
            r"|\bPreferred\b|\bPfd\b|\bLP\b|\bRight\b",
            case=False, regex=True, na=True,
        )
        df = df[~exclude]
        tickers = df["Symbol"].dropna().str.strip().tolist()
        tickers = sorted(set(tickers))
        print(f"  NASDAQ: {len(tickers)} common stocks")
        return tickers
    except Exception as e:
        print(f"  NASDAQ fetch failed ({e})")
        return []


def get_sp500_universe() -> list:
    """Pull S&P 500 constituents from Wikipedia."""
    print("  Fetching S&P 500 universe…")
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        tickers = (
            tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        )
        print(f"  S&P 500: {len(tickers)} stocks")
        return tickers
    except Exception as e:
        print(f"  S&P 500 fetch failed ({e}) — using fallback")
        return _fallback_universe()


def _fallback_universe() -> list:
    return sorted(set([
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","AVGO","JPM",
        "LLY","V","UNH","XOM","MA","JNJ","PG","HD","COST","MRK","ABBV","CVX",
        "BAC","WMT","NFLX","CRM","KO","PEP","TMO","ORCL","ACN","MCD","ABT",
        "CSCO","ADBE","WFC","TXN","NKE","DHR","LIN","PM","NEE","DIS","INTU",
        "AMGN","RTX","LOW","SPGI","UPS","CAT","HON","GS","MS","AMAT","ISRG",
        "NOW","BLK","SYK","ELV","GE","BKNG","ADP","VRTX","PLD","CB","REGN",
        "ADI","MDLZ","GILD","TJX","MMC","AMT","CI","ETN","ZTS","CME","PGR",
        "AON","SHW","LRCX","KLAC","MO","CL","SO","DUK","NOC","LMT","GD","BA",
        "PLTR","SNOW","DDOG","CRWD","PANW","ZS","NET","FTNT","APP","MNDY",
        "UBER","LYFT","DASH","AMD","INTC","QCOM","MU","ARM","MRVL","ON",
        "C","USB","PNC","TFC","COF","AXP","SCHW","ICE","APO","KKR","BX",
        "PFE","BMY","MRNA","BIIB","ILMN","IDXX","IQV","DXCM","VEEV","ALNY",
        "SBUX","CMG","YUM","TGT","DLTR","DG","ETSY","SLB","HAL","COP","OXY",
        "EMR","ITW","ROK","AXON","KTOS","RKLB","F","GM","DE","FCX","NEM",
        "WM","RSG","O","SPG","VICI","EQR","AVB",
    ]))


# ─────────────────────────────────────────────────────────────────────────────
# Price download
# ─────────────────────────────────────────────────────────────────────────────

def download_prices(tickers: list, period: str = "1y") -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers + SPY.
    Returns a DataFrame with tickers as columns, dates as index.
    """
    all_tickers = sorted(set(tickers + ["SPY"]))
    total = len(all_tickers)
    CHUNK = 100
    batches = [all_tickers[i:i+CHUNK] for i in range(0, total, CHUNK)]

    print(f"\n  Downloading {total} tickers in {len(batches)} batches…")
    all_closes = {}

    for idx, batch in enumerate(batches, 1):
        print(f"  Batch {idx}/{len(batches)}  ({len(batch)} tickers)…", end=" ", flush=True)
        try:
            raw = yf.download(
                batch, period=period,
                auto_adjust=True, progress=False,
                group_by="ticker", threads=False,
            )
            fetched = 0
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        s = raw["Close"]
                    else:
                        if ticker in raw.columns.get_level_values(0):
                            s = raw[ticker]["Close"]
                        else:
                            continue
                    if s is not None and not s.dropna().empty:
                        all_closes[ticker] = s
                        fetched += 1
                except Exception:
                    continue
            print(f"✓ {fetched}/{len(batch)} ok")
        except Exception as e:
            print(f"✗ batch failed: {e}")

    df = pd.DataFrame(all_closes)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RS calculations
# ─────────────────────────────────────────────────────────────────────────────

def rs_score(stock_series: pd.Series, spy_series: pd.Series) -> float:
    """
    IBD-style composite RS score (0–99 scale before ranking).
    Weighted % change over 4 periods: last 63/126/189/252 trading days.
    Recent performance weighted 2x.
    """
    if len(stock_series) < 63 or len(spy_series) < 63:
        return np.nan

    def pct(s, n):
        if len(s) < n + 1:
            return np.nan
        return (s.iloc[-1] / s.iloc[-n] - 1) * 100

    def rel(n):
        sp = pct(stock_series, n)
        bp = pct(spy_series, n)
        if np.isnan(sp) or np.isnan(bp):
            return np.nan
        return sp - bp

    # Weights: 63d × 2, 126d × 1, 189d × 1, 252d × 1
    weights = [(63, 2), (126, 1), (189, 1), (252, 1)]
    total_w = sum(w for _, w in weights)
    score = 0.0
    for n, w in weights:
        r = rel(n)
        if np.isnan(r):
            r = 0.0
        score += r * w
    return score / total_w


def rs_line_slope(stock_series: pd.Series, spy_series: pd.Series, days: int = 20) -> float:
    """Slope of the RS line (stock/SPY) over the last N days. Positive = strengthening."""
    if len(stock_series) < days or len(spy_series) < days:
        return np.nan
    rs_line = (stock_series / spy_series).iloc[-days:]
    if rs_line.empty or rs_line.isna().all():
        return np.nan
    x = np.arange(len(rs_line))
    slope = np.polyfit(x, rs_line.values.astype(float), 1)[0]
    # Normalize by mean RS level so units are comparable
    mean_rs = rs_line.mean()
    return (slope / mean_rs * 100) if mean_rs != 0 else np.nan


def above_ma(series: pd.Series, window: int) -> bool:
    """True if the latest close is above the N-day MA."""
    if len(series) < window:
        return False
    ma = series.rolling(window=window, min_periods=window).mean().iloc[-1]
    return bool(series.iloc[-1] > ma)


def pct_from_high(series: pd.Series, window: int = 252) -> float:
    """% below the N-day high (negative means below high)."""
    if series.empty:
        return np.nan
    high = series.iloc[-window:].max() if len(series) >= window else series.max()
    return (series.iloc[-1] / high - 1) * 100


# ─────────────────────────────────────────────────────────────────────────────
# Main scan
# ─────────────────────────────────────────────────────────────────────────────

def scan(prices: pd.DataFrame, min_price: float = 5.0) -> pd.DataFrame:
    """
    Compute RS metrics for every ticker vs SPY.
    Returns DataFrame sorted by RS Score descending.
    """
    if "SPY" not in prices.columns:
        raise ValueError("SPY not in price data")

    spy = prices["SPY"].dropna()
    spy_dd = pct_from_high(spy)

    rows = []
    tickers = [c for c in prices.columns if c != "SPY"]
    total = len(tickers)
    print(f"\n  Computing RS metrics for {total} stocks…")

    for i, ticker in enumerate(tickers, 1):
        if i % 200 == 0:
            print(f"    {i}/{total}…")
        s = prices[ticker].dropna()
        if s.empty or s.iloc[-1] < min_price or len(s) < 63:
            continue

        score      = rs_score(s, spy)
        slope      = rs_line_slope(s, spy, days=20)
        dd         = pct_from_high(s)
        spy_delta  = dd - spy_dd          # positive = holding up better than SPY
        price      = round(s.iloc[-1], 2)
        chg_1m     = (s.iloc[-1] / s.iloc[-21] - 1) * 100 if len(s) >= 22 else np.nan
        chg_3m     = (s.iloc[-1] / s.iloc[-63] - 1) * 100 if len(s) >= 64 else np.nan

        rows.append({
            "Ticker":       ticker,
            "Price":        price,
            "RS_Score":     round(score, 1)    if not np.isnan(score) else np.nan,
            "RS_Slope_20d": round(slope, 2)    if not np.isnan(slope) else np.nan,
            "Drawdown%":    round(dd, 1)       if not np.isnan(dd)    else np.nan,
            "vs_SPY_DD":    round(spy_delta, 1) if not np.isnan(spy_delta) else np.nan,
            "Chg_1M%":      round(chg_1m, 1)  if not np.isnan(chg_1m) else np.nan,
            "Chg_3M%":      round(chg_3m, 1)  if not np.isnan(chg_3m) else np.nan,
            "Above_21MA":   above_ma(s, 21),
            "Above_50MA":   above_ma(s, 50),
            "Above_200MA":  above_ma(s, 200),
        })

    df = pd.DataFrame(rows).dropna(subset=["RS_Score"])
    # Rank RS_Score 1–99
    df["RS_Rank"] = df["RS_Score"].rank(pct=True).mul(99).round(0).astype(int)
    df.sort_values("RS_Score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def print_results(df: pd.DataFrame, top: int, spy_info: dict):
    spy_dd = spy_info["drawdown"]
    spy_1m = spy_info["chg_1m"]
    spy_3m = spy_info["chg_3m"]

    print(f"\n{'='*75}")
    print(f"  RELATIVE STRENGTH SCAN  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*75}")
    print(f"  SPY benchmark:  1M {spy_1m:+.1f}%   3M {spy_3m:+.1f}%   "
          f"Drawdown from 52w high: {spy_dd:.1f}%")
    print(f"{'─'*75}")
    print(f"  {'#':<4} {'Ticker':<8} {'Price':>7} {'RS':>4} {'1M%':>6} {'3M%':>6} "
          f"{'DD%':>6} {'vsSPY':>6} {'Slope':>7}  MAs")
    print(f"  {'─'*4} {'─'*8} {'─'*7} {'─'*4} {'─'*6} {'─'*6} "
          f"{'─'*6} {'─'*6} {'─'*7}  {'─'*12}")

    for rank, (_, row) in enumerate(df.head(top).iterrows(), 1):
        ma_flags = (
            ("21✓" if row["Above_21MA"]  else "21✗") + " " +
            ("50✓" if row["Above_50MA"]  else "50✗") + " " +
            ("200✓" if row["Above_200MA"] else "200✗")
        )
        slope_str = f"{row['RS_Slope_20d']:+.2f}" if not pd.isna(row["RS_Slope_20d"]) else "  n/a"
        print(
            f"  {rank:<4} {row['Ticker']:<8} {row['Price']:>7.2f} {int(row['RS_Rank']):>4} "
            f"{row['Chg_1M%']:>+6.1f} {row['Chg_3M%']:>+6.1f} "
            f"{row['Drawdown%']:>+6.1f} {row['vs_SPY_DD']:>+6.1f} "
            f"{slope_str:>7}  {ma_flags}"
        )

    print(f"{'='*75}")
    print(f"\n  Columns: RS=relative strength rank (99=best), DD%=drawdown from 52w high,")
    print(f"  vsSPY=your drawdown minus SPY drawdown (positive=holding up better),")
    print(f"  Slope=20-day RS line slope (positive=RS strengthening)\n")


def save_html(df: pd.DataFrame, top: int, spy_info: dict, out_path: str):
    rows_html = ""
    for rank, (_, row) in enumerate(df.head(top).iterrows(), 1):
        mas = []
        for w, col in [(21, "Above_21MA"), (50, "Above_50MA"), (200, "Above_200MA")]:
            color = "#2ecc71" if row[col] else "#e74c3c"
            mas.append(f"<span style='color:{color}'>{w}</span>")
        ma_str = " ".join(mas)

        # Color the vs_SPY_DD column
        vs = row["vs_SPY_DD"]
        vs_color = "#2ecc71" if vs > 0 else "#e74c3c"

        rows_html += (
            f"<tr>"
            f"<td>{rank}</td>"
            f"<td><b>{row['Ticker']}</b></td>"
            f"<td>${row['Price']:.2f}</td>"
            f"<td><b>{int(row['RS_Rank'])}</b></td>"
            f"<td>{row['Chg_1M%']:+.1f}%</td>"
            f"<td>{row['Chg_3M%']:+.1f}%</td>"
            f"<td>{row['Drawdown%']:+.1f}%</td>"
            f"<td style='color:{vs_color}'><b>{vs:+.1f}%</b></td>"
            f"<td>{row['RS_Slope_20d']:+.2f}</td>"
            f"<td>{ma_str}</td>"
            f"</tr>\n"
        )

    spy_dd  = spy_info["drawdown"]
    spy_1m  = spy_info["chg_1m"]
    spy_3m  = spy_info["chg_3m"]
    run_ts  = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Relative Strength Scan</title>
<style>
  body  {{ background:#1a1a2e; color:#eee; font-family:Arial,sans-serif; margin:20px; }}
  h1   {{ color:#00d4ff; margin-bottom:4px; }}
  .sub {{ color:#aaa; font-size:0.9em; margin-bottom:16px; }}
  .spy-box {{ background:#0f3460; border-radius:6px; padding:10px 18px;
              display:inline-block; margin-bottom:18px; }}
  .spy-box span {{ margin-right:20px; }}
  table {{ border-collapse:collapse; width:100%; }}
  th,td {{ border:1px solid #2a2a4a; padding:7px 12px; text-align:right; }}
  th    {{ background:#0f3460; color:#00d4ff; text-align:center; }}
  td:nth-child(2) {{ text-align:left; }}
  tr:nth-child(even) {{ background:#16213e; }}
  tr:hover {{ background:#1e2a5e; }}
  .legend {{ margin-top:12px; color:#888; font-size:0.82em; }}
</style>
</head>
<body>
<h1>Relative Strength Scan — Correction Leaders</h1>
<div class="sub">Run: {run_ts}</div>

<div class="spy-box">
  <b>SPY Benchmark</b> &nbsp;
  <span>1M: <b>{spy_1m:+.1f}%</b></span>
  <span>3M: <b>{spy_3m:+.1f}%</b></span>
  <span>52w Drawdown: <b>{spy_dd:.1f}%</b></span>
</div>

<table>
<tr>
  <th>#</th><th>Ticker</th><th>Price</th><th>RS Rank</th>
  <th>1M Chg</th><th>3M Chg</th><th>Drawdown</th>
  <th>vs SPY DD</th><th>RS Slope 20d</th><th>MAs (21/50/200)</th>
</tr>
{rows_html}
</table>

<div class="legend">
  <b>RS Rank</b>: 1–99 percentile vs universe (99 = strongest). &nbsp;
  <b>vs SPY DD</b>: stock drawdown minus SPY drawdown — positive means holding up better. &nbsp;
  <b>RS Slope</b>: 20-day trend of (stock/SPY) ratio — positive = RS line rising. &nbsp;
  <b>MAs</b>: green = above that MA, red = below.
</div>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"  HTML report saved → {out_path}")


def make_rs_chart(df: pd.DataFrame, prices: pd.DataFrame, top_n: int, out_path: str):
    """
    Plot the RS line (stock / SPY) for the top N stocks over the last year.
    Each line normalized to 100 at start so they're comparable.
    """
    leaders = df.head(top_n)["Ticker"].tolist()
    spy = prices["SPY"].dropna()

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    colors = plt.cm.tab20.colors
    for i, ticker in enumerate(leaders):
        if ticker not in prices.columns:
            continue
        s = prices[ticker].dropna()
        # Align with SPY
        common = s.index.intersection(spy.index)
        if len(common) < 20:
            continue
        rs = (s[common] / spy[common])
        rs_norm = rs / rs.iloc[0] * 100  # normalize to 100
        ax.plot(rs_norm.index, rs_norm.values,
                color=colors[i % len(colors)], linewidth=1.3,
                label=ticker, alpha=0.85)

    # SPY flat line at 100
    ax.axhline(100, color="white", linewidth=0.8, linestyle="--", alpha=0.4, label="SPY (flat)")

    ax.set_title(
        f"RS Lines — Top {top_n} Relative Strength Leaders  "
        f"(normalized, stock / SPY × 100)",
        color="white", fontsize=12, pad=10,
    )
    ax.set_ylabel("Relative Strength vs SPY (normalized)", color="white")
    ax.tick_params(colors="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right", color="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.3,
              labelcolor="white", facecolor="#1a1a2e", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  RS chart saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Email
# ─────────────────────────────────────────────────────────────────────────────

def send_email(df: pd.DataFrame, top: int, spy_info: dict, html_path: str, chart_path: str):
    """Send the RS scan results via Resend with the HTML report inline and chart attached."""
    today     = datetime.now().strftime("%Y-%m-%d")
    spy_dd    = spy_info["drawdown"]
    spy_1m    = spy_info["chg_1m"]
    latest_val = df.iloc[0]["RS_Rank"] if not df.empty else 0

    # Build a compact HTML summary table for the email body
    rows_html = ""
    for rank, (_, row) in enumerate(df.head(top).iterrows(), 1):
        mas = []
        for w, col in [(21, "Above_21MA"), (50, "Above_50MA"), (200, "Above_200MA")]:
            color = "#2ecc71" if row[col] else "#e74c3c"
            mas.append(f"<span style='color:{color}'>{w}</span>")
        vs      = row["vs_SPY_DD"]
        vs_clr  = "#2ecc71" if vs > 0 else "#e74c3c"
        rows_html += (
            f"<tr>"
            f"<td style='padding:5px 10px'>{rank}</td>"
            f"<td style='padding:5px 10px'><b>{row['Ticker']}</b></td>"
            f"<td style='padding:5px 10px'>${row['Price']:.2f}</td>"
            f"<td style='padding:5px 10px;text-align:center'><b>{int(row['RS_Rank'])}</b></td>"
            f"<td style='padding:5px 10px;text-align:right'>{row['Chg_1M%']:+.1f}%</td>"
            f"<td style='padding:5px 10px;text-align:right'>{row['Chg_3M%']:+.1f}%</td>"
            f"<td style='padding:5px 10px;text-align:right;color:{vs_clr}'><b>{vs:+.1f}%</b></td>"
            f"<td style='padding:5px 10px'>{' '.join(mas)}</td>"
            f"</tr>\n"
        )

    body = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body style="background:#1a1a2e;color:#eee;font-family:Arial,sans-serif;padding:20px">
<h2 style="color:#00d4ff;margin-bottom:4px">RS Scan — Correction Leaders &nbsp;·&nbsp; {today}</h2>
<p style="color:#aaa;margin-top:0">NYSE + NASDAQ · {len(df)} stocks ranked · top {top} shown</p>
<div style="background:#0f3460;border-radius:6px;padding:10px 16px;display:inline-block;margin-bottom:16px">
  <b>SPY:</b> &nbsp;
  1M <b>{spy_1m:+.1f}%</b> &nbsp;|&nbsp;
  52w Drawdown <b>{spy_dd:.1f}%</b>
</div>
<table style="border-collapse:collapse;width:100%">
<tr style="background:#0f3460;color:#00d4ff">
  <th style="padding:6px 10px">#</th>
  <th style="padding:6px 10px">Ticker</th>
  <th style="padding:6px 10px">Price</th>
  <th style="padding:6px 10px">RS</th>
  <th style="padding:6px 10px">1M</th>
  <th style="padding:6px 10px">3M</th>
  <th style="padding:6px 10px">vs SPY DD</th>
  <th style="padding:6px 10px">MAs 21/50/200</th>
</tr>
{rows_html}
</table>
<p style="color:#888;font-size:0.8em;margin-top:12px">
  RS = percentile rank vs universe (99=best) &nbsp;·&nbsp;
  vs SPY DD = your drawdown minus SPY's (positive = holding up better)
</p>
</body></html>"""

    # Attach chart PNG
    attachments = []
    if os.path.exists(chart_path):
        with open(chart_path, "rb") as f:
            attachments.append({
                "filename": os.path.basename(chart_path),
                "content":  list(f.read()),
            })

    try:
        resend.Emails.send({
            "from":        EMAIL_FROM,
            "to":          EMAIL_TO,
            "subject":     f"RS Scan {today} — Top leaders: {', '.join(df.head(5)['Ticker'].tolist())}",
            "html":        body,
            "attachments": attachments,
        })
        print(f"  Email sent → {EMAIL_TO}")
    except Exception as e:
        print(f"  Email failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Relative Strength Scanner")
    parser.add_argument("--top",       type=int,   default=30,     help="Show top N stocks (default 30)")
    parser.add_argument("--min-price", type=float, default=5.0,    help="Min stock price (default $5)")
    parser.add_argument("--universe",  default="sp500",
                        choices=["nyse","nasdaq","all","sp500","fallback"],
                        help="Stock universe (default: sp500). 'all' = NYSE + NASDAQ combined")
    parser.add_argument("--period",    default="1y",               help="Data period (default: 1y)")
    parser.add_argument("--out-dir",   default=".",                help="Output directory")
    parser.add_argument("--no-chart",  action="store_true",        help="Skip PNG chart")
    parser.add_argument("--no-email",  action="store_true",        help="Skip email")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  Relative Strength Scanner — {today}")
    print(f"  Universe: {args.universe}  |  Min price: ${args.min_price}")
    print(f"{'='*60}")

    # Get universe
    if args.universe == "nyse":
        tickers = get_nyse_universe()
    elif args.universe == "nasdaq":
        tickers = get_nasdaq_universe()
    elif args.universe == "all":
        tickers = sorted(set(get_nyse_universe() + get_nasdaq_universe()))
        print(f"  Combined universe: {len(tickers)} stocks")
    elif args.universe == "sp500":
        tickers = get_sp500_universe()
    else:
        tickers = _fallback_universe()

    # Download prices
    prices = download_prices(tickers, period=args.period)

    if "SPY" not in prices.columns:
        print("  ERROR: Could not download SPY data. Aborting.")
        return

    spy = prices["SPY"].dropna()
    spy_info = {
        "drawdown": pct_from_high(spy),
        "chg_1m":   (spy.iloc[-1] / spy.iloc[-21] - 1) * 100 if len(spy) >= 22 else 0,
        "chg_3m":   (spy.iloc[-1] / spy.iloc[-63] - 1) * 100 if len(spy) >= 64 else 0,
    }

    # Scan
    results = scan(prices, min_price=args.min_price)

    if results.empty:
        print("  No results found.")
        return

    # Terminal output
    print_results(results, args.top, spy_info)

    # Save CSV
    csv_path = os.path.join(args.out_dir, f"rs_scan_{today}.csv")
    results.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"  CSV saved → {csv_path}")

    # HTML report
    html_path = os.path.join(args.out_dir, f"rs_scan_{today}.html")
    save_html(results, args.top, spy_info, html_path)

    # RS line chart for top 15
    chart_path = ""
    if not args.no_chart:
        chart_n    = min(15, args.top)
        chart_path = os.path.join(args.out_dir, f"rs_scan_{today}.png")
        make_rs_chart(results, prices, chart_n, chart_path)

    # Email
    if not args.no_email:
        send_email(results, args.top, spy_info, html_path, chart_path)

    print(f"\n  Done — {len(results)} stocks ranked, top {args.top} shown.\n")


if __name__ == "__main__":
    main()
