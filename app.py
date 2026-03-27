#!/usr/bin/env python3
"""
Relative Strength Scanner — Web Dashboard
Powered by yfinance (free, no API key required)
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import base64
import io
import json
import os
import threading
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request

app = Flask(__name__)

# ── Universe helpers ──────────────────────────────────────────────────────────

def get_sp500_universe():
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"  S&P 500: {len(tickers)} stocks")
        return tickers
    except Exception as e:
        print(f"  S&P 500 fetch failed ({e}) — using fallback")
        return _fallback_universe()


def _fallback_universe():
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
        "SBUX","CMG","YUM","TGT","DLTR","DG","SLB","HAL","COP","OXY",
        "EMR","ITW","ROK","AXON","KTOS","RKLB","F","GM","DE","FCX","NEM",
        "WM","RSG","O","SPG","VICI","EQR","AVB",
    ]))


# ── Price download ────────────────────────────────────────────────────────────

def download_prices(tickers, period="1y"):
    all_tickers = sorted(set(tickers + ["SPY"]))
    total = len(all_tickers)
    CHUNK = 100
    batches = [all_tickers[i:i+CHUNK] for i in range(0, total, CHUNK)]
    print(f"\n  Downloading {total} tickers in {len(batches)} batches…")
    all_closes = {}

    for idx, batch in enumerate(batches, 1):
        print(f"  Batch {idx}/{len(batches)} ({len(batch)} tickers)…", end=" ", flush=True)
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


# ── RS calculations ───────────────────────────────────────────────────────────

def rs_score(stock_series, spy_series):
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

    weights = [(63, 2), (126, 1), (189, 1), (252, 1)]
    total_w = sum(w for _, w in weights)
    score = 0.0
    for n, w in weights:
        r = rel(n)
        if np.isnan(r):
            r = 0.0
        score += r * w
    return score / total_w


def rs_line_slope(stock_series, spy_series, days=20):
    if len(stock_series) < days or len(spy_series) < days:
        return np.nan
    rs_line = (stock_series / spy_series).iloc[-days:]
    if rs_line.empty or rs_line.isna().all():
        return np.nan
    x = np.arange(len(rs_line))
    slope = np.polyfit(x, rs_line.values.astype(float), 1)[0]
    mean_rs = rs_line.mean()
    return (slope / mean_rs * 100) if mean_rs != 0 else np.nan


def above_ma(series, window):
    if len(series) < window:
        return False
    ma = series.rolling(window=window, min_periods=window).mean().iloc[-1]
    return bool(series.iloc[-1] > ma)


def pct_from_high(series, window=252):
    if series.empty:
        return np.nan
    high = series.iloc[-window:].max() if len(series) >= window else series.max()
    return (series.iloc[-1] / high - 1) * 100


def run_scan_df(prices, min_price=5.0):
    spy = prices["SPY"].dropna()
    spy_dd = pct_from_high(spy)
    rows = []
    tickers = [c for c in prices.columns if c != "SPY"]
    total = len(tickers)
    print(f"\n  Computing RS metrics for {total} stocks…")

    for i, ticker in enumerate(tickers, 1):
        if i % 100 == 0:
            print(f"    {i}/{total}…")
        s = prices[ticker].dropna()
        if s.empty or s.iloc[-1] < min_price or len(s) < 63:
            continue
        score = rs_score(s, spy)
        slope = rs_line_slope(s, spy, days=20)
        dd = pct_from_high(s)
        spy_delta = dd - spy_dd
        price = round(float(s.iloc[-1]), 2)
        chg_1m = (s.iloc[-1] / s.iloc[-21] - 1) * 100 if len(s) >= 22 else np.nan
        chg_3m = (s.iloc[-1] / s.iloc[-63] - 1) * 100 if len(s) >= 64 else np.nan

        rows.append({
            "Ticker":       ticker,
            "Price":        price,
            "RS_Score":     round(float(score), 1)     if not np.isnan(score) else np.nan,
            "RS_Slope_20d": round(float(slope), 2)     if not np.isnan(slope) else np.nan,
            "Drawdown%":    round(float(dd), 1)        if not np.isnan(dd)    else np.nan,
            "vs_SPY_DD":    round(float(spy_delta), 1) if not np.isnan(spy_delta) else np.nan,
            "Chg_1M%":      round(float(chg_1m), 1)   if not np.isnan(chg_1m) else np.nan,
            "Chg_3M%":      round(float(chg_3m), 1)   if not np.isnan(chg_3m) else np.nan,
            "Above_21MA":   above_ma(s, 21),
            "Above_50MA":   above_ma(s, 50),
            "Above_200MA":  above_ma(s, 200),
        })

    df = pd.DataFrame(rows).dropna(subset=["RS_Score"])
    df["RS_Rank"] = df["RS_Score"].rank(pct=True).mul(99).round(0).astype(int)
    df.sort_values("RS_Score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── RS Line Chart (base64 PNG) ────────────────────────────────────────────────

def make_chart_b64(df, prices, top_n=15):
    leaders = df.head(top_n)["Ticker"].tolist()
    spy = prices["SPY"].dropna()

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    colors = plt.cm.tab20.colors

    for i, ticker in enumerate(leaders):
        if ticker not in prices.columns:
            continue
        s = prices[ticker].dropna()
        common = s.index.intersection(spy.index)
        if len(common) < 20:
            continue
        rs = s[common] / spy[common]
        rs_norm = rs / rs.iloc[0] * 100
        ax.plot(rs_norm.index, rs_norm.values,
                color=colors[i % len(colors)], linewidth=1.4,
                label=ticker, alpha=0.9)

    ax.axhline(100, color="#8b949e", linewidth=0.8, linestyle="--", alpha=0.5, label="SPY baseline")
    ax.set_title(f"RS Lines — Top {top_n} Leaders  (stock / SPY, normalized to 100)",
                 color="#e6edf3", fontsize=11, pad=10)
    ax.set_ylabel("Relative Strength vs SPY", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=25, ha="right")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.4,
              labelcolor="#e6edf3", facecolor="#0d1117", ncol=2)
    ax.grid(color="#30363d", linewidth=0.4, alpha=0.6)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Scan state (in-memory cache) ─────────────────────────────────────────────

_state = {
    "status":       "idle",   # idle | scanning | ready | error
    "started_at":   None,
    "finished_at":  None,
    "universe":     None,
    "total_ranked": 0,
    "results":      None,     # list of dicts, top 50
    "spy_info":     None,
    "chart_b64":    None,
    "error":        None,
}
_lock = threading.Lock()


def _do_scan(universe="sp500"):
    with _lock:
        _state["status"]     = "scanning"
        _state["started_at"] = datetime.now().isoformat()
        _state["universe"]   = universe
        _state["error"]      = None

    try:
        if universe == "sp500":
            tickers = get_sp500_universe()
        else:
            tickers = _fallback_universe()

        prices = download_prices(tickers)

        if "SPY" not in prices.columns:
            raise RuntimeError("SPY data not available — check network")

        spy = prices["SPY"].dropna()
        spy_info = {
            "price":    round(float(spy.iloc[-1]), 2),
            "drawdown": round(float(pct_from_high(spy)), 2),
            "chg_1m":   round(float((spy.iloc[-1] / spy.iloc[-21] - 1) * 100), 2) if len(spy) >= 22 else 0,
            "chg_3m":   round(float((spy.iloc[-1] / spy.iloc[-63] - 1) * 100), 2) if len(spy) >= 64 else 0,
        }

        df = run_scan_df(prices)

        results = []
        for rank, (_, row) in enumerate(df.head(50).iterrows(), 1):
            results.append({
                "rank":      rank,
                "ticker":    row["Ticker"],
                "price":     row["Price"],
                "rs_rank":   int(row["RS_Rank"]),
                "chg_1m":    row["Chg_1M%"],
                "chg_3m":    row["Chg_3M%"],
                "drawdown":  row["Drawdown%"],
                "vs_spy":    row["vs_SPY_DD"],
                "slope":     row["RS_Slope_20d"] if not pd.isna(row["RS_Slope_20d"]) else None,
                "above_21":  bool(row["Above_21MA"]),
                "above_50":  bool(row["Above_50MA"]),
                "above_200": bool(row["Above_200MA"]),
            })

        print("  Building chart…")
        chart_b64 = make_chart_b64(df, prices, top_n=15)

        with _lock:
            _state["status"]       = "ready"
            _state["finished_at"]  = datetime.now().isoformat()
            _state["results"]      = results
            _state["spy_info"]     = spy_info
            _state["total_ranked"] = len(df)
            _state["chart_b64"]    = chart_b64

        print(f"\n  Scan complete — {len(df)} stocks ranked.\n")

    except Exception as e:
        import traceback
        traceback.print_exc()
        with _lock:
            _state["status"] = "error"
            _state["error"]  = str(e)


def start_scan(universe="sp500"):
    t = threading.Thread(target=_do_scan, args=(universe,), daemon=True)
    t.start()


# ── Kick off initial scan when first request arrives ──────────────────────────

_init_done = False
_init_lock = threading.Lock()

@app.before_request
def _init_on_first_request():
    global _init_done
    if not _init_done:
        with _init_lock:
            if not _init_done:
                _init_done = True
                start_scan()


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify({k: v for k, v in _state.items() if k != "chart_b64"})


@app.route("/api/results")
def api_results():
    with _lock:
        return jsonify({
            "status":       _state["status"],
            "finished_at":  _state["finished_at"],
            "universe":     _state["universe"],
            "total_ranked": _state["total_ranked"],
            "spy_info":     _state["spy_info"],
            "results":      _state["results"],
            "chart_b64":    _state["chart_b64"],
        })


@app.route("/api/rescan", methods=["POST"])
def api_rescan():
    with _lock:
        if _state["status"] == "scanning":
            return jsonify({"error": "Scan already running"}), 409
    universe = request.json.get("universe", "sp500") if request.is_json else "sp500"
    start_scan(universe)
    return jsonify({"ok": True, "universe": universe})


# ── Frontend ──────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RS Scanner — Relative Strength Leaders</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:      #0d1117;
  --surface: #161b22;
  --border:  #30363d;
  --text:    #e6edf3;
  --muted:   #8b949e;
  --green:   #3fb950;
  --red:     #f85149;
  --amber:   #f0a500;
  --blue:    #58a6ff;
  --purple:  #bc8cff;
}
body { background: var(--bg); color: var(--text);
       font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       min-height: 100vh; }

header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 28px; border-bottom: 1px solid var(--border);
  background: var(--surface); position: sticky; top: 0; z-index: 10;
  flex-wrap: wrap; gap: 10px;
}
header h1 { font-size: 1.1rem; font-weight: 700; white-space: nowrap; }
header h1 span { color: var(--amber); }
.hdr-right { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
#scan-meta { font-size: .76rem; color: var(--muted); }

.btn {
  background: #238636; border: none; color: #fff; padding: 6px 14px;
  border-radius: 6px; cursor: pointer; font-size: .82rem; font-weight: 600;
  transition: background .2s; white-space: nowrap;
}
.btn:hover { background: #2ea043; }
.btn:disabled { background: #1c3526; color: #4a7c59; cursor: not-allowed; }
.btn-secondary { background: #1f3a5f; }
.btn-secondary:hover { background: #2255a0; }

main { max-width: 1400px; margin: 0 auto; padding: 20px 16px; }

/* Scanning overlay */
#scan-overlay {
  display: none; text-align: center; padding: 80px 20px;
  color: var(--muted);
}
#scan-overlay.visible { display: block; }
.spinner {
  width: 40px; height: 40px; border: 3px solid var(--border);
  border-top-color: var(--amber); border-radius: 50%;
  animation: spin .8s linear infinite; margin: 0 auto 20px;
}
@keyframes spin { to { transform: rotate(360deg); } }
#scan-overlay p { font-size: 1rem; margin-bottom: 8px; }
#scan-overlay small { font-size: .78rem; color: var(--muted); }

/* SPY bar */
#spy-bar {
  display: flex; gap: 24px; align-items: center; flex-wrap: wrap;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 18px; margin-bottom: 18px;
  font-size: .82rem;
}
#spy-bar .label { color: var(--muted); font-size: .74rem; font-weight: 600;
                  letter-spacing: .5px; text-transform: uppercase; }
#spy-bar .val { font-weight: 700; }

/* Table */
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: .82rem; }
thead th {
  background: var(--surface); color: var(--muted); font-size: .72rem;
  font-weight: 700; letter-spacing: .5px; text-transform: uppercase;
  padding: 9px 10px; border-bottom: 1px solid var(--border);
  text-align: right; white-space: nowrap; cursor: pointer;
  user-select: none;
}
thead th:first-child,
thead th:nth-child(2) { text-align: left; }
thead th:hover { color: var(--text); }
thead th.sort-asc::after  { content: " ↑"; color: var(--amber); }
thead th.sort-desc::after { content: " ↓"; color: var(--amber); }

tbody tr { border-bottom: 1px solid var(--border); transition: background .1s; }
tbody tr:hover { background: #1c2128; }
tbody td { padding: 9px 10px; text-align: right; }
tbody td:first-child { text-align: left; color: var(--muted); font-size: .75rem; }
tbody td:nth-child(2) { text-align: left; }

.ticker { font-weight: 700; font-size: .88rem; color: var(--text); }
.price  { color: var(--muted); font-size: .8rem; margin-top: 1px; }
.rs-pill {
  display: inline-block; padding: 2px 8px; border-radius: 10px;
  font-weight: 800; font-size: .78rem; min-width: 36px; text-align: center;
}
.rs-high   { background: #0d3320; color: #3fb950; border: 1px solid #3fb950; }
.rs-mid    { background: #1f3a5f; color: #58a6ff; border: 1px solid #58a6ff; }
.rs-low    { background: #2a1f00; color: #f0a500; border: 1px solid #f0a500; }
.up   { color: var(--green); }
.dn   { color: var(--red); }
.neu  { color: var(--muted); }

.ma-dots { display: flex; gap: 3px; justify-content: flex-end; }
.dot {
  width: 22px; height: 18px; border-radius: 3px; font-size: .6rem;
  font-weight: 700; display: flex; align-items: center; justify-content: center;
}
.dot.yes { background: #0d3320; color: #3fb950; }
.dot.no  { background: #300; color: #f85149; }

.slope-up  { color: var(--green); }
.slope-dn  { color: var(--red); }
.slope-neu { color: var(--muted); }

/* Chart */
#chart-section { margin-top: 24px; }
#chart-section h3 { font-size: .85rem; color: var(--muted); margin-bottom: 12px; font-weight: 600; }
#chart-img { width: 100%; border-radius: 8px; border: 1px solid var(--border); display: none; }
#chart-img.visible { display: block; }

.error-msg {
  color: var(--red); background: #300; border: 1px solid var(--red);
  border-radius: 8px; padding: 12px 16px; margin: 16px 0; font-size: .88rem;
}

/* Legend */
.legend {
  margin-top: 14px; padding: 10px 14px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; font-size: .74rem; color: var(--muted); line-height: 1.7;
}
.legend b { color: var(--text); }
</style>
</head>
<body>

<header>
  <h1>Relative <span>Strength</span> Scanner</h1>
  <div class="hdr-right">
    <span id="scan-meta">Starting scan…</span>
    <select id="universe-sel" style="background:#1c2128;border:1px solid var(--border);color:var(--text);padding:5px 8px;border-radius:6px;font-size:.82rem">
      <option value="sp500">S&amp;P 500</option>
      <option value="fallback">Core 150</option>
    </select>
    <button class="btn" id="rescan-btn" onclick="triggerRescan()">&#8635; Rescan</button>
  </div>
</header>

<main>
  <div id="error-box"></div>

  <!-- Scanning state -->
  <div id="scan-overlay">
    <div class="spinner"></div>
    <p>Scanning market for relative strength leaders…</p>
    <small>Downloading ~500 tickers via yfinance. This takes 3–5 minutes on first load.</small>
  </div>

  <!-- SPY benchmark -->
  <div id="spy-bar" style="display:none">
    <span class="label">SPY</span>
    <span><span class="label">Price</span> &nbsp;<span class="val" id="spy-price">—</span></span>
    <span><span class="label">1M</span> &nbsp;<span class="val" id="spy-1m">—</span></span>
    <span><span class="label">3M</span> &nbsp;<span class="val" id="spy-3m">—</span></span>
    <span><span class="label">52W DD</span> &nbsp;<span class="val" id="spy-dd">—</span></span>
    <span style="margin-left:auto;color:var(--muted);font-size:.75rem" id="ranked-count"></span>
  </div>

  <!-- Leaderboard -->
  <div id="results-section" style="display:none">
    <div class="table-wrap">
      <table id="rs-table">
        <thead>
          <tr>
            <th>#</th>
            <th data-col="ticker">Ticker</th>
            <th data-col="rs_rank">RS Rank</th>
            <th data-col="price">Price</th>
            <th data-col="chg_1m">1M Chg</th>
            <th data-col="chg_3m">3M Chg</th>
            <th data-col="drawdown">Drawdown</th>
            <th data-col="vs_spy">vs SPY DD</th>
            <th data-col="slope">RS Slope</th>
            <th>MAs 21/50/200</th>
          </tr>
        </thead>
        <tbody id="rs-tbody"></tbody>
      </table>
    </div>

    <div class="legend">
      <b>RS Rank</b>: IBD-style percentile vs universe (99 = strongest, weighted recent performance). &nbsp;
      <b>vs SPY DD</b>: your drawdown minus SPY's — positive means holding up better than the market. &nbsp;
      <b>RS Slope</b>: 20-day trend of the stock/SPY ratio — positive = RS line is rising (gaining on market). &nbsp;
      <b>MAs</b>: whether price is above the 21 / 50 / 200-day moving average.
    </div>

    <!-- Chart -->
    <div id="chart-section">
      <h3>RS Lines — Top 15 Leaders &nbsp;<span style="font-size:.72rem;color:var(--muted)">(stock/SPY normalized to 100 at start of year)</span></h3>
      <img id="chart-img" alt="RS Lines chart">
    </div>
  </div>
</main>

<script>
let _data = [];
let _sortCol = null;
let _sortAsc = true;
let _pollTimer = null;

const fmt2 = n => n == null ? '—' : n.toFixed(2);
const fmtPct = n => n == null ? '—' : (n >= 0 ? '+' : '') + n.toFixed(1) + '%';
const fmtPctC = (n, el) => {
  const s = fmtPct(n);
  if (n == null) { el.textContent = s; el.className = 'neu'; return; }
  el.textContent = s;
  el.className = n >= 0 ? 'up' : 'dn';
};

function rsPillClass(rs) {
  if (rs >= 80) return 'rs-high';
  if (rs >= 60) return 'rs-mid';
  return 'rs-low';
}

function renderTable(rows) {
  const tbody = document.getElementById('rs-tbody');
  tbody.innerHTML = '';
  rows.forEach((r, i) => {
    const tr = document.createElement('tr');

    // MA dots
    const makeDot = (label, val) =>
      `<div class="dot ${val ? 'yes' : 'no'}">${label}</div>`;

    const slopeClass = r.slope == null ? 'slope-neu' : r.slope > 0 ? 'slope-up' : 'slope-dn';
    const slopeText  = r.slope == null ? '—' : (r.slope > 0 ? '+' : '') + r.slope.toFixed(2);

    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>
        <div class="ticker">${r.ticker}</div>
        <div class="price">$${fmt2(r.price)}</div>
      </td>
      <td><span class="rs-pill ${rsPillClass(r.rs_rank)}">${r.rs_rank}</span></td>
      <td>$${fmt2(r.price)}</td>
      <td class="${r.chg_1m >= 0 ? 'up' : 'dn'}">${fmtPct(r.chg_1m)}</td>
      <td class="${r.chg_3m >= 0 ? 'up' : 'dn'}">${fmtPct(r.chg_3m)}</td>
      <td class="${(r.drawdown || 0) >= 0 ? 'up' : 'dn'}">${fmtPct(r.drawdown)}</td>
      <td class="${(r.vs_spy || 0) >= 0 ? 'up' : 'dn'}">${fmtPct(r.vs_spy)}</td>
      <td class="${slopeClass}">${slopeText}</td>
      <td>
        <div class="ma-dots">
          ${makeDot('21', r.above_21)}
          ${makeDot('50', r.above_50)}
          ${makeDot('200', r.above_200)}
        </div>
      </td>`;
    tbody.appendChild(tr);
  });
}

function sortData(col) {
  if (_sortCol === col) {
    _sortAsc = !_sortAsc;
  } else {
    _sortCol = col;
    _sortAsc = col === 'ticker';
  }
  document.querySelectorAll('thead th').forEach(th => {
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.col === col) th.classList.add(_sortAsc ? 'sort-asc' : 'sort-desc');
  });

  const sorted = [..._data].sort((a, b) => {
    let va = a[col], vb = b[col];
    if (typeof va === 'string') va = va.toLowerCase();
    if (typeof vb === 'string') vb = vb.toLowerCase();
    if (va == null) return 1;
    if (vb == null) return -1;
    return _sortAsc ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  });
  renderTable(sorted);
}

// Wire up sortable headers
document.querySelectorAll('thead th[data-col]').forEach(th => {
  th.addEventListener('click', () => sortData(th.dataset.col));
});

function showSpy(spy) {
  const bar = document.getElementById('spy-bar');
  bar.style.display = 'flex';
  document.getElementById('spy-price').textContent = '$' + fmt2(spy.price);
  const set1m = document.getElementById('spy-1m');
  set1m.textContent = fmtPct(spy.chg_1m);
  set1m.className = 'val ' + (spy.chg_1m >= 0 ? 'up' : 'dn');
  const set3m = document.getElementById('spy-3m');
  set3m.textContent = fmtPct(spy.chg_3m);
  set3m.className = 'val ' + (spy.chg_3m >= 0 ? 'up' : 'dn');
  const setdd = document.getElementById('spy-dd');
  setdd.textContent = fmtPct(spy.drawdown);
  setdd.className = 'val dn';
}

function applyResults(data) {
  clearInterval(_pollTimer);
  document.getElementById('scan-overlay').classList.remove('visible');
  document.getElementById('results-section').style.display = 'block';

  _data = data.results || [];
  renderTable(_data);

  if (data.spy_info) showSpy(data.spy_info);

  const ts = data.finished_at
    ? 'Scanned ' + new Date(data.finished_at).toLocaleTimeString()
    : '';
  document.getElementById('scan-meta').textContent =
    (ts ? ts + ' · ' : '') + (data.universe === 'sp500' ? 'S&P 500' : 'Core 150') +
    ' · ' + (data.total_ranked || 0) + ' stocks ranked';

  document.getElementById('ranked-count').textContent =
    'Showing top ' + _data.length + ' of ' + (data.total_ranked || 0) + ' ranked';

  document.getElementById('rescan-btn').disabled = false;

  if (data.chart_b64) {
    const img = document.getElementById('chart-img');
    img.src = 'data:image/png;base64,' + data.chart_b64;
    img.classList.add('visible');
  }
}

function poll() {
  fetch('/api/results')
    .then(r => r.json())
    .then(data => {
      if (data.status === 'ready') {
        applyResults(data);
      } else if (data.status === 'error') {
        clearInterval(_pollTimer);
        document.getElementById('scan-overlay').classList.remove('visible');
        document.getElementById('error-box').innerHTML =
          `<div class="error-msg">Scan failed: ${data.error || 'unknown error'}</div>`;
        document.getElementById('rescan-btn').disabled = false;
      }
      // else still scanning — keep polling
    })
    .catch(() => {});
}

function triggerRescan() {
  document.getElementById('rescan-btn').disabled = true;
  document.getElementById('error-box').innerHTML = '';
  document.getElementById('results-section').style.display = 'none';
  document.getElementById('scan-overlay').classList.add('visible');
  document.getElementById('scan-meta').textContent = 'Scanning…';
  document.getElementById('chart-img').classList.remove('visible');

  const universe = document.getElementById('universe-sel').value;
  fetch('/api/rescan', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({universe}),
  })
  .then(r => r.json())
  .then(() => {
    _pollTimer = setInterval(poll, 5000);
  })
  .catch(e => {
    document.getElementById('error-box').innerHTML =
      `<div class="error-msg">Could not start scan: ${e.message}</div>`;
    document.getElementById('rescan-btn').disabled = false;
  });
}

// On load: check if scan is already ready or in progress
(function init() {
  fetch('/api/results')
    .then(r => r.json())
    .then(data => {
      if (data.status === 'ready') {
        applyResults(data);
      } else if (data.status === 'scanning' || data.status === 'idle') {
        document.getElementById('scan-overlay').classList.add('visible');
        document.getElementById('scan-meta').textContent = 'Scanning…';
        _pollTimer = setInterval(poll, 5000);
      } else if (data.status === 'error') {
        document.getElementById('error-box').innerHTML =
          `<div class="error-msg">Last scan failed: ${data.error}. Click Rescan to retry.</div>`;
        document.getElementById('rescan-btn').disabled = false;
      }
    });
})();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return HTML


if __name__ == "__main__":
    start_scan()
    port = int(os.environ.get("PORT", 5050))
    print(f"\n  RS Scanner Dashboard → http://localhost:{port}\n")
    app.run(debug=False, port=port)
