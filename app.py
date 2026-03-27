#!/usr/bin/env python3
"""
Mag 7 Stock Dashboard — powered by Alpaca Market Data API
"""
from flask import Flask, jsonify, render_template_string
import requests
import os
from datetime import datetime, timedelta, timezone

app = Flask(__name__)

# ── Alpaca credentials ──────────────────────────────────────────────────────
ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

DATA_BASE_URL = "https://data.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID":     ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

NAMES = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN":  "Amazon",
    "NVDA":  "NVIDIA",
    "META":  "Meta",
    "TSLA":  "Tesla",
}

COLORS = {
    "AAPL":  "#A2AAAD",
    "MSFT":  "#00A4EF",
    "GOOGL": "#4285F4",
    "AMZN":  "#FF9900",
    "NVDA":  "#76B900",
    "META":  "#0866FF",
    "TSLA":  "#CC0000",
}


def get_snapshots():
    """Fetch latest snapshots (quote + daily bar) for all Mag7 tickers."""
    symbols = ",".join(MAG7)
    url = f"{DATA_BASE_URL}/v2/stocks/snapshots"
    params = {"symbols": symbols, "feed": "iex"}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_bars(symbol, days=30):
    """Fetch daily bars for a symbol over the last N calendar days."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 10)   # buffer for weekends/holidays
    url   = f"{DATA_BASE_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Day",
        "start":     start.strftime("%Y-%m-%d"),
        "end":       end.strftime("%Y-%m-%d"),
        "limit":     days,
        "feed":      "iex",
    }
    resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    bars = resp.json().get("bars", [])
    return bars[-days:] if len(bars) > days else bars


@app.route("/api/overview")
def api_overview():
    data = get_snapshots()
    result = []
    for sym in MAG7:
        snap = data.get(sym, {})
        daily = snap.get("dailyBar", {})
        prev  = snap.get("prevDailyBar", {})
        quote = snap.get("latestQuote", {})

        close  = daily.get("c") or quote.get("ap") or 0
        prev_c = prev.get("c") or close
        chg    = close - prev_c
        pct    = (chg / prev_c * 100) if prev_c else 0

        result.append({
            "symbol":  sym,
            "name":    NAMES[sym],
            "color":   COLORS[sym],
            "price":   round(close, 2),
            "change":  round(chg, 2),
            "pct":     round(pct, 2),
            "open":    round(daily.get("o") or 0, 2),
            "high":    round(daily.get("h") or 0, 2),
            "low":     round(daily.get("l") or 0, 2),
            "volume":  daily.get("v") or 0,
            "vwap":    round(daily.get("vw") or 0, 2),
        })
    return jsonify(result)


@app.route("/api/chart/<symbol>")
def api_chart(symbol):
    if symbol not in MAG7:
        return jsonify({"error": "invalid symbol"}), 400
    bars = get_bars(symbol, days=30)
    dates  = [b["t"][:10] for b in bars]
    closes = [round(b["c"], 2) for b in bars]
    # normalise to % from first close
    base = closes[0] if closes else 1
    normed = [round((c - base) / base * 100, 2) for c in closes]
    return jsonify({"dates": dates, "closes": closes, "normed": normed})


@app.route("/api/compare")
def api_compare():
    """Return 30-day normalised series for all Mag7 — for the compare chart."""
    out = {}
    for sym in MAG7:
        bars = get_bars(sym, days=30)
        dates  = [b["t"][:10] for b in bars]
        closes = [round(b["c"], 2) for b in bars]
        base   = closes[0] if closes else 1
        normed = [round((c - base) / base * 100, 2) for c in closes]
        out[sym] = {"dates": dates, "normed": normed, "color": COLORS[sym]}
    return jsonify(out)


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mag 7 Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
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
  }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; min-height: 100vh; }

  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 32px; border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  header h1 { font-size: 1.35rem; font-weight: 700; letter-spacing: .5px; }
  header h1 span { color: #f0a500; }
  #last-updated { font-size: .78rem; color: var(--muted); }
  #refresh-btn {
    background: #238636; border: none; color: #fff; padding: 6px 16px;
    border-radius: 6px; cursor: pointer; font-size: .85rem; font-weight: 600;
    transition: background .2s;
  }
  #refresh-btn:hover { background: #2ea043; }

  main { max-width: 1400px; margin: 0 auto; padding: 28px 24px; }

  /* ── Cards grid ─────────────────────────────────────── */
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 16px; margin-bottom: 36px; }
  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px 16px; cursor: pointer; transition: border-color .2s, transform .15s;
    position: relative; overflow: hidden;
  }
  .card:hover, .card.active { border-color: var(--accent); transform: translateY(-2px); }
  .card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--accent); }
  .card .ticker { font-size: .75rem; font-weight: 700; color: var(--muted); letter-spacing: 1px; margin-bottom: 2px; }
  .card .company { font-size: .82rem; color: var(--muted); margin-bottom: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .card .price { font-size: 1.45rem; font-weight: 700; }
  .card .change { font-size: .82rem; font-weight: 600; margin-top: 4px; }
  .card .change.up   { color: var(--green); }
  .card .change.down { color: var(--red);   }
  .card .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 10px; margin-top: 12px; border-top: 1px solid var(--border); padding-top: 10px; }
  .card .stat-label { font-size: .68rem; color: var(--muted); }
  .card .stat-value { font-size: .78rem; font-weight: 600; }

  /* ── Chart section ──────────────────────────────────── */
  .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 800px) { .charts-row { grid-template-columns: 1fr; } }
  .chart-box {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px;
  }
  .chart-box h3 { font-size: .9rem; color: var(--muted); margin-bottom: 14px; font-weight: 600; }
  .chart-box h3 span { color: var(--text); }
  canvas { width: 100% !important; }

  .spinner { text-align: center; padding: 60px; color: var(--muted); font-size: 1rem; }
  .error-msg { color: var(--red); background: #300; border: 1px solid var(--red); border-radius: 8px; padding: 14px 18px; margin: 20px 0; font-size: .9rem; }
</style>
</head>
<body>

<header>
  <h1>Mag <span>7</span> Dashboard</h1>
  <div style="display:flex; align-items:center; gap:16px;">
    <span id="last-updated">Loading…</span>
    <button id="refresh-btn" onclick="loadAll()">↻ Refresh</button>
  </div>
</header>

<main>
  <div id="error-box"></div>
  <div id="cards-container" class="cards"><div class="spinner">Fetching live data…</div></div>
  <div class="charts-row">
    <div class="chart-box">
      <h3>Price — <span id="chart-title">select a stock</span></h3>
      <canvas id="priceChart" height="220"></canvas>
    </div>
    <div class="chart-box">
      <h3>30-Day Performance Comparison <span style="font-size:.75rem;color:var(--muted)">(% return, equal start)</span></h3>
      <canvas id="compareChart" height="220"></canvas>
    </div>
  </div>
</main>

<script>
let priceChart   = null;
let compareChart = null;
let activeSymbol = null;
const COLORS = {{ colors | tojson }};

function fmt(n)    { return n.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}); }
function fmtVol(v) { if(v>=1e9) return (v/1e9).toFixed(2)+'B'; if(v>=1e6) return (v/1e6).toFixed(2)+'M'; return v.toLocaleString(); }

function showError(msg) {
  document.getElementById('error-box').innerHTML =
    `<div class="error-msg">⚠ ${msg}</div>`;
}

function renderCards(stocks) {
  const el = document.getElementById('cards-container');
  el.innerHTML = '';
  stocks.forEach(s => {
    const up    = s.pct >= 0;
    const arrow = up ? '▲' : '▼';
    const cls   = up ? 'up' : 'down';
    const div   = document.createElement('div');
    div.className = 'card' + (activeSymbol === s.symbol ? ' active' : '');
    div.style.setProperty('--accent', s.color);
    div.innerHTML = `
      <div class="ticker">${s.symbol}</div>
      <div class="company">${s.name}</div>
      <div class="price">$${fmt(s.price)}</div>
      <div class="change ${cls}">${arrow} ${fmt(Math.abs(s.change))} (${fmt(Math.abs(s.pct))}%)</div>
      <div class="stats">
        <div class="stat-label">Open</div>  <div class="stat-value">$${fmt(s.open)}</div>
        <div class="stat-label">High</div>  <div class="stat-value">$${fmt(s.high)}</div>
        <div class="stat-label">Low</div>   <div class="stat-value">$${fmt(s.low)}</div>
        <div class="stat-label">Vol</div>   <div class="stat-value">${fmtVol(s.volume)}</div>
        <div class="stat-label">VWAP</div>  <div class="stat-value">$${fmt(s.vwap)}</div>
      </div>`;
    div.addEventListener('click', () => loadPriceChart(s.symbol, s.name, s.color));
    el.appendChild(div);
  });
}

async function loadPriceChart(symbol, name, color) {
  activeSymbol = symbol;
  document.querySelectorAll('.card').forEach(c => c.classList.remove('active'));
  const clicked = [...document.querySelectorAll('.card')].find(c => c.querySelector('.ticker')?.textContent === symbol);
  if (clicked) clicked.classList.add('active');

  document.getElementById('chart-title').textContent = name + ' (' + symbol + ')';
  const res  = await fetch('/api/chart/' + symbol);
  const data = await res.json();

  if (priceChart) priceChart.destroy();
  const ctx = document.getElementById('priceChart').getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, 260);
  grad.addColorStop(0, color + '55');
  grad.addColorStop(1, color + '00');

  priceChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.dates,
      datasets: [{
        data: data.closes,
        borderColor: color,
        backgroundColor: grad,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        fill: true,
        tension: 0.3,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false }, tooltip: {
        callbacks: { label: ctx => ' $' + fmt(ctx.parsed.y) }
      }},
      scales: {
        x: { grid: { color: '#30363d' }, ticks: { color: '#8b949e', maxTicksLimit: 6 } },
        y: { grid: { color: '#30363d' }, ticks: { color: '#8b949e', callback: v => '$' + v } }
      }
    }
  });
}

async function loadCompareChart() {
  const res  = await fetch('/api/compare');
  const data = await res.json();
  const symbols = Object.keys(data);
  const labels  = data[symbols[0]].dates;

  if (compareChart) compareChart.destroy();
  const ctx = document.getElementById('compareChart').getContext('2d');
  compareChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: symbols.map(sym => ({
        label: sym,
        data:  data[sym].normed,
        borderColor:     data[sym].color,
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.3,
      }))
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#e6edf3', boxWidth: 12, padding: 14, font: { size: 11 } } },
        tooltip: { callbacks: { label: ctx => ' ' + ctx.dataset.label + ': ' + fmt(ctx.parsed.y) + '%' } }
      },
      scales: {
        x: { grid: { color: '#30363d' }, ticks: { color: '#8b949e', maxTicksLimit: 6 } },
        y: { grid: { color: '#30363d' }, ticks: { color: '#8b949e', callback: v => v + '%' } }
      }
    }
  });
}

async function loadAll() {
  document.getElementById('error-box').innerHTML = '';
  try {
    const res    = await fetch('/api/overview');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const stocks = await res.json();
    renderCards(stocks);
    document.getElementById('last-updated').textContent = 'Updated ' + new Date().toLocaleTimeString();

    // auto-select first card
    if (!activeSymbol && stocks.length) {
      const s = stocks[0];
      loadPriceChart(s.symbol, s.name, s.color);
    }

    loadCompareChart();
  } catch(e) {
    showError('Could not fetch data. Check your Alpaca API keys and try again. (' + e.message + ')');
  }
}

loadAll();
// Auto-refresh every 60 seconds
setInterval(loadAll, 60000);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    import json
    colors_json = json.dumps(COLORS)
    # Render with a simple string replace so we don't need Jinja2 templates
    html = HTML.replace("{{ colors | tojson }}", colors_json)
    return html


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n  Mag 7 Dashboard running at http://localhost:{port}\n")
    app.run(debug=False, port=port)
