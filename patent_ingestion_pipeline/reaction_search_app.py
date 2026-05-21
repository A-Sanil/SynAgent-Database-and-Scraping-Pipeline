"""
Standalone Reaction Search Frontend
=====================================
A fully dedicated search UI for the reaction database.

Usage:
    cd patent_ingestion_pipeline
    set PATENT_DATA_DIR=D:\\SynAgent
    python reaction_search_app.py

Then open: http://localhost:8001
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

# ── DB discovery (same logic as ordtool.py) ─────────────────────────────────

def _find_db() -> Optional[Path]:
    candidates = [
        os.environ.get("PATENT_DATA_DIR"),
        r"D:\SynAgent",
        r"E:\SynAgent",
        r"F:\SynAgent",
        Path(__file__).parent / "data",
    ]
    for c in candidates:
        if c is None:
            continue
        db = Path(c) / "db" / "patent_pipeline.db"
        if db.exists():
            return db
    return None


def _conn():
    db = _find_db()
    if db is None:
        raise RuntimeError(
            "Database not found. Set PATENT_DATA_DIR to your USB path "
            "(e.g. D:\\SynAgent) or run the ingestion pipeline first."
        )
    c = sqlite3.connect(str(db))
    c.row_factory = sqlite3.Row
    return c


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Reaction Search")


# ── API routes ───────────────────────────────────────────────────────────────

@app.get("/api/stats")
def api_stats():
    try:
        conn = _conn()
        total    = conn.execute("SELECT COUNT(*) FROM reactions").fetchone()[0]
        ord_c    = conn.execute("SELECT COUNT(*) FROM reactions WHERE source='ord'").fetchone()[0]
        patent_c = conn.execute("SELECT COUNT(*) FROM reactions WHERE source='patent'").fetchone()[0]
        hi_conf  = conn.execute("SELECT COUNT(*) FROM reactions WHERE confidence >= 0.6").fetchone()[0]
        with_yield = conn.execute("SELECT COUNT(*) FROM reactions WHERE yield_percent IS NOT NULL").fetchone()[0]
        conn.close()
        return {"total": total, "ord": ord_c, "patent": patent_c,
                "high_confidence": hi_conf, "with_yield": with_yield,
                "db_path": str(_find_db())}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/filters")
def api_filters():
    """Return distinct solvents and catalysts for filter dropdowns."""
    try:
        conn = _conn()
        solvents = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT solvent FROM reactions "
                "WHERE solvent IS NOT NULL AND solvent != '' "
                "ORDER BY solvent LIMIT 200"
            )
        ]
        catalysts = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT catalyst FROM reactions "
                "WHERE catalyst IS NOT NULL AND catalyst != '' "
                "ORDER BY catalyst LIMIT 200"
            )
        ]
        conn.close()
        return {"solvents": solvents, "catalysts": catalysts}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/search")
def api_search(
    q: str = Query("", description="Keyword / SMILES / SMARTS search"),
    source: str = Query("all", description="all | ord | patent"),
    min_yield: Optional[float] = Query(None),
    max_yield: Optional[float] = Query(None),
    min_confidence: float = Query(0.0),
    solvent: str = Query(""),
    catalyst: str = Query(""),
    sort: str = Query("confidence", description="confidence | yield | temperature"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    try:
        conn = _conn()
        params: list = []
        conditions: list[str] = []

        # ── keyword / FTS search ──────────────────────────────────────────
        use_fts = bool(q.strip())
        base_select = "SELECT r.*"
        base_from   = "FROM reactions r"

        if use_fts:
            # Try FTS5 first
            try:
                fts_rows = conn.execute(
                    "SELECT reaction_id FROM reactions_fts WHERE reactions_fts MATCH ? LIMIT 2000",
                    (q.strip(),),
                ).fetchall()
                ids = [row[0] for row in fts_rows]
                if ids:
                    placeholders = ",".join("?" * len(ids))
                    conditions.append(f"r.reaction_id IN ({placeholders})")
                    params.extend(ids)
                else:
                    # FTS found nothing — fall back to LIKE
                    like = f"%{q.strip()[:60]}%"
                    conditions.append(
                        "(r.reaction_smarts LIKE ? OR r.product_smiles LIKE ? "
                        "OR r.solvent LIKE ? OR r.catalyst LIKE ? OR r.notes LIKE ?)"
                    )
                    params.extend([like, like, like, like, like])
            except sqlite3.OperationalError:
                # No FTS table — LIKE fallback
                like = f"%{q.strip()[:60]}%"
                conditions.append(
                    "(r.reaction_smarts LIKE ? OR r.product_smiles LIKE ? "
                    "OR r.solvent LIKE ? OR r.catalyst LIKE ? OR r.notes LIKE ?)"
                )
                params.extend([like, like, like, like, like])

        # ── filters ───────────────────────────────────────────────────────
        if source in ("ord", "patent"):
            conditions.append("r.source = ?")
            params.append(source)

        if min_yield is not None:
            conditions.append("r.yield_percent >= ?")
            params.append(min_yield)

        if max_yield is not None:
            conditions.append("r.yield_percent <= ?")
            params.append(max_yield)

        if min_confidence > 0:
            conditions.append("r.confidence >= ?")
            params.append(min_confidence)

        if solvent.strip():
            conditions.append("r.solvent LIKE ?")
            params.append(f"%{solvent.strip()}%")

        if catalyst.strip():
            conditions.append("r.catalyst LIKE ?")
            params.append(f"%{catalyst.strip()}%")

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        # ── sort ──────────────────────────────────────────────────────────
        sort_map = {
            "confidence":   "r.confidence DESC",
            "yield":        "r.yield_percent DESC NULLS LAST",
            "temperature":  "r.temperature_celsius ASC NULLS LAST",
        }
        order_by = sort_map.get(sort, "r.confidence DESC")

        # ── count ─────────────────────────────────────────────────────────
        count_sql = f"SELECT COUNT(*) {base_from} {where}"
        total_count = conn.execute(count_sql, params).fetchone()[0]

        # ── paginate ──────────────────────────────────────────────────────
        offset = (page - 1) * page_size
        data_sql = (
            f"{base_select} {base_from} {where} "
            f"ORDER BY {order_by} LIMIT ? OFFSET ?"
        )
        rows = conn.execute(data_sql, params + [page_size, offset]).fetchall()
        conn.close()

        results = []
        for row in rows:
            d = dict(row)
            # Parse reactant list
            try:
                d["reactants"] = json.loads(d.get("reactant_smiles_json") or "[]")
            except Exception:
                d["reactants"] = []
            results.append(d)

        return {
            "total": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total_count + page_size - 1) // page_size),
            "results": results,
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Frontend HTML ─────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Reaction Search — SynAgent</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body { background:#0f1117; color:#e2e8f0; font-family:'Inter',system-ui,sans-serif; }
  ::-webkit-scrollbar { width:6px; } ::-webkit-scrollbar-track { background:#1e2130; }
  ::-webkit-scrollbar-thumb { background:#3b4262; border-radius:3px; }
  .card { background:#1a1f2e; border:1px solid #2a3050; }
  .badge-ord { background:#0d3b2e; color:#34d399; border:1px solid #065f46; }
  .badge-patent { background:#1e1a3b; color:#a78bfa; border:1px solid #4c1d95; }
  .smiles-chip { background:#0f1117; border:1px solid #2a3050; font-family:'Fira Mono','Courier New',monospace;
    font-size:0.72rem; padding:2px 8px; border-radius:4px; cursor:pointer; transition:border-color .15s; }
  .smiles-chip:hover { border-color:#6366f1; color:#a5b4fc; }
  .filter-label { font-size:0.7rem; font-weight:600; text-transform:uppercase;
    letter-spacing:.05em; color:#64748b; }
  input[type=range] { accent-color:#6366f1; }
  .stat-card { background:#151b2d; border:1px solid #1e2744; }
  #search-input::placeholder { color:#475569; }
  .result-enter { animation: fadeIn .2s ease; }
  @keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
  .copy-toast { position:fixed; bottom:1.5rem; right:1.5rem; background:#1e293b;
    border:1px solid #334155; padding:.5rem 1rem; border-radius:.5rem;
    font-size:.8rem; color:#94a3b8; z-index:999; display:none; }
  select, input[type=text], input[type=number] {
    background:#0f1117; border:1px solid #2a3050; color:#e2e8f0;
    border-radius:.375rem; padding:.4rem .7rem; width:100%; font-size:.82rem;
  }
  select:focus, input:focus { outline:none; border-color:#6366f1; }
</style>
</head>
<body class="min-h-screen">

<!-- Header -->
<header class="border-b border-[#1e2744] px-6 py-4 flex items-center justify-between sticky top-0 z-10" style="background:#0d1120cc;backdrop-filter:blur(8px)">
  <div class="flex items-center gap-3">
    <div class="w-8 h-8 rounded-lg flex items-center justify-center text-lg" style="background:#1e2744">⚗️</div>
    <div>
      <h1 class="font-bold text-white text-base leading-none">Reaction Search</h1>
      <p class="text-xs text-slate-500 mt-0.5">SynAgent · Patent &amp; ORD Database</p>
    </div>
  </div>
  <div id="db-path" class="text-xs text-slate-600 hidden md:block"></div>
</header>

<!-- Stats bar -->
<div class="px-6 py-3 border-b border-[#1e2744] flex gap-4 flex-wrap" id="stats-bar">
  <div class="stat-card rounded-lg px-4 py-2 flex gap-3 items-center">
    <span class="text-slate-500 text-xs">Total</span>
    <span id="stat-total" class="font-bold text-white text-sm">—</span>
  </div>
  <div class="stat-card rounded-lg px-4 py-2 flex gap-3 items-center">
    <span class="text-xs" style="color:#34d399">ORD</span>
    <span id="stat-ord" class="font-bold text-sm" style="color:#34d399">—</span>
  </div>
  <div class="stat-card rounded-lg px-4 py-2 flex gap-3 items-center">
    <span class="text-xs" style="color:#a78bfa">Patent</span>
    <span id="stat-patent" class="font-bold text-sm" style="color:#a78bfa">—</span>
  </div>
  <div class="stat-card rounded-lg px-4 py-2 flex gap-3 items-center">
    <span class="text-slate-500 text-xs">w/ Yield</span>
    <span id="stat-yield" class="font-bold text-white text-sm">—</span>
  </div>
  <div class="stat-card rounded-lg px-4 py-2 flex gap-3 items-center">
    <span class="text-slate-500 text-xs">High Confidence</span>
    <span id="stat-hiconf" class="font-bold text-white text-sm">—</span>
  </div>
</div>

<!-- Main layout -->
<div class="flex">

  <!-- Sidebar filters -->
  <aside class="w-64 shrink-0 border-r border-[#1e2744] p-4 space-y-5 sticky top-[105px] h-[calc(100vh-105px)] overflow-y-auto">
    <div>
      <p class="filter-label mb-2">Source</p>
      <div class="space-y-1.5">
        <label class="flex items-center gap-2 cursor-pointer text-sm text-slate-300">
          <input type="radio" name="source" value="all" checked class="accent-indigo-500"> All sources
        </label>
        <label class="flex items-center gap-2 cursor-pointer text-sm" style="color:#34d399">
          <input type="radio" name="source" value="ord" class="accent-indigo-500"> ORD (expert)
        </label>
        <label class="flex items-center gap-2 cursor-pointer text-sm" style="color:#a78bfa">
          <input type="radio" name="source" value="patent" class="accent-indigo-500"> Patent (LLM)
        </label>
      </div>
    </div>

    <div>
      <p class="filter-label mb-2">Yield (%)</p>
      <div class="flex gap-2">
        <input type="number" id="min-yield" placeholder="Min" min="0" max="100" style="width:50%">
        <input type="number" id="max-yield" placeholder="Max" min="0" max="100" style="width:50%">
      </div>
    </div>

    <div>
      <p class="filter-label mb-2">Min Confidence: <span id="conf-val" class="text-indigo-400">0.0</span></p>
      <input type="range" id="min-confidence" min="0" max="1" step="0.05" value="0" class="w-full">
    </div>

    <div>
      <p class="filter-label mb-2">Solvent</p>
      <input type="text" id="solvent-input" placeholder="e.g. water, ethanol">
    </div>

    <div>
      <p class="filter-label mb-2">Catalyst</p>
      <input type="text" id="catalyst-input" placeholder="e.g. palladium, copper">
    </div>

    <div>
      <p class="filter-label mb-2">Sort by</p>
      <select id="sort-select">
        <option value="confidence">Confidence ↓</option>
        <option value="yield">Yield ↓</option>
        <option value="temperature">Temperature ↑</option>
      </select>
    </div>

    <button onclick="doSearch(1)" class="w-full py-2 rounded-lg text-sm font-semibold text-white transition-colors"
      style="background:#4f46e5">
      Apply Filters
    </button>
    <button onclick="resetFilters()" class="w-full py-2 rounded-lg text-sm text-slate-400 hover:text-white transition-colors border border-[#2a3050]">
      Reset
    </button>
  </aside>

  <!-- Main content -->
  <main class="flex-1 p-6">

    <!-- Search bar -->
    <div class="relative mb-5">
      <span class="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 text-lg">🔍</span>
      <input id="search-input" type="text"
        placeholder="Search by keyword, reaction SMARTS, product SMILES, solvent, catalyst..."
        class="w-full pl-11 pr-4 py-3 rounded-xl text-sm border border-[#2a3050] focus:border-indigo-500"
        style="background:#151b2d; font-size:.9rem;"
        onkeydown="if(event.key==='Enter') doSearch(1)">
      <button onclick="doSearch(1)"
        class="absolute right-3 top-1/2 -translate-y-1/2 px-4 py-1.5 rounded-lg text-sm font-semibold text-white"
        style="background:#4f46e5">Search</button>
    </div>

    <!-- Results header -->
    <div class="flex items-center justify-between mb-4">
      <p id="result-count" class="text-sm text-slate-400"></p>
      <div id="pagination-top" class="flex gap-2"></div>
    </div>

    <!-- Results -->
    <div id="results" class="space-y-3"></div>

    <!-- Pagination bottom -->
    <div id="pagination-bottom" class="flex justify-center gap-2 mt-6"></div>

  </main>
</div>

<!-- Copy toast -->
<div id="copy-toast" class="copy-toast">✓ Copied to clipboard</div>

<script>
let currentPage = 1;
let totalPages = 1;

// Load stats
async function loadStats() {
  const r = await fetch('/api/stats');
  const d = await r.json();
  if (d.error) return;
  document.getElementById('stat-total').textContent = d.total.toLocaleString();
  document.getElementById('stat-ord').textContent = d.ord.toLocaleString();
  document.getElementById('stat-patent').textContent = d.patent.toLocaleString();
  document.getElementById('stat-yield').textContent = d.with_yield.toLocaleString();
  document.getElementById('stat-hiconf').textContent = d.high_confidence.toLocaleString();
  if (d.db_path) document.getElementById('db-path').textContent = d.db_path;
}

// Confidence slider
document.getElementById('min-confidence').addEventListener('input', function() {
  document.getElementById('conf-val').textContent = parseFloat(this.value).toFixed(2);
});

// Search
async function doSearch(page) {
  currentPage = page || 1;
  const q = document.getElementById('search-input').value.trim();
  const source = document.querySelector('input[name=source]:checked').value;
  const minYield = document.getElementById('min-yield').value;
  const maxYield = document.getElementById('max-yield').value;
  const minConf = document.getElementById('min-confidence').value;
  const solvent = document.getElementById('solvent-input').value.trim();
  const catalyst = document.getElementById('catalyst-input').value.trim();
  const sort = document.getElementById('sort-select').value;

  const params = new URLSearchParams({
    q, source, min_confidence: minConf, sort, page: currentPage, page_size: 20
  });
  if (minYield) params.set('min_yield', minYield);
  if (maxYield) params.set('max_yield', maxYield);
  if (solvent) params.set('solvent', solvent);
  if (catalyst) params.set('catalyst', catalyst);

  document.getElementById('results').innerHTML = '<p class="text-slate-500 text-sm text-center py-12">Searching...</p>';

  const res = await fetch('/api/search?' + params);
  const data = await res.json();

  if (data.error) {
    document.getElementById('results').innerHTML =
      `<div class="card rounded-xl p-6 text-red-400 text-sm">Error: ${data.error}</div>`;
    return;
  }

  totalPages = data.total_pages;
  document.getElementById('result-count').textContent =
    `${data.total.toLocaleString()} result${data.total !== 1 ? 's' : ''}` +
    (q ? ` for "${q}"` : '');

  renderResults(data.results);
  renderPagination(data.page, data.total_pages);
}

function renderResults(results) {
  const el = document.getElementById('results');
  if (!results.length) {
    el.innerHTML = '<p class="text-slate-500 text-sm text-center py-16">No reactions found. Try a different search or adjust filters.</p>';
    return;
  }

  el.innerHTML = results.map((r, i) => {
    const isOrd = r.source === 'ord';
    const badge = isOrd
      ? '<span class="badge-ord text-xs px-2 py-0.5 rounded-full font-semibold">ORD · Expert</span>'
      : `<span class="badge-patent text-xs px-2 py-0.5 rounded-full font-semibold">Patent · LLM</span>`;

    const conf = typeof r.confidence === 'number'
      ? `<span class="text-xs ${r.confidence >= 0.7 ? 'text-green-400' : r.confidence >= 0.4 ? 'text-yellow-400' : 'text-red-400'}">${(r.confidence * 100).toFixed(0)}%</span>`
      : '';

    const yieldStr = r.yield_percent != null
      ? `<div class="flex items-center gap-1"><span class="text-slate-500 text-xs">Yield</span><span class="text-white font-semibold text-sm">${r.yield_percent.toFixed(1)}%</span></div>`
      : '';

    const tempStr = r.temperature_celsius != null
      ? `<div class="flex items-center gap-1"><span class="text-slate-500 text-xs">Temp</span><span class="text-white text-sm">${r.temperature_celsius.toFixed(0)}°C</span></div>`
      : '';

    const timeStr = r.time_hours != null
      ? `<div class="flex items-center gap-1"><span class="text-slate-500 text-xs">Time</span><span class="text-white text-sm">${r.time_hours.toFixed(1)}h</span></div>`
      : '';

    const solvent = r.solvent
      ? `<div class="flex items-start gap-1 flex-wrap"><span class="text-slate-500 text-xs shrink-0 mt-0.5">Solvent</span><span class="smiles-chip" onclick="copyText(this,'${esc(r.solvent)}')">${r.solvent.length > 60 ? r.solvent.slice(0,60)+'…' : r.solvent}</span></div>`
      : '';

    const catalyst = r.catalyst
      ? `<div class="flex items-start gap-1 flex-wrap"><span class="text-slate-500 text-xs shrink-0 mt-0.5">Catalyst</span><span class="smiles-chip" onclick="copyText(this,'${esc(r.catalyst)}')">${r.catalyst.length > 80 ? r.catalyst.slice(0,80)+'…' : r.catalyst}</span></div>`
      : '';

    const smarts = r.reaction_smarts
      ? `<div class="mt-2"><span class="text-slate-500 text-xs">SMARTS</span><div class="mt-1"><span class="smiles-chip inline-block max-w-full overflow-hidden text-ellipsis whitespace-nowrap" style="max-width:100%" onclick="copyText(this,'${esc(r.reaction_smarts)}')" title="${esc(r.reaction_smarts)}">${r.reaction_smarts.length > 120 ? r.reaction_smarts.slice(0,120)+'…' : r.reaction_smarts}</span></div></div>`
      : '';

    const product = r.product_smiles
      ? `<div><span class="text-slate-500 text-xs">Product</span><div class="mt-1"><span class="smiles-chip inline-block" onclick="copyText(this,'${esc(r.product_smiles)}')">${r.product_smiles.length > 100 ? r.product_smiles.slice(0,100)+'…' : r.product_smiles}</span></div></div>`
      : '';

    const reactants = r.reactants && r.reactants.length
      ? `<div><span class="text-slate-500 text-xs">Reactants</span><div class="flex flex-wrap gap-1 mt-1">${r.reactants.map(s => `<span class="smiles-chip" onclick="copyText(this,'${esc(s)}')">${s.length > 50 ? s.slice(0,50)+'…' : s}</span>`).join('')}</div></div>`
      : '';

    const notes = r.notes
      ? `<p class="text-slate-400 text-xs mt-2 leading-relaxed">${r.notes.slice(0,300)}${r.notes.length > 300 ? '…' : ''}</p>`
      : '';

    return `<div class="card rounded-xl p-4 result-enter" style="animation-delay:${i*0.03}s">
      <div class="flex items-center justify-between mb-3 flex-wrap gap-2">
        <div class="flex items-center gap-2 flex-wrap">
          ${badge}
          ${conf}
          <span class="text-slate-600 text-xs font-mono">${r.reaction_id ? r.reaction_id.slice(0,16) : ''}</span>
        </div>
        <div class="flex gap-4 flex-wrap">
          ${yieldStr}${tempStr}${timeStr}
        </div>
      </div>
      <div class="space-y-2">
        ${product}${reactants}${smarts}${solvent}${catalyst}
      </div>
      ${notes}
    </div>`;
  }).join('');
}

function renderPagination(page, total) {
  if (total <= 1) { ['pagination-top','pagination-bottom'].forEach(id => document.getElementById(id).innerHTML=''); return; }
  const html = `
    <button onclick="doSearch(${page-1})" ${page<=1?'disabled':''} class="px-3 py-1.5 rounded-lg text-sm border border-[#2a3050] text-slate-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed">← Prev</button>
    <span class="px-3 py-1.5 text-sm text-slate-400">Page ${page} of ${total}</span>
    <button onclick="doSearch(${page+1})" ${page>=total?'disabled':''} class="px-3 py-1.5 rounded-lg text-sm border border-[#2a3050] text-slate-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed">Next →</button>`;
  document.getElementById('pagination-top').innerHTML = html;
  document.getElementById('pagination-bottom').innerHTML = html;
}

function copyText(el, text) {
  navigator.clipboard.writeText(text).then(() => {
    const toast = document.getElementById('copy-toast');
    toast.style.display = 'block';
    setTimeout(() => { toast.style.display = 'none'; }, 1800);
  });
}

function esc(s) {
  if (!s) return '';
  return String(s).replace(/\\/g,'\\\\').replace(/'/g,"\\'").replace(/"/g,'&quot;');
}

function resetFilters() {
  document.getElementById('search-input').value = '';
  document.querySelector('input[name=source][value=all]').checked = true;
  document.getElementById('min-yield').value = '';
  document.getElementById('max-yield').value = '';
  document.getElementById('min-confidence').value = 0;
  document.getElementById('conf-val').textContent = '0.0';
  document.getElementById('solvent-input').value = '';
  document.getElementById('catalyst-input').value = '';
  document.getElementById('sort-select').value = 'confidence';
  doSearch(1);
}

// Init
loadStats();
doSearch(1);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    db = _find_db()
    if db:
        print(f"[OK] Database found: {db}")
    else:
        print("[WARNING] Database not found — set PATENT_DATA_DIR=D:\\SynAgent")
    print("[*] Starting Reaction Search at http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
