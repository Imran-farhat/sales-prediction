/* ================================================================
   app.js — Stock Prediction Dashboard
   All AJAX calls, chart rendering, UI interactions
   ================================================================ */

'use strict';

// ── Popular tickers for autocomplete ─────────────────────────────
const POPULAR_TICKERS = [
    { ticker: 'AAPL', name: 'Apple Inc.', type: 'stock' },
    { ticker: 'GOOGL', name: 'Alphabet Inc.', type: 'stock' },
    { ticker: 'MSFT', name: 'Microsoft Corp.', type: 'stock' },
    { ticker: 'AMZN', name: 'Amazon.com Inc.', type: 'stock' },
    { ticker: 'TSLA', name: 'Tesla Inc.', type: 'stock' },
    { ticker: 'NVDA', name: 'NVIDIA Corp.', type: 'stock' },
    { ticker: 'META', name: 'Meta Platforms Inc.', type: 'stock' },
    { ticker: 'NFLX', name: 'Netflix Inc.', type: 'stock' },
    { ticker: 'JPM', name: 'JPMorgan Chase', type: 'stock' },
    { ticker: 'BTC-USD', name: 'Bitcoin', type: 'crypto' },
    { ticker: 'ETH-USD', name: 'Ethereum', type: 'crypto' },
    { ticker: 'BNB-USD', name: 'Binance Coin', type: 'crypto' },
    { ticker: 'EURUSD=X', name: 'EUR/USD', type: 'forex' },
    { ticker: 'GBPUSD=X', name: 'GBP/USD', type: 'forex' },
    { ticker: 'SPY', name: 'S&P 500 ETF', type: 'etf' },
    { ticker: 'QQQ', name: 'Nasdaq-100 ETF', type: 'etf' },
];

const DAY_VALUES = [7, 30, 60, 90, 180, 365];

// ── App State ─────────────────────────────────────────────────────
const State = {
    currentTicker: null,
    currentDays: 30,
    currentModel: 'prophet',
    currentData: null,
    indicatorData: null,
    chartMode: 'line',   // line | candlestick | ohlc
    watchlist: [],
    watchlistAdded: false,
    mainChart: null,
    indicChart: null,
    compareChart: null,
};

// ── DOM refs ──────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// Helper to inject HTML and execute child <script> tags (required for Plotly to_html)
function setPanelHTML(id, html) {
    const el = document.getElementById(id);
    if (!el || !html) return;
    el.innerHTML = '';
    const range = document.createRange();
    const frag = range.createContextualFragment(html);
    el.appendChild(frag);
}

// ── Init ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initNavbar();
    initSidebar();
    initDaysSlider();
    initAutocomplete();
    initHistory();
    initWatchlist();
    initCSVUpload();
    initManualTable();
    loadWatchlistFromAPI();
    initSuggestionCards();
    checkModelAvailability();
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
    document.querySelector('.modal-backdrop')?.addEventListener('click', e => {
        if (e.target.classList.contains('modal-backdrop')) closeModal();
    });
});

// ── Sidebar Nav ───────────────────────────────────────────────────
function initSidebar() {
    document.querySelectorAll('.nav-item[data-section]').forEach(btn => {
        btn.addEventListener('click', () => {
            const sec = btn.dataset.section;
            switchSection(sec);
            document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
}

function switchSection(name) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    const el = $('section-' + name);
    if (el) el.classList.add('active');
}

// ── Navbar ────────────────────────────────────────────────────────
function initNavbar() {
    const predictBtn = $('btn-predict');
    if (predictBtn) {
        predictBtn.addEventListener('click', () => {
            const ticker = $('search-input')?.value?.trim().toUpperCase();
            if (!ticker) { showToast('Please enter a ticker symbol', 'warn'); return; }
            State.currentTicker = ticker;
            State.currentDays = State.currentDays;
            State.currentModel = $('model-select')?.value || 'prophet';
            runPredict();
        });
    }
    // Enter key on search
    $('search-input')?.addEventListener('keydown', e => {
        if (e.key === 'Enter') $('btn-predict')?.click();
    });
}

// ── Days Slider ───────────────────────────────────────────────────
function initDaysSlider() {
    const slider = $('days-slider');
    const display = $('days-value');
    if (!slider) return;
    slider.min = 0;
    slider.max = DAY_VALUES.length - 1;
    slider.value = 1; // default 30 days
    if (display) display.textContent = DAY_VALUES[1] + 'd';
    State.currentDays = DAY_VALUES[1];
    slider.addEventListener('input', () => {
        const idx = parseInt(slider.value);
        State.currentDays = DAY_VALUES[idx];
        if (display) display.textContent = DAY_VALUES[idx] + 'd';
    });
}

// ── Autocomplete ──────────────────────────────────────────────────
function initAutocomplete() {
    const input = $('search-input');
    const list = $('autocomplete-list');
    if (!input || !list) return;

    let debounce;
    input.addEventListener('input', () => {
        clearTimeout(debounce);
        debounce = setTimeout(() => renderAutocomplete(input.value.trim()), 200);
    });
    input.addEventListener('focus', () => { if (input.value.trim()) renderAutocomplete(input.value.trim()); });
    document.addEventListener('click', e => {
        if (!e.target.closest('.search-wrap')) { list.classList.remove('visible'); }
    });
}

function renderAutocomplete(query) {
    const list = $('autocomplete-list');
    if (!list) return;
    if (!query) { list.classList.remove('visible'); return; }
    const q = query.toUpperCase();
    const matches = POPULAR_TICKERS.filter(t =>
        t.ticker.includes(q) || t.name.toUpperCase().includes(q)
    ).slice(0, 8);
    if (!matches.length) { list.classList.remove('visible'); return; }
    list.innerHTML = matches.map(t => `
    <div class="autocomplete-item" data-ticker="${t.ticker}">
      <span style="font-family:var(--mono);font-weight:700;color:var(--accent)">${t.ticker}</span>
      <span style="color:var(--muted);flex:1">${t.name}</span>
      <span class="badge">${t.type}</span>
    </div>`).join('');
    list.classList.add('visible');
    list.querySelectorAll('.autocomplete-item').forEach(item => {
        item.addEventListener('click', () => {
            $('search-input').value = item.dataset.ticker;
            list.classList.remove('visible');
        });
    });
}

// ── Suggestion Cards ──────────────────────────────────────────────
function initSuggestionCards() {
    document.querySelectorAll('.suggestion-card[data-ticker]').forEach(card => {
        card.addEventListener('click', () => {
            const ticker = card.dataset.ticker;
            if ($('search-input')) $('search-input').value = ticker;
            State.currentTicker = ticker;
            State.currentModel = $('model-select')?.value || 'prophet';
            runPredict();
        });
    });
}

// ── Main Predict ──────────────────────────────────────────────────
async function runPredict() {
    const ticker = State.currentTicker;
    const days = State.currentDays;
    const model = State.currentModel;
    if (!ticker) return;

    // Switch to dashboard section
    switchSection('dashboard');
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    document.querySelector('.nav-item[data-section="dashboard"]')?.classList.add('active');

    // Hide welcome, show skeletons
    const welcome = $('welcome');
    if (welcome) welcome.style.display = 'none';
    showSkeletons(true);

    setLoading(true);
    State.watchlistAdded = State.watchlist.includes(ticker);
    updateWatchlistBtn(State.watchlistAdded);

    try {
        const res = await fetch(`/api/predict?ticker=${encodeURIComponent(ticker)}&days=${days}&model=${encodeURIComponent(model)}`);
        const data = await res.json();
        if (!res.ok || !data.success) {
            showToast(data.error || 'Prediction failed', 'error');
            showSkeletons(false);
            return;
        }
        State.currentData = data;
        if (data.demo) showDemoBanner(ticker);
        else hideDemoBanner();

        renderMainChart(data);
        renderStats(data);

        // Inject backend-generated charts (using createContextualFragment to execute <script> tags)
        setPanelHTML('technical-chart', data.technical_chart);
        setPanelHTML('rmse-chart', data.rmse_chart);

        refreshHistory();
        showSkeletons(false);
        showToast(`${ticker} — ${data.model} prediction ready`, 'success');
    } catch (err) {
        showToast('Network error: ' + err.message, 'error');
        showSkeletons(false);
    } finally {
        setLoading(false);
    }
}

// ── Chart Mode Toggle ─────────────────────────────────────────────
function setChartMode(mode) {
    State.chartMode = mode;
    document.querySelectorAll('.chart-toggle button').forEach(b => b.classList.remove('active'));
    const btn = document.querySelector(`.chart-toggle button[data-mode="${mode}"]`);
    if (btn) btn.classList.add('active');
    if (State.currentData) renderMainChart(State.currentData);
}
window.setChartMode = setChartMode;

// ── Main Chart ────────────────────────────────────────────────────
function renderMainChart(data) {
    const div = $('main-chart');
    if (!div) return;

    const layout = {
        paper_bgcolor: '#0d1117',
        plot_bgcolor: '#161b22',
        font: { color: '#c9d1d9', family: 'Inter, sans-serif', size: 12 },
        xaxis: { gridcolor: '#21262d', showgrid: true, zeroline: false },
        yaxis: { gridcolor: '#21262d', showgrid: true, zeroline: false, tickprefix: '$' },
        legend: { bgcolor: 'rgba(0,0,0,0)', bordercolor: '#30363d', borderwidth: 1 },
        hovermode: 'x unified',
        margin: { t: 20, r: 20, b: 40, l: 60 },
        showlegend: true,
    };

    let traces = [];

    if (State.chartMode === 'candlestick' && data.ohlcv?.dates?.length) {
        traces.push({
            type: 'candlestick',
            x: data.ohlcv.dates,
            open: data.ohlcv.open,
            high: data.ohlcv.high,
            low: data.ohlcv.low,
            close: data.ohlcv.close,
            name: 'OHLC',
            increasing: { line: { color: '#3fb950' } },
            decreasing: { line: { color: '#f85149' } },
        });
    } else if (State.chartMode === 'ohlc' && data.ohlcv?.dates?.length) {
        traces.push({
            type: 'ohlc',
            x: data.ohlcv.dates,
            open: data.ohlcv.open,
            high: data.ohlcv.high,
            low: data.ohlcv.low,
            close: data.ohlcv.close,
            name: 'OHLC',
        });
    } else {
        traces.push({
            type: 'scatter', mode: 'lines',
            x: data.historical_dates, y: data.historical_close,
            name: 'Historical',
            line: { color: '#58a6ff', width: 1.5 },
        });
    }

    // Predicted line
    traces.push({
        type: 'scatter', mode: 'lines',
        x: data.predicted_dates, y: data.predicted_values,
        name: `${data.model} Forecast`,
        line: { color: '#3fb950', width: 2, dash: 'dash' },
    });

    // Confidence band (filled area)
    traces.push({
        type: 'scatter',
        x: [...data.predicted_dates, ...[...data.predicted_dates].reverse()],
        y: [...data.upper, ...[...data.lower].reverse()],
        fill: 'toself',
        fillcolor: 'rgba(63,185,80,0.08)',
        line: { color: 'rgba(0,0,0,0)' },
        name: 'Confidence Band',
        showlegend: false,
        hoverinfo: 'skip',
    });

    Plotly.react(div, traces, layout, { responsive: true, displayModeBar: false });
}


// Redundant client-side charts removed in favor of backend HTML injection


// ── Stat Cards ────────────────────────────────────────────────────
function renderStats(data) {
    const s = data.stats || {};
    // Current price
    setStatCard('stat-current', formatPrice(s.last_close), '', data.ticker);
    // Predicted price
    const isUp = s.predicted_end >= s.last_close;
    setStatCard('stat-predicted', formatPrice(s.predicted_end), isUp ? 'up' : 'down', `in ${data.days}d`);
    // % change
    const pct = s.pct_change || 0;
    const pctEl = $('stat-pct');
    if (pctEl) {
        pctEl.querySelector('.stat-value').textContent = (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%';
        pctEl.querySelector('.stat-value').className = 'stat-value ' + (pct >= 0 ? 'up' : 'down');
        const badge = pctEl.querySelector('.stat-badge');
        if (badge) { badge.textContent = pct >= 0 ? '▲ Bullish' : '▼ Bearish'; badge.className = 'stat-badge ' + (pct >= 0 ? 'up' : 'down'); }
    }
    // Confidence
    const conf = s.confidence || 0;
    const confEl = $('stat-conf');
    if (confEl) {
        confEl.querySelector('.stat-value').textContent = conf.toFixed(1) + '%';
        const fill = confEl.querySelector('.confidence-fill');
        if (fill) fill.style.width = conf + '%';
    }
    // Model used
    const modelEl = $('stat-model');
    if (modelEl) {
        modelEl.querySelector('.stat-value').textContent = data.model;
        modelEl.querySelector('.stat-sub').textContent = `RMSE: ${data.rmse}  MAE: ${data.mae}`;
    }

    // Watchlist btn ticker label
    const wlBtn = $('btn-watchlist-main');
    if (wlBtn) wlBtn.dataset.ticker = data.ticker;
}

function setStatCard(id, value, direction, sub) {
    const el = $(id);
    if (!el) return;
    el.querySelector('.stat-value').textContent = value;
    el.querySelector('.stat-value').className = 'stat-value' + (direction ? ' ' + direction : '');
    if (el.querySelector('.stat-sub')) el.querySelector('.stat-sub').textContent = sub || '';
}

function formatPrice(v) {
    if (v == null || isNaN(v)) return 'N/A';
    return '$' + Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// ── Skeletons ─────────────────────────────────────────────────────
function showSkeletons(show) {
    document.querySelectorAll('.skeleton-wrap').forEach(el => {
        el.style.display = show ? 'block' : 'none';
    });
    document.querySelectorAll('.chart-wrap, .stats-wrap').forEach(el => {
        el.style.visibility = show ? 'hidden' : 'visible';
    });
}

// ── Loading Button ────────────────────────────────────────────────
function setLoading(on) {
    const btn = $('btn-predict');
    if (!btn) return;
    btn.disabled = on;
    btn.classList.toggle('loading', on);
}

// ── Demo Banner ───────────────────────────────────────────────────
function showDemoBanner(ticker) {
    const el = $('demo-banner');
    if (el) { el.innerHTML = `⚠️ Demo Mode: Real data unavailable for <strong>${ticker}</strong>. Showing sample data.`; el.classList.remove('hidden'); }
}
function hideDemoBanner() {
    const el = $('demo-banner');
    if (el) el.classList.add('hidden');
}

// ── Watchlist ─────────────────────────────────────────────────────
function initWatchlist() {
    const btn = $('btn-watchlist-main');
    if (btn) {
        btn.addEventListener('click', async () => {
            const ticker = btn.dataset.ticker || State.currentTicker;
            if (!ticker) return;
            if (State.watchlistAdded) {
                await removeFromWatchlist(ticker);
            } else {
                await addToWatchlist(ticker);
            }
        });
    }
}

async function loadWatchlistFromAPI() {
    try {
        const res = await fetch('/api/watchlist');
        const data = await res.json();
        if (data.success) {
            State.watchlist = data.watchlist;
            renderWatchlist();
        }
    } catch (e) { /* silent */ }
}

async function addToWatchlist(ticker) {
    try {
        const res = await fetch('/api/watchlist', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ticker }) });
        const data = await res.json();
        if (data.success) {
            State.watchlist = data.watchlist;
            State.watchlistAdded = true;
            updateWatchlistBtn(true);
            renderWatchlist();
            showToast(`${ticker} added to watchlist ⭐`, 'success');
        }
    } catch (e) { showToast('Failed to update watchlist', 'error'); }
}

async function removeFromWatchlist(ticker) {
    try {
        const res = await fetch(`/api/watchlist/${encodeURIComponent(ticker)}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            State.watchlist = data.watchlist;
            State.watchlistAdded = false;
            updateWatchlistBtn(false);
            renderWatchlist();
            showToast(`${ticker} removed from watchlist`, 'info');
        }
    } catch (e) { showToast('Failed to update watchlist', 'error'); }
}

function updateWatchlistBtn(added) {
    const btn = $('btn-watchlist-main');
    if (!btn) return;
    btn.textContent = added ? '⭐ In Watchlist' : '☆ Add to Watchlist';
    btn.classList.toggle('active', added);
}

function renderWatchlist() {
    const container = $('watchlist-container');
    if (!container) return;
    if (!State.watchlist.length) {
        container.innerHTML = '<div class="wl-empty">No tickers in watchlist.<br><small>Click "☆ Add to Watchlist" after a prediction.</small></div>';
        return;
    }
    container.innerHTML = State.watchlist.map(ticker => `
    <div class="watchlist-item" data-ticker="${ticker}">
      <span class="wl-ticker">${ticker}</span>
      <div class="wl-sparkline" id="spark-${ticker}"></div>
      <button class="wl-remove" title="Remove" onclick="event.stopPropagation(); removeFromWatchlist('${ticker}')">✕</button>
    </div>`).join('');
    container.querySelectorAll('.watchlist-item').forEach(item => {
        item.addEventListener('click', () => {
            if ($('search-input')) $('search-input').value = item.dataset.ticker;
            State.currentTicker = item.dataset.ticker;
            runPredict();
        });
    });
    // Render mini sparklines
    State.watchlist.forEach(ticker => renderSparkline(ticker));
}

async function renderSparkline(ticker) {
    const div = $(`spark-${ticker}`);
    if (!div) return;
    try {
        const res = await fetch(`/api/indicators/${encodeURIComponent(ticker)}`);
        const data = await res.json();
        if (!data.success) return;
        const last30 = data.close.slice(-30);
        const color = last30.at(-1) >= last30[0] ? '#3fb950' : '#f85149';
        Plotly.react(div, [{
            type: 'scatter', mode: 'lines',
            y: last30,
            line: { color, width: 1.5 },
            fill: 'tozeroy',
            fillcolor: color.replace(')', ',0.1)').replace('rgb', 'rgba'),
        }], {
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            margin: { t: 0, r: 0, b: 0, l: 0 },
            xaxis: { visible: false }, yaxis: { visible: false },
            showlegend: false, height: 32,
        }, { responsive: true, displayModeBar: false, staticPlot: true });
    } catch (e) { /* silent */ }
}

// ── History ───────────────────────────────────────────────────────
function initHistory() {
    const btn = $('history-btn');
    const dropdown = $('history-dropdown');
    if (!btn || !dropdown) return;
    btn.addEventListener('click', e => {
        e.stopPropagation();
        dropdown.classList.toggle('visible');
        if (dropdown.classList.contains('visible')) renderHistoryDropdown();
    });
    document.addEventListener('click', () => dropdown.classList.remove('visible'));
}

async function refreshHistory() {
    try {
        const res = await fetch('/api/history');
        const data = await res.json();
        if (data.success) renderHistoryList(data.history);
    } catch (e) { /* silent */ }
}

function renderHistoryDropdown() { refreshHistory(); }

function renderHistoryList(history) {
    const dropdown = $('history-dropdown');
    if (!dropdown) return;
    if (!history.length) {
        dropdown.innerHTML = '<div class="history-header">Recent Searches</div><div class="history-empty">No history yet</div>';
        return;
    }
    dropdown.innerHTML = '<div class="history-header">Recent Searches</div>' +
        history.map(h => `
      <div class="history-item" data-ticker="${h.ticker}" data-model="${h.model}" data-days="${h.days}">
        <span class="history-ticker">${h.ticker}</span>
        <span class="history-meta">${h.model} · ${h.days}d · ${h.timestamp}</span>
      </div>`).join('');
    dropdown.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => {
            State.currentTicker = item.dataset.ticker;
            State.currentModel = item.dataset.model;
            State.currentDays = parseInt(item.dataset.days);
            if ($('search-input')) $('search-input').value = State.currentTicker;
            if ($('model-select')) $('model-select').value = State.currentModel;
            dropdown.classList.remove('visible');
            runPredict();
        });
    });
}

// ── CSV Upload ────────────────────────────────────────────────────
function initCSVUpload() {
    const zone = $('csv-dropzone');
    const input = $('csv-file-input');
    if (!zone || !input) return;

    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        const file = e.dataTransfer?.files[0];
        if (file) processCSVFile(file);
    });
    input.addEventListener('change', () => { if (input.files[0]) processCSVFile(input.files[0]); });

    $('btn-csv-predict')?.addEventListener('click', () => {
        if (!window._csvData) { showToast('Please upload a CSV file first', 'warn'); return; }
        submitCSVPredict();
    });
}

async function processCSVFile(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showToast('Please upload a CSV file', 'error'); return;
    }
    const formData = new FormData();
    formData.append('file', file);
    showToast('Processing CSV...', 'info');
    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (!data.success) {
            showToast(data.error || 'Failed to parse CSV', 'error');
            showColumnMap(data.columns || []);
            return;
        }
        window._csvFile = file;
        window._csvData = data;
        showCSVPreview(data);
        showToast(`CSV loaded: ${data.total_rows} rows (${data.date_range.start} → ${data.date_range.end})`, 'success');
    } catch (e) { showToast('Upload error: ' + e.message, 'error'); }
}

function showCSVPreview(data) {
    const preview = $('csv-preview');
    if (!preview) return;
    preview.classList.remove('hidden');
    const wrap = $('csv-table-wrap');
    if (!wrap || !data.preview?.length) return;
    const cols = Object.keys(data.preview[0]);
    wrap.innerHTML = `<table class="preview-table">
    <thead><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead>
    <tbody>${data.preview.map(row => `<tr>${cols.map(c => `<td>${row[c] ?? ''}</td>`).join('')}</tr>`).join('')}</tbody>
  </table>`;
}

function showColumnMap(cols) {
    const mapEl = $('col-map');
    if (!mapEl || !cols.length) return;
    mapEl.classList.remove('hidden');
    mapEl.innerHTML = `<p style="color:var(--muted);font-size:13px;margin-bottom:8px">Could not auto-detect columns. Please map them:</p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      ${['Date', 'Close', 'Open', 'High', 'Low', 'Volume'].map(req =>
        `<div><label class="form-label">${req}</label>
         <select class="form-select" id="map-${req}">
           <option value="">-- Select --</option>
           ${cols.map(c => `<option value="${c}">${c}</option>`).join('')}
         </select></div>`
    ).join('')}
    </div>`;
}

async function submitCSVPredict() {
    if (!window._csvFile) return;
    const formData = new FormData();
    formData.append('file', window._csvFile);
    setLoading(true);
    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const html = await res.text();
        const win = window.open('', '_blank');
        win.document.write(html);
    } catch (e) { showToast('Prediction failed: ' + e.message, 'error'); }
    finally { setLoading(false); }
}

// ── Manual Data Table ─────────────────────────────────────────────
function initManualTable() {
    $('btn-add-row')?.addEventListener('click', addManualRow);
    $('btn-paste-clipboard')?.addEventListener('click', pasteFromClipboard);
    $('btn-manual-predict')?.addEventListener('click', submitManualPredict);

    // Real-time listener for any input in the manual table body
    const tbody = $('manual-table-body');
    if (tbody) {
        tbody.addEventListener('input', (e) => {
            if (e.target.classList.contains('manual-date') || e.target.classList.contains('manual-price')) {
                updateManualPreview();
            }
        });
    }
}

function addManualRow() {
    const tbody = $('manual-table-body');
    if (!tbody) return;
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><input type="date" class="manual-date"></td>
    <td><input type="number" step="0.01" class="manual-price" placeholder="0.00"></td>
    <td><button class="btn btn-sm btn-danger" onclick="this.closest('tr').remove(); updateManualPreview()">✕</button></td>`;
    tbody.appendChild(tr);
}

async function pasteFromClipboard() {
    try {
        const text = await navigator.clipboard.readText();
        const lines = text.trim().split('\n');
        const tbody = $('manual-table-body');
        if (!tbody) return;

        // Clear existing for a clean paste
        tbody.innerHTML = '';

        lines.forEach(line => {
            const parts = line.split(/[\t,;]/);
            if (parts.length < 2) return;

            const tr = document.createElement('tr');
            // Attempt to clean date/price
            const rawDate = parts[0].trim();
            const rawPrice = parseFloat(parts[1].trim().replace(/[$,]/g, ''));

            tr.innerHTML = `<td><input type="date" class="manual-date" value="${rawDate}"></td>
        <td><input type="number" step="0.01" class="manual-price" value="${isNaN(rawPrice) ? '' : rawPrice}"></td>
        <td><button class="btn btn-sm btn-danger" onclick="this.closest('tr').remove(); updateManualPreview()">✕</button></td>`;
            tbody.appendChild(tr);
        });
        updateManualPreview();
        showToast(`Pasted ${lines.length} rows. Updates showing in preview.`, 'success');
    } catch (e) {
        showToast('Clipboard access denied or invalid format', 'warn');
    }
}

/**
 * Real-time "Zero-API" Local Forecast
 * Calculates a simple linear trend internally to show updates instantly
 */
function updateManualPreview() {
    const dateInputs = [...document.querySelectorAll('.manual-date')];
    const priceInputs = [...document.querySelectorAll('.manual-price')];

    const data = [];
    for (let i = 0; i < dateInputs.length; i++) {
        const d = dateInputs[i].value;
        const p = parseFloat(priceInputs[i].value);
        if (d && !isNaN(p)) {
            data.push({ x: d, y: p, ts: new Date(d).getTime() });
        }
    }

    // Sort by date
    data.sort((a, b) => a.ts - b.ts);

    const div = $('manual-preview-chart');
    if (!div || data.length < 2) {
        if (div) Plotly.purge(div);
        return;
    }

    const predDays = parseInt($('manual-days')?.value || 30);

    // Simple Linear Regression for "Real-Time" feel without backend hit
    const x = data.map((_, i) => i);
    const y = data.map(d => d.y);
    const n = data.length;

    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((a, b, i) => a + (b * y[i]), 0);
    const sumXX = x.reduce((a, b) => a + (b * b), 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    // Generate future points
    const lastDate = new Date(data[data.length - 1].ts);
    const futureX = [];
    const futureY = [];

    for (let i = 1; i <= predDays; i++) {
        const nextDate = new Date(lastDate);
        nextDate.setDate(lastDate.getDate() + i);
        futureX.push(nextDate.toISOString().split('T')[0]);
        futureY.push(slope * (n - 1 + i) + intercept);
    }

    const traces = [
        {
            type: 'scatter', mode: 'lines+markers',
            x: data.map(d => d.x), y: data.map(d => d.y),
            name: 'Actual Data', line: { color: '#58a6ff', width: 2 },
            marker: { size: 6 }
        },
        {
            type: 'scatter', mode: 'lines',
            x: futureX, y: futureY,
            name: 'Local Trend', line: { color: '#3fb950', dash: 'dash', width: 2 }
        }
    ];

    const layout = {
        paper_bgcolor: '#0d1117', plot_bgcolor: '#161b22',
        font: { color: '#c9d1d9', size: 10 },
        margin: { t: 10, r: 10, b: 30, l: 40 },
        xaxis: { gridcolor: '#21262d', zeroline: false },
        yaxis: { gridcolor: '#21262d', zeroline: false },
        showlegend: true,
        legend: { x: 0, y: 1.1, orientation: 'h', font: { size: 9 } },
        height: 250,
    };

    Plotly.react(div, traces, layout, { responsive: true, displayModeBar: false });
}

async function submitManualPredict() {
    const dates = [...document.querySelectorAll('.manual-date')].map(i => i.value).filter(Boolean);
    const prices = [...document.querySelectorAll('.manual-price')].map(i => i.value).filter(Boolean);
    const predDays = parseInt($('manual-days')?.value || 30);
    const model = $('manual-model')?.value || 'prophet';
    if (dates.length < 2 || prices.length < 2) {
        showToast('Please enter at least 2 data rows', 'warn'); return;
    }
    const form = document.createElement('form');
    form.method = 'POST'; form.action = '/manual_predict'; form.style.display = 'none';
    dates.forEach(d => { const i = document.createElement('input'); i.name = 'dates[]'; i.value = d; form.appendChild(i); });
    prices.forEach(p => { const i = document.createElement('input'); i.name = 'sales[]'; i.value = p; form.appendChild(i); });
    const pi = document.createElement('input'); pi.name = 'prediction_days'; pi.value = predDays; form.appendChild(pi);
    const mi = document.createElement('input'); mi.name = 'model'; mi.value = model; form.appendChild(mi);
    document.body.appendChild(form);
    form.submit();
}

// ── Modal ─────────────────────────────────────────────────────────
function openModal() {
    const backdrop = document.querySelector('.modal-backdrop');
    if (!backdrop) return;
    if (State.currentData) {
        $('modal-title').textContent = `${State.currentTicker} — ${State.currentData.model} Prediction`;
        renderModalChart(State.currentData);
    }
    backdrop.classList.add('visible');
}
function closeModal() {
    document.querySelector('.modal-backdrop')?.classList.remove('visible');
}
window.openModal = openModal;
window.closeModal = closeModal;

function renderModalChart(data) {
    const div = $('modal-chart');
    if (!div || !data) return;
    Plotly.react(div, [
        {
            type: 'scatter', mode: 'lines', x: data.historical_dates, y: data.historical_close,
            name: 'Historical', line: { color: '#58a6ff' }
        },
        {
            type: 'scatter', mode: 'lines', x: data.predicted_dates, y: data.predicted_values,
            name: 'Predicted', line: { color: '#3fb950', dash: 'dash' }
        },
    ], {
        paper_bgcolor: '#0d1117', plot_bgcolor: '#161b22',
        font: { color: '#c9d1d9' }, margin: { t: 20, r: 20, b: 40, l: 60 },
        hovermode: 'x unified',
    }, { responsive: true });
}

// ── Download helpers ──────────────────────────────────────────────
function downloadPNG() {
    const div = $('main-chart');
    if (!div) return;
    Plotly.toImage(div, { format: 'png', width: 1400, height: 700 }).then(url => {
        const a = document.createElement('a'); a.href = url;
        a.download = `${State.currentTicker || 'chart'}_prediction.png`; a.click();
    });
}
function downloadCSV() {
    const data = State.currentData;
    if (!data) { showToast('No prediction data to download', 'warn'); return; }
    const rows = [['Date', 'Type', 'Value', 'Lower', 'Upper']];
    data.historical_dates.forEach((d, i) => rows.push([d, 'Historical', data.historical_close[i], '', '']));
    data.predicted_dates.forEach((d, i) => rows.push([d, 'Predicted', data.predicted_values[i], data.lower[i], data.upper[i]]));
    const csv = rows.map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${data.ticker}_${data.model}_prediction.csv`; a.click();
    URL.revokeObjectURL(a.href);
}
function shareURL() {
    const url = new URL(window.location.href);
    if (State.currentTicker) url.searchParams.set('ticker', State.currentTicker);
    url.searchParams.set('days', State.currentDays);
    url.searchParams.set('model', State.currentModel);
    navigator.clipboard.writeText(url.toString())
        .then(() => showToast('Link copied to clipboard!', 'success'))
        .catch(() => showToast('Could not copy link', 'warn'));
}
window.downloadPNG = downloadPNG;
window.downloadCSV = downloadCSV;
window.shareURL = shareURL;

// ── Parse URL params on load ──────────────────────────────────────
(function parseURLParams() {
    const p = new URLSearchParams(window.location.search);
    if (p.has('ticker')) {
        State.currentTicker = p.get('ticker').toUpperCase();
        if ($('search-input')) $('search-input').value = State.currentTicker;
    }
    if (p.has('model') && $('model-select')) {
        State.currentModel = p.get('model');
        $('model-select').value = State.currentModel;
    }
    if (p.has('days')) {
        const d = parseInt(p.get('days'));
        const idx = DAY_VALUES.indexOf(d);
        State.currentDays = d;
        if (idx >= 0 && $('days-slider')) $('days-slider').value = idx;
        if ($('days-value')) $('days-value').textContent = d + 'd';
    }
    if (State.currentTicker) {
        setTimeout(() => runPredict(), 400);
    }
})();

// ── Toast Notifications ───────────────────────────────────────────
function showToast(message, type = 'info', title = null) {
    let container = $('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        document.body.appendChild(container);
    }
    const icons = { success: '✅', error: '❌', warn: '⚠️', info: 'ℹ️' };
    const titles = { success: 'Success', error: 'Error', warn: 'Warning', info: 'Info' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ️'}</span>
    <div class="toast-body">
      <div class="toast-title">${title || titles[type] || 'Notice'}</div>
      <div class="toast-msg">${message}</div>
    </div>
    <button class="close-btn" onclick="dismissToast(this.parentElement)">✕</button>`;
    container.appendChild(toast);
    toast.addEventListener('click', () => dismissToast(toast));
    setTimeout(() => dismissToast(toast), 5000);
}
function dismissToast(toast) {
    if (!toast || toast._removing) return;
    toast._removing = true;
    toast.classList.add('removing');
    setTimeout(() => toast.remove(), 300);
}
window.showToast = showToast;
window.removeFromWatchlist = removeFromWatchlist;
window.updateManualPreview = updateManualPreview;

async function checkModelAvailability() {
    try {
        const res = await fetch('/status');
        const data = await res.json();
        const models = data.models || {};

        // Update main dropdown
        const select = $('model-select');
        if (select) {
            [...select.options].forEach(opt => {
                const val = opt.value.toLowerCase();
                if (val !== 'ensemble' && models[val] === false) {
                    opt.disabled = true;
                    opt.textContent += ' (Not installed)';
                }
            });
            // If default prophet is missing, switch to linear
            if (models.prophet === false && select.value === 'prophet') {
                select.value = 'linear';
                State.currentModel = 'linear';
            }
        }

        // Show global alert if key libraries are missing
        if (models.prophet === false || models.tensorflow === false) {
            const banner = document.createElement('div');
            banner.className = 'status-alert-top';
            banner.innerHTML = `
                <div class="alert-content">
                    <strong>⚠️ Limited Environment:</strong> Some ML models (Prophet/LSTM) are missing in Python 3.14. 
                    Please run <code>start.bat</code> to use Python 3.13.
                </div>
            `;
            document.body.prepend(banner);
        }
    } catch (e) { console.error('Availability check failed:', e); }
}

window.checkModelAvailability = checkModelAvailability;
