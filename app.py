# app.py — Stock Prediction Dashboard (Revamped)
from flask import Flask, render_template, request, session, jsonify
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, time, random, os, pickle, hashlib, json
from datetime import datetime, timedelta

# ── Optional ML imports ─────────────────────────────────────────────────────
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    Prophet = None
    PROPHET_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# ── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'stock-dashboard-secret-2024')

# ── Rate Limiting ────────────────────────────────────────────────────────────
REQUEST_TIMESTAMPS = {}
MIN_REQUEST_INTERVAL = 5  # seconds

def is_rate_limited(key):
    now = time.time()
    if key in REQUEST_TIMESTAMPS:
        if now - REQUEST_TIMESTAMPS[key] < MIN_REQUEST_INTERVAL:
            return True
    REQUEST_TIMESTAMPS[key] = now
    return False

def get_session_key():
    if 'session_id' not in session:
        session['session_id'] = hashlib.md5(str(random.random()).encode()).hexdigest()
    return session['session_id']

def cleanup_old_timestamps():
    now = time.time()
    cutoff = now - (MIN_REQUEST_INTERVAL * 10)
    for key in [k for k, t in REQUEST_TIMESTAMPS.items() if t < cutoff]:
        del REQUEST_TIMESTAMPS[key]

# ── Cache ────────────────────────────────────────────────────────────────────
CACHE_DIR = 'cache'
CACHE_EXPIRY_HOURS = 6
DEMO_MODE_ENABLED = True
ENABLE_PRELOADING = False

os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_filename(ticker, period):
    return os.path.join(CACHE_DIR, f"{ticker}_{period}.pkl")

def is_cache_valid(cache_file):
    if not os.path.exists(cache_file):
        return False
    file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
    return datetime.now() - file_time < timedelta(hours=CACHE_EXPIRY_HOURS)

def save_to_cache(data, cache_file):
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Cache save error: {e}")

def load_from_cache(cache_file):
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Cache load error: {e}")
        return None

def cleanup_cache():
    try:
        if os.path.exists(CACHE_DIR):
            for fn in os.listdir(CACHE_DIR):
                fp = os.path.join(CACHE_DIR, fn)
                if fn.endswith('.pkl'):
                    ft = datetime.fromtimestamp(os.path.getmtime(fp))
                    if datetime.now() - ft > timedelta(hours=CACHE_EXPIRY_HOURS * 2):
                        os.remove(fp)
    except Exception as e:
        print(f"Cache cleanup error: {e}")

# ── Data Fetching ────────────────────────────────────────────────────────────
def fetch_stock_data_with_retry(ticker, period="5y", max_retries=2):
    cache_file = get_cache_filename(ticker, period)
    
    # 1. Attempt to use valid cache
    if is_cache_valid(cache_file):
        cached = load_from_cache(cache_file)
        if cached is not None and not cached.empty:
            return cached, False

    # Store any existing cache (even if expired) as a final fallback
    fallback_cache = load_from_cache(cache_file) if os.path.exists(cache_file) else None

    # 2. Try fetching live data
    periods_to_try = [period, "1y"] if period != "1y" else ["1y"]
    
    for p in periods_to_try:
        current_cache_file = get_cache_filename(ticker, p)
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(2)
                stock = yf.Ticker(ticker)
                hist = stock.history(period=p)
                if not hist.empty:
                    save_to_cache(hist, current_cache_file)
                    return hist, False
                else:
                    print(f"DEBUG: yfinance returned EMPTY history for {ticker} (period={p}, attempt={attempt+1})")
            except Exception as e:
                print(f"DEBUG: Fetch error for {ticker} ({p}) attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

    # 3. Use expired cache if live fetch failed
    if fallback_cache is not None and not fallback_cache.empty:
        print(f"DEBUG: Using EXPIRED cache fallback for {ticker}")
        return fallback_cache, False

    # 4. Final fallback to Demo Mode if enabled
    if DEMO_MODE_ENABLED:
        print(f"DEBUG: Triggering Demo Mode for {ticker} (No live data or cache)")
        return create_sample_data(ticker), True
        
    return pd.DataFrame(), True

def create_sample_data(ticker, period="5y"):
    print(f"Creating sample data for {ticker}")
    days = 365 * 5
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    base_price, prices = 100.0, []
    current = base_price
    for _ in dates:
        change = random.gauss(0.0005, 0.02)
        current = max(5, current * (1 + change))
        prices.append(current)
    df = pd.DataFrame({
        'Open':   [p * random.uniform(0.98, 1.01) for p in prices],
        'High':   [p * random.uniform(1.00, 1.04) for p in prices],
        'Low':    [p * random.uniform(0.96, 1.00) for p in prices],
        'Close':  prices,
        'Volume': [random.randint(5_000_000, 50_000_000) for _ in prices],
    }, index=dates)
    df.index.name = 'Date'
    return df

def detect_asset_type(ticker):
    t = ticker.upper()
    if t.endswith('-USD') or t.endswith('USDT') or t in ('BTC', 'ETH', 'BNB', 'SOL', 'XRP'):
        return 'crypto'
    if t.endswith('=X'):
        return 'forex'
    return 'stock'

# ── Feature Engineering ──────────────────────────────────────────────────────
def compute_indicators(df):
    """Add MA, RSI, MACD, Bollinger Bands, Volume trend columns."""
    close = df['Close'].copy()
    vol   = df['Volume'].copy() if 'Volume' in df.columns else None

    # Moving Averages
    df['MA7']  = close.rolling(7).mean()
    df['MA30'] = close.rolling(30).mean()
    df['MA90'] = close.rolling(90).mean()

    # RSI (User requested: 14-period using pandas rolling mean)
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (User requested: 12/26 EMA, signal = 9 EMA)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']

    # Bollinger Bands (20-day)
    df['BB_mid']   = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * bb_std
    df['BB_lower'] = df['BB_mid'] - 2 * bb_std

    # Volume trend
    if vol is not None:
        df['Vol_MA20'] = vol.rolling(20).mean()

    return df

def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float('nan')
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))

def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

# ── Model Runners ────────────────────────────────────────────────────────────
def split_series(series, test_frac=0.2):
    """Return (train, test) DataFrames given a Close price series."""
    n = len(series)
    split = int(n * (1 - test_frac))
    return series.iloc[:split], series.iloc[split:]

def run_prophet(df, days):
    if not PROPHET_AVAILABLE:
        return None, "Prophet not installed"
    try:
        close = df['Close'].dropna()
        train, test = split_series(close)

        def fit_predict(series, future_days):
            pdf = series.reset_index()
            pdf.columns = ['ds', 'y']
            pdf['ds'] = pd.to_datetime(pdf['ds']).dt.tz_localize(None)
            m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
            m.fit(pdf)
            future = m.make_future_dataframe(periods=future_days)
            return m.predict(future)

        # Backtest on test portion
        fc_full = fit_predict(train, len(test) + days)
        test_preds = fc_full['yhat'].values[len(train):len(train) + len(test)]
        rmse_val = rmse(test.values, test_preds)
        mae_val  = mae(test.values, test_preds)

        # Final forecast on all data
        fc = fit_predict(close, days)
        historical_fc = fc['yhat'].iloc[:len(close)]
        future_fc     = fc.iloc[len(close):]

        return {
            'model': 'Prophet',
            'historical_dates': close.index.strftime('%Y-%m-%d').tolist(),
            'historical_close': close.round(4).tolist(),
            'predicted_dates':  future_fc['ds'].dt.strftime('%Y-%m-%d').tolist(),
            'predicted_values': future_fc['yhat'].round(4).tolist(),
            'lower':            future_fc['yhat_lower'].round(4).tolist(),
            'upper':            future_fc['yhat_upper'].round(4).tolist(),
            'rmse': round(rmse_val, 4),
            'mae':  round(mae_val, 4),
        }, None
    except Exception as e:
        return None, str(e)

def run_linear(df, days):
    if not SKLEARN_AVAILABLE:
        return None, "scikit-learn not installed"
    try:
        close = df['Close'].dropna()
        train, test = split_series(close)

        def fit_predict(series, future_days):
            X = np.arange(len(series)).reshape(-1, 1)
            poly = PolynomialFeatures(degree=2)
            Xp = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(Xp, series.values)
            X_all = np.arange(len(series) + future_days).reshape(-1, 1)
            Xp_all = poly.transform(X_all)
            preds = model.predict(Xp_all)
            residuals = series.values - model.predict(Xp)
            std = residuals.std()
            return preds, std

        train_preds, _ = fit_predict(train, len(test))
        test_preds = train_preds[len(train):len(train) + len(test)]
        rmse_val = rmse(test.values, test_preds)
        mae_val  = mae(test.values, test_preds)

        all_preds, std = fit_predict(close, days)
        future_preds = all_preds[len(close):]
        future_dates = pd.date_range(close.index[-1] + timedelta(days=1), periods=days, freq='B')

        return {
            'model': 'Linear Regression',
            'historical_dates': close.index.strftime('%Y-%m-%d').tolist(),
            'historical_close': close.round(4).tolist(),
            'predicted_dates':  future_dates.strftime('%Y-%m-%d').tolist(),
            'predicted_values': future_preds.round(4).tolist(),
            'lower':            (future_preds - 1.96 * std).round(4).tolist(),
            'upper':            (future_preds + 1.96 * std).round(4).tolist(),
            'rmse': round(rmse_val, 4),
            'mae':  round(mae_val, 4),
        }, None
    except Exception as e:
        return None, str(e)

def run_arima(df, days):
    if not STATSMODELS_AVAILABLE:
        return None, "statsmodels not installed"
    try:
        close = df['Close'].dropna()
        # Use last 2 years for speed
        close = close.iloc[-min(len(close), 504):]
        train, test = split_series(close)

        def fit_arima(series):
            try:
                m = ARIMA(series, order=(5, 1, 0))
                return m.fit()
            except Exception:
                m = ARIMA(series, order=(1, 1, 0))
                return m.fit()

        fit_tr = fit_arima(train)
        test_preds = fit_tr.forecast(steps=len(test))
        rmse_val = rmse(test.values, test_preds)
        mae_val  = mae(test.values, test_preds)

        fit_all = fit_arima(close)
        fc = fit_all.get_forecast(steps=days)
        fc_mean = fc.predicted_mean
        fc_ci   = fc.conf_int(alpha=0.05)
        future_dates = pd.date_range(close.index[-1] + timedelta(days=1), periods=days, freq='B')

        return {
            'model': 'ARIMA',
            'historical_dates': close.index.strftime('%Y-%m-%d').tolist(),
            'historical_close': close.round(4).tolist(),
            'predicted_dates':  future_dates.strftime('%Y-%m-%d').tolist(),
            'predicted_values': fc_mean.values.round(4).tolist(),
            'lower':            fc_ci.iloc[:, 0].values.round(4).tolist(),
            'upper':            fc_ci.iloc[:, 1].values.round(4).tolist(),
            'rmse': round(rmse_val, 4),
            'mae':  round(mae_val, 4),
        }, None
    except Exception as e:
        return None, str(e)

def run_lstm(df, days):
    if not TENSORFLOW_AVAILABLE:
        return None, "TensorFlow not installed"
    try:
        close = df['Close'].dropna().values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close)

        SEQ_LEN = 60
        X, y = [], []
        for i in range(SEQ_LEN, len(scaled)):
            X.append(scaled[i - SEQ_LEN:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        X_train = X_train.reshape(-1, SEQ_LEN, 1)
        X_test  = X_test.reshape(-1, SEQ_LEN, 1)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1),
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)

        test_preds_scaled = model.predict(X_test, verbose=0).flatten()
        test_preds = scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        rmse_val = rmse(y_test_actual, test_preds)
        mae_val  = mae(y_test_actual, test_preds)

        # Autoregressive future forecast
        last_seq = scaled[-SEQ_LEN:].flatten()
        future_preds_scaled = []
        for _ in range(days):
            seq = last_seq[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
            pred = model.predict(seq, verbose=0)[0, 0]
            future_preds_scaled.append(pred)
            last_seq = np.append(last_seq, pred)
        future_preds = scaler.inverse_transform(
            np.array(future_preds_scaled).reshape(-1, 1)).flatten()

        close_index = df['Close'].dropna().index
        future_dates = pd.date_range(close_index[-1] + timedelta(days=1), periods=days, freq='B')
        std = float(np.std(close_index[-90:].to_frame().apply(lambda x: x))) if hasattr(close_index, 'to_frame') else float(abs(float(future_preds.std())))
        ci_width = 1.96 * max(std, abs(float(np.mean(future_preds)) * 0.05))

        return {
            'model': 'LSTM',
            'historical_dates': close_index.strftime('%Y-%m-%d').tolist(),
            'historical_close': df['Close'].dropna().round(4).tolist(),
            'predicted_dates':  future_dates.strftime('%Y-%m-%d').tolist(),
            'predicted_values': future_preds.round(4).tolist(),
            'lower':            (future_preds - ci_width).round(4).tolist(),
            'upper':            (future_preds + ci_width).round(4).tolist(),
            'rmse': round(rmse_val, 4),
            'mae':  round(mae_val, 4),
        }, None
    except Exception as e:
        return None, str(e)

def run_ensemble(df, days, results):
    """Average predictions from all successful model results."""
    try:
        valid = [r for r in results if r is not None]
        if not valid:
            return None, "No models succeeded for ensemble"

        n = min(days, min(len(v['predicted_values']) for v in valid))
        avg_preds = np.mean([v['predicted_values'][:n] for v in valid], axis=0)
        avg_lower = np.mean([v['lower'][:n] for v in valid], axis=0)
        avg_upper = np.mean([v['upper'][:n] for v in valid], axis=0)
        avg_rmse  = float(np.mean([v['rmse'] for v in valid if not np.isnan(v['rmse'])]))
        avg_mae   = float(np.mean([v['mae']  for v in valid if not np.isnan(v['mae'])]))

        ref = valid[0]
        return {
            'model': 'Ensemble',
            'historical_dates': ref['historical_dates'],
            'historical_close': ref['historical_close'],
            'predicted_dates':  ref['predicted_dates'][:n],
            'predicted_values': avg_preds.round(4).tolist(),
            'lower':            avg_lower.round(4).tolist(),
            'upper':            avg_upper.round(4).tolist(),
            'rmse': round(avg_rmse, 4),
            'mae':  round(avg_mae, 4),
        }, None
    except Exception as e:
        return None, str(e)

def run_model(name, df, days, all_results=None):
    runners = {
        'prophet': run_prophet,
        'linear':  run_linear,
        'arima':   run_arima,
        'lstm':    run_lstm,
    }
    if name.lower() == 'ensemble':
        return run_ensemble(df, days, all_results or [])
    fn = runners.get(name.lower())
    if fn is None:
        return None, f"Unknown model: {name}"
    return fn(df, days)

# ── Helper: Stock Info ───────────────────────────────────────────────────────
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'ticker':    ticker,
            'name':      info.get('longName', ticker),
            'sector':    info.get('sector', 'N/A'),
            'currency':  info.get('currency', 'USD'),
            'current_price': info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio':  info.get('trailingPE', 0),
            'asset_type': detect_asset_type(ticker),
        }
    except Exception:
        return {'ticker': ticker, 'name': ticker, 'current_price': 0, 'asset_type': detect_asset_type(ticker)}

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

# ── API: Stock Info ──────────────────────────────────────────────────────────
@app.route('/api/stock/<ticker>')
def api_stock(ticker):
    ticker = ticker.upper().strip()
    info = get_stock_info(ticker)
    return jsonify({'success': True, 'data': info})

# ── API: Technical Indicators ────────────────────────────────────────────────
@app.route('/api/indicators/<ticker>')
def api_indicators(ticker):
    ticker = ticker.upper().strip()
    hist, is_demo = fetch_stock_data_with_retry(ticker, period="1y")
    if hist.empty:
        return jsonify({'success': False, 'error': 'No data'}), 404

    df = compute_indicators(hist.copy())
    df = df.dropna(subset=['MA7'])
    dates = df.index.strftime('%Y-%m-%d').tolist()

    result = {
        'success': True,
        'demo': is_demo,
        'ticker': ticker,
        'dates':       dates,
        'close':       df['Close'].round(4).tolist(),
        'ma7':         df['MA7'].round(4).tolist(),
        'ma30':        df['MA30'].round(4).tolist(),
        'ma90':        df['MA90'].round(4).tolist(),
        'rsi':         df['RSI'].round(4).tolist(),
        'macd':        df['MACD'].round(4).tolist(),
        'macd_signal': df['MACD_signal'].round(4).tolist(),
        'macd_hist':   df['MACD_hist'].round(4).tolist(),
        'bb_upper':    df['BB_upper'].round(4).tolist(),
        'bb_mid':      df['BB_mid'].round(4).tolist(),
        'bb_lower':    df['BB_lower'].round(4).tolist(),
    }
    return jsonify(result)

# ── API: Single Model Predict ─────────────────────────────────────────────────
@app.route('/api/predict')
def api_predict():
    ticker = request.args.get('ticker', '').upper().strip()
    days   = int(request.args.get('days', 30))
    model  = request.args.get('model', 'prophet').lower()

    if not ticker:
        return jsonify({'success': False, 'error': 'Ticker is required'}), 400
    if days < 1 or days > 365:
        return jsonify({'success': False, 'error': 'Days must be 1–365'}), 400

    # Rate limit
    client_ip = request.remote_addr
    session_key = get_session_key()
    if is_rate_limited(f"{client_ip}_{session_key}"):
        return jsonify({'success': False, 'error': 'Rate limited. Please wait a few seconds.'}), 429
    cleanup_old_timestamps()

    hist, is_demo = fetch_stock_data_with_retry(ticker, period="5y")
    if hist.empty:
        return jsonify({'success': False, 'error': f'Could not fetch data for {ticker}'}), 404

    df = compute_indicators(hist.copy())

    # For ensemble: run all individual models first
    if model == 'ensemble':
        individual_results = []
        for m in ['prophet', 'linear', 'arima']:
            res, _ = run_model(m, df, days)
            individual_results.append(res)
        result, err = run_ensemble(df, days, individual_results)
    else:
        result, err = run_model(model, df, days)

    if result is None:
        return jsonify({'success': False, 'error': err or 'Model failed'}), 500

    # Current + predicted stats
    last_close = float(hist['Close'].iloc[-1])
    pred_last  = result['predicted_values'][-1] if result['predicted_values'] else last_close
    pct_change = ((pred_last - last_close) / last_close * 100) if last_close != 0 else 0

    # Indicators for chart
    ind_data = {}
    for col in ['dates', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_mid', 'bb_lower', 'ma7', 'ma30']:
        pass  # included separately via /api/indicators

    # Store in session history
    if 'history' not in session:
        session['history'] = []
    entry = {
        'ticker': ticker, 'days': days, 'model': model,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'rmse': result['rmse'], 'mae': result['mae'],
        'last_close': round(last_close, 4), 'predicted_end': round(pred_last, 4),
    }
    history = session.get('history', [])
    history.insert(0, entry)
    session['history'] = history[:10]
    session.modified = True

    # ── Finalize Response with Charts ─────────────────────────────────────────
    rmse_val = float(result['rmse']) if not np.isnan(result['rmse']) else 0
    mae_val  = float(result['mae']) if not np.isnan(result['mae']) else 0

    # 1. Technical Indicators Chart (Last 100 days for clarity)
    df_plot = df.tail(100)
    fig_ind = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.4, 0.6])
    
    # RSI Row 1
    fig_ind.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name='RSI', 
                                line=dict(color='#d2a8ff', width=2)), row=1, col=1)
    fig_ind.add_hline(y=70, line_dash="dot", line_color="#f85149", row=1, col=1)
    fig_ind.add_hline(y=30, line_dash="dot", line_color="#3fb950", row=1, col=1)
    fig_ind.update_yaxes(title_text="RSI", range=[0, 100], row=1, col=1)

    # MACD Row 2
    fig_ind.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD'], name='MACD', 
                                line=dict(color='#58a6ff', width=1.5)), row=2, col=1)
    fig_ind.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD_signal'], name='Signal', 
                                line=dict(color='#f85149', dash='dot', width=1)), row=2, col=1)
    
    colors = ['#3fb950' if v >= 0 else '#f85149' for v in df_plot['MACD_hist']]
    fig_ind.add_trace(go.Bar(x=df_plot.index, y=df_plot['MACD_hist'], name='Hist', 
                             marker_color=colors, opacity=0.7), row=2, col=1)
    fig_ind.update_yaxes(title_text="MACD", row=2, col=1)

    fig_ind.update_layout(template='plotly_dark', paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                          height=350, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    tech_chart_html = fig_ind.to_html(full_html=False, include_plotlyjs=False)

    # 2. Model Accuracy RMSE chart
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(
        x=["RMSE", "MAE"], 
        y=[rmse_val, mae_val],
        text=[f"{rmse_val:.2f}", f"{mae_val:.2f}"],
        textposition='auto',
        marker_color=['#3fb950', '#f85149']
    ))
    fig_metrics.update_layout(template='plotly_dark', paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                              height=250, margin=dict(l=20, r=20, t=30, b=20),
                              title={'text': "Model Accuracy", 'y':0.95, 'x':0.5, 'xanchor': 'center'})
    rmse_chart_html = fig_metrics.to_html(full_html=False, include_plotlyjs=False)

    return jsonify({
        'success': True,
        'demo':    is_demo,
        'ticker':  ticker,
        'model':   result['model'],
        'days':    days,
        'asset_type': detect_asset_type(ticker),
        'historical_dates':  result['historical_dates'][-252:],
        'historical_close':  result['historical_close'][-252:],
        'predicted_dates':   result['predicted_dates'],
        'predicted_values':  result['predicted_values'],
        'lower':             result['lower'],
        'upper':             result['upper'],
        'rmse':              result['rmse'],
        'mae':               result['mae'],
        'technical_chart':   tech_chart_html,
        'rmse_chart':        rmse_chart_html,
        'stats': {
            'last_close':  round(last_close, 4),
            'predicted_end': round(pred_last, 4),
            'pct_change':  round(pct_change, 2),
            'confidence':  max(0, round(100 - min(result['rmse'] / max(last_close, 1) * 100, 100), 1)),
        },
        'ohlcv': {
            'dates':  hist.index.strftime('%Y-%m-%d').tolist()[-252:],
            'open':   hist['Open'].round(4).tolist()[-252:],
            'high':   hist['High'].round(4).tolist()[-252:],
            'low':    hist['Low'].round(4).tolist()[-252:],
            'close':  hist['Close'].round(4).tolist()[-252:],
            'volume': hist['Volume'].tolist()[-252:] if 'Volume' in hist.columns else [],
        },
    })

# ── API: Compare All Models ──────────────────────────────────────────────────
@app.route('/api/compare')
def api_compare():
    ticker = request.args.get('ticker', '').upper().strip()
    days   = int(request.args.get('days', 30))

    if not ticker:
        return jsonify({'success': False, 'error': 'Ticker required'}), 400

    client_ip = request.remote_addr
    session_key = get_session_key()
    if is_rate_limited(f"{client_ip}_{session_key}_cmp"):
        return jsonify({'success': False, 'error': 'Rate limited'}), 429

    hist, is_demo = fetch_stock_data_with_retry(ticker, period="5y")
    if hist.empty:
        return jsonify({'success': False, 'error': f'No data for {ticker}'}), 404

    df = compute_indicators(hist.copy())
    model_names = ['prophet', 'linear', 'arima']
    if TENSORFLOW_AVAILABLE:
        model_names.append('lstm')

    individual_results = []
    comparison = {}
    for m in model_names:
        res, err = run_model(m, df, days)
        if res:
            individual_results.append(res)
            comparison[res['model']] = {
                'rmse': res['rmse'], 'mae': res['mae'],
                'predicted_end': res['predicted_values'][-1] if res['predicted_values'] else None,
            }

    ens_res, ens_err = run_ensemble(df, days, individual_results)
    if ens_res:
        comparison['Ensemble'] = {
            'rmse': ens_res['rmse'], 'mae': ens_res['mae'],
            'predicted_end': ens_res['predicted_values'][-1] if ens_res['predicted_values'] else None,
        }

    return jsonify({'success': True, 'demo': is_demo, 'ticker': ticker, 'days': days, 'comparison': comparison})

# ── API: CSV Upload ──────────────────────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'success': False, 'error': f'CSV parse error: {e}'}), 400

    # Auto column detection
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ('date', 'datetime', 'time', 'timestamp', 'ds'):
            col_map['date'] = col
        elif cl in ('close', 'price', 'adj close', 'adjusted close', 'y'):
            col_map['close'] = col
        elif cl in ('open',):
            col_map['open'] = col
        elif cl in ('high',):
            col_map['high'] = col
        elif cl in ('low',):
            col_map['low'] = col
        elif cl in ('volume', 'vol'):
            col_map['volume'] = col

    if 'date' not in col_map or 'close' not in col_map:
        return jsonify({
            'success': False,
            'error': 'Could not detect Date and Close columns',
            'columns': df.columns.tolist(),
            'col_map': col_map,
        }), 400

    df = df.rename(columns={col_map['date']: 'Date', col_map['close']: 'Close'})
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
    df = df.dropna(subset=['Date', 'Close']).sort_values('Date')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    df = df.set_index('Date')

    preview = df.head(5).reset_index().to_dict('records')
    # Convert dates to strings for JSON
    for row in preview:
        for k, v in row.items():
            if hasattr(v, 'strftime'):
                row[k] = v.strftime('%Y-%m-%d')

    return jsonify({
        'success': True,
        'columns': df.reset_index().columns.tolist(),
        'total_rows': len(df),
        'preview': preview,
        'date_range': {
            'start': df.index.min().strftime('%Y-%m-%d'),
            'end':   df.index.max().strftime('%Y-%m-%d'),
        },
    })

# ── API: Watchlist ───────────────────────────────────────────────────────────
@app.route('/api/watchlist', methods=['GET', 'POST'])
def api_watchlist():
    if 'watchlist' not in session:
        session['watchlist'] = []
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        ticker = data.get('ticker', '').upper().strip()
        if ticker and ticker not in session['watchlist']:
            wl = session['watchlist'][:]
            wl.append(ticker)
            session['watchlist'] = wl[:20]
            session.modified = True
        return jsonify({'success': True, 'watchlist': session['watchlist']})
    return jsonify({'success': True, 'watchlist': session.get('watchlist', [])})

@app.route('/api/watchlist/<ticker>', methods=['DELETE'])
def api_watchlist_remove(ticker):
    ticker = ticker.upper().strip()
    wl = session.get('watchlist', [])
    session['watchlist'] = [t for t in wl if t != ticker]
    session.modified = True
    return jsonify({'success': True, 'watchlist': session['watchlist']})

# ── API: Prediction History ──────────────────────────────────────────────────
@app.route('/api/history')
def api_history():
    return jsonify({'success': True, 'history': session.get('history', [])})

# ── Legacy Routes (kept) ─────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        ticker = request.form['ticker'].upper().strip()
        days   = int(request.form['days'])
        model  = request.form.get('model', 'prophet').lower()
        if not ticker:
            return render_template('results.html', graph='<b>Please enter a valid ticker symbol.</b>', ticker=ticker, days=days)

        client_ip = request.remote_addr
        session_key = get_session_key()
        if is_rate_limited(f"{client_ip}_{session_key}"):
            return render_template('results.html', graph='<b>Rate limit exceeded. Please wait.</b>', ticker=ticker, days=days)
        cleanup_old_timestamps()

        hist, is_demo = fetch_stock_data_with_retry(ticker, period="5y")
        demo_notice = ''
        if is_demo:
            demo_notice = f"<div class='demo-banner'>⚠ Demo Mode: Real data unavailable for {ticker}. Showing sample data.</div>"

        if hist.empty:
            return render_template('results.html', graph='<b>Unable to fetch data for this ticker.</b>', ticker=ticker, days=days)

        df = compute_indicators(hist.copy())
        result, err = run_model(model, df, days)
        if result is None:
            return render_template('results.html', graph=f'<b>Model error: {err}</b>', ticker=ticker, days=days)

        close = df['Close'].dropna()
        pdf = pd.DataFrame({'ds': pd.to_datetime(result['historical_dates']), 'y': result['historical_close']})
        future_df = pd.DataFrame({'ds': pd.to_datetime(result['predicted_dates']), 'yhat': result['predicted_values'],
                                  'yhat_lower': result['lower'], 'yhat_upper': result['upper']})

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pdf['ds'], y=pdf['y'], mode='lines', name='Historical', line=dict(color='#58a6ff')))
        fig.add_trace(go.Scatter(x=future_df['ds'], y=future_df['yhat'], mode='lines', name='Predicted', line=dict(color='#3fb950', dash='dash')))
        fig.add_trace(go.Scatter(x=pd.concat([future_df['ds'], future_df['ds'][::-1]]),
            y=pd.concat([future_df['yhat_upper'], future_df['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(63,185,80,0.1)', line=dict(color='rgba(255,255,255,0)'), name='Confidence Band'))
        fig.update_layout(template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
                          title=f'{ticker} — {result["model"]} Prediction ({days} days)',
                          font=dict(color='#c9d1d9'), hovermode='x unified')
        graph = fig.to_html(full_html=False)
        return render_template('results.html', graph=demo_notice + graph, ticker=ticker, days=days,
                               rmse=result['rmse'], mae=result['mae'], model=result['model'])
    except Exception as e:
        return render_template('results.html', graph=f'<b>Error: {e}</b>',
                               ticker=request.form.get('ticker', '?'), days=request.form.get('days', 30))

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return render_template('csv_results.html', csv_graph='<b>No file uploaded.</b>')
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return render_template('csv_results.html', csv_graph=f'<b>CSV parse error: {e}</b>')

    if not {'Date', 'Close'}.issubset(df.columns):
        # Try lowercase
        df.columns = [c.strip().title() for c in df.columns]
    if not {'Date', 'Close'}.issubset(df.columns):
        return render_template('csv_results.html', csv_graph='<b>CSV must contain Date and Close columns.</b>')

    df['MovingAvg'] = pd.to_numeric(df['Close'], errors='coerce').rolling(window=20).mean()
    fig = px.line(df, x='Date', y=['Close', 'MovingAvg'], title='CSV Data Analysis',
                  template='plotly_dark', color_discrete_sequence=['#58a6ff', '#3fb950'])
    fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#161b22', font=dict(color='#c9d1d9'))
    csv_graph = fig.to_html(full_html=False)
    return render_template('csv_results.html', csv_graph=csv_graph)

@app.route('/manual_predict', methods=['POST'])
def manual_predict():
    try:
        dates = request.form.getlist('dates[]')
        sales = request.form.getlist('sales[]')
        prediction_days = int(request.form.get('prediction_days', 30))
        model = request.form.get('model', 'prophet').lower()

        if not dates or not sales or len(dates) != len(sales):
            return render_template('results.html', graph='<b>Invalid data.</b>', ticker="Manual", days=prediction_days)

        data = {'Date': pd.to_datetime(dates), 'Close': [float(s) for s in sales]}
        df = pd.DataFrame(data).set_index('Date').sort_index()
        df = compute_indicators(df)

        result, err = run_model(model, df, prediction_days)
        if result is None:
            return render_template('results.html', graph=f'<b>Model error: {err}</b>', ticker="Manual", days=prediction_days)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pd.to_datetime(result['historical_dates']), y=result['historical_close'],
                                  mode='lines+markers', name='Actual', line=dict(color='#58a6ff')))
        fig.add_trace(go.Scatter(x=pd.to_datetime(result['predicted_dates']), y=result['predicted_values'],
                                  mode='lines', name='Predicted', line=dict(color='#3fb950', dash='dash')))
        fig.update_layout(template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
                          title=f'Manual Data Prediction — {result["model"]}', font=dict(color='#c9d1d9'))
        return render_template('results.html', graph=fig.to_html(full_html=False), ticker="Manual Data", days=prediction_days,
                               rmse=result['rmse'], mae=result['mae'], model=result['model'])
    except Exception as e:
        return render_template('results.html', graph=f'<b>Error: {e}</b>', ticker="Manual", days=30)

# ── Status Route (kept) ──────────────────────────────────────────────────────
@app.route('/status')
def status():
    cache_files = []
    if os.path.exists(CACHE_DIR):
        for fn in os.listdir(CACHE_DIR):
            if fn.endswith('.pkl'):
                fp = os.path.join(CACHE_DIR, fn)
                ft = datetime.fromtimestamp(os.path.getmtime(fp))
                cache_files.append({'file': fn, 'age': str(datetime.now() - ft), 'valid': is_cache_valid(fp)})
    return jsonify({
        'cache_files': len(cache_files),
        'active_rate_limits': len(REQUEST_TIMESTAMPS),
        'models': {
            'prophet':    PROPHET_AVAILABLE,
            'sklearn':    SKLEARN_AVAILABLE,
            'statsmodels': STATSMODELS_AVAILABLE,
            'tensorflow': TENSORFLOW_AVAILABLE,
        },
        'cache_details': cache_files[:10],
    })

if __name__ == '__main__':
    cleanup_cache()
    app.run(debug=True, host='0.0.0.0', port=5000)
