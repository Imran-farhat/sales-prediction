# app.py (Flask Backend)
from flask import Flask, render_template, request, session
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
import pandas as pd
import plotly.express as px
import io
import time
import random
from datetime import datetime, timedelta
import os
import pickle
import hashlib

app = Flask(__name__)

# Rate limiting configuration
REQUEST_TIMESTAMPS = {}
MIN_REQUEST_INTERVAL = 5  # seconds between requests for the same IP (reduced from 10)

# Add session configuration
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key

def is_rate_limited(ip_address):
    """Check if IP is rate limited"""
    now = time.time()
    if ip_address in REQUEST_TIMESTAMPS:
        if now - REQUEST_TIMESTAMPS[ip_address] < MIN_REQUEST_INTERVAL:
            return True
    REQUEST_TIMESTAMPS[ip_address] = now
    return False

def get_session_key():
    """Get a unique session key for rate limiting"""
    if 'session_id' not in session:
        session['session_id'] = hashlib.md5(str(random.random()).encode()).hexdigest()
    return session['session_id']

# Clean up old timestamps periodically
def cleanup_old_timestamps():
    """Remove old timestamps to prevent memory buildup"""
    now = time.time()
    cutoff = now - (MIN_REQUEST_INTERVAL * 10)  # Keep only last 10 intervals
    keys_to_remove = [ip for ip, timestamp in REQUEST_TIMESTAMPS.items() if timestamp < cutoff]
    for key in keys_to_remove:
        del REQUEST_TIMESTAMPS[key]

# Cache configuration
CACHE_DIR = 'cache'
CACHE_EXPIRY_HOURS = 6  # Increased to 6 hours to reduce API calls

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_filename(ticker, period):
    """Generate cache filename for a ticker and period"""
    return os.path.join(CACHE_DIR, f"{ticker}_{period}.pkl")

def is_cache_valid(cache_file):
    """Check if cache file exists and is not expired"""
    if not os.path.exists(cache_file):
        return False
    
    file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
    return datetime.now() - file_time < timedelta(hours=CACHE_EXPIRY_HOURS)

def save_to_cache(data, cache_file):
    """Save data to cache file"""
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving to cache: {e}")

def load_from_cache(cache_file):
    """Load data from cache file"""
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading from cache: {e}")
        return None

def fetch_stock_data_with_retry(ticker, period="5y", max_retries=2):  # Reduced retries
    """Fetch stock data with retry logic and caching"""
    cache_file = get_cache_filename(ticker, period)
    
    # Check cache first
    if is_cache_valid(cache_file):
        cached_data = load_from_cache(cache_file)
        if cached_data is not None:
            print(f"Using cached data for {ticker}")
            return cached_data
    
    # Check if we have expired cache as immediate fallback
    if os.path.exists(cache_file):
        cached_data = load_from_cache(cache_file)
        if cached_data is not None:
            print(f"Found expired cache for {ticker}, will use as fallback")
            fallback_data = cached_data
        else:
            fallback_data = None
    else:
        fallback_data = None
    
    # Try to fetch from API with retry logic
    for attempt in range(max_retries):
        try:
            print(f"Fetching data for {ticker} (attempt {attempt + 1}/{max_retries})")
            
            # Add a small delay between requests to avoid rate limiting
            if attempt > 0:
                time.sleep(5)
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if not hist.empty:
                # Save to cache
                save_to_cache(hist, cache_file)
                print(f"Successfully fetched and cached {ticker}")
                return hist
            else:
                print(f"No data returned for {ticker}")
                
        except YFRateLimitError:
            print(f"Rate limit hit for {ticker}, attempt {attempt + 1}")
            if attempt < max_retries - 1:
                # Shorter wait times but still effective
                wait_time = 15 + random.uniform(5, 10)  # 15-25 seconds
                print(f"Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Max retries reached for {ticker}")
                
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            if attempt < max_retries - 1:
                wait_time = 5 + random.uniform(1, 3)
                print(f"Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
    
    # If all attempts failed, use fallback data if available
    if fallback_data is not None:
        print(f"Using expired cache data for {ticker}")
        return fallback_data
    
    print(f"No data available for {ticker}")
    return pd.DataFrame()  # Return empty DataFrame if all attempts fail

def create_sample_data(ticker, period="5y"):
    """Create sample stock data for demonstration when API is unavailable"""
    print(f"Creating sample data for {ticker}")
    
    # Create date range
    if period == "5y":
        days = 365 * 5
    elif period == "1y":
        days = 365
    else:
        days = 365  # Default to 1 year
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate sample price data with some volatility
    base_price = 100
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Add some random walk behavior
        change = random.uniform(-0.05, 0.05)  # ±5% daily change
        current_price = max(10, current_price * (1 + change))  # Minimum price of $10
        prices.append(current_price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': [p * random.uniform(0.98, 1.02) for p in prices],
        'High': [p * random.uniform(1.00, 1.05) for p in prices],
        'Low': [p * random.uniform(0.95, 1.00) for p in prices],
        'Close': prices,
        'Volume': [random.randint(1000000, 10000000) for _ in prices]
    })
    
    df.set_index('Date', inplace=True)
    return df

def preload_popular_stocks():
    """Preload popular stocks to cache during startup (optional)"""
    popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']  # Reduced list
    preloaded_count = 0
    
    for ticker in popular_stocks:
        cache_file = get_cache_filename(ticker, "5y")
        if not is_cache_valid(cache_file):
            print(f"Attempting to preload {ticker} to cache...")
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5y")
                if not hist.empty:
                    save_to_cache(hist, cache_file)
                    print(f"Successfully cached {ticker}")
                    preloaded_count += 1
                else:
                    print(f"No data available for {ticker}")
                
                # Increase delay between requests
                time.sleep(3)
                
                # Stop preloading if we hit rate limits
                if preloaded_count >= 2:  # Limit to 2 successful preloads
                    print("Preload limit reached to avoid rate limiting")
                    break
                    
            except YFRateLimitError:
                print(f"Rate limit hit during preload for {ticker} - stopping preload")
                break
            except Exception as e:
                print(f"Failed to preload {ticker}: {e}")
                continue
    
    print(f"Preloaded {preloaded_count} stocks to cache")

# Configuration
DEMO_MODE_ENABLED = True  # Enable demo mode when API is unavailable
ENABLE_PRELOADING = False  # Disable preloading to avoid rate limits

# Initialize cache with popular stocks on startup (configurable)
if ENABLE_PRELOADING:
    print("Initializing cache with popular stocks...")
    try:
        preload_popular_stocks()
    except Exception as e:
        print(f"Error during cache initialization: {e}")
        print("Continuing without preloading - cache will be populated as needed")
else:
    print("Cache preloading disabled. Cache will be populated as needed.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        ticker = request.form['ticker'].upper().strip()
        days = int(request.form['days'])
        
        if not ticker:
            return render_template('results.html', 
                                graph='<b>Please enter a valid ticker symbol.</b>', 
                                ticker=ticker, 
                                days=days)
          # Check rate limiting (using both IP and session)
        client_ip = request.remote_addr
        session_key = get_session_key()
        rate_limit_key = f"{client_ip}_{session_key}"
        
        if is_rate_limited(rate_limit_key):
            return render_template('results.html', 
                                graph='<b>Rate limit exceeded. Please wait a few seconds before trying again.</b>', 
                                ticker=ticker, 
                                days=days)
        
        # Clean up old timestamps
        cleanup_old_timestamps()        # Fetch data with retry logic and caching
        hist = fetch_stock_data_with_retry(ticker, period="5y")
        
        if hist.empty:
            # Try to create sample data for demonstration if enabled
            if DEMO_MODE_ENABLED:
                print(f"Creating sample data for {ticker} due to API unavailability")
                hist = create_sample_data(ticker, period="5y")
                
                if not hist.empty:
                    # Add a notice that this is sample data
                    demo_notice = f"<div style='background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; border-radius: 5px;'><strong>Demo Mode:</strong> Unable to fetch real data for {ticker} due to API limitations. Showing sample data for demonstration purposes.</div>"
            
            if hist.empty:
                return render_template('results.html', 
                                    graph='<b>Unable to fetch data for this ticker. This could be due to:</b><br>• Invalid ticker symbol<br>• API rate limiting<br>• Network issues<br><br>Please verify the ticker symbol and try again in a few minutes.', 
                                    ticker=ticker, 
                                    days=days)
        
        if Prophet is None:
            return render_template('results.html', 
                                graph='<b>Prophet library is not installed. Please install prophet.</b>', 
                                ticker=ticker, 
                                days=days)
        
        # Prepare Prophet model
        df = hist.reset_index()[['Date', 'Close']]
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  # Remove timezone
        df.columns = ['ds', 'y']
        
        # Check if we have enough data
        if len(df) < 2:
            return render_template('results.html', 
                                graph='<b>Not enough historical data for prediction.</b>', 
                                ticker=ticker, 
                                days=days)
        
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
          # Generate Plotly graph
        fig = px.line(forecast, x='ds', y='yhat', 
                     title=f'{ticker} Price Prediction for next {days} days')
        
        # Add actual historical data
        fig.add_scatter(x=df['ds'], y=df['y'], mode='lines', name='Historical Data', line=dict(color='blue'))
        
        # Add prediction confidence intervals
        fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                       mode='lines', name='Upper Confidence', line=dict(color='lightgray', dash='dash'))
        fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], 
                       mode='lines', name='Lower Confidence', line=dict(color='lightgray', dash='dash'))
        
        graph = fig.to_html(full_html=False)
        
        # Add demo notice if using sample data
        if 'demo_notice' in locals():
            graph = demo_notice + graph
        
        return render_template('results.html', 
                             graph=graph,
                             ticker=ticker,
                             days=days)
    
    except Exception as e:
        return render_template('results.html', 
                            graph=f'<b>Unexpected error: {str(e)}</b>', 
                            ticker=ticker if 'ticker' in locals() else 'Unknown', 
                            days=days if 'days' in locals() else 30)

@app.route('/upload', methods=['POST'])
def upload():
    # Process CSV file
    file = request.files['file']
    df = pd.read_csv(file)
    if not set(['Date', 'Close']).issubset(df.columns):
        return render_template('csv_results.html', csv_graph='<b>CSV must contain Date and Close columns.</b>')
    # Process data (example calculation)
    df['MovingAvg'] = df['Close'].rolling(window=20).mean()
    # Generate visualization
    fig = px.line(df, x='Date', y=['Close', 'MovingAvg'],
                 title='CSV Data Analysis')
    csv_graph = fig.to_html(full_html=False)
    return render_template('csv_results.html',
                         csv_graph=csv_graph)

@app.route('/manual_predict', methods=['POST'])
def manual_predict():
    try:
        # Get data from form
        dates = request.form.getlist('dates[]')
        sales = request.form.getlist('sales[]')
        prediction_days = int(request.form.get('prediction_days', 30))
        
        if not dates or not sales or len(dates) != len(sales):
            return render_template('results.html', 
                                graph='<b>Invalid data. Please ensure dates and sales values match.</b>',
                                ticker="Manual Data", 
                                days=prediction_days)
        
        # Convert to DataFrame
        data = {'ds': dates, 'y': [float(s) for s in sales]}
        df = pd.DataFrame(data)
        df['ds'] = pd.to_datetime(df['ds'])
        
        if Prophet is None:
            return render_template('results.html', 
                                graph='<b>Prophet library is not installed. Please install prophet.</b>',
                                ticker="Manual Data", 
                                days=prediction_days)
        
        # Fit Prophet model
        model = Prophet()
        model.fit(df)
        
        # Make forecast
        future = model.make_future_dataframe(periods=prediction_days)
        forecast = model.predict(future)
        
        # Generate Plotly graph
        fig = px.line(forecast, x='ds', y='yhat', 
                    title=f'Sales Prediction for next {prediction_days} days')
        # Add actual data points
        fig.add_scatter(x=df['ds'], y=df['y'], mode='markers', name='Actual Data')
        graph = fig.to_html(full_html=False)
        
        return render_template('results.html', 
                            graph=graph,
                            ticker="Manual Data",
                            days=prediction_days)
    except Exception as e:
        return render_template('results.html', 
                            graph=f'<b>Error: {str(e)}</b>',
                            ticker="Manual Data", 
                            days=prediction_days if 'prediction_days' in locals() else 30)

def cleanup_cache():
    """Remove old cache files to prevent disk space issues"""
    try:
        if os.path.exists(CACHE_DIR):
            for filename in os.listdir(CACHE_DIR):
                filepath = os.path.join(CACHE_DIR, filename)
                if filename.endswith('.pkl'):
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if datetime.now() - file_time > timedelta(hours=CACHE_EXPIRY_HOURS * 2):
                        os.remove(filepath)
                        print(f"Removed old cache file: {filename}")
    except Exception as e:
        print(f"Error during cache cleanup: {e}")

@app.route('/status')
def status():
    """Status endpoint to check cache and rate limiting"""
    cache_files = []
    if os.path.exists(CACHE_DIR):
        for filename in os.listdir(CACHE_DIR):
            if filename.endswith('.pkl'):
                filepath = os.path.join(CACHE_DIR, filename)
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                cache_files.append({
                    'file': filename,
                    'age': str(datetime.now() - file_time),
                    'valid': is_cache_valid(filepath)
                })
    
    return {
        'cache_files': len(cache_files),
        'active_rate_limits': len(REQUEST_TIMESTAMPS),
        'cache_details': cache_files[:10]  # Show first 10 cache files
    }

if __name__ == '__main__':
    # Clean up old cache files on startup
    cleanup_cache()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
