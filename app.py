# app.py (Flask Backend)
from flask import Flask, render_template, request
import yfinance as yf
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
import pandas as pd
import plotly.express as px
import io

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs
    ticker = request.form['ticker']
    days = int(request.form['days'])
    
    # Fetch data
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")
    if hist.empty:
        return render_template('results.html', graph='<b>No data found for this ticker.</b>', ticker=ticker, days=days)
    if Prophet is None:
        return render_template('results.html', graph='<b>Prophet library is not installed. Please install prophet.</b>', ticker=ticker, days=days)
    # Prepare Prophet model
    df = hist.reset_index()[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  # Remove timezone
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    # Generate Plotly graph
    fig = px.line(forecast, x='ds', y='yhat', 
                 title=f'{ticker} Price Prediction')
    graph = fig.to_html(full_html=False)
    return render_template('results.html', 
                         graph=graph,
                         ticker=ticker,
                         days=days)

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

if __name__ == '__main__':
    app.run(debug=True)
