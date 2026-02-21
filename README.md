# 📈 StockAI: Advanced Prediction Dashboard

StockAI is a professional-grade, Bloomberg Terminal-inspired dashboard for stock market forecasting and technical analysis. Built with Flask, it integrates multiple machine learning models and real-time data processing to provide actionable financial insights.

![Dashboard Preview](file:///C:/Users/ACER/.gemini/antigravity/brain/32409471-d024-4378-b2a8-7c2030143881/verify_prediction_app_1771656070958.webp)

## 🚀 Key Features

- **Multi-Model Intelligence**: Compare predictions from **Prophet**, **ARIMA**, **LSTM**, and **Linear Regression** in one view.
- **"Zero-API" Manual Mode**: Paste or type your own data for instant frontend forecasting without any backend overhead.
- **Advanced Technical Indicators**: Real-time calculation of **RSI**, **MACD**, and **Bollinger Bands**.
- **Modern Bloomberg UI**: High-contrast dark theme with interactive Plotly charts (Candlestick, Line, OHLC).
- **Smart Data Pipeline**: File-based caching and intelligent fallback logic to handle Yahoo Finance rate limits gracefully.
- **Watchlist & History**: Session-based persistence for tracking your favorite tickers.

## 🛠️ Installation

### Prerequisites
- **Python 3.13** (Recommended for full ML support)
- *Note: Python 3.14 is currently in "Limited Mode" (Prophet/LSTM disabled).*

### Quick Start
1. **Clone & Install**:
   ```bash
   git clone https://github.com/yourusername/stock-prediction.git
   cd stock-prediction
   pip install -r requirements.txt
   ```

2. **Run the App**:
   - **Windows**: Double-click `start.bat` (this ensures the correct Python version is used).
   - **Manual**: `python app.py`

3. **Access**: Open `http://localhost:5000` in your browser.

## 📖 Usage Guide

### 1. Market Prediction
Search for any ticker (e.g., `AAPL`, `BTC-USD`, `EURUSD=X`). Adjust the forecast horizon (up to 365 days) and select your preferred model.

### 2. Technical Analysis
Switch to the **Technical Indicators** panel to view stacked RSI and MACD charts synced with historical price data.

### 3. Manual Data Entry
Navigate to the **Manual Data** section to input custom data. Use the **Paste from Clipboard** feature for tab-separated data from Excel or Google Sheets.

### 4. CSV Upload
Drag and drop your own CSV files. The app automatically detects `Date` and `Close` columns to generate analysis and forecasts.

## ⚠️ Troubleshooting & Rate Limits

The app uses **Yahoo Finance** for real-time data which has strict rate limits. Our smart pipeline handles this by:
- **Caching**: Data is cached for 1 hour to reduce API calls.
- **Fallbacks**: If the API is blocked, the app automatically attempts a shorter 1-year window or uses expired cache.
- **Demo Mode**: If all else fails, a high-quality sample dataset is generated for testing.

## 🏗️ Architecture

- **Backend**: Python (Flask)
- **Forecasting**: Prophet, Scikit-learn, Statsmodels, TensorFlow
- **Frontend**: JavaScript (Vanilla), Plotly.js, CSS3
- **Data Source**: yfinance API

## 📝 License
This project is licensed under the MIT License.
