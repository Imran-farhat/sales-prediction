# Stock Predictor Application

## Overview

This is a Flask-based web application that provides stock price predictions using Facebook's Prophet forecasting model. The application offers three main functionalities:

1. **Stock Price Prediction**: Predict future stock prices by entering a stock ticker symbol and number of days to predict.
2. **CSV Data Analysis**: Upload a CSV file with historical stock data for analysis and visualization.
3. **Manual Data Prediction**: Enter manual sales data with dates to generate predictions.

## Features

- Interactive web interface with responsive design
- Real-time stock data fetching using Yahoo Finance API (yfinance)
- Time series forecasting with Facebook's Prophet
- Data visualization using Plotly
- Support for both stock market data and custom CSV data
- Manual data entry with dynamic form fields

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-predictor.git
   cd stock-predictor
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Note: On some systems, you might need to install additional dependencies for Prophet:
   ```bash
   sudo apt-get install python3-dev python3-pip python3-venv
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Access the application in your browser at:
   ```
   http://localhost:5000
   ```

## Usage

1. **Stock Prediction**:
   - Enter a valid stock ticker (e.g., AAPL for Apple)
   - Specify the number of days to predict (1-365)
   - Click "Predict" to see the forecast

2. **CSV Analysis**:
   - Upload a CSV file containing at least 'Date' and 'Close' columns
   - The application will display a chart with the closing prices and moving average

3. **Manual Data Prediction**:
   - Add rows with dates and corresponding sales values
   - Specify prediction days
   - Click "Predict Sales" to generate forecast

## File Structure

```
stock-predictor/
├── app.py               # Flask application backend
├── requirements.txt     # Python dependencies
├── static/              # Static files (CSS, JS, images)
├── templates/
│   ├── index.html       # Main application page
│   ├── results.html     # Stock prediction results page
│   └── csv_results.html # CSV analysis results page
└── README.md            # This file
```

## Dependencies

- Python 3.7+
- Flask (web framework)
- yfinance (Yahoo Finance API)
- prophet (Facebook's forecasting tool)
- pandas (data manipulation)
- plotly (data visualization)

## Limitations

- Stock predictions are based on historical data and may not account for sudden market changes
- The Prophet model works best with at least several months of historical data
- CSV files must contain specific column names ('Date' and 'Close')

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For issues or questions, please open an issue on the GitHub repository.
