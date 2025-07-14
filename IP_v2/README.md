# Earnings Implied Volatility (EIV) Analysis

This project analyzes earnings-induced volatility using options data from WRDS (Wharton Research Data Services).

## Prerequisites

1. **WRDS Account**: You need a WRDS account with access to:
   - OptionMetrics (optionm schema)
   - Compustat (for earnings dates)
   - CRSP (for stock prices)

2. **Python Environment**: Python 3.8 or higher

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure WRDS Credentials

You need to update the WRDS credentials in `main.py`. Replace the placeholder credentials with your actual WRDS username and password:

```python
db = wrds.Connection(wrds_username="your_username",
                     password="your_password")
```

### 3. Project Structure

```
earnings_iv/
├── main.py              # Main execution script
├── pipeline.py          # Data pipeline class
├── analysis.py          # Analysis functions
├── queries.py           # SQL query builders
├── plotting.py          # Visualization functions
├── utils.py             # Utility functions
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Running the Analysis

### Option 1: Run as a Module

```bash
python -m earnings_iv.main
```

### Option 2: Run Directly

```bash
python main.py
```

### Option 3: Run from Parent Directory

If you're in the parent directory of `earnings_iv/`:

```bash
python -m earnings_iv.main
```

## Configuration

You can modify the analysis parameters in `main.py`:

```python
# Define analysis parameters
TICKERS = ['AAPL']                    # Stock tickers to analyze
START_DATE = '2020-01-01'            # Start date for data
END_DATE = '2024-12-31'              # End date for data

# Custom filter values
MIN_VOLUME = 1                       # Minimum option volume
MAX_BID_ASK_SPREAD = 1.0            # Maximum bid-ask spread
TTE_RANGE = (1, 90)                 # Time-to-expiry range (days)
MONEYNESS_RANGE = (0.8, 1.2)        # Moneyness range
```

## What the Pipeline Does

1. **Data Collection**: Fetches securities info, earnings dates, options data, and stock prices from WRDS
2. **Data Processing**: Calculates option metrics (TTE, moneyness, bid-ask spreads)
3. **Filtering**: Applies quality filters to remove low-quality data
4. **EIV Calculation**: Computes Earnings-Induced Volatility for each earnings event
5. **Analysis**: Runs IV vs Realized Volatility regression analysis
6. **Reporting**: Generates summary reports and visualizations

## Output

The pipeline generates:
- Summary statistics for the data
- EIV calculations for earnings events
- IV vs Realized Volatility analysis
- Plots (if matplotlib backend is configured)

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the correct directory and have installed all dependencies
2. **WRDS Connection**: Verify your WRDS credentials and internet connection
3. **Data Access**: Ensure your WRDS account has access to the required databases

### Linter Errors

The linter may show import errors for `wrds` and `earnings_iv.analysis`. These are expected if:
- The `wrds` package isn't installed
- You're running the script from outside the package context

These errors don't affect runtime if the dependencies are properly installed.

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `scipy`: Statistical functions
- `matplotlib`: Plotting
- `statsmodels`: Statistical modeling
- `wrds`: WRDS database connection 