# Earnings Implied Volatility Analysis

This project analyzes the relationship between implied volatility and realized volatility around earnings announcements for large-cap stocks.

## Setup Instructions

### 1. Install Python
- Download Python 3.8+ from https://www.python.org/downloads/
- Make sure to check "Add Python to PATH" during installation
- Restart your terminal after installation

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure WRDS Access
1. Get a WRDS account with access to the required databases
2. No need to edit any files - you'll be prompted for credentials when running the analysis

### 4. Test Options Surface Features (Optional)
```bash
python test_options_features.py
```

### 5. Run the Analysis
```bash
python main.py
```

## What the Code Does

This system analyzes earnings-related volatility by:

1. **REVR (Realized Earnings Volatility Ratio)**: Measures actual volatility increase after earnings
2. **IEVR (Implied Earnings Volatility Ratio)**: Measures expected volatility from options pricing
3. **Options Surface Features**: Advanced options market features including:
   - **TERM_RATIO**: 30-day ATM IV / 10-day ATM IV (term structure)
   - **SKEW**: (Call OTM IV - Put OTM IV) / ATM IV (volatility skew)
   - **KURT**: (Call OTM IV + Put OTM IV - 2×ATM IV) / ATM IV (volatility kurtosis)
   - **IV_RATIO**: Recent ATM IV / Earlier ATM IV (monthly change)
   - **SMIRK**: (Put OTM IV - Call ATM IV) / Call ATM IV (volatility smirk)
4. **Regression Analysis**: Tests the relationship REVR = α + β × IEVR with options surface controls

## Output Files

- **Data Files**: CSV files with analysis results in `data_files/`
- **Visualizations**: PNG charts in `output_files/`
- **Regression Results**: Statistical summaries and model diagnostics

## Key Features

- Analyzes 45+ large-cap stocks across Technology, Financial, and Healthcare sectors
- Covers 2015-2024 period
- Uses advanced statistical methods including kernel regression
- Incorporates comprehensive options surface features (term structure, skew, kurtosis, smirk)
- Generates comprehensive visualizations and reports
- Tests interaction effects between IEVR and options surface features

## Troubleshooting

- **Python not found**: Make sure Python is installed and added to PATH
- **WRDS connection error**: Verify your credentials when prompted
- **Missing dependencies**: Run `pip install -r requirements.txt`
