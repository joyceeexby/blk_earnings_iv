# 🎯 Option Surface Features Generation 2005-2023

This focused script generates option surface features for every earnings season from 2005-2023, using the top 500 high volume stocks for each season from the existing `top500_liquidity_2005_2023.csv` file.

## 📊 **What This Script Does**

### **🎯 Core Functionality**
- **📅 Time Period**: 2005-2023 (uses existing data from CSV file)
- **📈 Stock Selection**: Top 500 high volume stocks per earnings season (from existing CSV)
- **🎯 Features**: All 5 option surface features (TERM_RATIO, SKEW, KURT, IV_RATIO, SMIRK)
- **💾 Output**: Individual CSV files per season + combined summary

### **🔧 Features Generated**
1. **TERM_RATIO**: 30-day vs 10-day ATM implied volatility ratio
2. **SKEW**: Implied volatility skew measure
3. **KURT**: Implied volatility kurtosis measure
4. **IV_RATIO**: Monthly implied volatility change ratio
5. **SMIRK**: Volatility smirk measure

## 🏗️ **Files Created**

### **Main Scripts**
- **`generate_option_features_2005_2023.py`**: Main generation script
- **`run_option_features_generation.py`**: Simple execution script with WRDS connection

### **Input Files**
- **`data_files/top500_liquidity_2005_2023.csv`**: Existing file with top 500 stocks per season

### **Output Files**
- **Individual Season Files**: `option_features_YYYY_QX.csv` (one per season in data)
- **Combined Data**: `option_features_2005_2023_combined.csv`
- **Summary Report**: `option_features_summary.json`

## 🎯 **How to Use**

### **1. Verify Input Data**
Make sure the input file exists:
```bash
# Check if the file exists
ls data_files/top500_liquidity_2005_2023.csv
```

### **2. Set Up WRDS Connection**
Make sure you have WRDS credentials configured:
```bash
# Install WRDS if needed
pip install wrds

# Configure WRDS credentials (if not already done)
# The script will prompt for username/password if needed
```

### **3. Run the Generation**
```bash
# Simple execution
python run_option_features_generation.py

# Or run the main script directly
python generate_option_features_2005_2023.py
```

### **4. Monitor Progress**
The script will show:
- 📊 Loading of existing top 500 stocks data
- 📅 Total seasons found in the data
- ✅ Progress for each earnings season
- ❌ Any failures
- 📁 Generated file paths
- 📋 Final summary statistics

## 📅 **Earnings Seasons Covered**

The script automatically determines the seasons from your existing data file. Based on the file structure, it should cover:

### **Data Structure**
```csv
year,quarter,quarter_start_date,quarter_end_date,permno,ticker,cusip,comnam,dollar_vol
2005,1,2005-01-01,2005-03-31,10107,MSFT,59491810,MICROSOFT CORP,401426159.04
2005,1,2005-01-01,2005-03-31,90319,GOOG,02079K30,GOOGLE INC,400850223.74
...
```

### **Automatic Season Detection**
- The script reads all unique year-quarter combinations from your CSV file
- Converts quarter numbers (1,2,3,4) to quarter names (Q1,Q2,Q3,Q4)
- Processes only the seasons that exist in your data

## 📊 **Output Structure**

### **Individual Season Files**
```csv
ticker,TERM_RATIO,SKEW,KURT,IV_RATIO,SMIRK,year,quarter,earnings_date,computation_date
AAPL,1.156,0.023,0.045,1.089,0.012,2023,Q1,2023-01-26,2023-12-19 10:30:15
MSFT,1.134,0.019,0.038,1.076,0.009,2023,Q1,2023-01-26,2023-12-19 10:30:16
...
```

### **Combined Data File**
- All seasons combined into one file
- Additional column: `source_file` (original file name)
- Total records: Depends on your data (seasons × ~500 stocks)

### **Summary Report**
```json
{
  "total_records": 38000,
  "unique_tickers": 1200,
  "year_range": "2005-2023",
  "quarters": ["Q1", "Q2", "Q3", "Q4"],
  "feature_columns": ["TERM_RATIO", "SKEW", "KURT", "IV_RATIO", "SMIRK"]
}
```

## 🔧 **Configuration Options**

### **Earnings Dates**
The script uses approximate earnings dates:
- **Q1**: Late January (e.g., 2023-01-26)
- **Q2**: Late April (e.g., 2023-04-26)
- **Q3**: Late July (e.g., 2023-07-26)
- **Q4**: Late October (e.g., 2023-10-26)

### **Parameters**
- **n_lag**: 20 trading days before earnings (configurable)
- **Input file**: `data_files/top500_liquidity_2005_2023.csv` (configurable)
- **Output directory**: `data_files/` (configurable)

## ⚡ **Performance Considerations**

### **Expected Runtime**
- **Total Time**: Several hours (depending on database speed)
- **Per Season**: ~2-5 minutes
- **Total Seasons**: Depends on your data file

### **Database Load**
- Uses existing `compute_option_surface_features` function
- Only queries WRDS for option data (not stock selection)
- Includes delays between seasons to avoid overwhelming database

### **File Management**
- Checks for existing files to avoid regeneration
- Creates individual files per season for easier management
- Combines all data at the end

## 🎯 **Integration with Existing Code**

### **Uses Existing Functions**
- **`compute_option_surface_features()`**: Computes the 5 option features
- **Existing CSV data**: Uses your pre-computed top 500 stocks
- **Existing database queries**: Leverages your current WRDS setup

### **No Breaking Changes**
- Works with your existing codebase
- Uses same functions and methods
- Maintains same output format

## 🏁 **Getting Started**

### **1. Verify Dependencies**
```bash
pip install pandas numpy wrds
```

### **2. Check Input File**
```bash
# Verify the input file exists
ls data_files/top500_liquidity_2005_2023.csv
```

### **3. Check WRDS Access**
```python
import wrds
db = wrds.Connection()
# Should connect without errors
```

### **4. Run Generation**
```bash
python run_option_features_generation.py
```

### **5. Monitor Output**
- Watch console output for progress
- Check `data_files/` directory for generated files
- Review summary report for statistics

## 🎉 **Expected Results**

### **Files Generated**
- Individual season files (one per season in your data)
- 1 combined data file
- 1 summary report

### **Data Coverage**
- **Time Period**: Based on your existing data (likely 2005-2023)
- **Stocks**: Top 500 per season (from your existing data)
- **Features**: 5 option surface features per stock
- **Seasons**: All seasons in your existing data

### **Use Cases**
- **Research**: Comprehensive option surface analysis
- **Backtesting**: Historical option feature data
- **Analysis**: Cross-sectional and time-series studies
- **Modeling**: Feature engineering for ML models

## 🎯 **Summary**

This script provides:
- ✅ **Uses Existing Data**: Leverages your pre-computed top 500 stocks
- ✅ **Complete Coverage**: All seasons in your existing data
- ✅ **All Features**: 5 option surface features per stock
- ✅ **Efficient Processing**: Only queries WRDS for option data
- ✅ **Comprehensive Output**: Individual + combined files
- ✅ **Easy Execution**: Simple script with WRDS connection

Ready to generate comprehensive option surface features using your existing top 500 stocks data!
