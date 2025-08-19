# Batch IEVR Analysis for Multiple Stocks

## Overview

This new pipeline efficiently calculates **Implied Earnings Volatility Ratio (IEVR)** for 500+ stocks for a single earnings date, making it much more scalable than the previous approach.

## Key Features

### 🚀 **Efficiency Improvements**
- **Parallel Processing**: Uses ThreadPoolExecutor for concurrent analysis
- **Single Earnings Date**: Focuses on one earnings date for all stocks
- **Optimized Queries**: Streamlined database queries for batch processing
- **Memory Efficient**: Processes stocks one at a time, not loading all data

### 📊 **What It Calculates**
- **IEVR**: Post-earnings IV / Pre-earnings IV
- **Skew Ratio**: 90P/110C volatility ratio
- **Pre/Post IV**: Average implied volatility before/after earnings
- **Data Quality Metrics**: Number of data points, underlying price, etc.

### 🎯 **Target Use Case**
- **Large Stock Universe**: 500+ stocks efficiently
- **Single Earnings Date**: Focus on one earnings announcement
- **Production Scale**: Designed for real-world analysis

## Files

### `batch_ievr_analysis.py`
- Main batch analysis class
- Handles stock selection, data fetching, and IEVR calculation
- Parallel processing with configurable workers

### `test_batch_ievr.py`
- Test script with small sample (10 stocks)
- Use this to verify functionality before scaling up

## Usage

### 1. **Test First (Recommended)**
```bash
cd analysis_scripts
python3 test_batch_ievr.py
```

This runs with 10 stocks to verify everything works.

### 2. **Run Full Analysis**
```bash
cd analysis_scripts
python3 batch_ievr_analysis.py
```

This runs with 100 stocks by default. To change to 500:
- Edit `num_stocks = 500` in the main function
- Or modify the call to `run_batch_analysis()`

### 3. **Custom Parameters**
```python
# Initialize with custom settings
analyzer = BatchIEVRAnalysis(db, max_workers=8)  # More workers for faster processing

# Run with custom parameters
results = analyzer.run_batch_analysis(
    earnings_date=pd.to_datetime('2023-04-30'),  # Different earnings date
    num_stocks=500,                              # Target 500 stocks
    analysis_days_before=45                      # More days before earnings
)
```

## Configuration

### **Performance Tuning**
```python
# Adjust based on your system and WRDS limits
max_workers = 4      # Default: 4 concurrent threads
num_stocks = 500     # Target number of stocks
analysis_days_before = 30  # Days before earnings to analyze
```

### **Data Quality Filters**
```python
# Built-in filters in calculate_single_ievr()
min_market_cap = 1e9        # $1B minimum market cap
min_iv_data_points = 10     # Minimum IV data points required
min_pre_post_points = 3     # Minimum pre/post earnings data
moneyness_range = (0.8, 1.2)  # Reasonable strike prices
tte_range = (10, 90)        # Reasonable time to expiry
```

## Output

### **CSV Results**
- **Filename**: `data_files/batch_ievr_results_YYYYMMDD_HHMMSS.csv`
- **Columns**: ticker, permno, earnings_date, ievr, skew_ratio, avg_pre_iv, avg_post_iv, etc.

### **Console Output**
- **Progress**: Real-time updates for each stock
- **Summary**: Success rate, timing, failed stocks
- **Statistics**: IEVR distribution, skew ratio stats

## Performance Expectations

### **Speed**
- **10 stocks**: ~1-2 minutes
- **100 stocks**: ~10-15 minutes  
- **500 stocks**: ~45-60 minutes

### **Success Rate**
- **Expected**: 70-80% success rate
- **Common failures**: No options data, insufficient IV data
- **Failed stocks**: Listed in console output

## Troubleshooting

### **Common Issues**

1. **"No secid found"**
   - Stock may not have options data
   - Check if ticker exists in optionm.securd1

2. **"No IV data found"**
   - Options table may not exist for that year
   - Stock may not have liquid options

3. **"Insufficient data"**
   - Not enough pre/post earnings options
   - Adjust `analysis_days_before` or data filters

### **Debug Mode**
The script includes extensive debug output. Look for:
- `[DEBUG]` messages showing data availability
- Stock-by-stock progress updates
- Detailed error messages for failed stocks

## Integration with Existing Pipeline

### **Next Steps**
After running batch IEVR analysis:
1. **Add REVR**: Calculate realized volatility for successful IEVR stocks
2. **Add Features**: Integrate dispersion, option surface, Fama-French factors
3. **Run Models**: Use the results in `nonlinear_models.py`

### **Data Flow**
```
Batch IEVR → CSV Results → Feature Integration → Model Training
```

## Advantages Over Previous Approach

| Aspect | Previous Pipeline | New Batch Pipeline |
|--------|------------------|-------------------|
| **Scope** | 45 stocks, multiple dates | 500+ stocks, single date |
| **Efficiency** | Sequential processing | Parallel processing |
| **Memory** | Loads all data at once | Processes incrementally |
| **Scalability** | Limited by memory | Limited by time/workers |
| **Focus** | Multiple earnings dates | Single earnings date |

## Recommendations

1. **Start Small**: Test with 10-50 stocks first
2. **Monitor Progress**: Watch console output for issues
3. **Adjust Workers**: Increase `max_workers` if your system can handle it
4. **Check Results**: Verify IEVR values are reasonable (typically 0.5-2.0)
5. **Scale Up**: Once working, increase to 500+ stocks

This new pipeline should handle your 500-stock universe much more efficiently! 🚀
