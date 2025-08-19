# Option Surface Features Integration

## Overview
The `option_surface_features.py` module has been successfully integrated into the main earnings volatility analysis pipeline. This integration adds sophisticated option market features that capture the shape and dynamics of the volatility surface around earnings events.

## 🎯 **Integrated Features**

### **1. TERM_RATIO**
- **Definition**: Term structure difference between 30-day and 10-day implied volatility
- **Calculation**: `IV_30day / IV_10day`
- **Interpretation**: Values > 1 indicate upward-sloping term structure (normal), < 1 indicate inverted term structure

### **2. SKEW**
- **Definition**: Volatility skew between OTM call and OTM put options
- **Calculation**: `(IV_OTM_Call - IV_OTM_Put) / IV_ATM`
- **Interpretation**: Positive values indicate call skew, negative values indicate put skew

### **3. KURT**
- **Definition**: Volatility kurtosis (OTM vs ATM options)
- **Calculation**: `(IV_OTM_Call + IV_OTM_Put - 2*IV_ATM) / IV_ATM`
- **Interpretation**: Measures the "fatness" of volatility tails

### **4. IV_RATIO**
- **Definition**: Monthly implied volatility change ratio
- **Calculation**: `IV_Recent / IV_Earlier` (21 days apart)
- **Interpretation**: Values > 1 indicate increasing volatility, < 1 indicate decreasing volatility

### **5. SMIRK**
- **Definition**: Volatility smirk (OTM put vs ATM call)
- **Calculation**: `(IV_OTM_Put - IV_ATM_Call) / IV_ATM_Call`
- **Interpretation**: Positive values indicate put smirk (crash protection premium)

## 🔧 **Integration Details**

### **Pipeline Flow**
```
1. Calculate REVR (ST/MT methodology)
2. Calculate IEVR (with S&P 500 comparison)
3. Get analyst dispersion (T-21 business days)
4. Load Fama-French factors (monthly)
5. Calculate option surface features (T-15 trading days) ← NEW
6. Combine all results into comprehensive dataset
```

### **Data Sources**
- **Option Surface**: `optionm_all.vsurfd{year}` tables
- **Security Info**: `optionm_all.secnmd` for secid mapping
- **Timing**: T-15 trading days before earnings (configurable)

### **Feature Calculation**
- **Delta-based**: Uses specific delta levels (25, 50) for consistent strikes
- **Maturity-based**: Focuses on 30-day and 10-day options
- **Quality filters**: Requires sufficient data for reliable calculations

## 📊 **Output Structure**

The integrated CSV now contains all features:

### **Core Volatility Measures**
- `earnings_date`, `ticker`
- `revr`, `ievr`, `ratio`
- `vol_st`, `vol_mt`, `avg_pre`, `avg_post`

### **Analyst Features**
- `analyst_dispersion`, `num_analysts`

### **Market Risk Factors**
- `mkt_rf`, `smb`, `hml`, `rmw`, `cma`, `rf`, `mkt_return`

### **Option Surface Features** ← NEW
- `TERM_RATIO`, `SKEW`, `KURT`, `IV_RATIO`, `SMIRK`
- `surface_date` (when features were calculated)

## 🚀 **Usage Examples**

### **Basic Analysis**
```python
from automated_analysis import AutomatedEarningsAnalysis

analyzer = AutomatedEarningsAnalysis(db)
results = analyzer.analyze_multiple_events(
    ticker='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

### **Feature Access**
```python
# Option surface features
term_ratio = results['TERM_RATIO']
skew = results['SKEW']
kurt = results['KURT']
iv_ratio = results['IV_RATIO']
smirk = results['SMIRK']

# All features together
all_features = results[['revr', 'ievr', 'analyst_dispersion', 'TERM_RATIO', 'SKEW']]
```

## 📈 **Enhanced Analysis Capabilities**

### **1. Volatility Surface Dynamics**
- **Term Structure**: How volatility evolves across maturities
- **Skew Patterns**: Market sentiment and crash protection demand
- **Kurtosis**: Tail risk and extreme event pricing

### **2. Earnings-Specific Insights**
- **Pre-Earnings Volatility**: Surface shape before announcements
- **Risk Premiums**: Option market pricing of earnings uncertainty
- **Market Expectations**: Implied volatility patterns

### **3. Multi-Factor Models**
- **REVR = f(IEVR, Dispersion, TERM_RATIO, SKEW, KURT, IV_RATIO, SMIRK)**
- **Enhanced explanatory power** with option market information
- **Cross-sectional analysis** across different volatility dimensions

## 🛠️ **Configuration Options**

### **Timing Parameters**
```python
# Default: 15 trading days before earnings
option_features = analyzer.get_option_surface_features(
    ticker='AAPL',
    earnings_date='2023-02-02',
    n_lag=15  # Configurable
)
```

### **Quality Filters**
- **Minimum data requirements**: Sufficient option contracts
- **Delta constraints**: Specific strike selection (25, 50)
- **Maturity focus**: 10-day and 30-day options

## 📊 **Expected Output**

### **Console Output**
```
6. Calculating option surface features...
  Calculating option surface features for AAPL...
    Found secid: 12345
    ✓ Surface features calculated successfully
      TERM_RATIO: 1.2345, SKEW: 0.1234, KURT: 0.5678
      IV_RATIO: 1.1111, SMIRK: 0.2222
```

### **CSV Output**
```csv
earnings_date,revr,ievr,TERM_RATIO,SKEW,KURT,IV_RATIO,SMIRK,surface_date
2023-02-02,1.104,1.007,1.2345,0.1234,0.5678,1.1111,0.2222,2023-01-20
2023-05-04,1.167,1.205,1.3456,0.2345,0.6789,1.2222,0.3333,2023-04-20
```

## 🎉 **Benefits of Integration**

### **1. Comprehensive Volatility Analysis**
- **Realized volatility** (REVR)
- **Implied volatility** (IEVR)
- **Analyst uncertainty** (dispersion)
- **Market risk factors** (Fama-French)
- **Option surface dynamics** (new features)

### **2. Enhanced Predictive Power**
- **Multiple volatility dimensions**
- **Market microstructure insights**
- **Risk premium information**
- **Cross-asset relationships**

### **3. Research Applications**
- **Earnings volatility prediction**
- **Option market efficiency**
- **Risk premium modeling**
- **Cross-sectional asset pricing**

## 🔍 **Testing the Integration**

### **Run Full Test**
```bash
python3 test_integration.py
```

### **Expected Results**
- ✅ **All features populated** (no empty columns)
- ✅ **Option surface features calculated** for each earnings event
- ✅ **Comprehensive coverage reporting** for all feature types
- ✅ **Correlation analysis** between REVR and option features

## 🚀 **Next Steps**

1. **Test the integration** with your data
2. **Verify feature quality** and coverage
3. **Explore correlations** between REVR and option features
4. **Build enhanced models** using all available features
5. **Analyze cross-sectional patterns** across different volatility dimensions

This integration provides a comprehensive foundation for sophisticated earnings volatility analysis with multiple data sources and feature types!
