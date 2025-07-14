# Earnings Implied Volatility Analysis - Complete Summary

## Project Overview
This project analyzes the relationship between **Implied Earnings Volatility Ratio (IEVR)** and **Realized Earnings Volatility Ratio (REVR)** using options data from WRDS. The analysis covers 100+ large-cap stocks from 2020-2023.

## Methodology

### 1. Realized Earnings Volatility Ratio (REVR)
- **Definition**: REVR = vol_t+4 / vol_t-3
- **Calculation**: 
  - Rolling volatility with 30-day window and 7-day half-life
  - T-3: Friday before earnings announcement
  - T+4: Friday after earnings announcement

### 2. Implied Earnings Volatility Ratio (IEVR)
- **Definition**: IEVR = IV at kink / Normative IV at maturity
- **Calculation**:
  - Position 30 days before earnings event
  - Use kernel regression to estimate normative IV curve
  - Find volatility kink between 20-40 days to earnings
  - Calculate ratio of kink IV to normative IV

### 3. Data Pipeline
- **Source**: WRDS OptionMetrics (opprcd tables)
- **Stocks**: 100+ large-cap S&P 500 constituents
- **Period**: 2020-2023 (4 years)
- **Direct IV Fetching**: Simplified pipeline using `impl_volatility` column directly

## Key Results (Expanded Analysis, July 2025)

### Overall Sample Statistics
- **Total Events Analyzed**: 1,935 earnings events
- **Stocks with Data**: 173 stocks
- **Date Range**: 2020-2023
- **Success Rate**: 65% of attempted stock regressions

### Regression Results (All Stocks)

- **Successful regressions**: 113 out of 173 stocks (65.3%)
- **Mean IEVR coefficient**: 0.28
- **Median IEVR coefficient**: 0.24
- **Std Dev (IEVR coef)**: 0.93
- **Mean R-squared**: 0.11
- **Median R-squared**: 0.05
- **Significant at 5% level**: 8 stocks (7.1%)
- **Positive and significant**: 7 stocks
- **Negative and significant**: 1 stock

*Note: All statistics are from individual stock-specific regressions (REVR = α + β × IEVR for each stock separately), not from pooled models.*

#### Top 10 Stocks by IEVR Coefficient:
*Results from individual stock regressions (REVR = α + β × IEVR for each stock)*

| Ticker | IEVR Coef | T-stat | P-value | R² | N Events |
|--------|-----------|--------|---------|-----|----------|
| ISRG   | 3.38      | 3.68   | 0.003   | 0.51| 15       |
| NFLX   | 2.80      | 0.73   | 0.480   | 0.04| 15       |
| TSLA   | 2.50      | 2.05   | 0.061   | 0.24| 15       |
| CMG    | 2.23      | 1.09   | 0.294   | 0.08| 15       |
| MS     | 2.20      | 1.74   | 0.105   | 0.19| 15       |
| INTC   | 2.13      | 1.44   | 0.174   | 0.14| 15       |
| VNO    | 1.96      | 1.98   | 0.105   | 0.44| 7        |
| EQIX   | 1.93      | 2.42   | 0.094   | 0.66| 5        |
| AVGO   | 1.80      | 1.42   | 0.178   | 0.13| 15       |
| AMGN   | 1.63      | 2.95   | 0.011   | 0.40| 15       |

#### Top 10 Stocks by R-squared:
*Results from individual stock regressions (REVR = α + β × IEVR for each stock)*

| Ticker | R²    | IEVR Coef | T-stat | P-value | N Events |
|--------|-------|-----------|--------|---------|----------|
| XEL    | 0.68  | 0.25      | 2.93   | 0.043   | 6        |
| CMS    | 0.67  | 0.18      | 2.03   | 0.179   | 4        |
| EQIX   | 0.66  | 1.93      | 2.42   | 0.094   | 5        |
| SRE    | 0.60  | -1.14     | -1.24  | 0.433   | 3        |
| TXN    | 0.55  | 1.34      | 4.00   | 0.002   | 15       |
| ISRG   | 0.51  | 3.38      | 3.68   | 0.003   | 15       |
| FTV    | 0.50  | 0.37      | 1.00   | 0.500   | 3        |
| VNO    | 0.44  | 1.96      | 1.98   | 0.105   | 7        |
| AMGN   | 0.40  | 1.63      | 2.95   | 0.011   | 15       |
| REGN   | 0.39  | 1.01      | 2.52   | 0.030   | 12       |

#### Sample Size Analysis:
- **Mean events per stock**: 14.0
- **Median events per stock**: 15.0
- **Stocks with 10+ events**: 103
- **Stocks with 20+ events**: 0

### Temporal Analysis
- **No year-by-year regression results** could be created due to missing or invalid data (NaNs or infs) in the IEVR/REVR columns for each year.

### Future Model Specifications

The following pooled regression models will be tested in future analysis:

- **Model 1**: REVR = α + β × IEVR (already done)
- **Model 2**: REVR = α + β × IEVR + Stock Fixed Effects
- **Model 3**: REVR = α + β × IEVR + Time Fixed Effects
- **Model 4**: REVR = α + β × IEVR + Stock + Time Fixed Effects

## Key Findings

### 1. Limited Predictive Power
- **Low R-squared values**: Most regressions show R² < 0.20
- **Mixed coefficients**: Some positive, some negative IEVR coefficients
- **High p-values**: Most coefficients not statistically significant

### 2. Stock-Specific Variation
- **A few stocks show strong relationships** (e.g., ISRG, TXN, AMGN)
- **Most stocks**: Weak or insignificant relationship between IEVR and REVR

### 3. Time Period Effects
- **No year-by-year results** due to data quality issues (NaNs/infs in annual splits)
- **Market conditions**: Not directly analyzed due to above

## Technical Achievements

- **Automated analysis for 173 stocks**
- **Comprehensive results saved for all stocks**
- **Robust error handling and reporting**
- **Direct IV fetching and pipeline optimization**

## Lessons Learned

- **Data quality is critical**: NaNs/infs can block regression analysis, especially in temporal splits
- **IEVR has limited predictive power** for REVR in most cases
- **Stock-level analysis is more robust than year-by-year splits with current data

## Next Steps

- [ ] Investigate and clean NaN/inf values in IEVR/REVR for year-by-year analysis
- [ ] Explore sector-specific or regime-based regressions
- [ ] Consider alternative volatility measures or additional control variables

## Files Generated
- `expanded_earnings_analysis_results.csv`: Complete dataset
- `all_stocks_regression_results.csv`: Individual stock regressions
- `significant_regressions.csv`: Stocks with significant IEVR coefficients
- `top_performers_by_r2.csv`: Top 20 by R²
- `summary_statistics.csv`: Descriptive stats
- `correlation_matrix.csv`: Correlations
- `summary_analysis.png`: Summary plots

## Conclusion
The expanded analysis confirms that **IEVR has limited predictive power for REVR** across a large sample of stocks. While a handful of stocks show stronger relationships, the overall evidence suggests that implied volatility from options markets does not reliably predict realized volatility around earnings events. Data quality remains a key challenge for temporal analysis. These findings have important implications for options trading strategies and volatility forecasting models. 