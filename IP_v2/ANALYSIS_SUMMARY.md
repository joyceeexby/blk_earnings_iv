# Earnings Implied Volatility Analysis - Summary (Updated July 2025)

## Project Overview
This project analyzes the relationship between **Implied Earnings Volatility Ratio (IEVR)** and **Realized Earnings Volatility Ratio (REVR)** using options data from WRDS. The analysis covers 50 large-cap stocks from 2000-2023, using both pooled and individual stock regression approaches.

---

## Methodology

### 1. Realized Earnings Volatility Ratio (REVR)
- **Definition:**
  - REVR = (Post-earnings volatility) / (Pre-earnings volatility)
  - Specifically, REVR = σ<sub>t+4</sub> / σ<sub>t-3</sub>, where:
    - σ<sub>t-3</sub>: Realized volatility on the Friday before earnings (T-3)
    - σ<sub>t+4</sub>: Realized volatility on the Friday after earnings (T+4)
- **Calculation Steps:**
  1. For each earnings event, identify the announcement date.
  2. Calculate daily returns for the underlying stock.
  3. Compute rolling volatility (30-day window, 7-day half-life) for each day.
  4. Extract volatility values for T-3 and T+4 (closest available trading days).
  5. Compute REVR as the ratio of these two volatilities.
- **Purpose:** Measures how much realized volatility changes after earnings relative to before.

### 2. Implied Earnings Volatility Ratio (IEVR)
- **Definition:**
  - IEVR = (Implied volatility at the "kink") / (Normative implied volatility at same maturity)
  - The "kink" is the point in the IV curve where implied volatility jumps due to the upcoming earnings event.
- **How IV is obtained:**
  - **Implied volatility values are taken directly from WRDS OptionMetrics** (`impl_volatility` field for each option contract).
  - **We do NOT compute IV from option prices ourselves**; we use the pre-calculated values provided by WRDS.
- **Options Considered:**
  - **Type:** Only **call options** are used (to avoid put-call parity complications and liquidity issues).
  - **Strikes:** At-the-money (ATM) or nearest-to-ATM strikes are selected for each event.
  - **Maturities:**
    - Focus on maturities that bracket the earnings date:
      - **Short-term:** Option expiring just after the earnings date (captures event volatility)
      - **Long-term:** Option expiring well after the event (serves as baseline)
    - Typical maturities: 20-40 days to expiry, with the "kink" usually 1-2 weeks after earnings.
  - **Filters:** Only options with sufficient liquidity (volume/open interest) are included.
- **Calculation Steps:**
  1. For each event, select the relevant call options (ATM, correct maturities).
  2. **Construct the implied volatility term structure** (IV vs. time-to-expiry) using the `impl_volatility` values from WRDS.
  3. **Estimate the "normative" (non-event) IV curve** using kernel regression or curve fitting (see below).
  4. Identify the "kink"—the IV jump at the maturity just after earnings.
  5. Compute IEVR as the ratio of the IV at the kink to the normative IV at the same maturity.
- **Purpose:** Quantifies the market's expectation of volatility due to the earnings event, relative to normal conditions.

#### More on Kernel Regression / Curve Fitting
- **Why do we need it?**
  - The observed IV term structure is "distorted" by the earnings event (the "kink"). To know what IV *would* be without the event, we need a smooth estimate of the "normal" IV curve.
- **What is kernel regression?**
  - **Kernel regression** is a non-parametric smoothing technique. It estimates the value of a function (here, IV as a function of time-to-expiry) at a given point by taking a weighted average of nearby observed values, with weights decreasing smoothly as you move away from the target point.
  - In this context, for each maturity, we estimate the "normative" IV by averaging the IVs of options with similar (but not identical) maturities, giving more weight to those closer in time.
  - This produces a smooth, continuous curve that represents what the IV term structure would look like *without* the earnings event.
- **Why not just use a polynomial fit?**
  - Kernel regression is more flexible and does not assume a specific functional form, making it well-suited for capturing the typical shape of the IV curve.
- **Result:**
  - The "normative" IV curve is used as the baseline, and the observed "kink" is compared to this baseline to compute IEVR.

### 3. Data Construction
- **Source:** WRDS OptionMetrics (option prices, implied volatilities, earnings dates)
- **Stocks:** 50 large-cap S&P 500 constituents
- **Period:** 2000-2023 (24 years)
- **Event Selection:** All quarterly earnings events with sufficient option data
- **Data Cleaning:** Remove NaNs, infinite values, and outliers (see Results)

### 4. Regression Analysis
- **Pooled regressions:** All stocks combined, with stock and time fixed effects
- **Individual regressions:** Separate regression for each stock

---

## Results

### 1. Data Cleaning
- Removed 2 rows with NaN values
- Removed 1,074 rows with infinite values
- Removed outliers from REVR and IEVR
- **Final sample for regressions:** 2,880 events from 50 stocks
- **Average IEVR:** 1.48
- **Average REVR:** 1.29

### 2. Pooled Regression Results
- **Model 1 (Basic):** No significant relationship (R² = 0.000, p = 0.365)
- **Model 2 (Stock Fixed Effects):**
  - **IEVR coefficient:** 0.182 (p < 0.001)
  - **R² = 0.102**
- **Model 4 (Stock + Time Fixed Effects):**
  - **IEVR coefficient:** 0.117 (p = 0.002)
  - **R² = 0.119**
- **Other models**: See `pooled_regression_summary.csv` for full details

### 3. Individual Stock Regression Results
- **50 individual regressions** (one per stock)
- **Significant regressions (p < 0.05):** 10 out of 50 (20.0%)
- **Mean R-squared:** 0.032
- **Median R-squared:** 0.008
- **Mean IEVR coefficient:** 0.215
- **Mean IEVR p-value:** 0.430
- **Top 5 performers by R-squared:**
    - AAPL: R²=0.184, β=0.987 (p<0.001)
    - ORCL: R²=0.161, β=0.685 (p=0.002)
    - AMGN: R²=0.117, β=0.663 (p=0.025)
    - MRK: R²=0.114, β=0.605 (p=0.020)
    - JNJ: R²=0.113, β=0.448 (p=0.010)
- **See:** `individual_stock_regression_results.csv` and `significant_individual_regressions.csv`

---

## Key Findings
- **IEVR has significant predictive power** for REVR in pooled regressions with stock and time controls.
- **Individual stock regressions** show significant results for 20% of stocks, but most have low R-squared (limited explanatory power).
- **Top stocks** (AAPL, ORCL, AMGN, MRK, JNJ) show the strongest IEVR-REVR relationship.
- **Data quality and outlier handling** are critical for robust results.

---

## Files Generated
- `expanded_earnings_analysis_results.csv` — Cleaned event-level data
- `pooled_regression_summary.csv` — Pooled regression models
- `individual_stock_regression_results.csv` — All individual stock regressions
- `significant_individual_regressions.csv` — Only significant individual regressions
- `top_performers_individual.csv` — Top stocks by R-squared

---

## Next Steps
- Explore additional controls and nonlinearities
- Investigate why some stocks show stronger relationships
- Consider alternative volatility measures

---

*Last updated: 14 July 2025* 