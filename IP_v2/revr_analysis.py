#!/usr/bin/env python3
"""
Realized Earnings Volatility Ratio (REVR) Analysis
Step 1: Calculate REVR for AAPL October 2022 earnings event
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import wrds

class REVRAnalysis:
    """
    Calculate Realized Earnings Volatility Ratio (REVR)
    REVR = vol_t+4 / vol_t-3 (volatility after earnings / volatility before earnings)
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.stock_data = None
        self.earnings_date = None
        
    def get_stock_data(self, ticker, start_date, end_date):
        """
        Fetch stock price data from WRDS.
        """
        print(f"Fetching stock data for {ticker} from {start_date} to {end_date}")
        
        try:
            # Get security info first
            sec_query = f"""
            SELECT secid, ticker
            FROM optionm.secnmd
            WHERE ticker = '{ticker}'
            ORDER BY effect_date DESC
            LIMIT 1
            """
            sec_info = self.db.raw_sql(sec_query)
            
            if sec_info.empty:
                print(f"Could not find secid for {ticker}")
                return None
                
            secid = sec_info.iloc[0]['secid']
            print(f"Found secid {secid} for {ticker}")
            
            # Get stock prices
            price_query = f"""
            SELECT date, close, volume, return
            FROM optionm.secprd
            WHERE secid = {secid}
              AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
            """
            
            stock_data = self.db.raw_sql(price_query)
            print(f"Retrieved {len(stock_data)} stock price records")
            print(f"Date range: {stock_data['date'].min()} to {stock_data['date'].max()}")
            
            return stock_data
            
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None
    
    def calculate_rolling_volatility(self, returns, window=30, half_life=7):
        """
        Calculate rolling volatility with exponential weighting.
        
        Args:
            returns: Series of daily returns
            window: Rolling window size (days)
            half_life: Half-life for exponential weighting (days)
        
        Returns:
            Series of rolling volatilities (annualized)
        """
        # Calculate exponential weights
        alpha = np.log(2) / half_life
        weights = np.exp(-alpha * np.arange(window))
        weights = weights / weights.sum()
        
        # Calculate exponentially weighted rolling volatility
        # Use pandas ewm for efficiency
        ewm_vol = returns.ewm(halflife=half_life, min_periods=window//2).std()
        
        # Annualize (multiply by sqrt(252))
        annualized_vol = ewm_vol * np.sqrt(252)
        
        return annualized_vol
    
    def calculate_revr(self, ticker, earnings_date, days_before=60, days_after=30):
        """
        Calculate Realized Earnings Volatility Ratio (REVR).
        
        Args:
            ticker: Stock ticker
            earnings_date: Earnings announcement date
            days_before: Days before earnings to analyze
            days_after: Days after earnings to analyze
        
        Returns:
            Dictionary with REVR and analysis details
        """
        print(f"\n{'='*80}")
        print(f"REVR ANALYSIS: {ticker} - {earnings_date}")
        print(f"{'='*80}")
        
        # Convert earnings date
        earnings_date = pd.to_datetime(earnings_date)
        
        # Define date range
        start_date = earnings_date - timedelta(days=days_before)
        end_date = earnings_date + timedelta(days=days_after)
        
        # Get stock data
        stock_data = self.get_stock_data(ticker, start_date, end_date)
        if stock_data is None:
            return None
        
        # Store data
        self.stock_data = stock_data
        self.earnings_date = earnings_date
        
        # Calculate daily returns
        stock_data['returns'] = stock_data['close'].pct_change()
        
        # Calculate rolling volatility
        stock_data['rolling_vol'] = self.calculate_rolling_volatility(
            stock_data['returns'], window=30, half_life=7
        )
        
        # Find key dates
        t_minus_3 = earnings_date - timedelta(days=3)  # Friday before earnings
        t_plus_4 = earnings_date + timedelta(days=4)   # Friday after earnings
        
        # Get volatility values - find closest available dates
        def find_closest_date(data, target_date):
            """Find the closest available date in the data."""
            data_dates = pd.to_datetime(data['date'])
            target_date = pd.to_datetime(target_date)
            
            # Find the closest date
            date_diff = abs(data_dates - target_date)
            closest_idx = date_diff.idxmin()
            closest_date = data_dates[closest_idx]
            
            print(f"  Target: {target_date.strftime('%Y-%m-%d')}, Found: {closest_date.strftime('%Y-%m-%d')}")
            return closest_idx
        
        # Find closest dates and get volatility values
        t_minus_3_idx = find_closest_date(stock_data, t_minus_3)
        t_plus_4_idx = find_closest_date(stock_data, t_plus_4)
        
        vol_t_minus_3 = stock_data.iloc[t_minus_3_idx]['rolling_vol']
        vol_t_plus_4 = stock_data.iloc[t_plus_4_idx]['rolling_vol']
        
        # Update the actual dates used
        t_minus_3_actual = pd.to_datetime(stock_data.iloc[t_minus_3_idx]['date'])
        t_plus_4_actual = pd.to_datetime(stock_data.iloc[t_plus_4_idx]['date'])
        
        # Calculate REVR
        revr = vol_t_plus_4 / vol_t_minus_3
        
        # Analysis results
        results = {
            'ticker': ticker,
            'earnings_date': earnings_date,
            't_minus_3': t_minus_3_actual,
            't_plus_4': t_plus_4_actual,
            'vol_t_minus_3': vol_t_minus_3,
            'vol_t_plus_4': vol_t_plus_4,
            'revr': revr,
            'stock_data': stock_data
        }
        
        # Print analysis
        self._print_revr_analysis(results)
        
        return results
    
    def _print_revr_analysis(self, results):
        """
        Print detailed REVR analysis.
        """
        print(f"\nREVR Analysis Results:")
        print(f"  Ticker: {results['ticker']}")
        print(f"  Earnings Date: {results['earnings_date'].strftime('%Y-%m-%d')}")
        print(f"  T-3 Date: {results['t_minus_3'].strftime('%Y-%m-%d')}")
        print(f"  T+4 Date: {results['t_plus_4'].strftime('%Y-%m-%d')}")
        print(f"  Volatility T-3: {results['vol_t_minus_3']:.3f} ({results['vol_t_minus_3']*100:.1f}%)")
        print(f"  Volatility T+4: {results['vol_t_plus_4']:.3f} ({results['vol_t_plus_4']*100:.1f}%)")
        print(f"  REVR: {results['revr']:.3f}")
        
        # Validate REVR
        if 0.9 <= results['revr'] <= 2.0:
            print(f"  ✓ REVR is in expected range (0.9-2.0)")
        else:
            print(f"  ⚠ REVR is outside expected range (0.9-2.0)")
        
        # Validate volatility levels
        vol_t_minus_3_pct = results['vol_t_minus_3'] * 100
        vol_t_plus_4_pct = results['vol_t_plus_4'] * 100
        
        if 20 <= vol_t_minus_3_pct <= 30:
            print(f"  ✓ Pre-earnings volatility is in expected range (20-30%)")
        else:
            print(f"  ⚠ Pre-earnings volatility ({vol_t_minus_3_pct:.1f}%) outside expected range")
        
        if 20 <= vol_t_plus_4_pct <= 60:  # Allow higher post-earnings vol
            print(f"  ✓ Post-earnings volatility is reasonable")
        else:
            print(f"  ⚠ Post-earnings volatility ({vol_t_plus_4_pct:.1f}%) seems unusual")
    
    def plot_volatility_timeline(self, results):
        """
        Plot volatility timeline around earnings event.
        """
        if results is None or 'stock_data' not in results:
            print("No data available for plotting")
            return
        
        stock_data = results['stock_data'].copy()
        earnings_date = pd.to_datetime(results['earnings_date'])
        
        # Ensure all dates are datetime objects
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Stock price
        ax1.plot(stock_data['date'], stock_data['close'], 'b-', linewidth=1.5)
        ax1.axvline(x=earnings_date, color='red', linestyle='--', alpha=0.7, label='Earnings')
        ax1.axvline(x=results['t_minus_3'], color='orange', linestyle=':', alpha=0.7, label='T-3')
        ax1.axvline(x=results['t_plus_4'], color='green', linestyle=':', alpha=0.7, label='T+4')
        ax1.set_title(f'{results["ticker"]} Stock Price Around Earnings')
        ax1.set_ylabel('Stock Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rolling volatility
        ax2.plot(stock_data['date'], stock_data['rolling_vol'] * 100, 'r-', linewidth=1.5)
        ax2.axvline(x=earnings_date, color='red', linestyle='--', alpha=0.7, label='Earnings')
        ax2.axvline(x=results['t_minus_3'], color='orange', linestyle=':', alpha=0.7, label='T-3')
        ax2.axvline(x=results['t_plus_4'], color='green', linestyle=':', alpha=0.7, label='T+4')
        ax2.axhline(y=results['vol_t_minus_3'] * 100, color='orange', alpha=0.5, linestyle='-')
        ax2.axhline(y=results['vol_t_plus_4'] * 100, color='green', alpha=0.5, linestyle='-')
        ax2.set_title(f'{results["ticker"]} Rolling Volatility (30-day EWM, 7-day half-life)')
        ax2.set_ylabel('Volatility (%)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nVolatility Timeline Analysis:")
        print(f"  Pre-earnings baseline: {results['vol_t_minus_3']*100:.1f}%")
        print(f"  Post-earnings level: {results['vol_t_plus_4']*100:.1f}%")
        print(f"  Volatility ratio: {results['revr']:.3f}x")

def main():
    """
    Main function to run REVR analysis for AAPL October 2022.
    """
    print("REALIZED EARNINGS VOLATILITY RATIO (REVR) ANALYSIS")
    print("="*80)
    
    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="sami_sellami",
                     password="xampok-9Hezfy-cahveq")
        print("✓ Connected to WRDS")
        
        # Initialize analysis
        analyzer = REVRAnalysis(db)
        
        # Calculate REVR for AAPL October 2022
        results = analyzer.calculate_revr(
            ticker='AAPL',
            earnings_date='2022-10-27',  # AAPL Q4 2022 earnings
            days_before=60,
            days_after=30
        )
        
        if results is not None:
            # Plot the volatility timeline
            analyzer.plot_volatility_timeline(results)
            
            print(f"\n✓ REVR analysis completed successfully!")
            print(f"  REVR = {results['revr']:.3f}")
            
        else:
            print("✗ REVR analysis failed")
        
        # Close connection
        db.close()
        print("\n✓ Database connection closed")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 