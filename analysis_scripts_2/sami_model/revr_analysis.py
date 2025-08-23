#!/usr/bin/env python3
"""
REVR Analysis Module
Calculate Realized Earnings Volatility Ratio (REVR)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import wrds
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class REVRAnalysis:
    """
    Calculate Realized Earnings Volatility Ratio (REVR)
    REVR = vol_t+4 / vol_t-3 (volatility after earnings / volatility before earnings)
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.stock_data = None
        self.earnings_date = None
        
    # Connection management removed - using simple error handling
        
    def get_stock_data(self, ticker, start_date, end_date):
        """
        Fetch stock price data from WRDS with minimal error handling.
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
    
    def calculate_revr(self, ticker, earnings_date, days_before=30, days_after=30):
        """
        Calculate Realized Earnings Volatility Ratio (REVR) using new methodology.
        
        Methodology:
        - Start 1 month before earnings
        - Normative realized vol: average from start until T-1 (day before earnings)
        - Post-earnings realized vol: average from T+1 until 1 month after earnings
        - Use 7-day half-life for both periods
        - REVR = post-earnings avg / pre-earnings avg
        
        Args:
            ticker: Stock ticker
            earnings_date: Earnings announcement date
            days_before: Days before earnings to analyze (default 30)
            days_after: Days after earnings to analyze (default 30)
        
        Returns:
            Dictionary with REVR and analysis details
        """
        print(f"\n{'='*80}")
        print(f"REVR ANALYSIS: {ticker} - {earnings_date}")
        print(f"{'='*80}")
        
        # Convert earnings date
        earnings_date = pd.to_datetime(earnings_date)
        
        # Define date range (1 month before to 1 month after)
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
        
        # Calculate rolling volatility with 7-day half-life
        stock_data['rolling_vol'] = self.calculate_rolling_volatility(
            stock_data['returns'], window=30, half_life=7
        )
        
        # Ensure date column is datetime
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        
        # Define key dates
        t_minus_1 = earnings_date - timedelta(days=1)  # Day before earnings
        t_plus_1 = earnings_date + timedelta(days=1)   # Day after earnings
        
        # Split data into pre-earnings and post-earnings periods
        pre_earnings_data = stock_data[stock_data['date'] <= t_minus_1]
        post_earnings_data = stock_data[stock_data['date'] >= t_plus_1]
        
        # Calculate average volatilities for each period
        if len(pre_earnings_data) > 0:
            # Use exponential weighted average for pre-earnings period
            pre_vol_avg = pre_earnings_data['rolling_vol'].ewm(halflife=7).mean().iloc[-1]
        else:
            pre_vol_avg = np.nan
            
        if len(post_earnings_data) > 0:
            # Use exponential weighted average for post-earnings period
            post_vol_avg = post_earnings_data['rolling_vol'].ewm(halflife=7).mean().iloc[-1]
        else:
            post_vol_avg = np.nan
        
        # Calculate REVR
        if not np.isnan(pre_vol_avg) and not np.isnan(post_vol_avg) and pre_vol_avg > 0:
            revr = post_vol_avg / pre_vol_avg
        else:
            revr = np.nan
        
        # Analysis results
        results = {
            'ticker': ticker,
            'earnings_date': earnings_date,
            't_minus_1': t_minus_1,
            't_plus_1': t_plus_1,
            'pre_earnings_avg_vol': pre_vol_avg,
            'post_earnings_avg_vol': post_vol_avg,
            'revr': revr,
            'stock_data': stock_data,
            'pre_earnings_data': pre_earnings_data,
            'post_earnings_data': post_earnings_data
        }
        
        # Print analysis
        self._print_revr_analysis(results)
        
        return results
    
    def calculate_revr_st_mt_ratio(self, ticker, earnings_date, days_before=120, days_after=60):
        """
        Calculate Realized Earnings Volatility Ratio (REVR) using ST/MT ratio methodology.
        
        Updated Methodology:
        - Calculate excess returns relative to SPX index
        - Compute expanding EWM volatility with two half-lives:
          * ST (Short-term): 5-day half-life - captures recent volatility dynamics
          * MT (Medium-term): 21-day half-life - provides stable baseline
        - REVR = ST_volatility / MT_volatility at earnings date
        - This ratio captures how much recent volatility exceeds the stable baseline
        
        Args:
            ticker: Stock ticker
            earnings_date: Earnings announcement date
            days_before: Days before earnings to analyze (default 120)
            days_after: Days after earnings to analyze (default 60)
        
        Returns:
            Dictionary with REVR and analysis details
        """
        print(f"\n{'='*80}")
        print(f"REVR ST/MT ANALYSIS: {ticker} - {earnings_date}")
        print(f"{'='*80}")
        
        # Convert earnings date
        earnings_date = pd.to_datetime(earnings_date)
        
        # Define date range (extended for better MT calculation)
        start_date = earnings_date - timedelta(days=days_before)
        end_date = earnings_date + timedelta(days=days_after)
        
        # Get stock data and SPX data
        stock_data = self.get_stock_data(ticker, start_date, end_date)
        spx_data = self.get_stock_data('SPX', start_date, end_date)
        
        if stock_data is None or spx_data is None:
            print(f"Failed to get data for {ticker} or SPX")
            return None
        
        # Ensure date columns are datetime
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        spx_data['date'] = pd.to_datetime(spx_data['date'])
        
        # Calculate returns
        stock_data['returns'] = stock_data['close'].pct_change()
        spx_data['returns'] = spx_data['close'].pct_change()
        
        # Merge stock and SPX data
        merged_data = stock_data[['date', 'returns']].merge(
            spx_data[['date', 'returns']].rename(columns={'returns': 'spx_returns'}),
            on='date', how='inner'
        )
        
        # Calculate excess returns
        merged_data['excess_returns'] = merged_data['returns'] - merged_data['spx_returns']
        
        # Remove NaN values
        merged_data = merged_data.dropna(subset=['excess_returns'])
        
        # Check data sufficiency with more reasonable requirements
        min_required = 30  # Reduced from 60
        if len(merged_data) < min_required:
            print(f"Insufficient data for {ticker}: {len(merged_data)} observations (need {min_required})")
            return None
        
        # Sort by date
        merged_data = merged_data.sort_values('date').reset_index(drop=True)
        
        # Calculate expanding EWM volatility with adjusted half-lives
        # ST (Short-term): 5-day half-life (reduced from 10)
        merged_data['vol_st'] = merged_data['excess_returns'].ewm(
            halflife=5, min_periods=20  # Reduced from 60
        ).std() * np.sqrt(252)  # Annualize
        
        # MT (Medium-term): 21-day half-life (reduced from 126)
        merged_data['vol_mt'] = merged_data['excess_returns'].ewm(
            halflife=21, min_periods=20  # Reduced from 60
        ).std() * np.sqrt(252)  # Annualize
        
        # Find the REVR on the 3rd trading day after earnings
        post_earnings_data = merged_data[merged_data['date'] > earnings_date].copy()
        post_earnings_data = post_earnings_data.sort_values('date')
        
        # Get the 3rd trading day after earnings (index 2 since we start from 0)
        if len(post_earnings_data) >= 3:
            third_day_row = post_earnings_data.iloc[2]  # 3rd trading day (index 2)
            
            # Check if we have valid volatility values
            vol_st = third_day_row['vol_st']
            vol_mt = third_day_row['vol_mt']
            
            if pd.notna(vol_st) and pd.notna(vol_mt) and vol_mt > 0:
                revr = vol_st / vol_mt
                analysis_date = third_day_row['date']
                print(f"Using 3rd trading day after earnings: {analysis_date.strftime('%Y-%m-%d')}")
            else:
                print(f"Invalid volatility values on 3rd trading day for {ticker}")
                vol_st = np.nan
                vol_mt = np.nan
                revr = np.nan
                analysis_date = earnings_date
        else:
            print(f"Insufficient post-earnings data for {ticker}: {len(post_earnings_data)} days (need at least 3)")
            vol_st = np.nan
            vol_mt = np.nan
            revr = np.nan
            analysis_date = earnings_date
        
        # Calculate normative realized volatility (at 30 days before earnings)
        # IMPORTANT: Use only data available at that point in time to avoid look-ahead bias
        normative_date = earnings_date - timedelta(days=30)
        normative_data = merged_data[merged_data['date'] <= normative_date].copy()
        
        if len(normative_data) > 0:
            # Calculate MT volatility using ONLY data up to normative date
            # This avoids look-ahead bias by not using future data
            normative_excess_returns = normative_data['excess_returns']
            normative_realized_vol = normative_excess_returns.ewm(
                halflife=21, min_periods=20
            ).std().iloc[-1] * np.sqrt(252)  # Annualize
            
            print(f"Normative realized vol calculation:")
            print(f"  Normative date: {normative_date.strftime('%Y-%m-%d')}")
            print(f"  Data points used: {len(normative_data)}")
            print(f"  Normative realized vol: {normative_realized_vol:.4f}")
        else:
            normative_realized_vol = np.nan
            print(f"Warning: No data available for normative date {normative_date}")
        
        # Analysis results
        results = {
            'ticker': ticker,
            'earnings_date': earnings_date,
            'analysis_date': analysis_date,
            'vol_st': vol_st,
            'vol_mt': vol_mt,
            'revr': revr,
            'normative_realized_vol': normative_realized_vol,  # Added normative realized vol
            'merged_data': merged_data,
            'methodology': 'ST/MT Ratio (Expanding EWM) - 3rd Trading Day Post-Earnings',
            'data_points': len(merged_data)
        }
        
        # Print analysis
        self._print_st_mt_analysis(results)
        
        return results
    
    def _print_revr_analysis(self, results):
        """
        Print detailed REVR analysis.
        """
        print(f"\nREVR Analysis Results:")
        print(f"  Ticker: {results['ticker']}")
        print(f"  Earnings Date: {results['earnings_date'].strftime('%Y-%m-%d')}")
        print(f"  T-1 Date: {results['t_minus_1'].strftime('%Y-%m-%d')}")
        print(f"  T+1 Date: {results['t_plus_1'].strftime('%Y-%m-%d')}")
        print(f"  Pre-earnings avg volatility: {results['pre_earnings_avg_vol']:.3f} ({results['pre_earnings_avg_vol']*100:.1f}%)")
        print(f"  Post-earnings avg volatility: {results['post_earnings_avg_vol']:.3f} ({results['post_earnings_avg_vol']*100:.1f}%)")
        print(f"  REVR: {results['revr']:.3f}")
        
        # Validate REVR
        if not np.isnan(results['revr']):
            if 0.9 <= results['revr'] <= 2.0:
                print(f"  ✓ REVR is in expected range (0.9-2.0)")
            else:
                print(f"  ⚠ REVR is outside expected range (0.9-2.0)")
        else:
            print(f"  ⚠ REVR calculation failed (NaN)")
        
        # Validate volatility levels
        pre_vol_pct = results['pre_earnings_avg_vol'] * 100
        post_vol_pct = results['post_earnings_avg_vol'] * 100
        
        if not np.isnan(pre_vol_pct):
            if 20 <= pre_vol_pct <= 30:
                print(f"  ✓ Pre-earnings volatility is in expected range (20-30%)")
            else:
                print(f"  ⚠ Pre-earnings volatility ({pre_vol_pct:.1f}%) outside expected range")
        
        if not np.isnan(post_vol_pct):
            if 20 <= post_vol_pct <= 60:  # Allow higher post-earnings vol
                print(f"  ✓ Post-earnings volatility is reasonable")
            else:
                print(f"  ⚠ Post-earnings volatility ({post_vol_pct:.1f}%) seems unusual")
    
    def _print_st_mt_analysis(self, results):
        """
        Print detailed ST/MT REVR analysis.
        """
        print(f"\nST/MT REVR Analysis Results:")
        print(f"  Ticker: {results['ticker']}")
        print(f"  Earnings Date: {results['earnings_date'].strftime('%Y-%m-%d')}")
        print(f"  Analysis Period: First 3 trading days after earnings")
        print(f"  Data Points: {results.get('data_points', 'N/A')}")
        print(f"  ST Volatility (5-day half-life): {results['vol_st']:.4f}")
        print(f"  MT Volatility (21-day half-life): {results['vol_mt']:.4f}")
        print(f"  REVR (ST/MT): {results['revr']:.3f}")
        
        # Show 3rd trading day information
        print(f"  3rd Trading Day Analysis:")
        print(f"    Analysis Date: {results['analysis_date'].strftime('%Y-%m-%d')}")
        print(f"    Trading Days from Earnings: T+3")
        print(f"    ST Volatility: {results['vol_st']:.4f}")
        print(f"    MT Volatility: {results['vol_mt']:.4f}")
        print(f"    REVR: {results['revr']:.3f}")
        
        # Validate REVR
        if not np.isnan(results['revr']):
            if 0.5 <= results['revr'] <= 3.0:
                print(f"  ✓ REVR is in expected range (0.5-3.0)")
            else:
                print(f"  ⚠ REVR is outside expected range (0.5-3.0)")
        else:
            print(f"  ⚠ REVR calculation failed (NaN)")
        
        # Interpretation
        if not np.isnan(results['revr']):
            if results['revr'] > 1.0:
                print(f"  📈 Post-earnings volatility ({results['vol_st']:.4f}) is {results['revr']:.1f}x higher than baseline ({results['vol_mt']:.4f})")
            else:
                print(f"  📉 Post-earnings volatility ({results['vol_st']:.4f}) is {1/results['revr']:.1f}x lower than baseline ({results['vol_mt']:.4f})")
    
    # REVR volatility timeline plotting removed
    
    # REVR plotting functionality removed

def main():
    """
    Main function to run REVR analysis for AAPL October 2022.
    Tests both old and new ST/MT methodology.
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
        
        # Test both methodologies for AAPL October 2022
        print("\n" + "="*80)
        print("TESTING BOTH REVR METHODOLOGIES")
        print("="*80)
        
        # 1. Original methodology
        print("\n1. ORIGINAL METHODOLOGY (Pre/Post Earnings Average)")
        results_original = analyzer.calculate_revr(
            ticker='AAPL',
            earnings_date='2022-10-27',
            days_before=30,
            days_after=30
        )
        
        # 2. New ST/MT methodology
        print("\n2. NEW ST/MT METHODOLOGY (Expanding EWM Ratio)")
        results_st_mt = analyzer.calculate_revr_st_mt_ratio(
            ticker='AAPL',
            earnings_date='2022-10-27',
            days_before=60,
            days_after=60
        )
        
        # Compare results
        if results_original and results_st_mt:
            print(f"\n" + "="*80)
            print("COMPARISON OF METHODOLOGIES")
            print("="*80)
            print(f"Original REVR: {results_original['revr']:.3f}")
            print(f"ST/MT REVR:    {results_st_mt['revr']:.3f}")
            
            if not np.isnan(results_original['revr']) and not np.isnan(results_st_mt['revr']):
                diff = abs(results_original['revr'] - results_st_mt['revr'])
                print(f"Difference:    {diff:.3f}")
                
                if diff < 0.1:
                    print("✓ Results are similar (< 0.1 difference)")
                else:
                    print("⚠ Results differ significantly")
        
        # Close connection
        db.close()
        print("\n✓ Database connection closed")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 