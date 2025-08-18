#!/usr/bin/env python3
"""
Batch IEVR Analysis for Multiple Stocks
Efficiently calculate IEVR for 500+ stocks for one earnings date
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import wrds
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class BatchIEVRAnalysis:
    """
    Efficient batch IEVR calculation for multiple stocks
    """
    
    def __init__(self, db_connection=None, max_workers=4):
        self.db = db_connection
        self.max_workers = max_workers
        self.results = []
        self.failed_stocks = []
        
    def get_top_stocks(self, num_stocks=500, min_market_cap=1e9):
        """
        Get top stocks by market cap for analysis
        """
        try:
            print(f"Fetching top {num_stocks} stocks by market cap...")
            
            # Get quarterly market cap data
            market_cap_query = f"""
            WITH quarterly_market_cap AS (
                SELECT 
                    a.permno,
                    a.ticker,
                    a.comnam,
                    b.date,
                    b.prc * b.shrout as market_cap,
                    ROW_NUMBER() OVER (PARTITION BY a.permno ORDER BY b.date DESC) as rn
                FROM crsp.stocknames a
                JOIN crsp.msf b ON a.permno = b.permno
                WHERE b.date >= '2023-01-01'
                  AND b.prc > 0 
                  AND b.shrout > 0
                  AND a.ticker IS NOT NULL
                  AND a.ticker != ''
            )
            SELECT DISTINCT
                permno,
                ticker,
                comnam,
                market_cap
            FROM quarterly_market_cap
            WHERE rn = 1
              AND market_cap >= {min_market_cap}
            ORDER BY market_cap DESC
            LIMIT {num_stocks}
            """
            
            result = self.db.raw_sql(market_cap_query)
            if isinstance(result, pd.DataFrame):
                stocks_df = result
            else:
                stocks_df = pd.DataFrame([dict(row) for row in result])
            
            print(f"✓ Found {len(stocks_df)} stocks")
            return stocks_df
            
        except Exception as e:
            print(f"Error fetching stocks: {e}")
            return pd.DataFrame()
    
    def get_earnings_dates(self, ticker, year=2023):
        """
        Get earnings dates for a specific ticker and year
        """
        try:
            # Get CUSIP from CRSP
            cusip_query = f"""
            SELECT DISTINCT cusip
            FROM crsp.stocknames
            WHERE ticker = '{ticker}'
            LIMIT 1
            """
            
            cusip_result = self.db.raw_sql(cusip_query)
            if isinstance(cusip_result, pd.DataFrame):
                cusip_df = cusip_result
            else:
                cusip_df = pd.DataFrame([dict(row) for row in cusip_result])
            
            if cusip_df.empty:
                return None
            
            cusip = cusip_df.iloc[0]['cusip']
            
            # Get earnings dates from Compustat
            earnings_query = f"""
            SELECT DISTINCT rdq as earnings_date
            FROM comp.fundq
            WHERE cusip = '{cusip}'
              AND rdq IS NOT NULL
              AND EXTRACT(YEAR FROM rdq) = {year}
            ORDER BY rdq
            """
            
            earnings_result = self.db.raw_sql(earnings_query)
            if isinstance(earnings_result, pd.DataFrame):
                earnings_df = earnings_result
            else:
                earnings_df = pd.DataFrame([dict(row) for row in earnings_result])
            
            if earnings_df.empty:
                return None
            
            # Return the first earnings date of the year
            return pd.to_datetime(earnings_df.iloc[0]['earnings_date'])
            
        except Exception as e:
            print(f"Error getting earnings dates for {ticker}: {e}")
            return None
    
    def calculate_single_ievr(self, stock_info, earnings_date, analysis_days_before=30):
        """
        Calculate IEVR for a single stock
        """
        ticker = stock_info['ticker']
        permno = stock_info['permno']
        
        try:
            print(f"  Analyzing {ticker} (permno: {permno})...")
            
            # Get secid for the ticker
            secid_query = f"""
            SELECT DISTINCT secid
            FROM optionm.securd1
            WHERE ticker = '{ticker}'
              AND exchange_d != 0
            LIMIT 1
            """
            
            secid_result = self.db.raw_sql(secid_query)
            if isinstance(secid_result, pd.DataFrame):
                secid_df = secid_result
            else:
                secid_df = pd.DataFrame([dict(row) for row in secid_result])
            
            if secid_df.empty:
                print(f"    No secid found for {ticker}")
                return None
            
            secid = secid_df.iloc[0]['secid']
            
            # Get underlying stock price
            price_query = f"""
            SELECT prc as price
            FROM crsp.dsf
            WHERE permno = {permno}
              AND date = '{earnings_date - timedelta(days=analysis_days_before):%Y-%m-%d}'
            LIMIT 1
            """
            
            price_result = self.db.raw_sql(price_query)
            if isinstance(price_result, pd.DataFrame):
                price_df = price_result
            else:
                price_df = pd.DataFrame([dict(row) for row in price_result])
            
            if price_df.empty:
                print(f"    No price data found for {ticker}")
                return None
            
            underlying_price = price_df.iloc[0]['price']
            
            # Get options data
            analysis_date = earnings_date - timedelta(days=analysis_days_before)
            year = analysis_date.year
            
            # Try to find the right options table
            table_name = f"opprcd{year}"
            
            # Check if table exists
            table_check_query = f"""
            SELECT COUNT(*) as table_exists
            FROM information_schema.tables
            WHERE table_schema = 'optionm'
              AND table_name = '{table_name}'
            """
            
            table_check_result = self.db.raw_sql(table_check_query)
            if isinstance(table_check_result, pd.DataFrame):
                table_exists = table_check_result.iloc[0]['table_exists'] > 0
            else:
                table_exists = any(row['table_exists'] > 0 for row in table_check_result)
            
            if not table_exists:
                print(f"    Options table {table_name} not available for {ticker}")
                return None
            
            # Get IV data
            start_date = (analysis_date - timedelta(days=15)).strftime('%Y-%m-%d')
            end_date = (analysis_date + timedelta(days=15)).strftime('%Y-%m-%d')
            
            iv_query = f"""
            SELECT 
                date, exdate, strike_price, cp_flag, impl_volatility,
                underlying_price
            FROM optionm.{table_name}
            WHERE secid = {secid}
              AND date BETWEEN '{start_date}' AND '{end_date}'
              AND impl_volatility > 0
              AND impl_volatility < 5.0
            ORDER BY date, exdate, strike_price
            """
            
            iv_result = self.db.raw_sql(iv_query)
            if isinstance(iv_result, pd.DataFrame):
                iv_df = iv_result
            else:
                iv_df = pd.DataFrame([dict(row) for row in iv_result])
            
            if iv_df.empty:
                print(f"    No IV data found for {ticker}")
                return None
            
            # Calculate time to expiry and moneyness
            iv_df['date'] = pd.to_datetime(iv_df['date'])
            iv_df['exdate'] = pd.to_datetime(iv_df['exdate'])
            iv_df['tte'] = (iv_df['exdate'] - iv_df['date']).dt.days
            iv_df['moneyness'] = iv_df['strike_price'] / iv_df['underlying_price']
            
            # Filter for reasonable data
            iv_df = iv_df[
                (iv_df['moneyness'].between(0.8, 1.2)) &
                (iv_df['tte'].between(10, 90))
            ]
            
            if len(iv_df) < 10:
                print(f"    Insufficient IV data for {ticker}: {len(iv_df)} points")
                return None
            
            # Separate puts and calls
            puts = iv_df[iv_df['cp_flag'] == 'P'].copy()
            calls = iv_df[iv_df['cp_flag'] == 'C'].copy()
            
            if puts.empty:
                print(f"    No put options data for {ticker}")
                return None
            
            # Calculate IEVR
            days_to_earnings = (earnings_date - analysis_date).days
            
            # Pre-earnings: TTE < days_to_earnings
            pre_earnings = puts[puts['tte'] < days_to_earnings]
            
            # Post-earnings: TTE in (days_to_earnings, days_to_earnings+20]
            post_earnings = puts[
                (puts['tte'] > days_to_earnings) & 
                (puts['tte'] <= days_to_earnings + 20)
            ]
            
            if len(pre_earnings) < 3 or len(post_earnings) < 3:
                print(f"    Insufficient pre/post earnings data for {ticker}")
                return None
            
            # Calculate average IVs
            avg_pre = pre_earnings['impl_volatility'].mean()
            avg_post = post_earnings['impl_volatility'].mean()
            
            if avg_pre == 0 or np.isnan(avg_pre) or np.isnan(avg_post):
                print(f"    Invalid IV values for {ticker}")
                return None
            
            ievr = avg_post / avg_pre
            
            # Calculate skew ratio (90P/110C if available)
            skew_ratio = np.nan
            if not calls.empty:
                put_90 = puts[puts['moneyness'].between(0.88, 0.92)]['impl_volatility'].mean()
                call_110 = calls[calls['moneyness'].between(1.08, 1.12)]['impl_volatility'].mean()
                if not (np.isnan(put_90) or np.isnan(call_110) or call_110 == 0):
                    skew_ratio = put_90 / call_110
            
            result = {
                'ticker': ticker,
                'permno': permno,
                'earnings_date': earnings_date,
                'analysis_date': analysis_date,
                'ievr': ievr,
                'avg_pre_iv': avg_pre,
                'avg_post_iv': avg_post,
                'skew_ratio': skew_ratio,
                'underlying_price': underlying_price,
                'days_to_earnings': days_to_earnings,
                'iv_data_points': len(iv_df),
                'pre_earnings_points': len(pre_earnings),
                'post_earnings_points': len(post_earnings)
            }
            
            print(f"    ✓ {ticker}: IEVR={ievr:.3f}, Pre={avg_pre:.3f}, Post={avg_post:.3f}")
            return result
            
        except Exception as e:
            print(f"    ✗ Error analyzing {ticker}: {e}")
            return None
    
    def run_batch_analysis(self, earnings_date, num_stocks=500, analysis_days_before=30):
        """
        Run batch IEVR analysis for multiple stocks
        """
        print(f"\n{'='*80}")
        print(f"BATCH IEVR ANALYSIS")
        print(f"{'='*80}")
        print(f"Earnings Date: {earnings_date}")
        print(f"Analysis Days Before: {analysis_days_before}")
        print(f"Target Stocks: {num_stocks}")
        print(f"Max Workers: {self.max_workers}")
        
        # Get top stocks
        stocks_df = self.get_top_stocks(num_stocks)
        if stocks_df.empty:
            print("No stocks found for analysis")
            return
        
        print(f"\nStarting IEVR calculation for {len(stocks_df)} stocks...")
        start_time = time.time()
        
        # Process stocks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_stock = {
                executor.submit(
                    self.calculate_single_ievr, 
                    stock_info, 
                    earnings_date, 
                    analysis_days_before
                ): stock_info 
                for _, stock_info in stocks_df.iterrows()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_stock):
                stock_info = future_to_stock[future]
                try:
                    result = future.result()
                    if result is not None:
                        self.results.append(result)
                    else:
                        self.failed_stocks.append(stock_info['ticker'])
                except Exception as e:
                    print(f"Exception for {stock_info['ticker']}: {e}")
                    self.failed_stocks.append(stock_info['ticker'])
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Summary
        print(f"\n{'='*80}")
        print(f"BATCH ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Total stocks processed: {len(stocks_df)}")
        print(f"Successful analyses: {len(self.results)}")
        print(f"Failed analyses: {len(self.failed_stocks)}")
        print(f"Success rate: {len(self.results)/len(stocks_df)*100:.1f}%")
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Average time per stock: {elapsed_time/len(stocks_df):.2f} seconds")
        
        if self.failed_stocks:
            print(f"\nFailed stocks: {', '.join(self.failed_stocks[:20])}")
            if len(self.failed_stocks) > 20:
                print(f"... and {len(self.failed_stocks) - 20} more")
        
        return self.results
    
    def save_results(self, filename=None):
        """
        Save results to CSV
        """
        if not self.results:
            print("No results to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_files/batch_ievr_results_{timestamp}.csv"
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save
        results_df.to_csv(filename, index=False)
        print(f"\n✓ Results saved to {filename}")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"  IEVR - Mean: {results_df['ievr'].mean():.3f}, Std: {results_df['ievr'].std():.3f}")
        print(f"  IEVR - Min: {results_df['ievr'].min():.3f}, Max: {results_df['ievr'].max():.3f}")
        print(f"  Skew Ratio - Mean: {results_df['skew_ratio'].mean():.3f}, Std: {results_df['skew_ratio'].std():.3f}")
        
        return filename

def main():
    """
    Main function to run batch IEVR analysis
    """
    print("BATCH IEVR ANALYSIS FOR MULTIPLE STOCKS")
    print("="*80)
    
    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="sami_sellami", password="xampok-9Hezfy-cahveq")
        print("✓ Connected to WRDS")
        
        # Initialize batch analyzer
        analyzer = BatchIEVRAnalysis(db, max_workers=4)
        
        # Set parameters
        earnings_date = pd.to_datetime('2023-01-31')  # Example earnings date
        num_stocks = 100  # Start with 100 for testing, then increase to 500
        
        # Run batch analysis
        results = analyzer.run_batch_analysis(
            earnings_date=earnings_date,
            num_stocks=num_stocks,
            analysis_days_before=30
        )
        
        if results:
            # Save results
            filename = analyzer.save_results()
            print(f"\n✓ Batch analysis completed successfully!")
            print(f"  Results saved to: {filename}")
        else:
            print("\n✗ Batch analysis failed - no results generated")
        
        # Close connection
        db.close()
        print("\n✓ Database connection closed")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
