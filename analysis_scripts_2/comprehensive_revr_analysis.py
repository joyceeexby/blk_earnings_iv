#!/usr/bin/env python3
"""
Comprehensive REVR Analysis Script (Modified for Static Stock List)
Loop through all earnings seasons from Q1 2005 to Q3 2023
Use static stock list from CSV and calculate REVR for comparison
Uses CUSIP for accurate data matching instead of ticker
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import wrds
from revr_analysis import REVRAnalysis
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveREVRAnalysis:
    """
    Comprehensive REVR analysis across multiple earnings seasons using static stock list.
    Uses CUSIP for accurate data matching.
    """
    
    def __init__(self, db):
        """
        Initialize the comprehensive analyzer.
        
        Parameters:
        -----------
        db : wrds.Connection
            Active WRDS connection
        """
        self.db = db
        self.revr_analyzer = REVRAnalysis(db)
        
    def load_static_stock_list(self, csv_file_path='data_files/top500_liquidity_2005_2023.csv', target_quarter=None, target_year=None):
        """
        Load static stock list from CSV file for a specific quarter.
        Uses ticker for data queries since CUSIPs may not match between data sources.
        
        Parameters:
        -----------
        csv_file_path : str
            Path to the CSV file containing stock list
        target_quarter : str, optional
            Target quarter (Q1, Q2, Q3, Q4) - if provided, filters stocks for that quarter
        target_year : int, optional
            Target year - if provided, filters stocks for that year
            
        Returns:
        --------
        list : List of stock tickers
        str : Type of identifier (always 'ticker' for simplicity)
        """
        try:
            print(f"📁 Loading static stock list from: {csv_file_path}")
            
            # Read the CSV file
            stocks_df = pd.read_csv(csv_file_path)
            print(f"📊 Total rows in CSV: {len(stocks_df):,}")
            print(f"📊 CSV columns: {list(stocks_df.columns)}")
            
            # Filter for specific quarter if provided
            if target_quarter is not None and target_year is not None:
                # Convert quarter string to number (Q1->1, Q2->2, etc.)
                quarter_num = int(target_quarter[1])
                
                print(f"🔍 Filtering for {target_quarter} {target_year} (quarter {quarter_num})")
                print(f"🔍 Available years: {sorted(stocks_df['year'].unique())}")
                print(f"🔍 Available quarters: {sorted(stocks_df['quarter'].unique())}")
                
                # Filter for the specific quarter and year
                filtered_df = stocks_df[
                    (stocks_df['year'] == target_year) & 
                    (stocks_df['quarter'] == quarter_num)
                ]
                
                print(f"📊 Filtered for {target_quarter} {target_year}: {len(filtered_df)} stocks")
                
                if len(filtered_df) == 0:
                    print(f"❌ No stocks found for {target_quarter} {target_year}")
                    print(f"❌ Check if year {target_year} and quarter {quarter_num} exist in CSV")
                    return None, None
                
                # Show sample of filtered stocks
                if 'ticker' in filtered_df.columns:
                    sample_tickers = filtered_df['ticker'].dropna().head(5).tolist()
                    print(f"📊 Sample tickers: {sample_tickers}")
                
                # Use filtered dataframe for further processing
                stocks_df = filtered_df
            else:
                print("📊 Using all stocks from file (no quarter filtering)")
            
            # Extract ticker column (prefer ticker over CUSIP due to data source mismatches)
            if 'ticker' in stocks_df.columns:
                tickers = stocks_df['ticker'].dropna().unique().tolist()
                print(f"✅ Loaded {len(tickers)} tickers from static list")
                print(f"📊 Sample tickers: {tickers[:3]}")
                return tickers, 'ticker'
            
            elif 'Ticker' in stocks_df.columns:
                tickers = stocks_df['Ticker'].dropna().unique().tolist()
                print(f"✅ Loaded {len(tickers)} tickers from static list")
                print(f"📊 Sample tickers: {tickers[:3]}")
                return tickers, 'ticker'
            
            else:
                # If no ticker column found, use the first column
                first_col = stocks_df.columns[0]
                values = stocks_df[first_col].dropna().unique().tolist()
                print(f"⚠️  No 'ticker' column found, using first column: '{first_col}'")
                print(f"📊 Sample values: {values[:3]}")
                return values, 'unknown'
            
        except FileNotFoundError:
            print(f"❌ Error: CSV file '{csv_file_path}' not found")
            print("Please ensure the file exists in the current directory")
            return None, None
        except Exception as e:
            print(f"❌ Error loading stock list: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def get_earnings_dates_for_season(self, ticker, target_quarter='Q1', target_year=2005):
        """
        Get earnings dates for a specific quarter and year using ticker.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        target_quarter : str
            Target quarter (Q1, Q2, Q3, Q4)
        target_year : int
            Target year
        """
        try:
            # Map quarter to month range (calendar months)
            quarter_months = {
                'Q1': (1, 3),    # Jan-Mar
                'Q2': (4, 6),    # Apr-Jun
                'Q3': (7, 9),    # Jul-Sep
                'Q4': (10, 12)   # Oct-Dec
            }
            
            start_month, end_month = quarter_months[target_quarter]
            start_date = f"{target_year}-{start_month:02d}-01"
            
            # Calculate the last day of the end month properly
            if end_month in [4, 6, 9, 11]:  # April, June, September, November
                end_day = 30
            elif end_month == 2:  # February
                # Check if it's a leap year
                if (target_year % 4 == 0 and target_year % 100 != 0) or (target_year % 400 == 0):
                    end_day = 29
                else:
                    end_day = 28
            else:  # January, March, May, July, August, October, December
                end_day = 31
            
            end_date = f"{target_year}-{end_month:02d}-{end_day:02d}"
            
            print(f"🔍 Fetching earnings for {ticker} in {target_quarter} {target_year}")
            print(f"🔍 Date range: {start_date} to {end_date}")
            
            # Use ticker-based query
            query = f"""
            SELECT cusip,
                   tic as ticker,
                   datadate,
                   rdq as earnings_date,
                   fyearq,
                   fqtr
            FROM comp.fundq
            WHERE tic = '{ticker}'
              AND rdq BETWEEN '{start_date}' AND '{end_date}'
              AND rdq IS NOT NULL
            ORDER BY rdq;
            """
            
            print(f"🔍 Executing ticker query for: {ticker}")
            earnings = self.db.raw_sql(query)
            print(f"🔍 Query returned {len(earnings)} earnings records")
            
            if len(earnings) > 0:
                print(f"🔍 Sample earnings dates: {earnings['earnings_date'].head(3).tolist()}")
            
            return earnings
            
        except Exception as e:
            print(f"❌ Error fetching earnings dates for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_single_revr_event(self, ticker, earnings_date, analysis_days_before=30):
        """
        Analyze a single REVR event using ticker.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        earnings_date : datetime
            Earnings announcement date
        analysis_days_before : int
            Days before earnings to analyze
        """
        try:
            print(f"🔍 Analyzing REVR for {ticker} on {earnings_date}")
            
            # Ensure earnings_date is a datetime object
            if isinstance(earnings_date, str):
                earnings_date = pd.to_datetime(earnings_date)
                print(f"🔍 Converted earnings_date to datetime: {earnings_date}")
            
            # Calculate REVR using ticker directly
            print(f"🔍 Calculating REVR using ticker: {ticker}")
            revr_result = self.revr_analyzer.calculate_revr(
                ticker=ticker,
                earnings_date=earnings_date,
                days_before=analysis_days_before
            )
            print(f"🔍 REVR calculation result: {revr_result}")
            
            if revr_result is not None and isinstance(revr_result, dict):
                # Extract the REVR value from the results dictionary
                revr_value = revr_result.get('revr')
                print(f"🔍 Extracted REVR value: {revr_value}")
                
                if revr_value is not None and not pd.isna(revr_value):
                    result = {
                        'ticker': ticker,
                        'earnings_date': earnings_date,
                        'analysis_date': earnings_date - timedelta(days=analysis_days_before),
                        'revr': revr_value
                    }
                    print(f"✅ Successfully created REVR result: {result}")
                    return result
                else:
                    print(f"⚠️  REVR value is None or NaN: {revr_value}")
            else:
                print(f"⚠️  REVR result is None or not a dict: {revr_result}")
            
            return None
            
        except Exception as e:
            print(f"❌ Error analyzing REVR for {ticker} on {earnings_date}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_single_season(self, target_quarter, target_year, stock_list, analysis_days_before=30):
        """
        Process a single earnings season using static stock list.
        """
        print(f"\n{'='*80}")
        print(f"PROCESSING {target_quarter} {target_year}")
        print(f"{'='*80}")
        
        print(f"Step 1: Using static stock list with {len(stock_list)} tickers...")
        
        # Step 2: Calculate REVR for each stock
        print(f"\nStep 2: Calculating REVR for {len(stock_list)} stocks...")
        
        season_results = []
        successful_count = 0
        failed_count = 0
        
        # Process stocks (now all are tickers)
        for i, ticker in enumerate(stock_list):
            if i % 50 == 0:  # Progress update every 50 stocks
                print(f"  Progress: {i}/{len(stock_list)} stocks processed")
            
            try:
                # Get earnings dates for this season
                earnings = self.get_earnings_dates_for_season(ticker, target_quarter, target_year)
                
                if earnings is None or earnings.empty:
                    failed_count += 1
                    continue
                
                # Only process the FIRST earnings event per stock per quarter
                if len(earnings) > 1:
                    print(f"  ⚠️  {ticker} has {len(earnings)} earnings, using first")
                
                # Take only the first earnings event
                first_earnings = earnings.iloc[0]
                earnings_date = first_earnings['earnings_date']
                
                # Calculate REVR
                event_results = self.analyze_single_revr_event(
                    ticker=ticker,
                    earnings_date=earnings_date,
                    analysis_days_before=analysis_days_before
                )
                
                if event_results is not None:
                    # Add season metadata
                    event_results['season'] = f"{target_quarter} {target_year}"
                    event_results['year'] = target_year
                    event_results['quarter'] = target_quarter
                    
                    season_results.append(event_results)
                    successful_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                print(f"  ❌ Error processing {ticker}: {e}")
                failed_count += 1
                continue
        
        print(f"\n✅ Season {target_quarter} {target_year} completed:")
        print(f"  Successful: {successful_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Total: {len(season_results)}")
        
        return season_results
    
    def run_comprehensive_analysis(self, start_quarter='Q1', start_year=2005, 
                                 end_quarter='Q3', end_year=2023, analysis_days_before=30,
                                 static_stocks_csv='data_files/top500_liquidity_2005_2023.csv'):
        """
        Run comprehensive REVR analysis across all specified seasons using static stock list.
        """
        print("COMPREHENSIVE REVR ANALYSIS (STATIC STOCK LIST - CUSIP ENHANCED)")
        print("="*80)
        print(f"Period: {start_quarter} {start_year} to {end_quarter} {end_year}")
        print(f"Analysis window: {analysis_days_before} days before earnings")
        print(f"Stock list source: {static_stocks_csv}")
        
        # Generate all quarters to process
        quarters_to_process = []
        current_year = start_year
        current_quarter = start_quarter
        
        while (current_year < end_year) or (current_year == end_year and current_quarter <= end_quarter):
            quarters_to_process.append((current_quarter, current_year))
            
            # Move to next quarter
            if current_quarter == 'Q1':
                current_quarter = 'Q2'
            elif current_quarter == 'Q2':
                current_quarter = 'Q3'
            elif current_quarter == 'Q3':
                current_quarter = 'Q4'
            elif current_quarter == 'Q4':
                current_quarter = 'Q1'
                current_year += 1
        
        print(f"Total quarters to process: {len(quarters_to_process)}")
        print(f"Quarters: {quarters_to_process[:5]}...{quarters_to_process[-5:]}")
        
        # Process each season
        all_results = []
        
        for i, (quarter, year) in enumerate(quarters_to_process):
            print(f"\n{'='*60}")
            print(f"PROCESSING SEASON {i+1}/{len(quarters_to_process)}: {quarter} {year}")
            print(f"{'='*60}")
            
            try:
                # Load stocks for THIS specific quarter
                stock_list, identifier_type = self.load_static_stock_list(
                    static_stocks_csv, 
                    target_quarter=quarter, 
                    target_year=year
                )
                
                if not stock_list:
                    print(f"⚠️  No stocks found for {quarter} {year}, skipping")
                    continue
                
                # Process this season with the quarter-specific stock list
                season_results = self.process_single_season(quarter, year, stock_list, analysis_days_before)
                all_results.extend(season_results)
                
                # Save intermediate results every 10 seasons
                if (i + 1) % 10 == 0:
                    self.save_intermediate_results(all_results, i + 1, "static_cusip")
                
            except Exception as e:
                print(f"❌ Error processing {quarter} {year}: {e}")
                continue
        
        # Final results
        if all_results:
            print(f"\n{'='*80}")
            print(f"COMPREHENSIVE ANALYSIS COMPLETED")
            print(f"{'='*80}")
            print(f"Total observations: {len(all_results)}")
            print(f"Unique stocks: {pd.DataFrame(all_results)['ticker'].nunique()}")
            print(f"Seasons covered: {len(quarters_to_process)}")
            
            # Save final results
            self.save_final_results(all_results, "static_cusip")
            
            return all_results
        else:
            print("❌ No results generated")
            return None
    
    def save_intermediate_results(self, results, season_count, prefix=""):
        """
        Save intermediate results every 10 seasons.
        """
        if not results:
            return
        
        df = pd.DataFrame(results)
        filename = f'data_files/bulk_revr_intermediate_{prefix}_{season_count}_seasons.csv'
        df.to_csv(filename, index=False)
        print(f"💾 Intermediate results saved: {filename}")
    
    def save_final_results(self, results, prefix=""):
        """
        Save final comprehensive results.
        """
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        column_order = ['ticker', 'season', 'year', 'quarter', 'earnings_date', 'analysis_date', 'revr']
        df = df[column_order]
        
        # Save to CSV with prefix to distinguish from original
        filename = f'data_files/bulk_revr_comprehensive_{prefix}_comparison.csv'
        df.to_csv(filename, index=False)
        print(f"💾 Final results saved: {filename}")
        
        # Summary statistics
        print(f"\n📊 Final Summary Statistics:")
        print(f"  Total observations: {len(df)}")
        print(f"  Unique stocks: {df['ticker'].nunique()}")
        print(f"  Seasons covered: {df['season'].nunique()}")
        print(f"  Date range: {df['earnings_date'].min()} to {df['earnings_date'].max()}")
        
        # REVR statistics
        print(f"\n📈 REVR Statistics:")
        print(f"  Mean: {df['revr'].mean():.3f}")
        print(f"  Std: {df['revr'].std():.3f}")
        print(f"  Min: {df['revr'].min():.3f}")
        print(f"  Max: {df['revr'].max():.3f}")
        print(f"  Median: {df['revr'].median():.3f}")
        
        # Season distribution
        print(f"\n📅 Season Distribution:")
        season_counts = df['season'].value_counts().sort_index()
        for season, count in season_counts.items():
            print(f"  {season}: {count} observations")

def main():
    """
    Main function to run comprehensive REVR analysis with static stock list.
    """
    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="", password="")
        print("✅ Connected to WRDS")
        
        # Initialize comprehensive analyzer
        analyzer = ComprehensiveREVRAnalysis(db)
        
        # Run comprehensive analysis with static stock list
        results = analyzer.run_comprehensive_analysis(
            start_quarter='Q1',
            start_year=2005,
            end_quarter='Q3',
            end_year=2023,
            analysis_days_before=30,
            static_stocks_csv='data_files/top500_liquidity_2005_2023.csv'
        )
        
        if results is not None:
            print(f"\n🎉 Comprehensive REVR analysis completed successfully!")
            print(f"Generated {len(results)} observations across all seasons")
            print(f"Results saved to: bulk_revr_comprehensive_static_cusip_comparison.csv")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
