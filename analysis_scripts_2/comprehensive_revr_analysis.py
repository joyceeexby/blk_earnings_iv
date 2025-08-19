#!/usr/bin/env python3
"""
Comprehensive REVR Analysis Script
Loop through all earnings seasons from Q1 2005 to Q3 2023
Get top 500 most liquid stocks for each quarter and calculate REVR
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import wrds
from revr_analysis import REVRAnalysis
import warnings
warnings.filterwarnings('ignore')

def get_top_dollar_volume_quarterly(db, start_year, end_year, num_top_stocks=500, add_year_quarter=True):
    """
    Fetch the top N stocks by dollar trading volume (|prc| * vol) for the first available
    monthly date of each calendar quarter, filtered to single common stocks on US exchanges.

    Parameters
    ----------
    db : wrds.Connection
        Active WRDS connection.
    start_year : int
        Start year (inclusive).
    end_year : int
        End year (inclusive).
    num_top_stocks : int
        Number of top stocks to return per quarter.
    add_year_quarter : bool
        Whether to add 'year' and 'quarter' columns.

    Returns
    -------
    pandas.DataFrame
        Top stocks ranked by dollar volume, with metadata and quarterly labels.
    """

    # Step 1: Get all available monthly dates in CRSP
    dates_df = db.raw_sql(f"""
        SELECT DISTINCT date
        FROM crsp.msf
        WHERE date >= '{start_year}-01-01' AND date <= '{end_year}-12-31'
    """)
    if dates_df.empty:
        cols = ['quarter_start_date','quarter_end_date','permno','ticker','comnam','dollar_vol']
        if add_year_quarter:
            cols = ['year','quarter'] + cols
        return pd.DataFrame(columns=cols)

    dates_df['date'] = pd.to_datetime(dates_df['date'])
    qstarts = (dates_df
               .assign(quarter=dates_df['date'].dt.to_period('Q'))
               .sort_values('date')
               .groupby('quarter')
               .first()
               .reset_index())
    quarter_start_dates = qstarts['date'].dt.strftime('%Y-%m-%d').tolist()

    # Step 2: Pull CRSP msf and compute dollar volume for each quarterly date
    frames = []
    for qdate in quarter_start_dates:
        df = db.raw_sql(f"""
            SELECT permno, date, prc, vol
            FROM crsp.msf
            WHERE date = '{qdate}'
        """)
        if df.empty:
            continue

        df = df[df['vol'] > 0].copy()
        df['prc'] = df['prc'].abs()
        df['dollar_vol'] = df['prc'] * df['vol']
        df = df[df['dollar_vol'] > 0].nlargest(num_top_stocks, 'dollar_vol')

        df['quarter_start'] = pd.to_datetime(qdate)
        qper = df['quarter_start'].dt.to_period('Q')
        df['quarter_start_date'] = qper.apply(lambda p: p.start_time).dt.date
        df['quarter_end_date']   = qper.apply(lambda p: p.end_time).dt.date

        frames.append(df[['quarter_start_date','quarter_end_date','permno','dollar_vol']])

    if not frames:
        cols = ['quarter_start_date','quarter_end_date','permno','ticker','comnam','dollar_vol']
        if add_year_quarter:
            cols = ['year','quarter'] + cols
        return pd.DataFrame(columns=cols)

    top_dollar_volume = pd.concat(frames, ignore_index=True)

    # Step 3: Pull stocknames for filtering and time-valid match
    permnos = sorted(top_dollar_volume['permno'].unique().tolist())
    min_dt = pd.to_datetime(top_dollar_volume['quarter_start_date']).min().strftime('%Y-%m-%d')
    max_dt = pd.to_datetime(top_dollar_volume['quarter_end_date']).max().strftime('%Y-%m-%d')

    def _chunk(lst, n=8000):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    stocknames_parts = []
    for chunk_permnos in _chunk(permnos):
        in_list = ",".join(map(str, chunk_permnos))
        part = db.raw_sql(f"""
            SELECT permno, ticker, comnam, namedt,
                   COALESCE(nameenddt,'9999-12-31') AS nameenddt,
                   shrcd, exchcd
            FROM crsp.stocknames
            WHERE permno IN ({in_list})
              AND namedt <= '{max_dt}'
              AND COALESCE(nameenddt,'9999-12-31') >= '{min_dt}'
        """)
        stocknames_parts.append(part)

    if not stocknames_parts:
        cols = ['quarter_start_date','quarter_end_date','permno','ticker','comnam','dollar_vol']
        if add_year_quarter:
            cols = ['year','quarter'] + cols
        return pd.DataFrame(columns=cols)

    stocknames_df = pd.concat(stocknames_parts, ignore_index=True)
    stocknames_df['namedt'] = pd.to_datetime(stocknames_df['namedt'])
    stocknames_df['nameenddt'] = pd.to_datetime(stocknames_df['nameenddt'])

    # Step 4: Merge and filter
    m = top_dollar_volume.merge(stocknames_df, on='permno', how='left')
    m['qstart_ts'] = pd.to_datetime(m['quarter_start_date'])

    m = m[
        (m['qstart_ts'] >= m['namedt']) &
        (m['qstart_ts'] <= m['nameenddt']) &
        (m['shrcd'].isin([10, 11])) &
        (m['exchcd'].isin([1, 2, 3]))
    ].copy()

    # Optional: add year/quarter columns
    if add_year_quarter:
        m['year'] = m['qstart_ts'].dt.year
        m['quarter'] = m['qstart_ts'].dt.quarter

    # Final sort and select
    base_cols = ['quarter_start_date','quarter_end_date','permno','ticker','comnam','dollar_vol']
    cols = ['year','quarter'] + base_cols if add_year_quarter else base_cols
    m = m[cols]

    m = m.sort_values(['quarter_start_date','dollar_vol'], ascending=[True, False])
    m = m.drop_duplicates(subset=['quarter_start_date','permno'], keep='first')

    return m.reset_index(drop=True)

class ComprehensiveREVRAnalysis:
    """
    Comprehensive REVR analysis across multiple earnings seasons.
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
        
    def get_earnings_dates_for_season(self, ticker, target_quarter='Q1', target_year=2005):
        """
        Get earnings dates for a specific quarter and year.
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
            
            # Query for earnings in the calendar quarter
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
            
            earnings = self.db.raw_sql(query)
            return earnings
            
        except Exception as e:
            print(f"Error fetching earnings dates: {e}")
            return None
    
    def analyze_single_revr_event(self, ticker, earnings_date, analysis_days_before=30):
        """
        Analyze a single REVR event.
        """
        try:
            # Ensure earnings_date is a datetime object
            if isinstance(earnings_date, str):
                earnings_date = pd.to_datetime(earnings_date)
            
            # Calculate REVR using the existing analyzer
            revr_result = self.revr_analyzer.calculate_revr(
                ticker=ticker,
                earnings_date=earnings_date,
                days_before=analysis_days_before
            )
            
            if revr_result is not None and isinstance(revr_result, dict):
                # Extract the REVR value from the results dictionary
                revr_value = revr_result.get('revr')
                
                if revr_value is not None and not pd.isna(revr_value):
                    return {
                        'ticker': ticker,
                        'earnings_date': earnings_date,
                        'analysis_date': earnings_date - timedelta(days=analysis_days_before),
                        'revr': revr_value
                    }
            
            return None
            
        except Exception as e:
            print(f"Error analyzing REVR for {ticker} on {earnings_date}: {e}")
            return None
    
    def process_single_season(self, target_quarter, target_year, analysis_days_before=30):
        """
        Process a single earnings season.
        """
        print(f"\n{'='*80}")
        print(f"PROCESSING {target_quarter} {target_year}")
        print(f"{'='*80}")
        
        # Step 1: Get top 500 most liquid stocks for this quarter
        print(f"Step 1: Getting top 500 stocks by dollar volume for {target_quarter} {target_year}...")
        
        try:
            stocks_df = get_top_dollar_volume_quarterly(
                db=self.db,
                start_year=target_year,
                end_year=target_year,
                num_top_stocks=500,
                add_year_quarter=True
            )
            
            if stocks_df.empty:
                print(f"⚠️  No stocks found for {target_quarter} {target_year}")
                return []
            
            # Filter for the specific quarter
            quarter_stocks = stocks_df[stocks_df['quarter'] == int(target_quarter[1])]
            
            if quarter_stocks.empty:
                print(f"⚠️  No stocks found for {target_quarter} {target_year}")
                return []
            
            # Get unique tickers
            tickers = quarter_stocks['ticker'].dropna().unique()
            print(f"✅ Found {len(tickers)} stocks for {target_quarter} {target_year}")
            
            # Show sample of stocks
            print(f"Sample stocks: {list(tickers[:10])}")
            
        except Exception as e:
            print(f"❌ Error getting stock list: {e}")
            return []
        
        # Step 2: Calculate REVR for each stock
        print(f"\nStep 2: Calculating REVR for {len(tickers)} stocks...")
        
        season_results = []
        successful_count = 0
        failed_count = 0
        
        for i, ticker in enumerate(tickers):
            if i % 50 == 0:  # Progress update every 50 stocks
                print(f"  Progress: {i}/{len(tickers)} stocks processed")
            
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
                                 end_quarter='Q3', end_year=2023, analysis_days_before=30):
        """
        Run comprehensive REVR analysis across all specified seasons.
        """
        print("COMPREHENSIVE REVR ANALYSIS")
        print("="*80)
        print(f"Period: {start_quarter} {start_year} to {end_quarter} {end_year}")
        print(f"Analysis window: {analysis_days_before} days before earnings")
        
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
                season_results = self.process_single_season(quarter, year, analysis_days_before)
                all_results.extend(season_results)
                
                # Save intermediate results every 10 seasons
                if (i + 1) % 10 == 0:
                    self.save_intermediate_results(all_results, i + 1)
                
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
            self.save_final_results(all_results)
            
            return all_results
        else:
            print("❌ No results generated")
            return None
    
    def save_intermediate_results(self, results, season_count):
        """
        Save intermediate results every 10 seasons.
        """
        if not results:
            return
        
        df = pd.DataFrame(results)
        filename = f'data_files/bulk_revr_intermediate_{season_count}_seasons.csv'
        df.to_csv(filename, index=False)
        print(f"💾 Intermediate results saved: {filename}")
    
    def save_final_results(self, results):
        """
        Save final comprehensive results.
        """
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        column_order = ['ticker', 'season', 'year', 'quarter', 'earnings_date', 'analysis_date', 'revr']
        df = df[column_order]
        
        # Save to CSV
        filename = 'data_files/bulk_revr_comprehensive.csv'
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
    Main function to run comprehensive REVR analysis.
    """
    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="", password="")
        print("✅ Connected to WRDS")
        
        # Initialize comprehensive analyzer
        analyzer = ComprehensiveREVRAnalysis(db)
        
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis(
            start_quarter='Q1',
            start_year=2005,
            end_quarter='Q3',
            end_year=2023,
            analysis_days_before=30
        )
        
        if results is not None:
            print(f"\n🎉 Comprehensive REVR analysis completed successfully!")
            print(f"Generated {len(results)} observations across all seasons")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
