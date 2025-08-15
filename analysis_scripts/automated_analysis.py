#!/usr/bin/env python3
"""
Automated REVR and IEVR Analysis for Multiple Earnings Events
Calculate both measures for all AAPL earnings events and create Events × 2 matrix
Enhanced with dispersion analysis and Fama-French factors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import wrds
from revr_analysis import REVRAnalysis
from ievr_analysis import IEVRAnalysis
from pandas.tseries.offsets import BDay
from option_surface_features import OptionSurfaceFeatures
import traceback

class AutomatedEarningsAnalysis:
    """
    Automated analysis of REVR and IEVR for multiple earnings events.
    Enhanced with dispersion analysis and Fama-French factors.
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.revr_analyzer = REVRAnalysis(db_connection)
        self.ievr_analyzer = IEVRAnalysis(db_connection)  # Pass the database connection
        self.option_surface_analyzer = OptionSurfaceFeatures(db_connection)  # Add option surface features
        self.results = []
        
    def get_earnings_dates(self, ticker, start_date, end_date):
        """
        Fetch earnings dates for the stock.
        """
        print(f"Fetching earnings dates for {ticker} from {start_date} to {end_date}")
        
        try:
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
            print(f"Retrieved {len(earnings)} earnings events")
            
            if not earnings.empty:
                print(f"Earnings date range: {earnings['earnings_date'].min()} to {earnings['earnings_date'].max()}")
                for _, row in earnings.iterrows():
                    print(f"  {row['earnings_date']}: Q{row['fqtr']} {row['fyearq']}")
            
            return earnings
            
        except Exception as e:
            print(f"Error fetching earnings dates: {e}")
            return None
    
    def get_stock_price_at_date(self, ticker, target_date, days_before=5):
        """
        Get approximate stock price at a given date for IEVR analysis.
        """
        try:
            # Get security info
            sec_query = f"""
            SELECT secid, ticker
            FROM optionm.secnmd
            WHERE ticker = '{ticker}'
            ORDER BY effect_date DESC
            LIMIT 1
            """
            sec_info = self.db.raw_sql(sec_query)
            
            if sec_info.empty:
                return None
                
            secid = sec_info.iloc[0].to_dict()['secid']
            
            # Get stock price around the date
            start_date = target_date - timedelta(days=days_before)
            end_date = target_date + timedelta(days=days_before)
            
            price_query = f"""
            SELECT date, close
            FROM optionm.secprd
            WHERE secid = {secid}
              AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY ABS(date - '{target_date}')
            LIMIT 1
            """
            
            price_data = self.db.raw_sql(price_query)
            
            if not price_data.empty:
                return price_data.iloc[0].to_dict()['close']
            else:
                return None
                
        except Exception as e:
            print(f"Error getting stock price: {e}")
            return None

    def get_analyst_dispersion(self, ticker, earnings_date, lookback_days=21):
        """
        Get analyst forecast dispersion for a specific earnings event.
        Uses IBES summary data to calculate dispersion at T-lookback_days.
        Adds [DEBUG] logging and stepwise try/except for each block.
        """
        print(f"[DEBUG] get_analyst_dispersion called with ticker={ticker}, earnings_date={earnings_date}, lookback_days={lookback_days}")
        # Step 1: Calculate lookup date
        try:
            earnings_date = pd.to_datetime(earnings_date)
            lookup_date = earnings_date - BDay(lookback_days)
            print(f"[DEBUG] Calculated lookup_date: {lookup_date}")
        except Exception as e:
            print(f"[DEBUG] Error parsing earnings_date or calculating lookup_date: {e}")
            traceback.print_exc()
            return None, None

        # Step 2: Query CUSIP
        try:
            cusip_query = f"""
            SELECT DISTINCT cusip
            FROM comp.fundq
            WHERE tic = '{ticker}'
            AND cusip IS NOT NULL
            LIMIT 1
            """
            cusip_result = self.db.raw_sql(cusip_query)
            print(f"[DEBUG] CUSIP query returned type: {type(cusip_result)}, shape: {getattr(cusip_result, 'shape', 'N/A')}")
        except Exception as e:
            print(f"[DEBUG] Error during CUSIP query for ticker {ticker}: {e}")
            traceback.print_exc()
            return None, None

        # Step 3: Check CUSIP query result
        try:
            if hasattr(cusip_result, 'empty') and cusip_result.empty:
                print(f"[DEBUG] No CUSIP found for {ticker}")
                return None, None
            if len(cusip_result) == 0:
                print(f"[DEBUG] No CUSIP found for {ticker} (len==0)")
                return None, None
            print(f"[DEBUG] CUSIP found: {cusip_result.iloc[0].to_dict()}")
        except Exception as e:
            print(f"[DEBUG] Error checking CUSIP query result: {e}")
            traceback.print_exc()
            return None, None

        # Step 4: Prepare CUSIP8
        try:
            cusip = cusip_result.iloc[0].to_dict()['cusip']
            cusip8 = cusip[:8] if len(cusip) >= 8 else cusip
            print(f"[DEBUG] Using CUSIP8: {cusip8}")
        except Exception as e:
            print(f"[DEBUG] Error extracting/formatting CUSIP: {e}")
            traceback.print_exc()
            return None, None

        # Step 5: Query IBES summary
        try:
            # Try different IBES table names - the table structure might be different
            ibes_tables = [
                "tr_ibes.statsum_epsus",
                "ibes.statsum_epsus", 
                "ibes.statsum",
                "tr_ibes.statsum"
            ]
            
            ibes_data = None
            successful_table = None
            
            for table_name in ibes_tables:
                try:
                    print(f"[DEBUG] Trying table: {table_name}")
                    test_query = f"SELECT COUNT(*) as count FROM {table_name} LIMIT 1"
                    test_result = self.db.raw_sql(test_query)
                    print(f"[DEBUG] Table {table_name} accessible, count: {test_result.iloc[0]['count']}")
                    
                    # Now try the actual query
                    ibes_query = f"""
                    SELECT ticker, cusip, statpers, meanest, stdev, numest
                    FROM {table_name}
                    WHERE cusip LIKE '{cusip8}%'
                      AND statpers <= '{lookup_date}'
                      AND meanest IS NOT NULL
                      AND stdev IS NOT NULL
                    ORDER BY statpers DESC
                    LIMIT 1
                    """
                    print(f"[DEBUG] Executing IBES query on {table_name}: {ibes_query}")
                    ibes_data = self.db.raw_sql(ibes_query)
                    print(f"[DEBUG] Query successful on {table_name}, returned type: {type(ibes_data)}, shape: {getattr(ibes_data, 'shape', 'N/A')}")
                    successful_table = table_name
                    break
                    
                except Exception as table_error:
                    print(f"[DEBUG] Table {table_name} failed: {table_error}")
                    continue
            
            if ibes_data is None:
                print(f"[DEBUG] All IBES tables failed for ticker {ticker}")
                return None, None
                
            print(f"[DEBUG] Successfully used table: {successful_table}")
            
        except Exception as e:
            print(f"[DEBUG] Error during IBES query for ticker {ticker}: {e}")
            traceback.print_exc()
            return None, None

        # Step 6: Check IBES query result
        try:
            if hasattr(ibes_data, 'empty') and ibes_data.empty:
                print(f"[DEBUG] No IBES data found for {ticker} at {lookup_date}")
                return None, None
            if len(ibes_data) == 0:
                print(f"[DEBUG] No IBES data found for {ticker} at {lookup_date} (len==0)")
                return None, None
            print(f"[DEBUG] IBES data first row: {ibes_data.iloc[0].to_dict()}")
        except Exception as e:
            print(f"[DEBUG] Error checking IBES query result: {e}")
            traceback.print_exc()
            return None, None

        # Step 7: Extract meanest, stdev, numest
        try:
            # Convert to pandas DataFrame if needed
            if not isinstance(ibes_data, pd.DataFrame):
                ibes_data = pd.DataFrame(ibes_data)
            # Try to access first row
            ibes_row = ibes_data.iloc[0]
            mean_est = ibes_row['meanest']
            stdev = ibes_row['stdev']
            num_analysts = ibes_row['numest']
            print(f"[DEBUG] Extracted IBES row: meanest={mean_est}, stdev={stdev}, numest={num_analysts}")
        except Exception as e:
            print(f"[DEBUG] Error extracting meanest/stdev/numest from IBES data: {e}")
            print(f"[DEBUG] ibes_data type: {type(ibes_data)}, shape: {getattr(ibes_data, 'shape', 'N/A')}")
            traceback.print_exc()
            return None, None

        # Step 8: Convert to numeric and calculate dispersion
        try:
            mean_est_num = float(mean_est)
            stdev_num = float(stdev)
            num_analysts_int = int(num_analysts)
            dispersion = stdev_num / abs(mean_est_num) if mean_est_num != 0 else None
            print(f"[DEBUG] Dispersion for {ticker}: {dispersion} (from {num_analysts_int} analysts)")
            print(f"[DEBUG] Raw values - Mean: {mean_est_num}, Std: {stdev_num}")
            return dispersion, num_analysts_int
        except Exception as e:
            print(f"[DEBUG] Error converting values to numeric or calculating dispersion: {e}")
            print(f"[DEBUG] meanest={mean_est} (type: {type(mean_est)}), stdev={stdev} (type: {type(stdev)}), numest={num_analysts} (type: {type(num_analysts)})")
            traceback.print_exc()
            return None, None

    def get_fama_french_factors(self, earnings_date, lookback_days=21):
        """
        Get Fama-French 5-factor data for the month containing the earnings date.
        Uses local CSV file for efficiency.
        
        Parameters:
        - earnings_date: Date of earnings announcement
        - lookback_days: Days before earnings to get factors
        
        Returns:
        - dict: Fama-French factors for the month
        """
        try:
            # Calculate the month for factor lookup
            earnings_date = pd.to_datetime(earnings_date)
            factor_date = earnings_date - timedelta(days=lookback_days)
            factor_month = factor_date.replace(day=1)  # First day of month
            
            # Load Fama-French data from local file
            ff_file = 'data_files/F-F_Research_Data_5_Factors_2x3.csv'
            
            try:
                ff_data = pd.read_csv(ff_file)
                
                # Clean column names
                ff_data.columns = ff_data.columns.str.strip()
                date_col = ff_data.columns[0]
                ff_data = ff_data.rename(columns={date_col: 'Date'})
                
                # Convert date format (YYYYMM to datetime)
                ff_data['Date'] = pd.to_datetime(ff_data['Date'], format='%Y%m', errors='coerce')
                
                # Filter for the specific month
                month_data = ff_data[ff_data['Date'] == factor_month]
                
                if month_data.empty:
                    print(f"No Fama-French data found for {factor_month}")
                    print(f"Available dates: {ff_data['Date'].dt.to_period('M').unique()}")
                    return None
                
                print(f"  Found Fama-French data for {factor_month}")
                print(f"  Raw factor values: {month_data.iloc[0].to_dict()}")
                
                # Get factors (convert from percentage to decimal)
                factors_dict = month_data.iloc[0].to_dict()
                factor_dict = {}
                
                # Safely convert each factor to numeric and then to decimal
                for factor_name, factor_value in factors_dict.items():
                    if factor_name in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
                        try:
                            # Convert to numeric first, then to decimal
                            numeric_value = pd.to_numeric(factor_value, errors='coerce')
                            if pd.notna(numeric_value):
                                # Convert from percentage to decimal
                                decimal_value = numeric_value / 100.0
                                # Map to our column names
                                if factor_name == 'Mkt-RF':
                                    factor_dict['mkt_rf'] = decimal_value
                                else:
                                    factor_dict[factor_name.lower()] = decimal_value
                            else:
                                factor_dict[factor_name.lower() if factor_name != 'Mkt-RF' else 'mkt_rf'] = None
                        except (ValueError, TypeError):
                            factor_dict[factor_name.lower() if factor_name != 'Mkt-RF' else 'mkt_rf'] = None
                
                # Calculate market return
                if factor_dict['mkt_rf'] is not None and factor_dict['rf'] is not None:
                    factor_dict['mkt_return'] = factor_dict['mkt_rf'] + factor_dict['rf']
                
                print(f"  Fama-French factors loaded for {factor_month}")
                return factor_dict
                
            except FileNotFoundError:
                print(f"Fama-French data file not found: {ff_file}")
                return None
                
        except Exception as e:
            print(f"Error loading Fama-French factors: {e}")
            return None

    def get_option_surface_features(self, ticker, earnings_date, n_lag=15):
        """
        Get option surface features for a specific earnings event.
        
        Parameters:
        - ticker: Stock ticker
        - earnings_date: Date of earnings announcement
        - n_lag: Number of trading days before earnings to calculate features
        
        Returns:
        - dict: Option surface features (TERM_RATIO, SKEW, KURT, IV_RATIO, SMIRK)
        """
        try:
            print(f"  Calculating option surface features for {ticker}...")
            
            # Get the latest secid for the ticker
            secid_query = f"""
            SELECT secid
            FROM optionm_all.secnmd
            WHERE ticker = '{ticker}'
            ORDER BY effect_date DESC
            LIMIT 1
            """
            
            secid_result = self.db.raw_sql(secid_query)
            
            if secid_result.empty:
                print(f"    ✗ No secid found for {ticker}")
                return None
            
            # Convert to pandas DataFrame if needed
            if not isinstance(secid_result, pd.DataFrame):
                secid_result = pd.DataFrame(secid_result)
            
            secid = secid_result.iloc[0]['secid']
            print(f"    Found secid: {secid}")
            
            # Calculate all option surface features
            features = self.option_surface_analyzer.calculate_surface_features(
                ticker=ticker,
                secid=secid,
                earnings_date=earnings_date,
                n_lag=n_lag
            )
            
            return features
            
        except Exception as e:
            print(f"  Error calculating option surface features for {ticker}: {e}")
            return None

    def analyze_single_event(self, ticker, earnings_date, analysis_days_before=30):
        """
        Analyze a single earnings event for REVR, IEVR, dispersion, and Fama-French factors.
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING: {ticker} - {earnings_date}")
        print(f"{'='*60}")
        
        earnings_date = pd.to_datetime(earnings_date)
        analysis_date = earnings_date - timedelta(days=analysis_days_before)
        
        # Calculate REVR using new ST/MT methodology
        print(f"\n1. Calculating REVR (ST/MT methodology)...")
        revr_results = self.revr_analyzer.calculate_revr_st_mt_ratio(
            ticker=ticker,
            earnings_date=earnings_date,
            days_before=120,  # Extended for better MT calculation
            days_after=60
        )
        
        if revr_results is None:
            print(f"✗ REVR calculation failed for {earnings_date}")
            return None
        
        # REVR plotting removed
        print(f"\n2. REVR calculation completed (plotting disabled)")
        
        # Calculate IEVR
        print(f"\n3. Calculating IEVR...")
        
        # Get approximate stock price for IEVR analysis
        underlying_price = self.get_stock_price_at_date(ticker, analysis_date)
        if underlying_price is None:
            print(f"Could not get stock price for {analysis_date}, using default")
            underlying_price = 160.0  # Default fallback
        
        ievr_results = self.ievr_analyzer.calculate_ievr(
            ticker=ticker,
            earnings_date=earnings_date,
            analysis_days_before=analysis_days_before,
            underlying_price=underlying_price,
            include_spx=True  # Include S&P 500 IEVR calculation
        )
        
        if ievr_results is None:
            print(f"✗ IEVR calculation failed for {earnings_date}")
            return None
        
        # Get analyst dispersion
        print(f"\n4. Calculating analyst dispersion...")
        dispersion, num_analysts = self.get_analyst_dispersion(ticker, earnings_date, lookback_days=21)
        
        # Get Fama-French factors
        print(f"\n5. Loading Fama-French factors...")
        ff_factors = self.get_fama_french_factors(earnings_date, lookback_days=21)
        
        # Get option surface features
        print(f"\n6. Calculating option surface features...")
        option_features = self.get_option_surface_features(ticker, earnings_date, n_lag=15)
        
        # Combine all results
        event_results = {
            'ticker': ticker,
            'earnings_date': earnings_date,
            'analysis_date': analysis_date,
            'revr': revr_results['revr'],
            'ievr': ievr_results['ievr'],
            'vol_st': revr_results.get('vol_st', None),
            'vol_mt': revr_results.get('vol_mt', None),
            'avg_pre': ievr_results.get('avg_pre', None),
            'avg_post': ievr_results.get('avg_post', None),
            'normative_implied_vol': ievr_results.get('normative_implied_vol', None),
            'normative_realized_vol': revr_results.get('normative_realized_vol', None),
            'skew_ratio': ievr_results.get('skew_ratio', None),
            'spx_ievr': ievr_results.get('spx_ievr', None),
            'underlying_price': underlying_price,
            'methodology': 'ST/MT Ratio (Expanding EWM)',
            # New features
            'analyst_dispersion': dispersion,
            'num_analysts': num_analysts,
            # Fama-French factors
            'mkt_rf': ff_factors.get('mkt_rf', None) if ff_factors else None,
            'smb': ff_factors.get('smb', None) if ff_factors else None,
            'hml': ff_factors.get('hml', None) if ff_factors else None,
            'rmw': ff_factors.get('rmw', None) if ff_factors else None,
            'cma': ff_factors.get('cma', None) if ff_factors else None,
            'rf': ff_factors.get('rf', None) if ff_factors else None,
            'mkt_return': ff_factors.get('mkt_return', None) if ff_factors else None,
            # Option surface features
            'TERM_RATIO': option_features.get('TERM_RATIO', None) if option_features else None,
            'SKEW': option_features.get('SKEW', None) if option_features else None,
            'KURT': option_features.get('KURT', None) if option_features else None,
            'IV_RATIO': option_features.get('IV_RATIO', None) if option_features else None,
            'SMIRK': option_features.get('SMIRK', None) if option_features else None,
            'surface_date': option_features.get('surface_date', None) if option_features else None
        }
        
        print(f"\n✓ Event Analysis Complete:")
        print(f"  REVR: {event_results['revr']:.3f}")
        print(f"  IEVR: {event_results['ievr']:.3f}")
        print(f"  S&P 500 IEVR: {event_results['spx_ievr']:.3f}" if event_results['spx_ievr'] is not None else "  S&P 500 IEVR: Not calculated")
        print(f"  Ratio (IEVR/REVR): {event_results['ievr']/event_results['revr']:.3f}")
        print(f"  Analyst Dispersion: {event_results['analyst_dispersion']:.4f}" if event_results['analyst_dispersion'] is not None else "  Analyst Dispersion: Not calculated")
        print(f"  Fama-French Factors: {'Loaded' if ff_factors else 'Not loaded'}")
        print(f"  Option Surface Features: {'Loaded' if option_features else 'Not loaded'}")
        if option_features:
            print(f"    TERM_RATIO: {option_features.get('TERM_RATIO', 'N/A')}, SKEW: {option_features.get('SKEW', 'N/A')}")
            print(f"    KURT: {option_features.get('KURT', 'N/A')}, IV_RATIO: {option_features.get('IV_RATIO', 'N/A')}, SMIRK: {option_features.get('SMIRK', 'N/A')}")
        
        return event_results
    
    def analyze_multiple_events(self, ticker, start_date, end_date, analysis_days_before=30):
        """
        Analyze multiple earnings events for a stock.
        """
        print(f"\n{'='*80}")
        print(f"AUTOMATED ANALYSIS: {ticker} Earnings Events")
        print(f"{'='*80}")
        
        # Get earnings dates
        earnings = self.get_earnings_dates(ticker, start_date, end_date)
        if earnings is None or earnings.empty:
            print(f"No earnings data found for {ticker}")
            return None
        
        # Analyze each event
        successful_events = []
        
        for _, row in earnings.iterrows():
            earnings_date = row['earnings_date']
            
            try:
                event_results = self.analyze_single_event(
                    ticker=ticker,
                    earnings_date=earnings_date,
                    analysis_days_before=analysis_days_before
                )
                
                if event_results is not None:
                    successful_events.append(event_results)
                    self.results.append(event_results)
                
            except Exception as e:
                print(f"Error analyzing {earnings_date}: {e}")
                continue
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"Total earnings events: {len(earnings)}")
        print(f"Successfully analyzed: {len(successful_events)}")
        print(f"Success rate: {len(successful_events)/len(earnings)*100:.1f}%")
        
        if successful_events:
            # Create comprehensive results DataFrame with all features
            results_df = pd.DataFrame([
                {
                    'earnings_date': event['earnings_date'],
                    'revr': event['revr'],
                    'ievr': event['ievr'],
                    'ratio': event['ievr'] / event['revr'] if event['revr'] and not np.isnan(event['revr']) else None,
                    'vol_st': event.get('vol_st', None),
                    'vol_mt': event.get('vol_mt', None),
                    'avg_pre': event.get('avg_pre', None),
                    'avg_post': event.get('avg_post', None),
                    'normative_implied_vol': event.get('normative_implied_vol', None),
                    'normative_realized_vol': event.get('normative_realized_vol', None),
                    'skew_ratio': event.get('skew_ratio', None),
                    'underlying_price': event.get('underlying_price', None),
                    'methodology': event.get('methodology', 'ST/MT Ratio (Expanding EWM)'),
                    # New features
                    'analyst_dispersion': event.get('analyst_dispersion', None),
                    'num_analysts': event.get('num_analysts', None),
                    # Fama-French factors
                    'mkt_rf': event.get('mkt_rf', None),
                    'smb': event.get('smb', None),
                    'hml': event.get('hml', None),
                    'rmw': event.get('rmw', None),
                    'cma': event.get('cma', None),
                    'rf': event.get('rf', None),
                    'mkt_return': event.get('mkt_return', None),
                    # Option surface features
                    'TERM_RATIO': event.get('TERM_RATIO', None),
                    'SKEW': event.get('SKEW', None),
                    'KURT': event.get('KURT', None),
                    'IV_RATIO': event.get('IV_RATIO', None),
                    'SMIRK': event.get('SMIRK', None),
                    'surface_date': event.get('surface_date', None)
                }
                for event in successful_events
            ])
            
            # Sort by earnings date
            results_df = results_df.sort_values('earnings_date')
            
            print(f"\nEvents × 2 Matrix (REVR, IEVR):")
            print(results_df[['earnings_date', 'revr', 'ievr', 'ratio']].to_string(index=False))
            
            # Summary statistics
            print(f"\nSummary Statistics:")
            print(f"  REVR - Mean: {results_df['revr'].mean():.3f}, Std: {results_df['revr'].std():.3f}")
            print(f"  IEVR - Mean: {results_df['ievr'].mean():.3f}, Std: {results_df['ievr'].std():.3f}")
            print(f"  Ratio - Mean: {results_df['ratio'].mean():.3f}, Std: {results_df['ratio'].std():.3f}")
            
            # New features summary
            if 'analyst_dispersion' in results_df.columns:
                disp_data = results_df['analyst_dispersion'].dropna()
                if len(disp_data) > 0:
                    print(f"  Analyst Dispersion - Mean: {disp_data.mean():.4f}, Std: {disp_data.std():.4f}")
                    print(f"  Coverage: {len(disp_data)}/{len(results_df)} events ({len(disp_data)/len(results_df)*100:.1f}%)")
            
            if 'mkt_rf' in results_df.columns:
                ff_coverage = results_df[['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf']].notna().all(axis=1).sum()
                print(f"  Fama-French Factors - Coverage: {ff_coverage}/{len(results_df)} events ({ff_coverage/len(results_df)*100:.1f}%)")
            
            # Option surface features summary
            if 'TERM_RATIO' in results_df.columns:
                surface_features = ['TERM_RATIO', 'SKEW', 'KURT', 'IV_RATIO', 'SMIRK']
                surface_coverage = results_df[surface_features].notna().all(axis=1).sum()
                print(f"  Option Surface Features - Coverage: {surface_coverage}/{len(results_df)} events ({surface_coverage/len(results_df)*100:.1f}%)")
                
                # Show individual feature coverage
                for feature in surface_features:
                    feature_coverage = results_df[feature].notna().sum()
                    print(f"    {feature}: {feature_coverage}/{len(results_df)} ({feature_coverage/len(results_df)*100:.1f}%)")
            
            # Correlation analysis
            correlation = results_df['revr'].corr(results_df['ievr'])
            print(f"  Correlation (REVR vs IEVR): {correlation:.3f}")
            
            # New correlation analysis
            if 'analyst_dispersion' in results_df.columns:
                disp_corr = results_df['revr'].corr(results_df['analyst_dispersion'])
                print(f"  Correlation (REVR vs Analyst Dispersion): {disp_corr:.3f}" if not np.isnan(disp_corr) else "  Correlation (REVR vs Analyst Dispersion): Not available")
            
            # Option surface features correlation analysis
            if 'TERM_RATIO' in results_df.columns:
                print(f"\n  Option Surface Features Correlations:")
                for feature in ['TERM_RATIO', 'SKEW', 'KURT', 'IV_RATIO', 'SMIRK']:
                    if feature in results_df.columns:
                        feature_corr = results_df['revr'].corr(results_df[feature])
                        print(f"    REVR vs {feature}: {feature_corr:.3f}" if not np.isnan(feature_corr) else f"    REVR vs {feature}: Not available")
            
            return results_df
        
        return None
    
    def plot_results(self, results_df):
        """
        Plot the results of the automated analysis.
        """
        if results_df is None or results_df.empty:
            print("No results to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: REVR over time
        ax1.plot(results_df['earnings_date'], results_df['revr'], 'bo-', linewidth=2, markersize=6)
        ax1.set_title(f'REVR Over Time')
        ax1.set_ylabel('REVR')
        ax1.set_xlabel('Earnings Date')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Change')
        ax1.legend()
        
        # Plot 2: IEVR over time
        ax2.plot(results_df['earnings_date'], results_df['ievr'], 'go-', linewidth=2, markersize=6)
        ax2.set_title(f'IEVR Over Time')
        ax2.set_ylabel('IEVR')
        ax2.set_xlabel('Earnings Date')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Change')
        ax2.legend()
        
        # Plot 3: REVR vs IEVR scatter
        ax3.scatter(results_df['revr'], results_df['ievr'], alpha=0.7, s=60)
        ax3.set_xlabel('REVR (Realized)')
        ax3.set_ylabel('IEVR (Implied)')
        ax3.set_title(f'REVR vs IEVR')
        ax3.grid(True, alpha=0.3)
        
        # Add 1:1 line
        min_val = min(results_df['revr'].min(), results_df['ievr'].min())
        max_val = max(results_df['revr'].max(), results_df['ievr'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='1:1 Line')
        ax3.legend()
        
        # Plot 4: Ratio over time
        ax4.plot(results_df['earnings_date'], results_df['ratio'], 'mo-', linewidth=2, markersize=6)
        ax4.set_title(f'IEVR/REVR Ratio Over Time')
        ax4.set_ylabel('Ratio (IEVR/REVR)')
        ax4.set_xlabel('Earnings Date')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Prediction')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nVisualization complete!")

def main():
    """
    Main function to run automated analysis for AAPL earnings events.
    """
    print("AUTOMATED REVR AND IEVR ANALYSIS")
    print("="*80)
    
    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="sami_sellami",
                           password="xampok-9Hezfy-cahveq")
        print("✓ Connected to WRDS")
        
        # Initialize analysis
        analyzer = AutomatedEarningsAnalysis(db)
        
        # Analyze AAPL earnings events from 2020-2023
        results_df = analyzer.analyze_multiple_events(
            ticker='AAPL',
            start_date='2020-01-01',
            end_date='2023-12-31',
            analysis_days_before=30
        )
        
        if results_df is not None:
            # Plot the results
            analyzer.plot_results(results_df)
            
            print(f"\n✓ Automated analysis completed successfully!")
            print(f"  Analyzed {len(results_df)} earnings events")
            
        else:
            print("✗ Automated analysis failed")
        
        # Close connection
        db.close()
        print("\n✓ Database connection closed")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 