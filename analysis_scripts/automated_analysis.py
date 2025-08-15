#!/usr/bin/env python3
"""
Automated REVR and IEVR Analysis for Multiple Earnings Events
Calculate both measures for all AAPL earnings events and create Events × 2 matrix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import wrds
from revr_analysis import REVRAnalysis
from ievr_analysis import IEVRAnalysis
from option_surface_features import compute_option_surface_features, compute_option_surface_features_no_quarter_filter

class AutomatedEarningsAnalysis:
    """
    Automated analysis of REVR and IEVR for multiple earnings events.
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.revr_analyzer = REVRAnalysis(db_connection)
        self.ievr_analyzer = IEVRAnalysis(db_connection)  # Pass the database connection
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
                
            secid = sec_info.iloc[0]['secid']
            
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
                return price_data.iloc[0]['close']
            else:
                return None
                
        except Exception as e:
            print(f"Error getting stock price: {e}")
            return None
    
    def get_options_surface_features(self, ticker, earnings_date, analysis_days_before=30):
        """
        Get options surface features for a specific earnings event.
        Uses the same logic as IEVR - direct calculation for the specific earnings date.
        """
        try:
            # Get secid for the ticker (same as IEVR)
            sec_query = f"""
            SELECT secid, ticker
            FROM optionm.secnmd
            WHERE ticker = '{ticker}'
            ORDER BY effect_date DESC
            LIMIT 1
            """
            sec_info = self.db.raw_sql(sec_query)
            
            if sec_info.empty:
                print(f"  ⚠ No secid found for {ticker}")
                return None
                
            secid = sec_info.iloc[0]['secid']
            print(f"  Found secid {secid} for {ticker}")
            
            # Try multiple lag days to handle calendar vs trading date issues
            max_attempts = 10  # Try up to 10 different lag days
            for attempt in range(max_attempts):
                current_lag = analysis_days_before - attempt
                if current_lag < 20:  # Don't go too far back
                    break
                    
                print(f"  Attempt {attempt + 1}: Trying {current_lag} days before earnings...")
                
                # Calculate options surface features directly (same approach as IEVR)
                try:
                    # Get surface date
                    from option_surface_features import get_relative_surface_date
                    surface_date = get_relative_surface_date(secid, earnings_date, current_lag, self.db)
                    
                    if surface_date is None:
                        print(f"    No surface date found for {current_lag} days lag")
                        continue
                    
                    print(f"    Surface date: {surface_date}")
                    
                    # Calculate each feature directly
                    from option_surface_features import (
                        extract_term_diff_feature, 
                        extract_skew_feature, 
                        extract_kurtosis_feature,
                        monthly_iv_change_ratio_feature,
                        extract_smirk_feature
                    )
                    
                    # TERM_RATIO
                    term_ratio, _ = extract_term_diff_feature(secid, earnings_date, self.db, current_lag)
                    
                    # SKEW
                    skew, _ = extract_skew_feature(secid, earnings_date, self.db, current_lag)
                    
                    # KURT
                    kurt, _ = extract_kurtosis_feature(secid, earnings_date, self.db, current_lag)
                    
                    # IV_RATIO
                    iv_ratio, _, _ = monthly_iv_change_ratio_feature(secid, earnings_date, self.db, current_lag)
                    
                    # SMIRK
                    smirk, _ = extract_smirk_feature(secid, earnings_date, self.db, current_lag)
                    if isinstance(smirk, tuple):
                        smirk = smirk[0]
                    
                    # Check if we got any valid features
                    features = {
                        'TERM_RATIO': term_ratio,
                        'SKEW': skew,
                        'KURT': kurt,
                        'IV_RATIO': iv_ratio,
                        'SMIRK': smirk
                    }
                    
                    valid_features = {k: v for k, v in features.items() if v is not None}
                    
                    if valid_features:
                        print(f"  ✓ Found options surface features using {current_lag} days lag")
                        print(f"    Valid features: {list(valid_features.keys())}")
                        return features
                    else:
                        print(f"    No valid features found for {current_lag} days lag")
                        
                except Exception as e:
                    print(f"    Error calculating features for {current_lag} days lag: {e}")
                    continue
            
            print(f"  ⚠ Could not find options surface features after {max_attempts} attempts")
            
            # Final fallback: Try to calculate basic features from IEVR data
            print(f"  Trying fallback calculation...")
            return self.get_options_surface_features_fallback(ticker, earnings_date, analysis_days_before)
            
        except Exception as e:
            print(f"Error computing options surface features for {ticker}: {e}")
            # Fallback: Try to calculate basic features from IEVR data
            print(f"  Trying fallback calculation...")
            return self.get_options_surface_features_fallback(ticker, earnings_date, analysis_days_before)
    
    def get_options_surface_features_fallback(self, ticker, earnings_date, analysis_days_before=30):
        """
        Fallback method to calculate basic options surface features using IEVR data source.
        """
        try:
            # Use the same data source as IEVR (keep original behavior)
            analysis_date = earnings_date - timedelta(days=analysis_days_before)
            
            # Get IV surface data (same as IEVR)
            from ievr_analysis import DirectIVData
            surface = DirectIVData(ticker, analysis_date.strftime('%Y-%m-%d'), None, self.db)
            iv_surface_data = surface.fetch_iv_data()
            
            if iv_surface_data is None or iv_surface_data.empty:
                print(f"  Fallback: No IV data available for {ticker}")
                return None
            
            # Calculate basic features from the IV surface
            features = {}
            
            # TERM_RATIO: 30-day IV / 10-day IV
            atm_30 = iv_surface_data[(iv_surface_data['tte'] == 30) & 
                                   (iv_surface_data['moneyness'].between(0.98, 1.02))]
            atm_10 = iv_surface_data[(iv_surface_data['tte'] == 10) & 
                                   (iv_surface_data['moneyness'].between(0.98, 1.02))]
            
            if not atm_30.empty and not atm_10.empty:
                iv_30 = atm_30['put_iv'].mean()
                iv_10 = atm_10['put_iv'].mean()
                if iv_10 > 0:
                    features['TERM_RATIO'] = iv_30 / iv_10
            
            # SKEW: (Call OTM - Put OTM) / ATM
            call_otm = iv_surface_data[(iv_surface_data['tte'] == 30) & 
                                     (iv_surface_data['moneyness'] > 1.05)]
            put_otm = iv_surface_data[(iv_surface_data['tte'] == 30) & 
                                    (iv_surface_data['moneyness'] < 0.95)]
            atm = iv_surface_data[(iv_surface_data['tte'] == 30) & 
                                (iv_surface_data['moneyness'].between(0.98, 1.02))]
            
            if not call_otm.empty and not put_otm.empty and not atm.empty:
                call_iv = call_otm['call_iv'].mean() if 'call_iv' in call_otm.columns else call_otm['put_iv'].mean()
                put_iv = put_otm['put_iv'].mean()
                atm_iv = atm['put_iv'].mean()
                if atm_iv > 0:
                    features['SKEW'] = (call_iv - put_iv) / atm_iv
            
            # Basic SMIRK: Put OTM / Call ATM
            if not put_otm.empty and not atm.empty:
                put_iv = put_otm['put_iv'].mean()
                atm_iv = atm['put_iv'].mean()
                if atm_iv > 0:
                    features['SMIRK'] = put_iv / atm_iv - 1
            
            # IV_RATIO: Current IV / Historical average (simplified)
            if not atm.empty:
                features['IV_RATIO'] = 1.0  # Placeholder
            
            # KURT: Simplified kurtosis measure
            if not atm.empty:
                features['KURT'] = 0.0  # Placeholder
            
            if features:
                print(f"  Fallback: Calculated {len(features)} basic features")
                return features
            else:
                print(f"  Fallback: Could not calculate any features")
                return None
                
        except Exception as e:
            print(f"  Fallback calculation failed: {e}")
            return None
    
    def analyze_single_event(self, ticker, earnings_date, analysis_days_before=30):
        """
        Analyze a single earnings event for both REVR and IEVR.
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
        
        # Calculate options surface features FIRST
        print(f"\n3. Calculating Options Surface Features...")
        surface_features = self.get_options_surface_features(ticker, earnings_date, analysis_days_before)
        
        if surface_features is not None:
            print(f"✓ Options surface features calculated successfully")
            print(f"  TERM_RATIO: {surface_features.get('TERM_RATIO', 'N/A')}")
            print(f"  SKEW: {surface_features.get('SKEW', 'N/A')}")
            print(f"  KURT: {surface_features.get('KURT', 'N/A')}")
            print(f"  IV_RATIO: {surface_features.get('IV_RATIO', 'N/A')}")
            print(f"  SMIRK: {surface_features.get('SMIRK', 'N/A')}")
        else:
            print(f"⚠ Options surface features calculation failed or returned no data")
        
        # Calculate IEVR (now with options surface context available)
        print(f"\n4. Calculating IEVR...")
        
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
        
        # Combine results (updated for ST/MT methodology)
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
            'normative_implied_vol': ievr_results.get('normative_implied_vol', None),  # Added
            'normative_realized_vol': revr_results.get('normative_realized_vol', None),  # Added
            'skew_ratio': ievr_results.get('skew_ratio', None),  # Added skew ratio
            'spx_ievr': ievr_results.get('spx_ievr', None),  # Added S&P 500 IEVR
            'term_ratio': surface_features.get('TERM_RATIO', None) if surface_features else None,
            'skew': surface_features.get('SKEW', None) if surface_features else None,
            'kurt': surface_features.get('KURT', None) if surface_features else None,
            'iv_ratio': surface_features.get('IV_RATIO', None) if surface_features else None,
            'smirk': surface_features.get('SMIRK', None) if surface_features else None,
            'underlying_price': underlying_price,
            'methodology': 'ST/MT Ratio (Expanding EWM)'
        }
        
        print(f"\n✓ Event Analysis Complete:")
        print(f"  REVR: {event_results['revr']:.3f}")
        print(f"  IEVR: {event_results['ievr']:.3f}")
        print(f"  S&P 500 IEVR: {event_results['spx_ievr']:.3f}" if event_results['spx_ievr'] is not None else "  S&P 500 IEVR: Not calculated")
        print(f"  Ratio (IEVR/REVR): {event_results['ievr']/event_results['revr']:.3f}")
        
        # Show options surface features summary
        if surface_features is not None:
            print(f"  Options Surface Features:")
            print(f"    TERM_RATIO: {event_results.get('term_ratio', 'N/A')}")
            print(f"    SKEW: {event_results.get('skew', 'N/A')}")
            print(f"    KURT: {event_results.get('kurt', 'N/A')}")
            print(f"    IV_RATIO: {event_results.get('iv_ratio', 'N/A')}")
            print(f"    SMIRK: {event_results.get('smirk', 'N/A')}")
        else:
            print(f"  Options Surface Features: Not available")
        
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
            # Create Events × 2 matrix (updated for ST/MT methodology)
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
                    'normative_implied_vol': event.get('normative_implied_vol', None),  # Added
                    'normative_realized_vol': event.get('normative_realized_vol', None),  # Added
                    'skew_ratio': event.get('skew_ratio', None),  # Added skew ratio
                    'spx_ievr': event.get('spx_ievr', None),  # Added S&P 500 IEVR
                    'term_ratio': event.get('term_ratio', None),  # Added options surface features
                    'skew': event.get('skew', None),
                    'kurt': event.get('kurt', None),
                    'iv_ratio': event.get('iv_ratio', None),
                    'smirk': event.get('smirk', None),
                    'underlying_price': event.get('underlying_price', None),
                    'methodology': event.get('methodology', 'ST/MT Ratio (Expanding EWM)')
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
            
            # Options surface features summary
            surface_features = ['term_ratio', 'skew', 'kurt', 'iv_ratio', 'smirk']
            print(f"\nOptions Surface Features Summary:")
            for feature in surface_features:
                if feature in results_df.columns:
                    non_null_count = results_df[feature].notna().sum()
                    if non_null_count > 0:
                        mean_val = results_df[feature].mean()
                        std_val = results_df[feature].std()
                        print(f"  {feature.upper()} - Mean: {mean_val:.3f}, Std: {std_val:.3f} ({non_null_count}/{len(results_df)} events)")
                    else:
                        print(f"  {feature.upper()} - No data available")
                else:
                    print(f"  {feature.upper()} - Column not found")
            
            # Correlation analysis
            correlation = results_df['revr'].corr(results_df['ievr'])
            print(f"\nCorrelation Analysis:")
            print(f"  Correlation (REVR vs IEVR): {correlation:.3f}")
            
            # Options surface features correlations with IEVR
            if 'ievr' in results_df.columns:
                print(f"  IEVR Correlations with Options Features:")
                for feature in surface_features:
                    if feature in results_df.columns:
                        corr_val = results_df['ievr'].corr(results_df[feature])
                        if not pd.isna(corr_val):
                            print(f"    IEVR vs {feature.upper()}: {corr_val:.3f}")
            
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