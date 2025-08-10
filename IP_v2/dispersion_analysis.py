#!/usr/bin/env python3
"""
Simplified dispersion analysis script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
import wrds
import statsmodels.api as sm

def main():
    print("DISPERSION-ENHANCED EARNINGS VOLATILITY ANALYSIS")
    print("="*80)
    
    # Load existing data
    print("Loading existing earnings analysis data...")
    existing_data = pd.read_csv('analysis_scripts/data_files/expanded_earnings_analysis_results.csv')
    existing_data['earnings_date'] = pd.to_datetime(existing_data['earnings_date'])
    
    print(f"Loaded {len(existing_data)} existing observations")
    print(f"Date range: {existing_data['earnings_date'].min()} to {existing_data['earnings_date'].max()}")
    print(f"Stocks: {existing_data['ticker'].nunique()}")
    
    # Get date range
    start_date = existing_data['earnings_date'].min().strftime('%Y-%m-%d')
    end_date = existing_data['earnings_date'].max().strftime('%Y-%m-%d')
    
    print(f"Getting dispersion data from {start_date} to {end_date}")
    
    # Connect to WRDS
    try:
        db = wrds.Connection()
        print("✓ Connected to WRDS")
    except Exception as e:
        print(f"✗ Error connecting to WRDS: {e}")
        return
    
    try:
        # Get S&P 500 constituents
        print("Getting S&P 500 constituents...")
        sp500_query = """
            SELECT *
            FROM comp_na_daily_all.wrds_idx_cst_current t
            WHERE indexname = 'S&P 500'
        """
        sp500_constituents = db.raw_sql(sp500_query)
        print(f"Retrieved {len(sp500_constituents)} S&P 500 constituents")
        
        # Get earnings data for S&P 500 constituents
        print("Getting earnings data...")
        gvkey_list = sp500_constituents['gvkey'].tolist()
        formatted_gvkeys = "', '".join(gvkey_list)
        
        earnings_query = f"""
        SELECT gvkey, cusip,
               tic as ticker,
               datadate,
               rdq as earnings_date,
               fyearq,
               fqtr
        FROM comp.fundq
        WHERE gvkey IN ('{formatted_gvkeys}')
          AND rdq BETWEEN '{start_date}' AND '{end_date}'
          AND rdq IS NOT NULL
        ORDER BY ticker, rdq
        """
        earnings_df = db.raw_sql(earnings_query)
        print(f"Retrieved {len(earnings_df)} earnings events")
        
        # Extract CUSIPs for IBES
        print("Processing CUSIPs for IBES...")
        earnings_df['cusip8'] = earnings_df['cusip'].str[:8]
        cusip_list = earnings_df['cusip8'].dropna().unique().tolist()
        
        # Get IBES estimates
        print("Getting IBES estimates...")
        cusip_list_str = ', '.join(f"'{cusip}'" for cusip in cusip_list)
        
        ibes_query = f"""
            SELECT ticker, cusip, statpers, fpedats, anndats_act,
                   meanest, stdev, numest, fpi
            FROM tr_ibes.statsum_epsus
            WHERE cusip IN ({cusip_list_str})
              AND statpers BETWEEN '{start_date}' AND '{end_date}'
              AND measure = 'EPS'
              AND fiscalp = 'QTR'
        """
        ibes_estimates = db.raw_sql(ibes_query, date_cols=['statpers', 'fpedats', 'anndats_act'])
        print(f"Retrieved {len(ibes_estimates)} IBES estimates")
        
        # Filter for one-quarter-ahead estimates
        print("Filtering for one-quarter-ahead estimates...")
        ibes_estimates['statpers'] = pd.to_datetime(ibes_estimates['statpers'])
        ibes_estimates['fpedats'] = pd.to_datetime(ibes_estimates['fpedats'])
        ibes_estimates['anndats_act'] = pd.to_datetime(ibes_estimates['anndats_act'])

        mask_future = (ibes_estimates['fpedats'] > ibes_estimates['statpers']) & (
            ibes_estimates['anndats_act'].isna() | (ibes_estimates['anndats_act'] > ibes_estimates['statpers'])
        )
        ibes_future = ibes_estimates[mask_future].copy()

        ibes_filtered = (
            ibes_future.sort_values(['cusip', 'statpers', 'fpedats'])
            .groupby(['cusip', 'statpers'], as_index=False)
            .first()
        )

        ibes_filtered['dispersion'] = ibes_filtered['stdev'] / ibes_filtered['meanest'].abs()
        print(f"Filtered to {len(ibes_filtered)} one-quarter-ahead estimates")
        
        # Create dispersion panel
        print("Creating dispersion panel...")
        ibes_filtered['statpers'] = pd.to_datetime(ibes_filtered['statpers'])
        full_dates = pd.date_range(ibes_filtered['statpers'].min(), ibes_filtered['statpers'].max(), freq='B')
        cusips = ibes_filtered['cusip'].unique()

        panel_index = pd.MultiIndex.from_product([cusips, full_dates], names=['cusip', 'date'])
        expanded_df = pd.DataFrame(index=panel_index).reset_index()

        df_renamed = ibes_filtered.rename(columns={'statpers': 'date'})
        dispersion_panel = pd.merge(expanded_df, df_renamed[['cusip', 'date', 'dispersion']], 
                                   on=['cusip', 'date'], how='left')

        dispersion_panel['dispersion'] = dispersion_panel.groupby('cusip')['dispersion'].ffill()
        
        # Merge dispersion with earnings
        print("Merging dispersion with earnings...")
        earnings_df['earnings_date'] = pd.to_datetime(earnings_df['earnings_date'])
        earnings_df['cusip8'] = earnings_df['cusip'].str[:8]
        earnings_df['dispersion_lookup_date'] = earnings_df['earnings_date'] - BDay(21)

        dispersion_merged = pd.merge(
            earnings_df,
            dispersion_panel,
            left_on=['cusip8', 'dispersion_lookup_date'],
            right_on=['cusip', 'date'],
            how='left'
        )
        
        # Clean up and prepare for final merge
        dispersion_clean = dispersion_merged[['ticker', 'earnings_date', 'dispersion']].copy()
        dispersion_clean['earnings_date'] = pd.to_datetime(dispersion_clean['earnings_date'])
        
        # Merge with existing data
        print("Merging with existing analysis data...")
        merged_data = pd.merge(
            existing_data,
            dispersion_clean,
            on=['ticker', 'earnings_date'],
            how='left'
        )
        
        # Check merge results
        dispersion_available = merged_data['dispersion'].notna().sum()
        print(f"✓ Successfully merged dispersion data for {dispersion_available} observations")
        print(f"  Total observations: {len(merged_data)}")
        print(f"  Observations with dispersion: {dispersion_available}")
        print(f"  Observations missing dispersion: {len(merged_data) - dispersion_available}")
        
        # Save merged data
        output_file = 'analysis_scripts/data_files/earnings_analysis_with_dispersion.csv'
        merged_data.to_csv(output_file, index=False)
        print(f"✓ Saved merged data to {output_file}")
        
        # Run enhanced regressions
        print("\n" + "="*80)
        print("RUNNING ENHANCED REGRESSIONS WITH DISPERSION")
        print("="*80)
        
        # Filter for observations with dispersion
        dispersion_data = merged_data[merged_data['dispersion'].notna()].copy()
        
        if len(dispersion_data) < 10:
            print("✗ Insufficient data with dispersion for regression")
            return
        
        print(f"Running regressions on {len(dispersion_data)} observations with dispersion data")
        
        # Convert all data types to numeric with more robust conversion
        numeric_columns = ['revr', 'ievr', 'dispersion', 'skew_ratio', 'normative_implied_vol', 'normative_realized_vol', 'ratio']
        for col in numeric_columns:
            if col in dispersion_data.columns:
                dispersion_data[col] = pd.to_numeric(dispersion_data[col], errors='coerce')
                dispersion_data[col] = dispersion_data[col].astype('float64')
        
        # Create normative IV/RV ratio
        if 'normative_implied_vol' in dispersion_data.columns and 'normative_realized_vol' in dispersion_data.columns:
            dispersion_data['normative_iv_rv_ratio'] = dispersion_data['normative_implied_vol'] / dispersion_data['normative_realized_vol']
            dispersion_data['normative_iv_rv_ratio'] = pd.to_numeric(dispersion_data['normative_iv_rv_ratio'], errors='coerce')
            print("✓ Created normative_iv_rv_ratio")
        
        # Create finite mask for all variables
        finite_mask = np.isfinite(dispersion_data['revr']) & np.isfinite(dispersion_data['ievr']) & np.isfinite(dispersion_data['dispersion'])
        clean_data = dispersion_data[finite_mask].copy()
        
        print(f"Data types after conversion:")
        for col in ['revr', 'ievr', 'dispersion', 'skew_ratio', 'normative_iv_rv_ratio', 'ratio']:
            if col in clean_data.columns:
                print(f"  {col.upper()}: {clean_data[col].dtype}")
        print(f"  Clean observations: {len(clean_data)}")
        
        # MODEL 1: REVR = α + β₁×IEVR + β₂×Dispersion
        print(f"\nMODEL 1: REVR = α + β₁×IEVR + β₂×Dispersion")
        print(f"{'='*50}")
        
        if len(clean_data) >= 10:
            X1 = pd.DataFrame({
                'const': 1,
                'ievr': clean_data['ievr'].values,
                'dispersion': clean_data['dispersion'].values
            })
            y = clean_data['revr'].values
            
            model1 = sm.OLS(y, X1).fit()
            print(model1.summary())
            
            # MODEL 2: REVR = α + β₁×IEVR + β₂×Dispersion + β₃×Skew + β₄×Normative_IV_RV
            print(f"\nMODEL 2: REVR = α + β₁×IEVR + β₂×Dispersion + β₃×Skew + β₄×Normative_IV_RV")
            print(f"{'='*65}")
            
            # Check which additional features are available (excluding ratio)
            available_features = []
            feature_names = []
            
            if 'skew_ratio' in clean_data.columns and clean_data['skew_ratio'].notna().sum() > 0:
                available_features.append(clean_data['skew_ratio'].values)
                feature_names.append('skew_ratio')
            
            if 'normative_iv_rv_ratio' in clean_data.columns and clean_data['normative_iv_rv_ratio'].notna().sum() > 0:
                available_features.append(clean_data['normative_iv_rv_ratio'].values)
                feature_names.append('normative_iv_rv_ratio')
            
            # Create comprehensive X matrix
            X2_dict = {
                'const': 1,
                'ievr': clean_data['ievr'].values,
                'dispersion': clean_data['dispersion'].values
            }
            
            for i, feature in enumerate(available_features):
                X2_dict[feature_names[i]] = feature
            
            X2 = pd.DataFrame(X2_dict)
            
            # Check for any infinite values
            X2 = X2.replace([np.inf, -np.inf], np.nan)
            X2 = X2.dropna()
            
            if len(X2) >= 10:
                y2 = clean_data['revr'].iloc[:len(X2)].values
                
                model2 = sm.OLS(y2, X2).fit()
                print(model2.summary())
                
                # Save comprehensive results
                results = {
                    'model': 'REVR = α + β₁×IEVR + β₂×Dispersion + β₃×Skew + β₄×Normative_IV_RV',
                    'ievr_coef': model2.params['ievr'],
                    'ievr_tstat': model2.tvalues['ievr'],
                    'ievr_pvalue': model2.pvalues['ievr'],
                    'dispersion_coef': model2.params['dispersion'],
                    'dispersion_tstat': model2.tvalues['dispersion'],
                    'dispersion_pvalue': model2.pvalues['dispersion'],
                    'r_squared': model2.rsquared,
                    'adj_r_squared': model2.rsquared_adj,
                    'nobs': model2.nobs
                }
                
                # Add other coefficients if available
                for feature in feature_names:
                    if feature in model2.params:
                        results[f'{feature}_coef'] = model2.params[feature]
                        results[f'{feature}_tstat'] = model2.tvalues[feature]
                        results[f'{feature}_pvalue'] = model2.pvalues[feature]
                
                results_df = pd.DataFrame([results])
                results_df.to_csv('analysis_scripts/data_files/comprehensive_regression_results.csv', index=False)
                print(f"✓ Comprehensive results saved to analysis_scripts/data_files/comprehensive_regression_results.csv")
                
                # Print key findings
                print(f"\n{'='*80}")
                print(f"COMPREHENSIVE MODEL KEY FINDINGS")
                print(f"{'='*80}")
                print(f"IEVR coefficient: {model2.params['ievr']:.4f} (t={model2.tvalues['ievr']:.3f}, p={model2.pvalues['ievr']:.4f})")
                print(f"Dispersion coefficient: {model2.params['dispersion']:.4f} (t={model2.tvalues['dispersion']:.3f}, p={model2.pvalues['dispersion']:.4f})")
                
                for feature in feature_names:
                    if feature in model2.params:
                        print(f"{feature.upper()} coefficient: {model2.params[feature]:.4f} (t={model2.tvalues[feature]:.3f}, p={model2.pvalues[feature]:.4f})")
                
                print(f"R-squared: {model2.rsquared:.4f}")
                print(f"Adjusted R-squared: {model2.rsquared_adj:.4f}")
                print(f"Observations: {model2.nobs}")
                
                # Significance tests
                print(f"\nSIGNIFICANCE TESTS (α = 0.05):")
                print(f"{'='*40}")
                if model2.pvalues['ievr'] < 0.05:
                    print("✓ IEVR coefficient is significant (p < 0.05)")
                else:
                    print("✗ IEVR coefficient is not significant (p >= 0.05)")
                    
                if model2.pvalues['dispersion'] < 0.05:
                    print("✓ Dispersion coefficient is significant (p < 0.05)")
                else:
                    print("✗ Dispersion coefficient is not significant (p >= 0.05)")
                
                for feature in feature_names:
                    if feature in model2.pvalues:
                        if model2.pvalues[feature] < 0.05:
                            print(f"✓ {feature.upper()} coefficient is significant (p < 0.05)")
                        else:
                            print(f"✗ {feature.upper()} coefficient is not significant (p >= 0.05)")
            else:
                print("✗ Insufficient data for comprehensive model")
        else:
            print("✗ Insufficient data for regression")
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"✓ Dispersion data retrieved and merged")
        print(f"✓ Enhanced regressions completed")
        print(f"✓ Results saved to analysis_scripts/data_files/")
        
    except Exception as e:
        print(f"✗ Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()