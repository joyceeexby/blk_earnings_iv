#!/usr/bin/env python3
"""
Fixed Regression Analysis for REVR and IEVR
Handle NaN values and multicollinearity issues

Updated for new REVR methodology:
- REVR now calculated as post-earnings avg vol / pre-earnings avg vol
- Uses 7-day half-life exponential weighted averages
- Pre-earnings period: start until T-1 (day before earnings)
- Post-earnings period: T+1 (day after earnings) until 1 month after
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

class FixedRegressionAnalysis:
    """
    Fixed regression analysis for REVR and IEVR.
    """
    
    def __init__(self, data_file='data_files/multi_stock_results.csv'):
        """
        Initialize with the multi-stock results data.
        """
        self.data = pd.read_csv(data_file)
        self.data['earnings_date'] = pd.to_datetime(self.data['earnings_date'])
        
        # Add sector mapping
        self.data['sector'] = self.data['ticker'].map(self.ticker_to_sector())
        
        # Clean data
        self.clean_data()
        
        # Create additional variables
        self.create_variables()
        
        print(f"Loaded {len(self.data)} observations from {self.data['ticker'].nunique()} stocks")
        print(f"Date range: {self.data['earnings_date'].min()} to {self.data['earnings_date'].max()}")
        print(f"Sectors: {self.data['sector'].nunique()} - {sorted(self.data['sector'].dropna().unique())}")
    
    @staticmethod
    def ticker_to_sector():
        """Hardcoded mapping of ticker to sector for comprehensive stock list."""
        return {
            # Technology (15 stocks)
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology', 'TSLA': 'Technology',
            'META': 'Technology', 'NVDA': 'Technology', 'NFLX': 'Technology', 'ADBE': 'Technology', 'CRM': 'Technology',
            'INTC': 'Technology', 'AMD': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology', 'AVGO': 'Technology',
            
            # Financial (15 stocks)
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'GS': 'Financial', 'MS': 'Financial',
            'C': 'Financial', 'USB': 'Financial', 'BLK': 'Financial', 'SCHW': 'Financial', 'AXP': 'Financial',
            'COF': 'Financial', 'PNC': 'Financial', 'TFC': 'Financial', 'KEY': 'Financial', 'RF': 'Financial',
            
            # Healthcare (15 stocks)
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare', 'MRK': 'Healthcare',
            'TMO': 'Healthcare', 'ABT': 'Healthcare', 'DHR': 'Healthcare', 'BMY': 'Healthcare', 'AMGN': 'Healthcare',
            'GILD': 'Healthcare', 'CVS': 'Healthcare', 'VRTX': 'Healthcare', 'REGN': 'Healthcare', 'LLY': 'Healthcare',
            
            # Consumer Discretionary (10 stocks)
            'HD': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary',
            'TJX': 'Consumer Discretionary', 'MAR': 'Consumer Discretionary', 'BKNG': 'Consumer Discretionary', 'CMG': 'Consumer Discretionary', 'YUM': 'Consumer Discretionary',
            
            # Consumer Staples (8 stocks)
            'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples', 'WMT': 'Consumer Staples', 'COST': 'Consumer Staples',
            'PM': 'Consumer Staples', 'MO': 'Consumer Staples', 'CL': 'Consumer Staples',
            
            # Industrial (8 stocks)
            'BA': 'Industrial', 'CAT': 'Industrial', 'GE': 'Industrial', 'MMM': 'Industrial', 'HON': 'Industrial',
            'UPS': 'Industrial', 'FDX': 'Industrial', 'RTX': 'Industrial',
            
            # Energy (6 stocks)
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy', 'SLB': 'Energy', 'PSX': 'Energy',
            
            # Communication Services (5 stocks)
            'DIS': 'Communication Services', 'CMCSA': 'Communication Services', 'VZ': 'Communication Services', 'T': 'Communication Services', 'TMUS': 'Communication Services',
            
            # Materials (4 stocks)
            'LIN': 'Materials', 'APD': 'Materials', 'FCX': 'Materials', 'NEM': 'Materials',
            
            # Real Estate (4 stocks)
            'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate',
            
            # Utilities (4 stocks)
            'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
        }

    def clean_data(self):
        """
        Clean the data by removing NaN values, infinite values, and outliers.
        Updated for ST/MT REVR methodology.
        """
        print("Cleaning data...")
        
        initial_count = len(self.data)
        
        # Remove rows with NaN values in key variables
        # Updated for ST/MT methodology
        key_vars = ['revr', 'ievr']
        
        # Add ST/MT methodology variables if they exist
        if 'vol_st' in self.data.columns:
            key_vars.append('vol_st')
        if 'vol_mt' in self.data.columns:
            key_vars.append('vol_mt')
        
        # Add old methodology variables if they exist (for backward compatibility)
        old_vars = ['pre_earnings_avg_vol', 'post_earnings_avg_vol', 'vol_t_minus_3', 'vol_t_plus_4', 'kink_tte']
        for var in old_vars:
            if var in self.data.columns:
                key_vars.append(var)
        
        # Only require REVR and IEVR to be non-NaN for analysis
        required_vars = ['revr', 'ievr']
        for var in required_vars:
            if var in self.data.columns:
                self.data = self.data.dropna(subset=[var])
        
        print(f"Removed {initial_count - len(self.data)} rows with NaN values")
        
        # Remove infinite values
        initial_count = len(self.data)
        for var in ['revr', 'ievr']:
            if var in self.data.columns:
                self.data = self.data[np.isfinite(self.data[var])]
        
        print(f"Removed {initial_count - len(self.data)} rows with infinite values")
        
        # Remove extreme outliers (beyond 3 standard deviations)
        for var in ['revr', 'ievr']:
            if var in self.data.columns:
                mean_val = self.data[var].mean()
                std_val = self.data[var].std()
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                
                outliers = self.data[(self.data[var] < lower_bound) | (self.data[var] > upper_bound)]
                if len(outliers) > 0:
                    print(f"Removing {len(outliers)} outliers from {var}")
                    self.data = self.data[(self.data[var] >= lower_bound) & (self.data[var] <= upper_bound)]
    
    def create_variables(self):
        """
        Create additional variables for regression analysis.
        """
        # Time variables
        self.data['year'] = self.data['earnings_date'].dt.year
        self.data['quarter'] = self.data['earnings_date'].dt.quarter
        self.data['month'] = self.data['earnings_date'].dt.month
        
        # Market conditions
        self.data['covid_period'] = (self.data['year'] >= 2020).astype(int)
        self.data['post_covid'] = (self.data['year'] >= 2022).astype(int)
        
        # Volatility measures (only for finite values)
        self.data['log_revr'] = np.where(np.isfinite(self.data['revr']) & (self.data['revr'] > 0), 
                                        np.log(self.data['revr']), np.nan)
        self.data['log_ievr'] = np.where(np.isfinite(self.data['ievr']) & (self.data['ievr'] > 0), 
                                        np.log(self.data['ievr']), np.nan)
        self.data['revr_deviation'] = np.where(np.isfinite(self.data['revr']), 
                                              self.data['revr'] - 1.0, np.nan)
        self.data['ievr_deviation'] = np.where(np.isfinite(self.data['ievr']), 
                                              self.data['ievr'] - 1.0, np.nan)
        
        # Prediction error (only for finite values)
        finite_mask = np.isfinite(self.data['revr']) & np.isfinite(self.data['ievr'])
        self.data['prediction_error'] = np.where(finite_mask, 
                                                self.data['revr'] - self.data['ievr'], np.nan)
        self.data['prediction_error_pct'] = np.where(finite_mask & (self.data['ievr'] != 0), 
                                                    (self.data['revr'] - self.data['ievr']) / self.data['ievr'], np.nan)
        
        # Additional variables for ST/MT REVR methodology
        if 'vol_st' in self.data.columns and 'vol_mt' in self.data.columns:
            # Log transformations for ST/MT volatility measures
            self.data['log_vol_st'] = np.where(
                np.isfinite(self.data['vol_st']) & (self.data['vol_st'] > 0),
                np.log(self.data['vol_st']), np.nan
            )
            self.data['log_vol_mt'] = np.where(
                np.isfinite(self.data['vol_mt']) & (self.data['vol_mt'] > 0),
                np.log(self.data['vol_mt']), np.nan
            )
            
            # Volatility spread (ST - MT)
            self.data['volatility_spread'] = np.where(
                np.isfinite(self.data['vol_st']) & np.isfinite(self.data['vol_mt']),
                self.data['vol_st'] - self.data['vol_mt'], np.nan
            )
            
            # Volatility ratio (ST / MT) - should be same as REVR
            self.data['volatility_ratio'] = np.where(
                np.isfinite(self.data['vol_st']) & np.isfinite(self.data['vol_mt']) & (self.data['vol_mt'] > 0),
                self.data['vol_st'] / self.data['vol_mt'], np.nan
            )
        
        # Create stock dummies (excluding one to avoid multicollinearity)
        tickers = self.data['ticker'].unique()
        for ticker in tickers[:-1]:  # Exclude last ticker
            self.data[f'dummy_{ticker}'] = (self.data['ticker'] == ticker).astype(int)
        
        # Create year dummies (excluding one to avoid multicollinearity)
        years = sorted(self.data['year'].unique())
        for year in years[:-1]:  # Exclude last year
            self.data[f'dummy_year_{year}'] = (self.data['year'] == year).astype(int)
        
        # Add sector dummies (excluding one to avoid multicollinearity)
        if 'sector' in self.data.columns:
            sectors = self.data['sector'].dropna().unique()
            for sector in sectors[:-1]:
                self.data[f'dummy_sector_{sector}'] = (self.data['sector'] == sector).astype(int)
        
        # Add options surface features variables if they exist
        options_features = ['term_ratio', 'skew', 'kurt', 'iv_ratio', 'smirk']
        for feature in options_features:
            if feature in self.data.columns:
                # Log transformation for positive values
                self.data[f'log_{feature}'] = np.where(
                    np.isfinite(self.data[feature]) & (self.data[feature] > 0),
                    np.log(self.data[feature]), np.nan
                )
                
                # Squared term for interaction effects
                self.data[f'{feature}_squared'] = np.where(
                    np.isfinite(self.data[feature]),
                    self.data[feature] ** 2, np.nan
                )
                
                # Interaction with IEVR
                if 'ievr' in self.data.columns:
                    self.data[f'ievr_{feature}'] = np.where(
                        np.isfinite(self.data['ievr']) & np.isfinite(self.data[feature]),
                        self.data['ievr'] * self.data[feature], np.nan
                    )
    
    def descriptive_statistics(self):
        """
        Generate comprehensive descriptive statistics.
        """
        print(f"\n{'='*100}")
        print(f"DESCRIPTIVE STATISTICS")
        print(f"{'='*100}")
        
        # Overall statistics
        print(f"\nOVERALL STATISTICS:")
        base_vars = ['revr', 'ievr', 'ratio', 'prediction_error']
        
        # Add ST/MT methodology variables if they exist
        if 'vol_st' in self.data.columns:
            base_vars.extend(['vol_st', 'vol_mt', 'volatility_spread'])
        
        # Add old methodology variables if they exist (for backward compatibility)
        if 'pre_earnings_avg_vol' in self.data.columns:
            base_vars.extend(['pre_earnings_avg_vol', 'post_earnings_avg_vol'])
        
        # Add options surface features if they exist
        options_features = ['term_ratio', 'skew', 'kurt', 'iv_ratio', 'smirk']
        for feature in options_features:
            if feature in self.data.columns:
                base_vars.append(feature)
        
        print(self.data[base_vars].describe())
        
        # By stock statistics
        print(f"\nBY-STOCK STATISTICS:")
        agg_vars = {
            'revr': ['count', 'mean', 'std', 'min', 'max'],
            'ievr': ['mean', 'std'],
            'prediction_error': ['mean', 'std']
        }
        
        # Add ST/MT methodology variables to aggregation if they exist
        if 'vol_st' in self.data.columns:
            agg_vars.update({
                'vol_st': ['mean', 'std'],
                'vol_mt': ['mean', 'std'],
                'volatility_spread': ['mean', 'std']
            })
        
        # Add old methodology variables to aggregation if they exist
        if 'pre_earnings_avg_vol' in self.data.columns:
            agg_vars.update({
                'pre_earnings_avg_vol': ['mean', 'std'],
                'post_earnings_avg_vol': ['mean', 'std']
            })
        
        stock_stats = self.data.groupby('ticker').agg(agg_vars).round(3)
        
        # Flatten column names
        stock_stats.columns = ['_'.join(col).strip() for col in stock_stats.columns]
        print(stock_stats.to_string())
        
        # Correlation matrix
        print(f"\nCORRELATION MATRIX:")
        corr_vars = ['revr', 'ievr', 'ratio', 'prediction_error']
        
        # Add ST/MT methodology variables to correlation matrix if they exist
        if 'vol_st' in self.data.columns:
            corr_vars.extend(['vol_st', 'vol_mt', 'volatility_spread'])
        
        # Add old methodology variables to correlation matrix if they exist
        if 'pre_earnings_avg_vol' in self.data.columns:
            corr_vars.extend(['pre_earnings_avg_vol', 'post_earnings_avg_vol'])
        
        correlation_matrix = self.data[corr_vars].corr()
        print(correlation_matrix.round(3))
        
        return correlation_matrix
    
    def plot_descriptive_analysis(self):
        """
        Create descriptive analysis plots.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: REVR vs IEVR scatter with regression line
        ax1.scatter(self.data['ievr'], self.data['revr'], alpha=0.6, s=50)
        
        # Add regression line
        z = np.polyfit(self.data['ievr'], self.data['revr'], 1)
        p = np.poly1d(z)
        ax1.plot(self.data['ievr'], p(self.data['ievr']), "r--", alpha=0.8)
        
        ax1.set_xlabel('IEVR (Implied)')
        ax1.set_ylabel('REVR (Realized)')
        ax1.set_title('REVR vs IEVR with Regression Line')
        ax1.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = self.data['revr'].corr(self.data['ievr'])
        ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 2: Prediction error distribution
        ax2.hist(self.data['prediction_error'], bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(self.data['prediction_error'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.data["prediction_error"].mean():.3f}')
        ax2.set_xlabel('Prediction Error (REVR - IEVR)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Prediction Errors')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: REVR by stock (box plot)
        self.data.boxplot(column='revr', by='ticker', ax=ax3)
        ax3.set_title('REVR Distribution by Stock (ST/MT Methodology)')
        ax3.set_ylabel('REVR')
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Plot 4: Prediction error by stock
        self.data.boxplot(column='prediction_error', by='ticker', ax=ax4)
        ax4.set_title('Prediction Error by Stock')
        ax4.set_ylabel('Prediction Error')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('output_files/regression_descriptive_analysis.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Disabled for batch mode
    
    def run_sector_specific_regressions(self):
        """
        Run regression for each sector: Log(REVR) ~ Log(IEVR) + Stock Dummies (and print summary)
        """
        print(f"\n{'='*80}")
        print(f"SECTOR-SPECIFIC LOG REGRESSION ANALYSIS (WITH STOCK DUMMIES)")
        print(f"{'='*80}")
        results = []
        for sector in sorted(self.data['sector'].dropna().unique()):
            sector_data = self.data[self.data['sector'] == sector]
            
            # Use log transformations - only for positive values
            log_mask = (sector_data['revr'] > 0) & (sector_data['ievr'] > 0) & np.isfinite(sector_data['revr']) & np.isfinite(sector_data['ievr'])
            clean_data = sector_data[log_mask].copy()
            
            if len(clean_data) < 10:
                print(f"  {sector}: Not enough data ({len(clean_data)})")
                continue
            
            # Create log variables
            clean_data['log_revr'] = np.log(clean_data['revr'])
            clean_data['log_ievr'] = np.log(clean_data['ievr'])
            
            # Get stock dummies for this sector
            stock_dummies = [col for col in clean_data.columns if col.startswith('dummy_') and not col.startswith('dummy_year_') and not col.startswith('dummy_sector_')]
            
            if stock_dummies:
                # Model with stock dummies: Log(REVR) = α + β × Log(IEVR) + Stock Dummies
                X = clean_data[['log_ievr'] + stock_dummies]
                X = sm.add_constant(X)
                y = clean_data['log_revr']
                model = sm.OLS(y, X).fit()
                
                results.append({
                    'sector': sector,
                    'n_obs': len(clean_data),
                    'n_stocks': clean_data['ticker'].nunique(),
                    'log_ievr_coef': model.params['log_ievr'],
                    'log_ievr_tstat': model.tvalues['log_ievr'],
                    'log_ievr_pval': model.pvalues['log_ievr'],
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj,
                    'model_type': 'Log Model with Stock Dummies'
                })
                
                print(f"  {sector}: {len(clean_data)} obs, {clean_data['ticker'].nunique()} stocks")
                print(f"    Log(IEVR) coef: {model.params['log_ievr']:.4f} (t={model.tvalues['log_ievr']:.3f}, p={model.pvalues['log_ievr']:.3f})")
                print(f"    R²: {model.rsquared:.3f}, Adj R²: {model.rsquared_adj:.3f}")
            else:
                # Fallback to simple log model without dummies
                X = sm.add_constant(clean_data['log_ievr'])
                y = clean_data['log_revr']
                model = sm.OLS(y, X).fit()
                
                results.append({
                    'sector': sector,
                    'n_obs': len(clean_data),
                    'n_stocks': clean_data['ticker'].nunique(),
                    'log_ievr_coef': model.params['log_ievr'],
                    'log_ievr_tstat': model.tvalues['log_ievr'],
                    'log_ievr_pval': model.pvalues['log_ievr'],
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj,
                    'model_type': 'Simple Log Model (No Dummies)'
                })
                
                print(f"  {sector}: {len(clean_data)} obs, {clean_data['ticker'].nunique()} stocks (no dummies)")
                print(f"    Log(IEVR) coef: {model.params['log_ievr']:.4f} (t={model.tvalues['log_ievr']:.3f}, p={model.pvalues['log_ievr']:.3f})")
                print(f"    R²: {model.rsquared:.3f}, Adj R²: {model.rsquared_adj:.3f}")
        
        df = pd.DataFrame(results)
        print(f"\nSector Log Regression Summary:")
        print(df.round(3).to_string(index=False))
        return df

    def run_basic_regressions(self):
        """
        Run basic regression models.
        """
        print(f"\n{'='*100}")
        print(f"BASIC REGRESSION MODELS")
        print(f"{'='*100}")
        
        models = []
        
        # Model 1: REVR on IEVR (basic relationship)
        print(f"\nMODEL 1: REVR = α + β × IEVR")
        print(f"{'='*50}")
        
        try:
            # Ensure we have finite data
            finite_mask = np.isfinite(self.data['revr']) & np.isfinite(self.data['ievr'])
            clean_data = self.data[finite_mask].copy()
            
            if len(clean_data) < 10:
                print("  Error: Insufficient clean data for regression")
                models.append(None)
            else:
                X1 = sm.add_constant(clean_data['ievr'])
                model1 = sm.OLS(clean_data['revr'], X1).fit()
                print(model1.summary())
                models.append(model1)
        except Exception as e:
            print(f"  Error: {e}")
            models.append(None)
        
        # Model 2: REVR on IEVR with stock fixed effects
        print(f"\nMODEL 2: REVR = α + β × IEVR + Stock Fixed Effects")
        print(f"{'='*50}")
        
        try:
            finite_mask = np.isfinite(self.data['revr']) & np.isfinite(self.data['ievr'])
            clean_data = self.data[finite_mask].copy()
            # Add stock and sector dummies
            dummies = [col for col in clean_data.columns if col.startswith('dummy_')]
            X = clean_data[['ievr'] + dummies]
            X = sm.add_constant(X)
            y = clean_data['revr']
            model2 = sm.OLS(y, X).fit()
            print(model2.summary())
            models.append(model2)
        except Exception as e:
            print(f"  Model 2 failed: {e}")
            models.append(None)
        
        # Model 3: REVR on IEVR with time fixed effects
        print(f"\nMODEL 3: REVR = α + β × IEVR + Time Fixed Effects")
        print(f"{'='*50}")
        
        try:
            time_dummies = [col for col in self.data.columns if col.startswith('dummy_year_')]
            if time_dummies:
                X3 = sm.add_constant(clean_data[['ievr'] + time_dummies])
                model3 = sm.OLS(clean_data['revr'], X3).fit()
                print(model3.summary())
                models.append(model3)
            else:
                print("  Error: No time dummies available")
                models.append(None)
        except Exception as e:
            print(f"  Error: {e}")
            models.append(None)
        
        # Model 4: REVR on IEVR with both fixed effects
        print(f"\nMODEL 4: REVR = α + β × IEVR + Stock + Time Fixed Effects")
        print(f"{'='*50}")
        
        try:
            if dummies and time_dummies:
                X4 = sm.add_constant(clean_data[['ievr'] + dummies + time_dummies])
                model4 = sm.OLS(clean_data['revr'], X4).fit()
                print(model4.summary())
                models.append(model4)
            else:
                print("  Error: Missing stock or time dummies")
                models.append(None)
        except Exception as e:
            print(f"  Error: {e}")
            models.append(None)
        
        # Model 5: REVR on IEVR with controls
        print(f"\nMODEL 5: REVR = α + β₁×IEVR + β₂×Controls")
        print(f"{'='*50}")
        
        try:
            # Ensure we have finite data
            finite_mask = np.isfinite(self.data['revr']) & np.isfinite(self.data['ievr'])
            clean_data = self.data[finite_mask].copy()
            
            controls = ['covid_period', 'post_covid']
            if 'vol_t_minus_3' in clean_data.columns:
                controls.append('vol_t_minus_3')
            if 'kink_tte' in clean_data.columns:
                controls.append('kink_tte')
            
            # Remove any controls with NaN values
            valid_controls = []
            for control in controls:
                if control in clean_data.columns and not clean_data[control].isna().all():
                    valid_controls.append(control)
            
            if len(clean_data) < 10:
                print("  Error: Insufficient clean data for regression")
                models.append(None)
            else:
                X5 = sm.add_constant(clean_data[['ievr'] + valid_controls])
                model5 = sm.OLS(clean_data['revr'], X5).fit()
                print(model5.summary())
                models.append(model5)
        except Exception as e:
            print(f"  Error: {e}")
            models.append(None)
        
        # Model 6: Log REVR on Log IEVR
        print(f"\nMODEL 6: Log(REVR) = α + β × Log(IEVR)")
        print(f"{'='*50}")
        
        try:
            # Check for finite log values
            log_finite_mask = np.isfinite(clean_data['log_revr']) & np.isfinite(clean_data['log_ievr'])
            log_clean_data = clean_data[log_finite_mask].copy()
            
            if len(log_clean_data) < 10:
                print("  Error: Insufficient clean log data for regression")
                models.append(None)
            else:
                X6 = sm.add_constant(log_clean_data['log_ievr'])
                model6 = sm.OLS(log_clean_data['log_revr'], X6).fit()
                print(model6.summary())
                models.append(model6)
        except Exception as e:
            print(f"  Error: {e}")
            models.append(None)
        
        # Model 7: Prediction error analysis
        print(f"\nMODEL 7: Prediction Error = α + β × Controls")
        print(f"{'='*50}")
        
        try:
            # Check for finite prediction error
            error_finite_mask = np.isfinite(clean_data['prediction_error'])
            error_clean_data = clean_data[error_finite_mask].copy()
            
            error_controls = ['covid_period', 'post_covid']
            if 'vol_t_minus_3' in error_clean_data.columns:
                error_controls.append('vol_t_minus_3')
            if 'kink_tte' in error_clean_data.columns:
                error_controls.append('kink_tte')
            
            # Remove any controls with NaN values
            valid_error_controls = []
            for control in error_controls:
                if control in error_clean_data.columns and not error_clean_data[control].isna().all():
                    valid_error_controls.append(control)
            
            if len(error_clean_data) < 10:
                print("  Error: Insufficient clean prediction error data for regression")
                models.append(None)
            else:
                X7 = sm.add_constant(error_clean_data[valid_error_controls])
                model7 = sm.OLS(error_clean_data['prediction_error'], X7).fit()
                print(model7.summary())
                models.append(model7)
        except Exception as e:
            print(f"  Error: {e}")
            models.append(None)
        
        return models
    
    def run_extended_regressions(self):
        """
        Run extended regression models with additional controls.
        """
        print(f"\n{'='*100}")
        print(f"EXTENDED REGRESSION MODELS")
        print(f"{'='*100}")
        
        models = []
        
        # Model 5: REVR on IEVR with controls
        print(f"\nMODEL 5: REVR = α + β₁×IEVR + β₂×Controls")
        print(f"{'='*50}")
        
        try:
            # Ensure we have finite data
            finite_mask = np.isfinite(self.data['revr']) & np.isfinite(self.data['ievr'])
            clean_data = self.data[finite_mask].copy()
            
            controls = ['covid_period', 'post_covid']
            if 'vol_t_minus_3' in clean_data.columns:
                controls.append('vol_t_minus_3')
            if 'kink_tte' in clean_data.columns:
                controls.append('kink_tte')
            
            # Remove any controls with NaN values
            valid_controls = []
            for control in controls:
                if control in clean_data.columns and not clean_data[control].isna().all():
                    valid_controls.append(control)
            
            if len(clean_data) < 10:
                print("  Error: Insufficient clean data for regression")
                models.append(None)
            else:
                X5 = sm.add_constant(clean_data[['ievr'] + valid_controls])
                model5 = sm.OLS(clean_data['revr'], X5).fit()
                print(model5.summary())
                models.append(model5)
        except Exception as e:
            print(f"  Error: {e}")
            models.append(None)
        
        # Model 6: Log REVR on Log IEVR
        print(f"\nMODEL 6: Log(REVR) = α + β × Log(IEVR)")
        print(f"{'='*50}")
        
        try:
            # Check for finite log values
            log_finite_mask = np.isfinite(clean_data['log_revr']) & np.isfinite(clean_data['log_ievr'])
            log_clean_data = clean_data[log_finite_mask].copy()
            
            if len(log_clean_data) < 10:
                print("  Error: Insufficient clean log data for regression")
                models.append(None)
            else:
                X6 = sm.add_constant(log_clean_data['log_ievr'])
                model6 = sm.OLS(log_clean_data['log_revr'], X6).fit()
                print(model6.summary())
                models.append(model6)
        except Exception as e:
            print(f"  Error: {e}")
            models.append(None)
        
        # Model 7: Prediction error analysis
        print(f"\nMODEL 7: Prediction Error = α + β × Controls")
        print(f"{'='*50}")
        
        try:
            # Check for finite prediction error
            error_finite_mask = np.isfinite(clean_data['prediction_error'])
            error_clean_data = clean_data[error_finite_mask].copy()
            
            error_controls = ['covid_period', 'post_covid']
            if 'vol_t_minus_3' in error_clean_data.columns:
                error_controls.append('vol_t_minus_3')
            if 'kink_tte' in error_clean_data.columns:
                error_controls.append('kink_tte')
            
            # Remove any controls with NaN values
            valid_error_controls = []
            for control in error_controls:
                if control in error_clean_data.columns and not error_clean_data[control].isna().all():
                    valid_error_controls.append(control)
            
            if len(error_clean_data) < 10:
                print("  Error: Insufficient clean prediction error data for regression")
                models.append(None)
            else:
                X7 = sm.add_constant(error_clean_data[valid_error_controls])
                model7 = sm.OLS(error_clean_data['prediction_error'], X7).fit()
                print(model7.summary())
                models.append(model7)
        except Exception as e:
            print(f"  Error: {e}")
            models.append(None)
        
        # Model 8: REVR on IEVR + Options Surface Features
        print(f"\nMODEL 8: REVR = α + β₁×IEVR + β₂×Options Surface Features")
        print(f"{'='*50}")
        
        try:
            # Check for options surface features
            options_features = ['term_ratio', 'skew', 'kurt', 'iv_ratio', 'smirk']
            available_features = [f for f in options_features if f in clean_data.columns]
            
            if available_features:
                # Check for finite values in options features
                feature_finite_mask = np.isfinite(clean_data[available_features]).all(axis=1)
                feature_clean_data = clean_data[feature_finite_mask].copy()
                
                if len(feature_clean_data) < 10:
                    print("  Error: Insufficient clean options features data for regression")
                    models.append(None)
                else:
                    X8 = sm.add_constant(feature_clean_data[['ievr'] + available_features])
                    model8 = sm.OLS(feature_clean_data['revr'], X8).fit()
                    print(model8.summary())
                    models.append(model8)
            else:
                print("  Error: No options surface features available")
                models.append(None)
        except Exception as e:
            print(f"  Error: {e}")
            models.append(None)
        
        # Model 9: REVR on IEVR + IEVR×Options Features (Interaction)
        print(f"\nMODEL 9: REVR = α + β₁×IEVR + β₂×IEVR×Options Features")
        print(f"{'='*50}")
        
        try:
            if available_features:
                # Create interaction terms
                interaction_terms = [f'ievr_{feature}' for feature in available_features]
                available_interactions = [term for term in interaction_terms if term in feature_clean_data.columns]
                
                if available_interactions:
                    X9 = sm.add_constant(feature_clean_data[['ievr'] + available_interactions])
                    model9 = sm.OLS(feature_clean_data['revr'], X9).fit()
                    print(model9.summary())
                    models.append(model9)
                else:
                    print("  Error: No interaction terms available")
                    models.append(None)
            else:
                print("  Error: No options surface features available for interactions")
                models.append(None)
        except Exception as e:
            print(f"  Error: {e}")
            models.append(None)
        
        return models
    
    
    def diagnostic_tests(self, model):
        """
        Run diagnostic tests for a regression model.
        """
        print(f"\nDIAGNOSTIC TESTS:")
        print(f"{'='*30}")
        
        # Heteroskedasticity test
        try:
            bp_test = het_breuschpagan(model.resid, model.model.exog)
            print(f"Breusch-Pagan Test for Heteroskedasticity:")
            print(f"  Statistic: {bp_test[0]:.3f}")
            print(f"  P-value: {bp_test[1]:.3f}")
            print(f"  Conclusion: {'Heteroskedastic' if bp_test[1] < 0.05 else 'Homoskedastic'}")
        except:
            print("Breusch-Pagan test failed")
        
        # Normality test
        try:
            stat, p_value = stats.normaltest(model.resid)
            print(f"\nNormality Test (D'Agostino K^2):")
            print(f"  Statistic: {stat:.3f}")
            print(f"  P-value: {p_value:.3f}")
            print(f"  Conclusion: {'Non-normal' if p_value < 0.05 else 'Normal'}")
        except:
            print("Normality test failed")
        
        # VIF for multicollinearity (if multiple variables)
        if model.model.exog.shape[1] > 2:
            try:
                vif_data = pd.DataFrame()
                vif_data["Variable"] = model.model.exog_names
                vif_data["VIF"] = [variance_inflation_factor(model.model.exog, i) 
                                  for i in range(model.model.exog.shape[1])]
                print(f"\nVariance Inflation Factors:")
                print(vif_data.to_string(index=False))
            except:
                print("VIF calculation failed")
    
    def create_regression_summary(self, models):
        """
        Create a summary table of all regression results.
        """
        print(f"\n{'='*100}")
        print(f"REGRESSION SUMMARY TABLE")
        print(f"{'='*100}")
        
        summary_data = []
        
        for i, model in enumerate(models, 1):
            if model is not None:
                # Get IEVR coefficient
                ievr_coef = None
                ievr_tstat = None
                ievr_pval = None
                
                if 'ievr' in model.params.index:
                    ievr_coef = model.params['ievr']
                    ievr_tstat = model.tvalues['ievr']
                    ievr_pval = model.pvalues['ievr']
                elif 'log_ievr' in model.params.index:
                    ievr_coef = model.params['log_ievr']
                    ievr_tstat = model.tvalues['log_ievr']
                    ievr_pval = model.pvalues['log_ievr']
                
                summary_data.append({
                    'Model': f'Model {i}',
                    'IEVR Coefficient': f"{ievr_coef:.3f}" if ievr_coef is not None else 'N/A',
                    'IEVR t-stat': f"{ievr_tstat:.3f}" if ievr_tstat is not None else 'N/A',
                    'IEVR p-value': f"{ievr_pval:.3f}" if ievr_pval is not None else 'N/A',
                    'R-squared': f"{model.rsquared:.3f}",
                    'Adj R-squared': f"{model.rsquared_adj:.3f}",
                    'N': model.nobs
                })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def plot_regression_results(self, models):
        """
        Create plots of regression results.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Actual vs Predicted
        if models[0] is not None:
            ax1.scatter(self.data['revr'], models[0].fittedvalues, alpha=0.6)
            ax1.plot([self.data['revr'].min(), self.data['revr'].max()], 
                    [self.data['revr'].min(), self.data['revr'].max()], 'r--', alpha=0.8)
            ax1.set_xlabel('Actual REVR')
            ax1.set_ylabel('Predicted REVR')
            ax1.set_title('Actual vs Predicted REVR (Model 1)')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residuals vs Fitted
        if models[0] is not None:
            ax2.scatter(models[0].fittedvalues, models[0].resid, alpha=0.6)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            ax2.set_xlabel('Fitted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals vs Fitted Values')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Residuals histogram
        if models[0] is not None:
            ax3.hist(models[0].resid, bins=20, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Residuals Distribution')
            ax3.grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        plt.savefig('output_files/regression_results.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Disabled for batch mode
    
    def plot_sector_analysis(self):
        """
        Plot REVR and IEVR analysis by sector.
        """
        print(f"\n{'='*80}")
        print(f"CREATING SECTOR ANALYSIS PLOTS")
        print(f"{'='*80}")
        
        # Filter data with valid sectors
        sector_data = self.data[self.data['sector'].notna()].copy()
        
        if len(sector_data) == 0:
            print("No sector data available for plotting")
            return
        
        # Create comprehensive sector analysis plots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot 1: REVR by Sector (Box Plot)
        sector_data.boxplot(column='revr', by='sector', ax=ax1)
        ax1.set_title('REVR Distribution by Sector')
        ax1.set_ylabel('REVR')
        ax1.set_xlabel('Sector')
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Change')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: IEVR by Sector (Box Plot)
        sector_data.boxplot(column='ievr', by='sector', ax=ax2)
        ax2.set_title('IEVR Distribution by Sector')
        ax2.set_ylabel('IEVR')
        ax2.set_xlabel('Sector')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Change')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Average REVR by Sector (Bar Plot)
        sector_revr_avg = sector_data.groupby('sector')['revr'].mean().sort_values(ascending=False)
        colors = plt.cm.Set3(np.linspace(0, 1, len(sector_revr_avg)))
        ax3.bar(sector_revr_avg.index, sector_revr_avg.values, color=colors, alpha=0.7)
        ax3.set_title('Average REVR by Sector')
        ax3.set_ylabel('Average REVR')
        ax3.set_xlabel('Sector')
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Change')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Average IEVR by Sector (Bar Plot)
        sector_ievr_avg = sector_data.groupby('sector')['ievr'].mean().sort_values(ascending=False)
        colors = plt.cm.Set3(np.linspace(0, 1, len(sector_ievr_avg)))
        ax4.bar(sector_ievr_avg.index, sector_ievr_avg.values, color=colors, alpha=0.7)
        ax4.set_title('Average IEVR by Sector')
        ax4.set_ylabel('Average IEVR')
        ax4.set_xlabel('Sector')
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Change')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        # Plot 5: REVR vs IEVR Scatter by Sector
        sectors = sector_data['sector'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(sectors)))
        
        for i, sector in enumerate(sectors):
            sector_subset = sector_data[sector_data['sector'] == sector]
            ax5.scatter(sector_subset['ievr'], sector_subset['revr'], 
                       alpha=0.6, s=30, color=colors[i], label=sector)
        
        # Add overall trend line
        z = np.polyfit(sector_data['ievr'], sector_data['revr'], 1)
        p = np.poly1d(z)
        ax5.plot(sector_data['ievr'], p(sector_data['ievr']), "k--", alpha=0.8, linewidth=2, label='Overall Trend')
        
        ax5.set_xlabel('IEVR')
        ax5.set_ylabel('REVR')
        ax5.set_title('REVR vs IEVR by Sector')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Sector Statistics Summary
        sector_stats = sector_data.groupby('sector').agg({
            'revr': ['count', 'mean', 'std'],
            'ievr': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        sector_stats.columns = ['_'.join(col).strip() for col in sector_stats.columns]
        
        # Create a text summary
        summary_text = "Sector Statistics Summary:\n\n"
        for sector in sector_stats.index:
            summary_text += f"{sector}:\n"
            summary_text += f"  Events: {sector_stats.loc[sector, 'revr_count']}\n"
            summary_text += f"  REVR: {sector_stats.loc[sector, 'revr_mean']:.3f} ± {sector_stats.loc[sector, 'revr_std']:.3f}\n"
            summary_text += f"  IEVR: {sector_stats.loc[sector, 'ievr_mean']:.3f} ± {sector_stats.loc[sector, 'ievr_std']:.3f}\n\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        ax6.set_title('Sector Statistics Summary')
        ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig('output_files/sector_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Sector analysis plots saved to output_files/sector_analysis.png")
        # plt.show()  # Disabled for batch mode
        
        return sector_stats

def main():
    """
    Main function to run fixed regression analysis.
    Updated for new REVR methodology with stock dummies in sector regressions.
    """
    print("FIXED REGRESSION ANALYSIS FOR REVR AND IEVR")
    print("Updated for new REVR methodology with stock dummies in sector regressions")
    print("="*100)
    
    # Load the data
    analysis = FixedRegressionAnalysis('data_files/expanded_earnings_analysis_results.csv')
    
    # Run sector-specific regressions with stock dummies
    sector_results = analysis.run_sector_specific_regressions()
    
    if not sector_results.empty:
        sector_results.to_csv('data_files/sector_regression_results.csv', index=False)
        print("✓ Sector regression results saved to data_files/sector_regression_results.csv")
    else:
        print("✗ No sector regression results generated")

if __name__ == "__main__":
    main() 