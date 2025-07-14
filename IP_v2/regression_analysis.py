#!/usr/bin/env python3
"""
Fixed Regression Analysis for REVR and IEVR
Handle NaN values and multicollinearity issues
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
    
    def __init__(self, data_file='multi_stock_results.csv'):
        """
        Initialize with the multi-stock results data.
        """
        self.data = pd.read_csv(data_file)
        self.data['earnings_date'] = pd.to_datetime(self.data['earnings_date'])
        
        # Clean data
        self.clean_data()
        
        # Create additional variables
        self.create_variables()
        
        print(f"Loaded {len(self.data)} observations from {self.data['ticker'].nunique()} stocks")
        print(f"Date range: {self.data['earnings_date'].min()} to {self.data['earnings_date'].max()}")
    
    def clean_data(self):
        """
        Clean the data by removing NaN values, infinite values, and outliers.
        """
        print("Cleaning data...")
        
        initial_count = len(self.data)
        
        # Remove rows with NaN values in key variables
        key_vars = ['revr', 'ievr', 'ratio', 'vol_t_minus_3', 'vol_t_plus_4', 'kink_tte']
        for var in key_vars:
            if var in self.data.columns:
                self.data = self.data.dropna(subset=[var])
        
        print(f"Removed {initial_count - len(self.data)} rows with NaN values")
        
        # Remove infinite values
        initial_count = len(self.data)
        for var in ['revr', 'ievr', 'ratio']:
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
        
        # Create stock dummies (excluding one to avoid multicollinearity)
        tickers = self.data['ticker'].unique()
        for ticker in tickers[:-1]:  # Exclude last ticker
            self.data[f'dummy_{ticker}'] = (self.data['ticker'] == ticker).astype(int)
        
        # Create year dummies (excluding one to avoid multicollinearity)
        years = sorted(self.data['year'].unique())
        for year in years[:-1]:  # Exclude last year
            self.data[f'dummy_year_{year}'] = (self.data['year'] == year).astype(int)
    
    def descriptive_statistics(self):
        """
        Generate comprehensive descriptive statistics.
        """
        print(f"\n{'='*100}")
        print(f"DESCRIPTIVE STATISTICS")
        print(f"{'='*100}")
        
        # Overall statistics
        print(f"\nOVERALL STATISTICS:")
        print(self.data[['revr', 'ievr', 'ratio', 'prediction_error']].describe())
        
        # By stock statistics
        print(f"\nBY-STOCK STATISTICS:")
        stock_stats = self.data.groupby('ticker').agg({
            'revr': ['count', 'mean', 'std', 'min', 'max'],
            'ievr': ['mean', 'std'],
            'prediction_error': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        stock_stats.columns = ['_'.join(col).strip() for col in stock_stats.columns]
        print(stock_stats.to_string())
        
        # Correlation matrix
        print(f"\nCORRELATION MATRIX:")
        corr_vars = ['revr', 'ievr', 'ratio', 'prediction_error']
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
        ax3.set_title('REVR Distribution by Stock')
        ax3.set_ylabel('REVR')
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Plot 4: Prediction error by stock
        self.data.boxplot(column='prediction_error', by='ticker', ax=ax4)
        ax4.set_title('Prediction Error by Stock')
        ax4.set_ylabel('Prediction Error')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('regression_descriptive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
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
            stock_dummies = [col for col in self.data.columns if col.startswith('dummy_') and not col.startswith('dummy_year_')]
            if stock_dummies:
                X2 = sm.add_constant(clean_data[['ievr'] + stock_dummies])
                model2 = sm.OLS(clean_data['revr'], X2).fit()
                print(model2.summary())
                models.append(model2)
            else:
                print("  Error: No stock dummies available")
                models.append(None)
        except Exception as e:
            print(f"  Error: {e}")
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
            if stock_dummies and time_dummies:
                X4 = sm.add_constant(clean_data[['ievr'] + stock_dummies + time_dummies])
                model4 = sm.OLS(clean_data['revr'], X4).fit()
                print(model4.summary())
                models.append(model4)
            else:
                print("  Error: Missing stock or time dummies")
                models.append(None)
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
        
        return models
    
    def run_stock_specific_regressions(self):
        """
        Run regressions for each stock separately.
        """
        print(f"\n{'='*100}")
        print(f"STOCK-SPECIFIC REGRESSIONS")
        print(f"{'='*100}")
        
        stock_results = {}
        
        for ticker in self.data['ticker'].unique():
            stock_data = self.data[self.data['ticker'] == ticker]
            
            if len(stock_data) >= 5:  # Need minimum observations
                print(f"\n{ticker} Regression (n={len(stock_data)}):")
                print(f"{'='*30}")
                
                X = sm.add_constant(stock_data['ievr'])
                model = sm.OLS(stock_data['revr'], X).fit()
                
                print(f"REVR = {model.params[0]:.3f} + {model.params[1]:.3f} × IEVR")
                print(f"R² = {model.rsquared:.3f}")
                print(f"β (IEVR coefficient) = {model.params[1]:.3f} (t={model.tvalues[1]:.3f})")
                print(f"P-value = {model.pvalues[1]:.3f}")
                
                stock_results[ticker] = {
                    'intercept': model.params[0],
                    'slope': model.params[1],
                    'r_squared': model.rsquared,
                    't_stat': model.tvalues[1],
                    'p_value': model.pvalues[1],
                    'n_obs': len(stock_data)
                }
        
        # Create summary table
        if stock_results:
            summary_df = pd.DataFrame(stock_results).T
            print(f"\nSTOCK-SPECIFIC REGRESSION SUMMARY:")
            print(f"{'='*50}")
            print(summary_df.round(3).to_string())
            
            return summary_df
        
        return None
    
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
        
        # Plot 4: Stock-specific slopes
        stock_slopes = self.run_stock_specific_regressions()
        if stock_slopes is not None:
            ax4.bar(range(len(stock_slopes)), stock_slopes['slope'])
            ax4.set_xticks(range(len(stock_slopes)))
            ax4.set_xticklabels(stock_slopes.index, rotation=45)
            ax4.set_ylabel('IEVR Coefficient')
            ax4.set_title('Stock-Specific IEVR Coefficients')
            ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Prediction')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regression_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main function to run fixed regression analysis.
    """
    print("FIXED REGRESSION ANALYSIS FOR REVR AND IEVR")
    print("="*100)
    
    try:
        # Initialize analysis
        analysis = FixedRegressionAnalysis()
        
        # Descriptive analysis
        correlation_matrix = analysis.descriptive_statistics()
        analysis.plot_descriptive_analysis()
        
        # Basic regressions
        basic_models = analysis.run_basic_regressions()
        
        # Extended regressions
        extended_models = analysis.run_extended_regressions()
        
        # Stock-specific regressions
        stock_results = analysis.run_stock_specific_regressions()
        
        # Diagnostic tests for main model
        print(f"\n{'='*100}")
        print(f"DIAGNOSTIC TESTS FOR MODEL 1")
        print(f"{'='*100}")
        analysis.diagnostic_tests(basic_models[0])
        
        # Create summary
        all_models = list(basic_models) + list(extended_models)
        summary_table = analysis.create_regression_summary(all_models)
        
        # Plot results
        analysis.plot_regression_results(basic_models)
        
        # Save results
        summary_table.to_csv('regression_summary.csv', index=False)
        if stock_results is not None:
            stock_results.to_csv('stock_specific_regressions.csv')
        
        print(f"\n✓ Fixed regression analysis completed successfully!")
        print(f"  Results saved to regression_summary.csv")
        print(f"  Stock-specific results saved to stock_specific_regressions.csv")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 