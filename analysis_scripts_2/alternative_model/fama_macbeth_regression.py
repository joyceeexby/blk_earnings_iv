"""
Fama-MacBeth Regression Analysis for IEVR-REVR Relationship

This module implements Fama-MacBeth regression analysis to test the significance
of the relationship between Implied Earnings Volatility Ratio (IEVR) and 
Realized Earnings Volatility Ratio (REVR) across time periods.

Key Features:
- Two-stage Fama-MacBeth regression
- Cross-sectional regressions for each time period
- Time-series analysis of coefficients
- Robust standard errors and t-statistics
- Multiple factor models and specifications
- Comprehensive statistical testing
- Visualization of results

The Fama-MacBeth approach:
1. Stage 1: Run cross-sectional regressions for each time period
2. Stage 2: Analyze the time series of coefficients
3. Provides robust inference for panel data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac_simple

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FamaMacBethRegression:
    """
    Implements Fama-MacBeth regression analysis for IEVR-REVR relationship.
    """
    
    def __init__(self, 
                 data_file: str = 'data_files/expanded_earnings_analysis_results_with_vix.csv',
                 output_dir: str = 'output_files',
                 data_dir: str = 'data_files'):
        """
        Initialize the Fama-MacBeth regression analysis.
        
        Parameters:
        -----------
        data_file : str
            Path to the CSV file containing the analysis results
        output_dir : str
            Directory for output files
        data_dir : str
            Directory for data files
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.data = None
        self.results = {}
        self.feature_columns = []
        
        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load and prepare data
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self) -> None:
        """
        Load data and prepare for Fama-MacBeth analysis.
        """
        logger.info("Loading and preparing data for Fama-MacBeth regression...")
        
        try:
            # Load data
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"Data file not found: {self.data_file}")
            
            self.data = pd.read_csv(self.data_file)
            logger.info(f"Loaded {len(self.data)} rows from {self.data_file}")
            
            # Validate required columns
            required_columns = ['earnings_date', 'revr', 'ievr', 'ticker']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert earnings_date to datetime
            self.data['earnings_date'] = pd.to_datetime(self.data['earnings_date'])
            
            # Create time period identifier (quarterly)
            self.data['year_quarter'] = self.data['earnings_date'].dt.to_period('Q')
            
            # Sort by date and ticker
            self.data = self.data.sort_values(['year_quarter', 'ticker']).reset_index(drop=True)
            
            # Clean data - remove NaN and infinite values
            initial_size = len(self.data)
            self.data = self.data[
                (self.data['revr'].notna()) & 
                (self.data['ievr'].notna()) &
                (np.isfinite(self.data['revr'])) & 
                (np.isfinite(self.data['ievr']))
            ].reset_index(drop=True)
            
            logger.info(f"Cleaned data: {initial_size} -> {len(self.data)} rows")
            
            # Set up feature columns
            self.setup_feature_columns()
            
            # Log data summary
            logger.info(f"Data spans: {self.data['year_quarter'].min()} to {self.data['year_quarter'].max()}")
            logger.info(f"Number of quarters: {self.data['year_quarter'].nunique()}")
            logger.info(f"Number of unique stocks: {self.data['ticker'].nunique()}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def setup_feature_columns(self) -> None:
        """
        Set up feature columns for the analysis.
        """
        # Base features for different specifications
        self.feature_columns = {
            'basic': ['ievr'],
            'extended': ['ievr', 'normative_implied_vol', 'skew_ratio'],
            'full': ['ievr', 'normative_implied_vol', 'skew_ratio'],
            'with_betas': ['ievr', 'normative_implied_vol', 'skew_ratio', 
                          'beta_market', 'beta_smb', 'beta_hml'],
            'with_vix': ['ievr', 'normative_implied_vol', 'skew_ratio', 
                        'beta_market', 'beta_smb', 'beta_hml', 'vix_momentum_5d']
        }
        
        # Filter available features for each specification
        for spec_name, features in self.feature_columns.items():
            self.feature_columns[spec_name] = [col for col in features if col in self.data.columns]
            logger.info(f"✓ {spec_name} specification: {len(self.feature_columns[spec_name])} features")
    
    def run_cross_sectional_regressions(self, specification: str = 'basic') -> Dict[str, Any]:
        """
        Run cross-sectional regressions for each time period (Stage 1 of Fama-MacBeth).
        
        Parameters:
        -----------
        specification : str
            Which feature specification to use ('basic', 'extended', 'full', 'with_betas', 'with_vix')
        
        Returns:
        --------
        dict : Results from cross-sectional regressions
        """
        logger.info(f"Running cross-sectional regressions for {specification} specification...")
        
        if specification not in self.feature_columns:
            raise ValueError(f"Unknown specification: {specification}")
        
        features = self.feature_columns[specification]
        if not features:
            raise ValueError(f"No available features for specification: {specification}")
        
        # Initialize results storage
        cross_sectional_results = []
        periods_with_data = []
        
        # Run regression for each time period
        for period in sorted(self.data['year_quarter'].unique()):
            period_data = self.data[self.data['year_quarter'] == period].copy()
            
            # Skip if insufficient observations
            if len(period_data) < 5:  # Minimum 5 observations per period (reduced from 10)
                continue
            
            # Prepare features and target
            X = period_data[features].fillna(0)
            y = period_data['revr']
            
            # Skip if all features are zero or constant
            if X.std().min() == 0:
                continue
            
            try:
                # Run regression
                model = LinearRegression()
                model.fit(X, y)
                
                # Get coefficients and predictions
                coefficients = model.coef_
                intercept = model.intercept_
                r_squared = model.score(X, y)
                
                # Calculate residuals and standard errors
                y_pred = model.predict(X)
                residuals = y - y_pred
                n = len(y)
                p = len(features)
                
                # Calculate standard errors
                X_with_const = sm.add_constant(X)
                model_sm = sm.OLS(y, X_with_const).fit()
                se = model_sm.bse[1:]  # Exclude intercept
                
                # Store results
                period_result = {
                    'period': period,
                    'n_observations': n,
                    'r_squared': r_squared,
                    'intercept': intercept,
                    'coefficients': coefficients,
                    'standard_errors': se,
                    'residuals': residuals,
                    'features': features
                }
                
                cross_sectional_results.append(period_result)
                periods_with_data.append(period)
                
            except Exception as e:
                logger.warning(f"Error in period {period}: {str(e)}")
                continue
        
        logger.info(f"✓ Completed cross-sectional regressions for {len(cross_sectional_results)} periods")
        
        return {
            'cross_sectional_results': cross_sectional_results,
            'periods_with_data': periods_with_data,
            'specification': specification,
            'features': features
        }
    
    def analyze_time_series_coefficients(self, cross_sectional_results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze the time series of coefficients (Stage 2 of Fama-MacBeth).
        
        Parameters:
        -----------
        cross_sectional_results : List[Dict]
            Results from cross-sectional regressions
        
        Returns:
        --------
        dict : Time series analysis results
        """
        logger.info("Analyzing time series of coefficients...")
        
        if not cross_sectional_results:
            raise ValueError("No cross-sectional results to analyze")
        
        # Extract coefficients over time
        periods = [result['period'] for result in cross_sectional_results]
        features = cross_sectional_results[0]['features']
        
        # Create coefficient matrix
        coefficient_matrix = np.array([result['coefficients'] for result in cross_sectional_results])
        
        # Calculate time series statistics
        time_series_results = {}
        
        for i, feature in enumerate(features):
            feature_coefficients = coefficient_matrix[:, i]
            
            # Basic statistics
            mean_coef = np.mean(feature_coefficients)
            std_coef = np.std(feature_coefficients, ddof=1)
            se_coef = std_coef / np.sqrt(len(feature_coefficients))
            t_stat = mean_coef / se_coef
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(feature_coefficients) - 1))
            
            # Additional statistics
            median_coef = np.median(feature_coefficients)
            min_coef = np.min(feature_coefficients)
            max_coef = np.max(feature_coefficients)
            positive_ratio = np.mean(feature_coefficients > 0)
            
            # Newey-West standard errors (for autocorrelation)
            try:
                # Calculate Newey-West standard errors
                nw_se = self.calculate_newey_west_se(feature_coefficients)
                nw_t_stat = mean_coef / nw_se
                nw_p_value = 2 * (1 - stats.t.cdf(abs(nw_t_stat), len(feature_coefficients) - 1))
            except:
                nw_se = se_coef
                nw_t_stat = t_stat
                nw_p_value = p_value
            
            time_series_results[feature] = {
                'mean_coefficient': mean_coef,
                'std_coefficient': std_coef,
                'standard_error': se_coef,
                't_statistic': t_stat,
                'p_value': p_value,
                'median_coefficient': median_coef,
                'min_coefficient': min_coef,
                'max_coefficient': max_coef,
                'positive_ratio': positive_ratio,
                'newey_west_se': nw_se,
                'newey_west_t_stat': nw_t_stat,
                'newey_west_p_value': nw_p_value,
                'time_series': feature_coefficients
            }
        
        # Overall model statistics
        r_squared_series = [result['r_squared'] for result in cross_sectional_results]
        avg_r_squared = np.mean(r_squared_series)
        
        return {
            'time_series_results': time_series_results,
            'periods': periods,
            'coefficient_matrix': coefficient_matrix,
            'avg_r_squared': avg_r_squared,
            'r_squared_series': r_squared_series
        }
    
    def calculate_newey_west_se(self, coefficients: np.ndarray, max_lags: int = 4) -> float:
        """
        Calculate Newey-West standard errors for autocorrelation.
        
        Parameters:
        -----------
        coefficients : np.ndarray
            Time series of coefficients
        max_lags : int
            Maximum number of lags for autocorrelation
        
        Returns:
        --------
        float : Newey-West standard error
        """
        n = len(coefficients)
        mean_coef = np.mean(coefficients)
        
        # Calculate autocovariances
        autocov = np.zeros(max_lags + 1)
        for lag in range(max_lags + 1):
            if lag == 0:
                autocov[lag] = np.mean((coefficients - mean_coef) ** 2)
            else:
                autocov[lag] = np.mean((coefficients[lag:] - mean_coef) * 
                                     (coefficients[:-lag] - mean_coef))
        
        # Newey-West variance
        variance = autocov[0] + 2 * np.sum([(1 - lag/(max_lags + 1)) * autocov[lag] 
                                           for lag in range(1, max_lags + 1)])
        
        return np.sqrt(variance / n)
    
    def run_fama_macbeth_analysis(self, specification: str = 'basic') -> Dict[str, Any]:
        """
        Run complete Fama-MacBeth regression analysis.
        
        Parameters:
        -----------
        specification : str
            Which feature specification to use
        
        Returns:
        --------
        dict : Complete Fama-MacBeth results
        """
        logger.info("="*80)
        logger.info(f"FAMA-MACBETH REGRESSION ANALYSIS - {specification.upper()} SPECIFICATION")
        logger.info("="*80)
        
        # Stage 1: Cross-sectional regressions
        cross_sectional_results = self.run_cross_sectional_regressions(specification)
        
        # Stage 2: Time series analysis
        time_series_results = self.analyze_time_series_coefficients(
            cross_sectional_results['cross_sectional_results']
        )
        
        # Combine results
        complete_results = {
            'specification': specification,
            'cross_sectional': cross_sectional_results,
            'time_series': time_series_results,
            'features': cross_sectional_results['features']
        }
        
        # Store results
        self.results[specification] = complete_results
        
        # Print summary
        self.print_fama_macbeth_summary(complete_results)
        
        return complete_results
    
    def print_fama_macbeth_summary(self, results: Dict[str, Any]) -> None:
        """
        Print summary of Fama-MacBeth results.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Complete Fama-MacBeth results
        """
        logger.info("\n" + "="*80)
        logger.info("FAMA-MACBETH REGRESSION RESULTS")
        logger.info("="*80)
        
        spec = results['specification']
        ts_results = results['time_series']['time_series_results']
        avg_r2 = results['time_series']['avg_r_squared']
        n_periods = len(results['time_series']['periods'])
        
        logger.info(f"Specification: {spec}")
        logger.info(f"Number of time periods: {n_periods}")
        logger.info(f"Average R²: {avg_r2:.4f}")
        logger.info(f"Features: {', '.join(results['features'])}")
        
        logger.info(f"\n{'Variable':<20} {'Mean Coef':<12} {'Std Error':<12} {'t-stat':<10} {'p-value':<10} {'NW t-stat':<10}")
        logger.info("-" * 80)
        
        for feature, stats in ts_results.items():
            logger.info(f"{feature:<20} {stats['mean_coefficient']:<12.4f} "
                       f"{stats['standard_error']:<12.4f} {stats['t_statistic']:<10.3f} "
                       f"{stats['p_value']:<10.3f} {stats['newey_west_t_stat']:<10.3f}")
        
        # Significance summary
        logger.info(f"\nSignificance Summary:")
        significant_vars = []
        for feature, stats in ts_results.items():
            if stats['p_value'] < 0.05:
                significant_vars.append(feature)
        
        if significant_vars:
            logger.info(f"Significant variables (p < 0.05): {', '.join(significant_vars)}")
        else:
            logger.info("No variables significant at 5% level")
    
    def create_fama_macbeth_plots(self, specification: str = 'basic') -> plt.Figure:
        """
        Create plots for Fama-MacBeth analysis.
        
        Parameters:
        -----------
        specification : str
            Which specification to plot
        
        Returns:
        --------
        plt.Figure : Fama-MacBeth plots
        """
        if specification not in self.results:
            logger.error(f"No results for specification: {specification}")
            return None
        
        results = self.results[specification]
        ts_results = results['time_series']['time_series_results']
        periods = results['time_series']['periods']
        coefficient_matrix = results['time_series']['coefficient_matrix']
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Fama-MacBeth Regression Results - {specification.upper()} Specification', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Coefficient time series
        for i, feature in enumerate(results['features']):
            feature_coefficients = coefficient_matrix[:, i]
            axes[0, 0].plot(range(len(periods)), feature_coefficients, 
                           marker='o', label=feature, alpha=0.7)
        
        axes[0, 0].set_title('Coefficient Time Series', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Time Period')
        axes[0, 0].set_ylabel('Coefficient')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 2: Coefficient distribution
        for feature in results['features']:
            coefficients = ts_results[feature]['time_series']
            axes[0, 1].hist(coefficients, bins=15, alpha=0.6, label=feature)
        
        axes[0, 1].set_title('Coefficient Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Coefficient Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 3: R-squared over time
        r2_series = results['time_series']['r_squared_series']
        axes[1, 0].plot(range(len(periods)), r2_series, marker='o', color='blue', alpha=0.7)
        axes[1, 0].set_title('R² Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Time Period')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: T-statistics comparison
        features = list(results['features'])
        t_stats = [ts_results[feature]['t_statistic'] for feature in features]
        nw_t_stats = [ts_results[feature]['newey_west_t_stat'] for feature in features]
        
        x = np.arange(len(features))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, t_stats, width, label='Standard t-stat', alpha=0.7)
        axes[1, 1].bar(x + width/2, nw_t_stats, width, label='Newey-West t-stat', alpha=0.7)
        
        axes[1, 1].set_title('T-Statistics Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Variables')
        axes[1, 1].set_ylabel('T-Statistic')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(features, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=1.96, color='red', linestyle='--', alpha=0.7, label='5% significance')
        axes[1, 1].axhline(y=-1.96, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'fama_macbeth_{specification}_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Fama-MacBeth plots saved to {plot_path}")
        
        return fig
    
    def save_fama_macbeth_results(self, specification: str = 'basic') -> None:
        """
        Save Fama-MacBeth results to CSV files.
        
        Parameters:
        -----------
        specification : str
            Which specification to save
        """
        if specification not in self.results:
            logger.error(f"No results for specification: {specification}")
            return
        
        results = self.results[specification]
        ts_results = results['time_series']['time_series_results']
        
        # Create summary table
        summary_data = []
        for feature, stats in ts_results.items():
            summary_data.append({
                'Variable': feature,
                'Mean_Coefficient': stats['mean_coefficient'],
                'Std_Error': stats['standard_error'],
                'T_Statistic': stats['t_statistic'],
                'P_Value': stats['p_value'],
                'Newey_West_SE': stats['newey_west_se'],
                'Newey_West_T_Stat': stats['newey_west_t_stat'],
                'Newey_West_P_Value': stats['newey_west_p_value'],
                'Median_Coefficient': stats['median_coefficient'],
                'Min_Coefficient': stats['min_coefficient'],
                'Max_Coefficient': stats['max_coefficient'],
                'Positive_Ratio': stats['positive_ratio']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_path = os.path.join(self.data_dir, f'fama_macbeth_{specification}_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"✓ Fama-MacBeth summary saved to {summary_path}")
        
        # Save coefficient time series
        coefficient_data = []
        periods = results['time_series']['periods']
        coefficient_matrix = results['time_series']['coefficient_matrix']
        
        for i, period in enumerate(periods):
            row = {'Period': period}
            for j, feature in enumerate(results['features']):
                row[f'{feature}_Coefficient'] = coefficient_matrix[i, j]
            coefficient_data.append(row)
        
        coefficient_df = pd.DataFrame(coefficient_data)
        coefficient_path = os.path.join(self.data_dir, f'fama_macbeth_{specification}_coefficients.csv')
        coefficient_df.to_csv(coefficient_path, index=False)
        logger.info(f"✓ Coefficient time series saved to {coefficient_path}")
    
    def run_complete_fama_macbeth_analysis(self) -> None:
        """
        Run complete Fama-MacBeth analysis for all specifications.
        """
        logger.info("="*80)
        logger.info("COMPLETE FAMA-MACBETH REGRESSION ANALYSIS")
        logger.info("="*80)
        
        specifications = ['basic', 'extended', 'full', 'with_betas', 'with_vix']
        
        for spec in specifications:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"ANALYZING SPECIFICATION: {spec.upper()}")
                logger.info(f"{'='*60}")
                
                # Run analysis
                results = self.run_fama_macbeth_analysis(spec)
                
                # Create plots
                self.create_fama_macbeth_plots(spec)
                
                # Save results
                self.save_fama_macbeth_results(spec)
                
            except Exception as e:
                logger.error(f"Error in specification {spec}: {str(e)}")
                continue
        
        logger.info("\n" + "="*80)
        logger.info("FAMA-MACBETH ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info("✓ All specifications analyzed")
        logger.info("✓ Results saved and visualized")
        logger.info("✓ Check output_files for plots")
        logger.info("✓ Check data_files for CSV results")

def main():
    """
    Main function to run Fama-MacBeth regression analysis.
    """
    try:
        # Initialize analysis
        fm_analysis = FamaMacBethRegression()
        
        # Run complete analysis
        fm_analysis.run_complete_fama_macbeth_analysis()
        
        print("\n✓ Fama-MacBeth regression analysis completed successfully!")
        print("✓ Check the output_files directory for plots")
        print("✓ Check the data_files directory for CSV results")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
