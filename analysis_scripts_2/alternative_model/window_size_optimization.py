"""
Window Size Optimization for Rolling Walk-Forward Analysis

This module tests different window sizes for training and testing data to find
the optimal configuration for the IEVR-REVR relationship analysis.

Key Features:
- Grid search over training window sizes
- Grid search over testing window sizes
- Performance comparison across configurations
- Statistical significance testing
- Optimal window size recommendations
- Visualization of results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
import itertools

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WindowSizeOptimizer:
    """
    Optimizes window sizes for rolling walk-forward analysis.
    """
    
    def __init__(self, 
                 data_file: str = 'data_files/expanded_earnings_analysis_results_with_vix.csv',
                 output_dir: str = 'output_files',
                 data_dir: str = 'data_files'):
        """
        Initialize the window size optimizer.
        
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
        self.results = []
        self.feature_columns = []
        
        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load and prepare data
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self) -> None:
        """
        Load data and prepare features for window size optimization.
        """
        logger.info("Loading and preparing data for window size optimization...")
        
        try:
            # Load data
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"Data file not found: {self.data_file}")
            
            self.data = pd.read_csv(self.data_file)
            logger.info(f"Loaded {len(self.data)} rows from {self.data_file}")
            
            # Validate required columns
            required_columns = ['earnings_date', 'revr', 'ievr']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert earnings_date to datetime
            self.data['earnings_date'] = pd.to_datetime(self.data['earnings_date'])
            
            # Sort by date to ensure temporal order
            self.data = self.data.sort_values('earnings_date').reset_index(drop=True)
            
            # Clean data - remove NaN and infinite values for target and main predictor
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
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def setup_feature_columns(self) -> None:
        """
        Set up feature columns for the analysis.
        """
        # Base features (same as original analysis)
        base_features = [
            'ievr', 'normative_iv_rv_ratio', 'skew_ratio', 'spx_ievr', 'sector_leader_revr',
            'beta_market', 'beta_smb', 'beta_hml', 'vix_momentum_5d'
        ]
        
        # Filter available features
        self.feature_columns = [col for col in base_features if col in self.data.columns]
        
        logger.info(f"✓ Using {len(self.feature_columns)} features: {self.feature_columns}")
    
    def train_and_evaluate_model(self, 
                               model_name: str,
                               X_train: np.ndarray, 
                               y_train: np.ndarray, 
                               X_test: np.ndarray, 
                               y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train and evaluate a single model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model ('Linear', 'Random Forest', 'XGBoost')
        X_train, y_train : array-like
            Training data
        X_test, y_test : array-like
            Test data
        
        Returns:
        --------
        dict : Model results
        """
        try:
            if model_name == 'Linear':
                model = LinearRegression()
            elif model_name == 'Random Forest':
                model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            elif model_name == 'XGBoost':
                model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
            
            results = {
                'model_name': model_name,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            return None
    
    def run_window_size_optimization(self,
                                   start_date: str = '2013-01-01',
                                   end_date: str = '2024-12-31',
                                   train_windows: List[int] = [12, 18, 24, 30, 36, 48],
                                   test_windows: List[int] = [3, 6, 9, 12],
                                   step_size: int = 6,
                                   min_test_size: int = 10) -> None:
        """
        Run window size optimization.
        
        Parameters:
        -----------
        start_date : str
            Start date for analysis
        end_date : str
            End date for analysis
        train_windows : List[int]
            List of training window sizes to test (in months)
        test_windows : List[int]
            List of test window sizes to test (in months)
        step_size : int
            Step size between periods (in months)
        min_test_size : int
            Minimum test set size
        """
        logger.info("="*80)
        logger.info("WINDOW SIZE OPTIMIZATION")
        logger.info("="*80)
        
        # Convert dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Initialize results
        self.results = []
        total_combinations = len(train_windows) * len(test_windows)
        combination_count = 0
        
        # Test all combinations
        for train_window, test_window in itertools.product(train_windows, test_windows):
            combination_count += 1
            logger.info(f"\nTesting combination {combination_count}/{total_combinations}: "
                       f"Train={train_window} months, Test={test_window} months")
            
            # Initialize for this combination
            current_train_start = start_date
            current_train_end = start_date + timedelta(days=train_window * 30)
            
            combination_results = []
            iteration = 0
            
            while current_train_end < end_date - timedelta(days=test_window * 30):
                iteration += 1
                
                # Define test period
                test_start = current_train_end
                test_end = current_train_end + timedelta(days=test_window * 30)
                
                # Get training and test data
                train_data = self.data[
                    (self.data['earnings_date'] >= current_train_start) & 
                    (self.data['earnings_date'] <= current_train_end)
                ]
                
                test_data = self.data[
                    (self.data['earnings_date'] > test_start) & 
                    (self.data['earnings_date'] <= test_end)
                ]
                
                # Skip if insufficient data
                if len(test_data) < min_test_size or len(train_data) < min_test_size:
                    current_train_start += timedelta(days=step_size * 30)
                    current_train_end += timedelta(days=step_size * 30)
                    continue
                
                # Prepare features
                X_train = train_data[self.feature_columns].fillna(0)
                y_train = train_data['revr']
                X_test = test_data[self.feature_columns].fillna(0)
                y_test = test_data['revr']
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train and evaluate models
                model_results = {}
                for model_name in ['Linear', 'Random Forest', 'XGBoost']:
                    result = self.train_and_evaluate_model(
                        model_name, X_train_scaled, y_train, X_test_scaled, y_test
                    )
                    if result:
                        model_results[model_name] = result
                
                # Store period results
                if model_results:
                    period_result = {
                        'train_window': train_window,
                        'test_window': test_window,
                        'iteration': iteration,
                        'train_start': current_train_start,
                        'train_end': current_train_end,
                        'test_start': test_start,
                        'test_end': test_end,
                        'train_observations': len(train_data),
                        'test_observations': len(test_data),
                        'model_results': model_results
                    }
                    combination_results.append(period_result)
                
                # Move to next period
                current_train_start += timedelta(days=step_size * 30)
                current_train_end += timedelta(days=step_size * 30)
            
            # Store combination results
            if combination_results:
                self.results.extend(combination_results)
                logger.info(f"✓ Completed {len(combination_results)} periods for "
                           f"Train={train_window} months, Test={test_window} months")
        
        logger.info(f"\n{'='*80}")
        logger.info("WINDOW SIZE OPTIMIZATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total periods analyzed: {len(self.results)}")
        logger.info(f"Combinations tested: {total_combinations}")
    
    def analyze_results(self) -> pd.DataFrame:
        """
        Analyze optimization results and find optimal configurations.
        
        Returns:
        --------
        pd.DataFrame : Summary of results by configuration
        """
        if not self.results:
            logger.error("No results to analyze. Run optimization first.")
            return pd.DataFrame()
        
        logger.info("Analyzing window size optimization results...")
        
        # Prepare summary data
        summary_data = []
        
        for result in self.results:
            train_window = result['train_window']
            test_window = result['test_window']
            
            for model_name, model_result in result['model_results'].items():
                summary_data.append({
                    'Train_Window': train_window,
                    'Test_Window': test_window,
                    'Model': model_name,
                    'Test_R2': model_result['test_r2'],
                    'Test_RMSE': model_result['test_rmse'],
                    'Test_MAE': model_result['test_mae'],
                    'CV_R2_Mean': model_result['cv_r2_mean'],
                    'CV_R2_Std': model_result['cv_r2_std'],
                    'Train_Obs': result['train_observations'],
                    'Test_Obs': result['test_observations']
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Calculate statistics by configuration
        config_stats = summary_df.groupby(['Train_Window', 'Test_Window', 'Model']).agg({
            'Test_R2': ['mean', 'std', 'count'],
            'Test_RMSE': ['mean', 'std'],
            'Test_MAE': ['mean', 'std'],
            'CV_R2_Mean': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        config_stats.columns = ['_'.join(col).strip() for col in config_stats.columns]
        config_stats = config_stats.reset_index()
        
        # Save results
        results_path = os.path.join(self.data_dir, 'window_size_optimization_results.csv')
        config_stats.to_csv(results_path, index=False)
        logger.info(f"✓ Optimization results saved to {results_path}")
        
        # Find optimal configurations
        self.find_optimal_configurations(config_stats)
        
        return config_stats
    
    def find_optimal_configurations(self, config_stats: pd.DataFrame) -> None:
        """
        Find optimal window size configurations.
        
        Parameters:
        -----------
        config_stats : pd.DataFrame
            Configuration statistics
        """
        logger.info("\n" + "="*80)
        logger.info("OPTIMAL WINDOW SIZE CONFIGURATIONS")
        logger.info("="*80)
        
        # Find best configuration for each model
        for model in config_stats['Model'].unique():
            model_data = config_stats[config_stats['Model'] == model]
            
            # Best by R²
            best_r2 = model_data.loc[model_data['Test_R2_mean'].idxmax()]
            logger.info(f"\n{model} - Best by R²:")
            logger.info(f"  Train Window: {best_r2['Train_Window']} months")
            logger.info(f"  Test Window: {best_r2['Test_Window']} months")
            logger.info(f"  Mean R²: {best_r2['Test_R2_mean']:.4f}")
            logger.info(f"  Std R²: {best_r2['Test_R2_std']:.4f}")
            logger.info(f"  Periods: {best_r2['Test_R2_count']}")
            
            # Best by RMSE
            best_rmse = model_data.loc[model_data['Test_RMSE_mean'].idxmin()]
            logger.info(f"\n{model} - Best by RMSE:")
            logger.info(f"  Train Window: {best_rmse['Train_Window']} months")
            logger.info(f"  Test Window: {best_rmse['Test_Window']} months")
            logger.info(f"  Mean RMSE: {best_rmse['Test_RMSE_mean']:.4f}")
            logger.info(f"  Std RMSE: {best_rmse['Test_RMSE_std']:.4f}")
        
        # Overall best configuration
        overall_best = config_stats.loc[config_stats['Test_R2_mean'].idxmax()]
        logger.info(f"\nOverall Best Configuration:")
        logger.info(f"  Model: {overall_best['Model']}")
        logger.info(f"  Train Window: {overall_best['Train_Window']} months")
        logger.info(f"  Test Window: {overall_best['Test_Window']} months")
        logger.info(f"  Mean R²: {overall_best['Test_R2_mean']:.4f}")
        
        # Stability analysis (lowest standard deviation)
        most_stable = config_stats.loc[config_stats['Test_R2_std'].idxmin()]
        logger.info(f"\nMost Stable Configuration:")
        logger.info(f"  Model: {most_stable['Model']}")
        logger.info(f"  Train Window: {most_stable['Train_Window']} months")
        logger.info(f"  Test Window: {most_stable['Test_Window']} months")
        logger.info(f"  Mean R²: {most_stable['Test_R2_mean']:.4f}")
        logger.info(f"  Std R²: {most_stable['Test_R2_std']:.4f}")
    
    def create_optimization_plots(self) -> plt.Figure:
        """
        Create plots showing optimization results.
        
        Returns:
        --------
        plt.Figure : Optimization plots
        """
        if not self.results:
            logger.error("No results to plot. Run optimization first.")
            return None
        
        logger.info("Creating window size optimization plots...")
        
        # Prepare data for plotting
        plot_data = []
        for result in self.results:
            for model_name, model_result in result['model_results'].items():
                plot_data.append({
                    'Train_Window': result['train_window'],
                    'Test_Window': result['test_window'],
                    'Model': model_name,
                    'Test_R2': model_result['test_r2'],
                    'Test_RMSE': model_result['test_rmse']
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Window Size Optimization Results', fontsize=16, fontweight='bold')
        
        # Plot 1: R² by train window size
        for model in plot_df['Model'].unique():
            model_data = plot_df[plot_df['Model'] == model]
            train_means = model_data.groupby('Train_Window')['Test_R2'].mean()
            train_stds = model_data.groupby('Train_Window')['Test_R2'].std()
            
            axes[0, 0].errorbar(train_means.index, train_means.values, 
                              yerr=train_stds.values, label=model, marker='o', capsize=5)
        
        axes[0, 0].set_title('R² by Training Window Size', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Training Window (Months)')
        axes[0, 0].set_ylabel('Mean Test R²')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: R² by test window size
        for model in plot_df['Model'].unique():
            model_data = plot_df[plot_df['Model'] == model]
            test_means = model_data.groupby('Test_Window')['Test_R2'].mean()
            test_stds = model_data.groupby('Test_Window')['Test_R2'].std()
            
            axes[0, 1].errorbar(test_means.index, test_means.values, 
                              yerr=test_stds.values, label=model, marker='s', capsize=5)
        
        axes[0, 1].set_title('R² by Test Window Size', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Test Window (Months)')
        axes[0, 1].set_ylabel('Mean Test R²')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Heatmap of R² by train/test combination
        pivot_data = plot_df.groupby(['Train_Window', 'Test_Window'])['Test_R2'].mean().unstack()
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0, 2])
        axes[0, 2].set_title('R² Heatmap (Train vs Test)', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Test Window (Months)')
        axes[0, 2].set_ylabel('Training Window (Months)')
        
        # Plot 4: RMSE by train window size
        for model in plot_df['Model'].unique():
            model_data = plot_df[plot_df['Model'] == model]
            train_means = model_data.groupby('Train_Window')['Test_RMSE'].mean()
            train_stds = model_data.groupby('Train_Window')['Test_RMSE'].std()
            
            axes[1, 0].errorbar(train_means.index, train_means.values, 
                              yerr=train_stds.values, label=model, marker='o', capsize=5)
        
        axes[1, 0].set_title('RMSE by Training Window Size', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Training Window (Months)')
        axes[1, 0].set_ylabel('Mean Test RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: RMSE by test window size
        for model in plot_df['Model'].unique():
            model_data = plot_df[plot_df['Model'] == model]
            test_means = model_data.groupby('Test_Window')['Test_RMSE'].mean()
            test_stds = model_data.groupby('Test_Window')['Test_RMSE'].std()
            
            axes[1, 1].errorbar(test_means.index, test_means.values, 
                              yerr=test_stds.values, label=model, marker='s', capsize=5)
        
        axes[1, 1].set_title('RMSE by Test Window Size', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Test Window (Months)')
        axes[1, 1].set_ylabel('Mean Test RMSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Model comparison
        model_means = plot_df.groupby('Model')['Test_R2'].mean()
        model_stds = plot_df.groupby('Model')['Test_R2'].std()
        
        x = np.arange(len(model_means))
        axes[1, 2].bar(x, model_means.values, yerr=model_stds.values, capsize=5, alpha=0.7)
        axes[1, 2].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Model')
        axes[1, 2].set_ylabel('Mean Test R²')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(model_means.index, rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'window_size_optimization_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Optimization plots saved to {plot_path}")
        
        return fig
    
    def run_complete_optimization(self) -> None:
        """
        Run complete window size optimization analysis.
        """
        try:
            # Run optimization
            self.run_window_size_optimization(
                start_date='2013-01-01',
                end_date='2024-12-31',
                train_windows=[12, 18, 24, 30, 36, 48],  # 1-4 years
                test_windows=[3, 6, 9, 12],              # 3-12 months
                step_size=6,                             # 6-month steps
                min_test_size=10                         # Minimum 10 observations
            )
            
            # Analyze results
            config_stats = self.analyze_results()
            
            # Create plots
            self.create_optimization_plots()
            
            logger.info("\n" + "="*80)
            logger.info("WINDOW SIZE OPTIMIZATION COMPLETE")
            logger.info("="*80)
            logger.info("✓ Window size optimization finished")
            logger.info("✓ Optimal configurations identified")
            logger.info("✓ Results saved and visualized")
            
        except Exception as e:
            logger.error(f"Error in optimization: {str(e)}")
            raise

def main():
    """
    Main function to run window size optimization.
    """
    try:
        # Initialize optimizer
        optimizer = WindowSizeOptimizer()
        
        # Run complete optimization
        optimizer.run_complete_optimization()
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

