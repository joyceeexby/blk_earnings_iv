"""
Rolling Walk-Forward Analysis for IEVR-REVR Relationship

This module implements a rolling walk-forward analysis to evaluate how the relationship
between Implied Earnings Volatility Ratio (IEVR) and Realized Earnings Volatility Ratio (REVR)
evolves over time and how model performance changes in different market conditions.

Key Features:
- Expanding window approach (starts with 2 years, adds 1 year at a time)
- Temporal validation (no future data leakage)
- Performance tracking over time
- Model stability analysis
- Market regime detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RollingWalkForwardAnalysis:
    """
    Class for implementing rolling walk-forward analysis for IEVR-REVR relationship.
    """
    
    def __init__(self, data_file='data_files/expanded_earnings_analysis_results.csv'):
        """
        Initialize the rolling walk-forward analysis.
        
        Parameters:
        -----------
        data_file : str
            Path to the CSV file containing the analysis results
        """
        self.data_file = data_file
        self.data = None
        self.results = []
        self.scaler = StandardScaler()
        
        # Load and prepare data
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """
        Load data and prepare features for rolling analysis.
        """
        print("Loading and preparing data for rolling walk-forward analysis...")
        
        # Load data
        self.data = pd.read_csv(self.data_file)
        
        # Convert earnings_date to datetime
        self.data['earnings_date'] = pd.to_datetime(self.data['earnings_date'])
        
        # Sort by date to ensure temporal order
        self.data = self.data.sort_values('earnings_date').reset_index(drop=True)
        
        # Clean data - remove NaN and infinite values
        self.data = self.data.dropna(subset=['revr', 'ievr'])
        self.data = self.data[np.isfinite(self.data['revr']) & np.isfinite(self.data['ievr'])]
        
        # Create additional features
        self.create_features()
        
        # Prepare features and target
        feature_columns = ['ievr', 'normative_iv_rv_ratio', 'skew_ratio']
        available_features = [col for col in feature_columns if col in self.data.columns]
        
        if len(available_features) < 2:
            available_features = ['ievr']
            print("Warning: Limited features available, using only IEVR")
        
        # Add features to data
        X = self.data[available_features].copy()
        y = self.data['revr'].copy()
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        self.data = self.data[mask].reset_index(drop=True)
        
        print(f"Final dataset: {len(self.data)} observations from {self.data['earnings_date'].min().strftime('%Y-%m-%d')} to {self.data['earnings_date'].max().strftime('%Y-%m-%d')}")
        print(f"Features: {available_features}")
        
        # Store feature columns for later use
        self.feature_columns = available_features
    
    def create_features(self):
        """
        Create additional features for the analysis.
        """
        # Create normative IV/RV ratio feature
        self.create_normative_iv_rv_ratio()
        
        # Create skew ratio feature
        self.create_skew_ratio()
    
    def create_normative_iv_rv_ratio(self):
        """
        Create normative IV/RV ratio feature.
        """
        print("Creating normative IV/RV ratio feature...")
        
        # Calculate normative IV/RV ratio
        self.data['normative_iv_rv_ratio'] = (
            self.data['normative_implied_vol'] / self.data['normative_realized_vol']
        )
        
        # Remove infinite values
        self.data = self.data[np.isfinite(self.data['normative_iv_rv_ratio'])]
        
        print(f"Created normative_iv_rv_ratio feature. Non-null values: {len(self.data)}")
        print(f"  Mean: {self.data['normative_iv_rv_ratio'].mean():.4f}")
        print(f"  Std: {self.data['normative_iv_rv_ratio'].std():.4f}")
        print(f"  Min: {self.data['normative_iv_rv_ratio'].min():.4f}")
        print(f"  Max: {self.data['normative_iv_rv_ratio'].max():.4f}")
        
        # Check if IV > RV on average (typical volatility risk premium)
        iv_rv_ratio_mean = self.data['normative_iv_rv_ratio'].mean()
        if iv_rv_ratio_mean > 1.0:
            print(f"  ✓ IV > RV on average (typical volatility risk premium)")
        else:
            print(f"  ⚠ IV ≤ RV on average (unusual)")
    
    def create_skew_ratio(self):
        """
        Create skew ratio feature (90Put / 110Call).
        """
        print("Creating skew ratio feature (90Put / 110Call)...")
        
        # Use existing skew_ratio if available, otherwise create it
        if 'skew_ratio' not in self.data.columns:
            # Create a simple skew ratio based on available data
            # This is a placeholder - you might want to implement actual skew calculation
            self.data['skew_ratio'] = 1.0  # Default value
        
        # Remove infinite values
        self.data = self.data[np.isfinite(self.data['skew_ratio'])]
        
        print(f"Created skew_ratio feature. Non-null values: {len(self.data)}")
        print(f"  Mean: {self.data['skew_ratio'].mean():.4f}")
        print(f"  Std: {self.data['skew_ratio'].std():.4f}")
        print(f"  Min: {self.data['skew_ratio'].min():.4f}")
        print(f"  Max: {self.data['skew_ratio'].max():.4f}")
        
        # Check correlation with REVR
        correlation = self.data['skew_ratio'].corr(self.data['revr'])
        print(f"  Correlation with REVR: {correlation:.4f}")
        
        if correlation > 0.1:
            print(f"  ✓ Skew ratio shows positive correlation with REVR")
        elif correlation < -0.1:
            print(f"  ✓ Skew ratio shows negative correlation with REVR")
        else:
            print(f"  ⚠ Skew ratio shows weak correlation with REVR")
    
    def run_rolling_analysis(self, initial_years=2, step_years=1, min_train_size=50):
        """
        Run rolling walk-forward analysis.
        
        Parameters:
        -----------
        initial_years : int
            Number of years to use for initial training set
        step_years : int
            Number of years to add to training set in each step
        min_train_size : int
            Minimum number of observations required for training
        """
        print(f"\n{'='*80}")
        print("ROLLING WALK-FORWARD ANALYSIS")
        print(f"{'='*80}")
        
        # Get unique years from the data
        self.data['year'] = self.data['earnings_date'].dt.year
        unique_years = sorted(self.data['year'].unique())
        
        print(f"Available years: {unique_years}")
        print(f"Initial training period: {initial_years} years")
        print(f"Step size: {step_years} year(s)")
        
        # Initialize results storage
        self.results = []
        
        # Start with initial training period
        start_year = unique_years[0]
        current_train_end_year = start_year + initial_years - 1
        
        while current_train_end_year < unique_years[-1] - 1:  # Leave at least 1 year for testing
            # Define training and test periods
            train_mask = self.data['year'] <= current_train_end_year
            test_mask = (self.data['year'] > current_train_end_year) & (self.data['year'] <= current_train_end_year + step_years)
            
            train_data = self.data[train_mask]
            test_data = self.data[test_mask]
            
            # Check if we have enough data
            if len(train_data) < min_train_size:
                print(f"Warning: Insufficient training data for {current_train_end_year} ({len(train_data)} < {min_train_size})")
                current_train_end_year += step_years
                continue
            
            if len(test_data) == 0:
                print(f"Warning: No test data for {current_train_end_year}")
                current_train_end_year += step_years
                continue
            
            print(f"\n{'='*60}")
            print(f"TRAINING PERIOD: {start_year}-{current_train_end_year}")
            print(f"TEST PERIOD: {current_train_end_year + 1}-{current_train_end_year + step_years}")
            print(f"{'='*60}")
            print(f"Training observations: {len(train_data)}")
            print(f"Test observations: {len(test_data)}")
            
            # Prepare features
            X_train = train_data[self.feature_columns]
            y_train = train_data['revr']
            X_test = test_data[self.feature_columns]
            y_test = test_data['revr']
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            rf_results = self.train_and_evaluate_model(
                'Random Forest', RandomForestRegressor(n_estimators=100, random_state=42),
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            
            xgb_results = self.train_and_evaluate_model(
                'XGBoost', xgb.XGBRegressor(n_estimators=100, random_state=42),
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            
            # Store results
            period_results = {
                'train_start_year': start_year,
                'train_end_year': current_train_end_year,
                'test_start_year': current_train_end_year + 1,
                'test_end_year': current_train_end_year + step_years,
                'train_observations': len(train_data),
                'test_observations': len(test_data),
                'rf_results': rf_results,
                'xgb_results': xgb_results
            }
            
            self.results.append(period_results)
            
            # Move to next period
            current_train_end_year += step_years
        
        print(f"\n{'='*80}")
        print("ROLLING ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Total periods analyzed: {len(self.results)}")
    
    def train_and_evaluate_model(self, model_name, model, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a single model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model : sklearn estimator
            The model to train
        X_train, y_train : array-like
            Training data
        X_test, y_test : array-like
            Test data
        
        Returns:
        --------
        dict : Model results
        """
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        

        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
        else:
            feature_importance = {col: 0.0 for col in self.feature_columns}
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Training RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Training MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        

        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'feature_importance': feature_importance,
            'predictions': y_test_pred,
            'actual': y_test.values
        }
    
    def create_performance_plots(self):
        """
        Create plots showing performance over time.
        """
        if not self.results:
            print("No results to plot. Run rolling analysis first.")
            return
        
        print("\nCreating performance plots...")
        
        # Prepare data for plotting
        periods = []
        rf_test_r2 = []
        xgb_test_r2 = []
        rf_test_rmse = []
        xgb_test_rmse = []
        
        for result in self.results:
            periods.append(f"{result['test_start_year']}-{result['test_end_year']}")
            rf_test_r2.append(result['rf_results']['test_r2'])
            xgb_test_r2.append(result['xgb_results']['test_r2'])
            rf_test_rmse.append(result['rf_results']['test_rmse'])
            xgb_test_rmse.append(result['xgb_results']['test_rmse'])
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Rolling Walk-Forward Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: R² over time
        axes[0, 0].plot(periods, rf_test_r2, 'o-', label='Random Forest', linewidth=2, markersize=6)
        axes[0, 0].plot(periods, xgb_test_r2, 's-', label='XGBoost', linewidth=2, markersize=6)
        axes[0, 0].set_title('Test R² Over Time')
        axes[0, 0].set_xlabel('Test Period')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: RMSE over time
        axes[0, 1].plot(periods, rf_test_rmse, 'o-', label='Random Forest', linewidth=2, markersize=6)
        axes[0, 1].plot(periods, xgb_test_rmse, 's-', label='XGBoost', linewidth=2, markersize=6)
        axes[0, 1].set_title('Test RMSE Over Time')
        axes[0, 1].set_xlabel('Test Period')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Training set size over time
        train_sizes = [result['train_observations'] for result in self.results]
        axes[1, 0].plot(periods, train_sizes, 'o-', color='green', linewidth=2, markersize=6)
        axes[1, 0].set_title('Training Set Size Over Time')
        axes[1, 0].set_xlabel('Test Period')
        axes[1, 0].set_ylabel('Number of Observations')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Model comparison (average performance)
        avg_rf_r2 = np.mean(rf_test_r2)
        avg_xgb_r2 = np.mean(xgb_test_r2)
        avg_rf_rmse = np.mean(rf_test_rmse)
        avg_xgb_rmse = np.mean(xgb_test_rmse)
        
        x = np.arange(2)
        width = 0.35
        
        axes[1, 1].bar(x - width/2, [avg_rf_r2, avg_rf_rmse], width, label='Random Forest', alpha=0.8)
        axes[1, 1].bar(x + width/2, [avg_xgb_r2, avg_xgb_rmse], width, label='XGBoost', alpha=0.8)
        axes[1, 1].set_title('Average Performance Comparison')
        axes[1, 1].set_xlabel('Metric')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['R²', 'RMSE'])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output_files/rolling_walk_forward_results.png', dpi=300, bbox_inches='tight')
        print("✓ Performance plots saved to output_files/rolling_walk_forward_results.png")
        
        return fig
    
    def create_summary_table(self):
        """
        Create a summary table of the rolling analysis results.
        """
        if not self.results:
            print("No results to summarize. Run rolling analysis first.")
            return
        
        print("\nCreating summary table...")
        
        # Prepare summary data
        summary_data = []
        
        for result in self.results:
            summary_data.append({
                'Train_Period': f"{result['train_start_year']}-{result['train_end_year']}",
                'Test_Period': f"{result['test_start_year']}-{result['test_end_year']}",
                'Train_Obs': result['train_observations'],
                'Test_Obs': result['test_observations'],
                'RF_R2': result['rf_results']['test_r2'],
                'XGB_R2': result['xgb_results']['test_r2'],
                'RF_RMSE': result['rf_results']['test_rmse'],
                'XGB_RMSE': result['xgb_results']['test_rmse'],
                'RF_MAE': result['rf_results']['test_mae'],
                'XGB_MAE': result['xgb_results']['test_mae']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_df.to_csv('data_files/rolling_walk_forward_summary.csv', index=False)
        print("✓ Summary table saved to data_files/rolling_walk_forward_summary.csv")
        
        # Print summary statistics
        print(f"\n{'='*80}")
        print("ROLLING WALK-FORWARD SUMMARY STATISTICS")
        print(f"{'='*80}")
        
        print(f"\nRandom Forest Performance:")
        print(f"  Average Test R²: {summary_df['RF_R2'].mean():.4f} (±{summary_df['RF_R2'].std():.4f})")
        print(f"  Average Test RMSE: {summary_df['RF_RMSE'].mean():.4f} (±{summary_df['RF_RMSE'].std():.4f})")
        print(f"  Average Test MAE: {summary_df['RF_MAE'].mean():.4f} (±{summary_df['RF_MAE'].std():.4f})")
        print(f"  Best Test R²: {summary_df['RF_R2'].max():.4f} (Period: {summary_df.loc[summary_df['RF_R2'].idxmax(), 'Test_Period']})")
        print(f"  Worst Test R²: {summary_df['RF_R2'].min():.4f} (Period: {summary_df.loc[summary_df['RF_R2'].idxmin(), 'Test_Period']})")
        
        print(f"\nXGBoost Performance:")
        print(f"  Average Test R²: {summary_df['XGB_R2'].mean():.4f} (±{summary_df['XGB_R2'].std():.4f})")
        print(f"  Average Test RMSE: {summary_df['XGB_RMSE'].mean():.4f} (±{summary_df['XGB_RMSE'].std():.4f})")
        print(f"  Average Test MAE: {summary_df['XGB_MAE'].mean():.4f} (±{summary_df['XGB_MAE'].std():.4f})")
        print(f"  Best Test R²: {summary_df['XGB_R2'].max():.4f} (Period: {summary_df.loc[summary_df['XGB_R2'].idxmax(), 'Test_Period']})")
        print(f"  Worst Test R²: {summary_df['XGB_R2'].min():.4f} (Period: {summary_df.loc[summary_df['XGB_R2'].idxmin(), 'Test_Period']})")
        
        # Model comparison
        rf_better = (summary_df['RF_R2'] > summary_df['XGB_R2']).sum()
        xgb_better = (summary_df['XGB_R2'] > summary_df['RF_R2']).sum()
        ties = len(summary_df) - rf_better - xgb_better
        
        print(f"\nModel Comparison:")
        print(f"  Random Forest better: {rf_better} periods")
        print(f"  XGBoost better: {xgb_better} periods")
        print(f"  Ties: {ties} periods")
        
        return summary_df
    
    def run_complete_analysis(self, initial_years=2, step_years=1):
        """
        Run the complete rolling walk-forward analysis.
        
        Parameters:
        -----------
        initial_years : int
            Number of years to use for initial training set
        step_years : int
            Number of years to add to training set in each step
        """
        # Run rolling analysis
        self.run_rolling_analysis(initial_years=initial_years, step_years=step_years)
        
        # Create plots
        self.create_performance_plots()
        
        # Create summary
        summary_df = self.create_summary_table()
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        
        return summary_df

def main():
    """
    Main function to run the rolling walk-forward analysis.
    """
    # Initialize analysis
    analysis = RollingWalkForwardAnalysis()
    
    # Run complete analysis with maximum possible training set (7 years)
    summary = analysis.run_complete_analysis(initial_years=7, step_years=1)
    
    return summary

if __name__ == "__main__":
    main() 