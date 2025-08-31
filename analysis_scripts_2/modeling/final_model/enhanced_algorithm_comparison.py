#!/usr/bin/env python3
"""
Enhanced Algorithm Comparison for REVR Prediction
Compares Linear Regression, Random Forest, and XGBoost using rolling windows
Features: IEVR + normative_iv_rv_ratio + SKEW + KURT + IV_RATIO + SMIRK + vol_hl7 + vol_hl10 + vol_hl21 + z_score_momentum
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost'])
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True

class EnhancedAlgorithmComparison:
    """
    Enhanced comparison of algorithms for REVR prediction using rolling windows
    """
    
    def __init__(self):
        self.df = None
        self.features = [
            'ievr', 'normative_iv_rv_ratio', 'SKEW', 'KURT', 'IV_RATIO', 
            'SMIRK', 'vol_hl7', 'vol_hl10', 'vol_hl21', 'z_score_momentum'
        ]
        self.target = 'revr'
        self.results = []
        self.algorithm_configs = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("📊 LOADING DATASET")
        print("="*50)
        
        try:
            # Try momentum dataset first
            try:
                self.df = pd.read_csv('data_files/final_merged_dataset_with_momentum_final.csv')
                print(f"✅ Loaded momentum dataset: {len(self.df):,} observations")
            except FileNotFoundError:
                # Fallback to main dataset
                self.df = pd.read_csv('data_files/final_merged_dataset.csv')
                print(f"✅ Loaded main dataset: {len(self.df):,} observations")
            
            # Convert earnings_date to datetime
            self.df['earnings_date'] = pd.to_datetime(self.df['earnings_date'])
            
            # Add year column for reference
            self.df['year'] = self.df['earnings_date'].dt.year
            
            print(f"📅 Date range: {self.df['year'].min()} - {self.df['year'].max()}")
            
            # Calculate normative_iv_rv_ratio if missing
            if 'normative_iv_rv_ratio' not in self.df.columns:
                if 'avg_pre' in self.df.columns and 'normative_realized_vol' in self.df.columns:
                    print("🔧 Calculating normative_iv_rv_ratio...")
                    self.df['normative_iv_rv_ratio'] = self.df['avg_pre'] / self.df['normative_realized_vol']
                    self.df['normative_iv_rv_ratio'] = self.df['normative_iv_rv_ratio'].replace([np.inf, -np.inf], np.nan)
                    print(f"✅ Created normative_iv_rv_ratio from avg_pre / normative_realized_vol")
                else:
                    print("❌ Cannot calculate normative_iv_rv_ratio - missing avg_pre or normative_realized_vol")
            
            # Add z_score_momentum if missing (simple calculation)
            if 'z_score_momentum' not in self.df.columns:
                print("🔧 Creating simple z_score_momentum feature...")
                # Sort by ticker and date
                self.df = self.df.sort_values(['ticker', 'earnings_date'])
                
                # Calculate rolling momentum (6-month)
                self.df['momentum_6m'] = self.df.groupby('ticker')['revr'].rolling(window=4, min_periods=2).mean().reset_index(0, drop=True)
                
                # Calculate z-score momentum
                self.df['z_score_momentum'] = (
                    (self.df['momentum_6m'] - self.df.groupby('ticker')['momentum_6m'].transform('mean')) /
                    self.df.groupby('ticker')['momentum_6m'].transform('std')
                ).fillna(0)
                
                print(f"✅ Created z_score_momentum feature")
            
            # Check feature availability
            missing_features = [f for f in self.features if f not in self.df.columns]
            if missing_features:
                print(f"❌ Still missing features: {missing_features}")
                print("Available columns sample:", list(self.df.columns[:20]))
                return False
                
            available_features = [f for f in self.features if f in self.df.columns]
            print(f"✅ Available features: {len(available_features)}/{len(self.features)}")
            
            # Check data coverage
            feature_coverage = {}
            for feature in self.features:
                non_null_count = self.df[feature].notna().sum()
                coverage_pct = (non_null_count / len(self.df)) * 100
                feature_coverage[feature] = coverage_pct
                print(f"  {feature}: {non_null_count:,} ({coverage_pct:.1f}% coverage)")
            
            return True
            
        except FileNotFoundError:
            print("❌ Dataset file not found!")
            return False
    
    def create_rolling_windows(self, train_years=5, val_months=6, test_months=6):
        """
        Create rolling time windows for walk-forward validation.
        Same methodology as momentum analysis.
        """
        print(f"\n🔄 CREATING ROLLING WINDOWS")
        print("="*50)
        print(f"Training window: {train_years} years")
        print(f"Validation window: {val_months} months")
        print(f"Testing window: {test_months} months")
        
        # Get unique dates and sort
        unique_dates = self.df['earnings_date'].dt.to_period('M').unique()
        unique_dates = np.sort(unique_dates)
        
        windows = []
        current_idx = 0
        
        while current_idx < len(unique_dates):
            # Training period
            train_end = current_idx + (train_years * 12) - 1
            if train_end >= len(unique_dates):
                break
                
            # Validation period
            val_start = train_end + 1
            val_end = val_start + val_months - 1
            if val_end >= len(unique_dates):
                break
                
            # Testing period
            test_start = val_end + 1
            test_end = test_start + test_months - 1
            if test_end >= len(unique_dates):
                break
                
            # Create window
            window = {
                'train_start': unique_dates[current_idx],
                'train_end': unique_dates[train_end],
                'val_start': unique_dates[val_start],
                'val_end': unique_dates[val_end],
                'test_start': unique_dates[test_start],
                'test_end': unique_dates[test_end],
                'window_id': len(windows) + 1
            }
            
            windows.append(window)
            current_idx += test_months  # Move forward by test window size
        
        print(f"Created {len(windows)} rolling windows")
        
        # Show first few windows
        for i, window in enumerate(windows[:3]):
            print(f"  Window {i+1}: Train {window['train_start']}-{window['train_end']}, "
                  f"Val {window['val_start']}-{window['val_end']}, "
                  f"Test {window['test_start']}-{window['test_end']}")
        
        return windows
    
    def get_window_data(self, window):
        """
        Extract data for a specific rolling window
        """
        # Convert periods to datetime for filtering
        train_start = window['train_start'].to_timestamp()
        train_end = window['train_end'].to_timestamp()
        val_start = window['val_start'].to_timestamp()
        val_end = window['val_end'].to_timestamp()
        test_start = window['test_start'].to_timestamp()
        test_end = window['test_end'].to_timestamp()
        
        # Filter data for each period
        train_data = self.df[(self.df['earnings_date'] >= train_start) & (self.df['earnings_date'] <= train_end)]
        val_data = self.df[(self.df['earnings_date'] >= val_start) & (self.df['earnings_date'] <= val_end)]
        test_data = self.df[(self.df['earnings_date'] >= test_start) & (self.df['earnings_date'] <= test_end)]
        
        return train_data, val_data, test_data
    
    def setup_algorithms(self):
        """
        Configure the algorithms for comparison
        """
        self.algorithm_configs = {
            'LinearRegression': {
                'model': LinearRegression(),
                'name': 'Linear Regression',
                'color': '#003366',  # BlackRock dark blue
                'marker': 'o',
                'description': 'Multifactor Linear Regression'
            },
            'RandomForest': {
                'model': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'name': 'Random Forest',
                'color': '#66CCFF',  # BlackRock light blue
                'marker': 's',
                'description': 'Random Forest Regressor'
            },
            'XGBoost': {
                'model': XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                ),
                'name': 'XGBoost',
                'color': '#FF6633',  # BlackRock orange accent
                'marker': '^',
                'description': 'XGBoost Regressor'
            }
        }
        
        print(f"\n🤖 ALGORITHM CONFIGURATIONS")
        print("="*40)
        for algo_name, config in self.algorithm_configs.items():
            print(f"✅ {config['name']}: {config['description']}")
    
    def evaluate_algorithm(self, train_data, val_data, test_data, algorithm_config, algorithm_name):
        """
        Evaluate a single algorithm on the given data splits
        """
        try:
            # Prepare data
            all_required_cols = self.features + [self.target]
            train_clean = train_data[all_required_cols].dropna()
            test_clean = test_data[all_required_cols].dropna()
            
            if len(train_clean) < 50 or len(test_clean) < 5:
                return None
            
            # Prepare features and target
            X_train = train_clean[self.features]
            y_train = train_clean[self.target]
            X_test = test_clean[self.features]
            y_test = test_clean[self.target]
            
            # Train model
            model = algorithm_config['model']
            
            # Create a fresh model instance to avoid contamination
            if algorithm_name == 'LinearRegression':
                model = LinearRegression()
            elif algorithm_name == 'RandomForest':
                model = RandomForestRegressor(
                    n_estimators=100, max_depth=10, min_samples_split=20,
                    min_samples_leaf=10, random_state=42, n_jobs=-1
                )
            elif algorithm_name == 'XGBoost':
                model = XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    n_jobs=-1, verbosity=0
                )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(self.features, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(self.features, np.abs(model.coef_)))
            
            return {
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'train_size': len(train_clean),
                'test_size': len(test_clean),
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            print(f"    ❌ {algorithm_name} failed: {str(e)[:100]}")
            return None
    
    def run_rolling_comparison(self):
        """
        Run the complete rolling window comparison across all algorithms
        """
        print(f"\n🏃‍♂️ RUNNING ROLLING WINDOW COMPARISON")
        print("="*60)
        
        # Setup algorithms
        self.setup_algorithms()
        
        # Create rolling windows
        windows = self.create_rolling_windows()
        
        if not windows:
            print("❌ No rolling windows created")
            return None
        
        # Run analysis for each window
        all_results = []
        
        for window in windows:
            print(f"\n📊 Analyzing Window {window['window_id']} (Test: {window['test_start']} to {window['test_end']})...")
            
            # Get data for this window
            train_data, val_data, test_data = self.get_window_data(window)
            
            if len(train_data) < 100 or len(test_data) < 10:
                print(f"  ⚠️ Insufficient data - skipping window")
                continue
            
            window_results = {
                'window_id': window['window_id'],
                'train_start': str(window['train_start']),
                'train_end': str(window['train_end']),
                'val_start': str(window['val_start']),
                'val_end': str(window['val_end']),
                'test_start': str(window['test_start']),
                'test_end': str(window['test_end']),
                'test_year': int(str(window['test_start'])[:4])
            }
            
            # Test each algorithm
            for algo_name, algo_config in self.algorithm_configs.items():
                result = self.evaluate_algorithm(train_data, val_data, test_data, algo_config, algo_name)
                
                if result:
                    window_results[f'{algo_name}_r2'] = result['test_r2']
                    window_results[f'{algo_name}_rmse'] = result['test_rmse']
                    window_results[f'{algo_name}_mae'] = result['test_mae']
                    window_results[f'{algo_name}_train_size'] = result['train_size']
                    window_results[f'{algo_name}_test_size'] = result['test_size']
                    
                    print(f"  ✅ {algo_config['name']:15s}: R² = {result['test_r2']:6.4f}, RMSE = {result['test_rmse']:6.4f}")
                else:
                    window_results[f'{algo_name}_r2'] = np.nan
                    window_results[f'{algo_name}_rmse'] = np.nan
                    window_results[f'{algo_name}_mae'] = np.nan
                    window_results[f'{algo_name}_train_size'] = 0
                    window_results[f'{algo_name}_test_size'] = 0
                    
                    print(f"  ❌ {algo_config['name']:15s}: Failed")
            
            all_results.append(window_results)
        
        # Convert to DataFrame
        self.results = pd.DataFrame(all_results)
        
        if len(self.results) > 0:
            self.analyze_results()
        
        return self.results
    
    def analyze_results(self):
        """
        Analyze and summarize the rolling window results
        """
        print(f"\n📈 ALGORITHM PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Calculate average performance
        print("AVERAGE PERFORMANCE ACROSS ALL WINDOWS:")
        print("-" * 50)
        
        for algo_name, algo_config in self.algorithm_configs.items():
            r2_col = f'{algo_name}_r2'
            rmse_col = f'{algo_name}_rmse'
            
            if r2_col in self.results.columns:
                avg_r2 = self.results[r2_col].mean()
                std_r2 = self.results[r2_col].std()
                avg_rmse = self.results[rmse_col].mean()
                
                print(f"{algo_config['name']:15s}: R² = {avg_r2:.4f} (±{std_r2:.4f}), RMSE = {avg_rmse:.4f}")
        
        # Best performing algorithm
        r2_means = {}
        for algo_name, algo_config in self.algorithm_configs.items():
            r2_col = f'{algo_name}_r2'
            if r2_col in self.results.columns:
                r2_means[algo_name] = self.results[r2_col].mean()
        
        if r2_means:
            best_algo = max(r2_means.keys(), key=lambda x: r2_means[x])
            best_config = self.algorithm_configs[best_algo]
            
            print(f"\nBEST PERFORMING ALGORITHM:")
            print("-" * 30)
            print(f"🏆 {best_config['name']}: R² = {r2_means[best_algo]:.4f}")
        
        # Year-by-year breakdown
        print(f"\nYEAR-BY-YEAR PERFORMANCE:")
        print("-" * 30)
        
        yearly_summary = self.results.groupby('test_year').agg({
            f'{algo_name}_r2': 'mean' for algo_name in self.algorithm_configs.keys()
        }).round(4)
        
        print(yearly_summary.to_string())
    
    def create_visualizations(self):
        """
        Create comprehensive visualizations comparing the algorithms
        """
        print(f"\n📊 CREATING VISUALIZATIONS")
        print("="*40)
        
        # Set BlackRock styling
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'grid.color': '#E5E5E5'
        })
        
        # Create main comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor('white')
        
        # Left plot: Performance over time
        self._plot_performance_over_time(ax1)
        
        # Right plot: Average performance comparison
        self._plot_average_performance(ax2)
        
        plt.tight_layout()
        
        # Save main comparison
        output_path = 'output_files/enhanced_algorithm_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Algorithm comparison saved: {output_path}")
        
        plt.close()
        
        # Create focused time series plot
        self._create_focused_time_series_plot()
    
    def _plot_performance_over_time(self, ax):
        """Plot R² performance over time for all algorithms"""
        # Create proper x-axis using test period dates
        if 'test_start' in self.results.columns:
            x_axis = pd.to_datetime(self.results['test_start'], format='%Y-%m')
            x_label = 'Test Period Start'
        else:
            x_axis = self.results['window_id']
            x_label = 'Window Number'
        
        for algo_name, algo_config in self.algorithm_configs.items():
            r2_col = f'{algo_name}_r2'
            if r2_col in self.results.columns:
                ax.plot(x_axis, self.results[r2_col], 
                       label=algo_config['name'], 
                       color=algo_config['color'], 
                       marker=algo_config['marker'],
                       markersize=6, linewidth=2, alpha=0.8)
        
        ax.set_xlabel(x_label, fontsize=12, color='#003366', fontweight='semibold')
        ax.set_ylabel('Test R²', fontsize=12, color='#003366', fontweight='semibold')
        ax.set_title('Algorithm Performance Over Time', 
                     fontsize=14, fontweight='bold', color='#003366', pad=20)
        ax.legend(fontsize=10, loc='upper left', frameon=True, fancybox=True, 
                  shadow=True, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        # Highlight 2018 underperformance
        if 'test_start' in self.results.columns:
            ax.axvspan(pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'), 
                       alpha=0.2, color='#FF6633', label='2018 Market Stress')
        elif 'window_id' in self.results.columns:
            windows_2018 = self.results[self.results['test_year'] == 2018]['window_id']
            if len(windows_2018) > 0:
                min_window = windows_2018.min() - 0.5
                max_window = windows_2018.max() + 0.5
                ax.axvspan(min_window, max_window, alpha=0.2, color='#FF6633')
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='#8C8C8C', linestyle='-', alpha=0.7, linewidth=1)
    
    def _plot_average_performance(self, ax):
        """Plot average performance comparison"""
        algorithms = []
        avg_r2s = []
        colors = []
        
        for algo_name, algo_config in self.algorithm_configs.items():
            r2_col = f'{algo_name}_r2'
            if r2_col in self.results.columns:
                avg_r2 = self.results[r2_col].mean()
                algorithms.append(algo_config['name'])
                avg_r2s.append(avg_r2)
                colors.append(algo_config['color'])
        
        bars = ax.bar(algorithms, avg_r2s, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_r2s):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylabel('Average Test R²', fontsize=12, color='#003366', fontweight='semibold')
        ax.set_title('Average Performance Comparison', 
                     fontsize=14, fontweight='bold', color='#003366', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='#8C8C8C', linestyle='-', alpha=0.7, linewidth=1)
    
    def _create_focused_time_series_plot(self):
        """Create a focused time series plot for presentations"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.patch.set_facecolor('white')
        
        # Plot performance over time
        if 'test_start' in self.results.columns:
            x_axis = pd.to_datetime(self.results['test_start'], format='%Y-%m')
            x_label = 'Test Period Start'
        else:
            x_axis = self.results['window_id']
            x_label = 'Window Number'
        
        for algo_name, algo_config in self.algorithm_configs.items():
            r2_col = f'{algo_name}_r2'
            if r2_col in self.results.columns:
                ax.plot(x_axis, self.results[r2_col], 
                       label=algo_config['name'], 
                       color=algo_config['color'], 
                       marker=algo_config['marker'],
                       markersize=8, linewidth=3, alpha=0.9,
                       markeredgewidth=0.5, markeredgecolor='white')
        
        ax.set_xlabel(x_label, fontsize=14, color='#003366', fontweight='bold')
        ax.set_ylabel('Test R²', fontsize=14, color='#003366', fontweight='bold')
        ax.set_title('REVR Prediction: Algorithm Performance Comparison\n'
                     'Rolling Window Analysis (5-Year Training, 6-Month Testing)', 
                     fontsize=16, fontweight='bold', color='#003366', pad=25)
        
        # Enhanced legend
        ax.legend(fontsize=12, loc='upper left', frameon=True, fancybox=True, 
                  shadow=True, framealpha=0.95, edgecolor='#003366')
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.2, top=0.4)
        
        # Highlight 2018 with better styling
        if 'test_start' in self.results.columns:
            ax.axvspan(pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'), 
                       alpha=0.15, color='red', label='2018 Market Regime')
            ax.text(pd.Timestamp('2018-06-15'), ax.get_ylim()[1]*0.9, '2018\nMarket Stress', 
                    ha='center', va='top', fontsize=11, color='darkred', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.8))
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='#8C8C8C', linestyle='-', alpha=0.7, linewidth=1.5)
        ax.text(ax.get_xlim()[0], 0.01, 'No Predictive Power', 
                fontsize=10, color='#8C8C8C', alpha=0.8, style='italic')
        
        plt.tight_layout()
        
        # Save focused plot
        output_path = 'output_files/enhanced_algorithm_time_series.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Focused time series saved: {output_path}")
        
        # Save SVG version
        output_path_svg = 'output_files/enhanced_algorithm_time_series.svg'
        plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white')
        print(f"✅ SVG version saved: {output_path_svg}")
        
        plt.close()
    
    def save_results(self):
        """Save detailed results to CSV files"""
        print(f"\n💾 SAVING RESULTS")
        print("="*30)
        
        if len(self.results) > 0:
            # Save detailed results
            results_path = 'output_files/enhanced_algorithm_results.csv'
            self.results.to_csv(results_path, index=False)
            print(f"✅ Detailed results saved: {results_path}")
            
            # Create summary statistics
            summary_stats = {}
            for algo_name, algo_config in self.algorithm_configs.items():
                r2_col = f'{algo_name}_r2'
                rmse_col = f'{algo_name}_rmse'
                
                if r2_col in self.results.columns:
                    summary_stats[f'{algo_name}_mean_r2'] = self.results[r2_col].mean()
                    summary_stats[f'{algo_name}_std_r2'] = self.results[r2_col].std()
                    summary_stats[f'{algo_name}_mean_rmse'] = self.results[rmse_col].mean()
                    summary_stats[f'{algo_name}_std_rmse'] = self.results[rmse_col].std()
            
            summary_df = pd.DataFrame([summary_stats])
            summary_path = 'output_files/enhanced_algorithm_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"✅ Summary statistics saved: {summary_path}")
        
        print("\n🎉 ENHANCED ALGORITHM COMPARISON COMPLETED!")
        print(f"Key outputs:")
        print(f"  • enhanced_algorithm_comparison.png - Main comparison")
        print(f"  • enhanced_algorithm_time_series.png - Focused time series")
        print(f"  • enhanced_algorithm_results.csv - Detailed results")
        print(f"  • enhanced_algorithm_summary.csv - Summary statistics")

def main():
    """
    Main function to run enhanced algorithm comparison
    """
    try:
        print("🚀 ENHANCED ALGORITHM COMPARISON FOR REVR PREDICTION")
        print("="*80)
        print("Features: IEVR + normative_iv_rv_ratio + SKEW + KURT + IV_RATIO + SMIRK + vol_hl7 + vol_hl10 + vol_hl21 + z_score_momentum")
        print("Algorithms: Linear Regression, Random Forest, XGBoost")
        print("Methodology: 5-Year Training, 6-Month Validation, 6-Month Testing Rolling Windows")
        print("="*80)
        
        # Initialize analyzer
        analyzer = EnhancedAlgorithmComparison()
        
        # Load data
        if not analyzer.load_data():
            return
        
        # Run rolling window comparison
        results = analyzer.run_rolling_comparison()
        
        if results is not None and len(results) > 0:
            # Create visualizations
            analyzer.create_visualizations()
            
            # Save results
            analyzer.save_results()
        else:
            print("❌ No results generated")
            
    except Exception as e:
        print(f"❌ Error in enhanced algorithm comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
