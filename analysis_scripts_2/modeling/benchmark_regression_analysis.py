#!/usr/bin/env python3
"""
Benchmark Regression Analysis: Simple vs IEVR/REVR Model Comparison
Creates a simple benchmark using average of last 4 earnings REVR and compares it with 
the sophisticated IEVR/REVR model to demonstrate the value of forward-looking features.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BenchmarkAnalysis:
    """
    Comprehensive benchmark analysis comparing simple historical average vs IEVR/REVR model
    """
    
    def __init__(self, data_file_path='data_files/final_merged_dataset_with_momentum_final.csv'):
        """
        Initialize the benchmark analysis
        """
        print("🏆 BENCHMARK REGRESSION ANALYSIS")
        print("="*80)
        print("Comparing Simple Historical Average vs IEVR/REVR Model")
        print("="*80)
        
        # Load data
        self.df = pd.read_csv(data_file_path)
        self.df['earnings_date'] = pd.to_datetime(self.df['earnings_date'])
        self.df['year'] = self.df['earnings_date'].dt.year
        self.df['quarter'] = self.df['earnings_date'].dt.quarter
        
        # Create normative IV/RV ratio
        self.df['normative_iv_rv_ratio'] = self.df['avg_pre'] / self.df['normative_realized_vol']
        self.df['normative_iv_rv_ratio'] = self.df['normative_iv_rv_ratio'].replace([np.inf, -np.inf], np.nan)
        
        print(f"✅ Loaded dataset: {len(self.df):,} observations")
        print(f"📅 Date range: {self.df['year'].min()} - {self.df['year'].max()}")
        
        # Sort by ticker and date for lag calculations
        self.df = self.df.sort_values(['ticker', 'earnings_date']).reset_index(drop=True)
        
    def create_benchmark_features(self):
        """
        Create benchmark features: average of last N earnings REVR values
        """
        print(f"\n📊 CREATING BENCHMARK FEATURES")
        print("-" * 50)
        
        # Create lagged REVR features
        lag_periods = [1, 2, 3, 4, 6, 8]  # Different lookback periods
        
        for lag in lag_periods:
            self.df[f'revr_lag_{lag}'] = self.df.groupby('ticker')['revr'].shift(lag)
        
        # Create rolling averages of different lengths
        rolling_windows = [2, 3, 4, 6, 8]
        
        for window in rolling_windows:
            self.df[f'revr_avg_{window}'] = self.df.groupby('ticker')['revr'].rolling(
                window=window, min_periods=window
            ).mean().reset_index(0, drop=True).shift(1)  # Shift to avoid look-ahead bias
        
        # Primary benchmark: Average of last 4 earnings
        self.df['benchmark_revr_4avg'] = self.df['revr_avg_4']
        
        # Alternative benchmarks
        self.df['benchmark_revr_last1'] = self.df['revr_lag_1']  # Just last earnings
        self.df['benchmark_revr_6avg'] = self.df['revr_avg_6']   # Longer average
        
        # Show coverage
        print("Benchmark feature coverage:")
        for feature in ['benchmark_revr_4avg', 'benchmark_revr_last1', 'benchmark_revr_6avg']:
            valid_count = self.df[feature].notna().sum()
            total_count = len(self.df)
            coverage = 100.0 * valid_count / total_count
            print(f"  {feature:20s}: {valid_count:6,} ({coverage:5.1f}% coverage)")
        
        # Check data quality
        self._validate_benchmark_features()
        
    def _validate_benchmark_features(self):
        """
        Validate that benchmark features make sense
        """
        print(f"\n🔍 BENCHMARK FEATURE VALIDATION")
        print("-" * 40)
        
        # Check for reasonable values
        for feature in ['benchmark_revr_4avg', 'benchmark_revr_last1']:
            if feature in self.df.columns:
                feature_data = self.df[feature].dropna()
                if len(feature_data) > 0:
                    print(f"{feature}:")
                    print(f"  Mean: {feature_data.mean():.4f}")
                    print(f"  Std:  {feature_data.std():.4f}")
                    print(f"  Range: [{feature_data.min():.4f}, {feature_data.max():.4f}]")
                    
                    # Check correlation with current REVR
                    current_revr = self.df.loc[feature_data.index, 'revr']
                    correlation = feature_data.corr(current_revr)
                    print(f"  Correlation with current REVR: {correlation:.4f}")
                    print()
        
        # Sample some cases to verify logic
        print("Sample validation (showing lag structure for first ticker):")
        first_ticker = self.df['ticker'].iloc[0]
        sample_data = self.df[self.df['ticker'] == first_ticker].head(10)
        
        columns_to_show = ['ticker', 'earnings_date', 'revr', 'revr_lag_1', 'revr_lag_2', 'benchmark_revr_4avg']
        available_columns = [col for col in columns_to_show if col in sample_data.columns]
        print(sample_data[available_columns].to_string(index=False))
        
    def run_benchmark_comparison(self, use_rolling_windows=True):
        """
        Run comprehensive benchmark comparison using proper rolling window methodology
        """
        print(f"\n🏁 RUNNING BENCHMARK COMPARISON")
        print("="*70)
        
        if use_rolling_windows:
            print("Using 5-year training, 6-month validation, 6-month testing rolling windows")
            return self._run_rolling_window_comparison()
        else:
            print("Using simple yearly train/test split")
            return self._run_yearly_comparison()
    
    def _run_rolling_window_comparison(self):
        """
        Run benchmark comparison using the same rolling window methodology as momentum analysis
        """
        # Create rolling windows using the same methodology as momentum analysis
        windows = self._create_rolling_windows()
        
        if not windows:
            print("No rolling windows created")
            return None
        
        print(f"Created {len(windows)} rolling windows")
        
        # Show first few windows for verification
        print("\nFirst 3 rolling windows:")
        for i, window in enumerate(windows[:3]):
            print(f"  Window {i+1}: Train {window['train_start']}-{window['train_end']}, "
                  f"Val {window['val_start']}-{window['val_end']}, "
                  f"Test {window['test_start']}-{window['test_end']}")
        
        # Run analysis on each window
        all_results = []
        
        for i, window in enumerate(windows):
            print(f"\nAnalyzing window {window['window_id']} (Test: {window['test_start']} to {window['test_end']})...")
            
            # Get data for this window
            train_data, val_data, test_data = self._get_window_data(window)
            
            if len(train_data) < 100 or len(test_data) < 10:
                print(f"  Insufficient data - skipping window")
                continue
            
            window_results = {
                'window_id': window['window_id'],
                'train_start': str(window['train_start']),
                'train_end': str(window['train_end']),
                'val_start': str(window['val_start']),
                'val_end': str(window['val_end']),
                'test_start': str(window['test_start']),
                'test_end': str(window['test_end']),
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data),
                'test_year': int(str(window['test_start'])[:4])
            }
            
            # Define model configurations with BlackRock styling
            models = {
                'Benchmark_4Avg': {
                    'features': ['benchmark_revr_4avg'],
                    'name': 'Benchmark:\nAvg Last 4 REVR',
                    'color': '#8C8C8C',
                    'linestyle': '--',
                    'marker': 'o'
                },
                'Simple_IEVR': {
                    'features': ['ievr'],
                    'name': 'Simple:\nREVR ~ IEVR',
                    'color': '#66CCFF',
                    'linestyle': '-',
                    'marker': 's'
                },
                'Multi_Factor': {
                    'features': ['ievr', 'normative_iv_rv_ratio', 'SKEW', 'KURT', 'IV_RATIO', 'SMIRK', 'vol_hl7', 'vol_hl10', 'vol_hl21', 'z_score_momentum'],
                    'name': 'Multi-Factor:\nREVR ~ IEVR + Features',
                    'color': '#003366',
                    'linestyle': '-',
                    'marker': '^'
                }
            }
            
            # Test each model on this window
            for model_name, model_config in models.items():
                try:
                    result = self._evaluate_model(
                        train_data, test_data, 
                        model_config['features'], 
                        model_name
                    )
                    
                    window_results[f'{model_name}_r2'] = result['test_r2']
                    window_results[f'{model_name}_rmse'] = result['test_rmse']
                    window_results[f'{model_name}_mae'] = result['test_mae']
                    window_results[f'{model_name}_n_obs'] = result['n_obs']
                    
                    print(f"  {model_config['name']:25s}: R² = {result['test_r2']:6.4f}, RMSE = {result['test_rmse']:6.4f}")
                    
                except Exception as e:
                    print(f"  {model_config['name']:25s}: Failed ({str(e)[:50]})")
                    window_results[f'{model_name}_r2'] = np.nan
                    window_results[f'{model_name}_rmse'] = np.nan
                    window_results[f'{model_name}_mae'] = np.nan
                    window_results[f'{model_name}_n_obs'] = 0
            
            all_results.append(window_results)
        
        # Convert to DataFrame and analyze
        self.yearly_results = pd.DataFrame(all_results)
        self.models = models
        
        if len(self.yearly_results) > 0:
            self._analyze_benchmark_results()
        
        return self.yearly_results
        
    def _evaluate_model(self, train_data, test_data, features, model_name):
        """
        Evaluate a single model configuration
        """
        # Prepare training data
        train_features = train_data[features + ['revr']].dropna()
        
        if len(train_features) < 50:
            raise ValueError(f"Insufficient training data: {len(train_features)}")
        
        X_train = train_features[features].values
        y_train = train_features['revr'].values
        
        # Prepare test data
        test_features = test_data[features + ['revr']].dropna()
        
        if len(test_features) < 10:
            raise ValueError(f"Insufficient test data: {len(test_features)}")
        
        X_test = test_features[features].values
        y_test = test_features['revr'].values
        
        # Handle benchmark models (no fitting required)
        if len(features) == 1 and 'benchmark' in features[0]:
            # For benchmark models, prediction is just the feature value
            y_pred = X_test.flatten()
        else:
            # For IEVR/REVR models, fit linear regression
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        
        return {
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'n_obs': len(test_features)
        }
    
    def _analyze_benchmark_results(self):
        """
        Analyze and summarize benchmark comparison results
        """
        print(f"\n📈 BENCHMARK ANALYSIS SUMMARY")
        print("="*60)
        
        if len(self.yearly_results) == 0:
            print("No results to analyze")
            return
        
        # Calculate average performance across all years
        print("AVERAGE PERFORMANCE ACROSS ALL YEARS:")
        print("-" * 50)
        
        model_performance = {}
        
        for model_name, model_config in self.models.items():
            r2_col = f'{model_name}_r2'
            rmse_col = f'{model_name}_rmse'
            
            if r2_col in self.yearly_results.columns:
                avg_r2 = self.yearly_results[r2_col].mean()
                avg_rmse = self.yearly_results[rmse_col].mean()
                std_r2 = self.yearly_results[r2_col].std()
                
                model_performance[model_name] = {
                    'avg_r2': avg_r2,
                    'avg_rmse': avg_rmse,
                    'std_r2': std_r2,
                    'name': model_config['name']
                }
                
                print(f"{model_config['name']:25s}: R² = {avg_r2:6.4f} (±{std_r2:5.4f}), RMSE = {avg_rmse:6.4f}")
        
        # Calculate improvements over benchmark
        print(f"\nIMPROVEMENT OVER BENCHMARKS:")
        print("-" * 40)
        
        if 'Benchmark_4Avg' in model_performance and 'IEVR_REVR_Enhanced' in model_performance:
            benchmark_r2 = model_performance['Benchmark_4Avg']['avg_r2']
            enhanced_r2 = model_performance['IEVR_REVR_Enhanced']['avg_r2']
            
            if benchmark_r2 > 0:
                improvement = (enhanced_r2 - benchmark_r2) / benchmark_r2 * 100
                absolute_improvement = enhanced_r2 - benchmark_r2
                
                print(f"IEVR/REVR Enhanced vs 4-Period Average Benchmark:")
                print(f"  Benchmark R²: {benchmark_r2:.4f}")
                print(f"  Enhanced R²:  {enhanced_r2:.4f}")
                print(f"  Improvement:  +{absolute_improvement:.4f} ({improvement:+.1f}%)")
                
                if improvement > 20:
                    print("  ✅ Strong improvement over simple benchmark!")
                elif improvement > 5:
                    print("  ✅ Meaningful improvement over benchmark")
                else:
                    print("  ⚠️  Limited improvement over benchmark")
        
        # Identify best performers
        print(f"\nBEST PERFORMING MODELS:")
        print("-" * 30)
        
        sorted_models = sorted(model_performance.items(), 
                             key=lambda x: x[1]['avg_r2'], reverse=True)
        
        for i, (model_name, perf) in enumerate(sorted_models[:5]):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            print(f"  {rank_emoji} {perf['name']:25s}: R² = {perf['avg_r2']:.4f}")
        
        self.model_performance = model_performance
    
    def _create_rolling_windows(self, train_years=5, val_months=6, test_months=6):
        """
        Create rolling time windows using the same methodology as momentum analysis
        """
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
        
        return windows
    
    def _get_window_data(self, window):
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
    
    def _run_yearly_comparison(self, start_year=2012, end_year=2023):
        """
        Run benchmark comparison using simple yearly train/test split (original method)
        """
        # Filter data for analysis period
        analysis_data = self.df[
            (self.df['year'] >= start_year) & 
            (self.df['year'] <= end_year)
        ].copy()
        
        print(f"Analysis dataset: {len(analysis_data):,} observations")
    
    def create_benchmark_visualizations(self):
        """
        Create comprehensive visualizations comparing benchmark vs IEVR/REVR models
        """
        print(f"\n📊 CREATING BENCHMARK VISUALIZATIONS")
        print("-" * 50)
        
        if not hasattr(self, 'yearly_results') or len(self.yearly_results) == 0:
            print("No results available for visualization")
            return
        
        # Set professional styling for BlackRock presentations
        plt.style.use('default')  # Start with clean slate
        plt.rcParams.update({
            'font.family': 'Arial',  # Professional font
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'axes.linewidth': 0.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5
        })
        
        # Create comprehensive comparison plot
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('white')
        
        # 1. Time series of R² performance
        ax1 = plt.subplot(2, 3, (1, 2))
        self._plot_performance_over_time(ax1)
        
        # 2. Average performance comparison
        ax2 = plt.subplot(2, 3, 3)
        self._plot_average_performance(ax2)
        
        # 3. Performance distribution
        ax3 = plt.subplot(2, 3, 4)
        self._plot_performance_distribution(ax3)
        
        # 4. Improvement over benchmark
        ax4 = plt.subplot(2, 3, 5)
        self._plot_improvement_analysis(ax4)
        
        # 5. Model stability
        ax5 = plt.subplot(2, 3, 6)
        self._plot_model_stability(ax5)
        
        plt.suptitle('Benchmark vs IEVR/REVR Model Comparison', 
                    fontsize=16, fontweight='bold', color='#003366')
        plt.tight_layout()
        
        # Save plot
        output_path = 'output_files/benchmark_comparison_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Benchmark visualization saved: {output_path}")
        
        # Create focused comparison plot
        self._create_focused_comparison_plot()
        
    def _plot_performance_over_time(self, ax):
        """Plot R² performance over time for all models"""
        # Create proper x-axis using test period dates or window sequence
        if 'test_start' in self.yearly_results.columns:
            x_axis = pd.to_datetime(self.yearly_results['test_start'], format='%Y-%m')
            x_label = 'Test Period Start'
        elif 'window_id' in self.yearly_results.columns:
            x_axis = self.yearly_results['window_id']
            x_label = 'Window Number'
        else:
            x_axis = self.yearly_results['test_year'] if 'test_year' in self.yearly_results.columns else self.yearly_results['year']
            x_label = 'Year'
            
        for model_name, model_config in self.models.items():
            r2_col = f'{model_name}_r2'
            if r2_col in self.yearly_results.columns:
                ax.plot(x_axis, self.yearly_results[r2_col], 
                       label=model_config['name'], 
                       color=model_config['color'], 
                       linestyle=model_config.get('linestyle', '-'),
                       marker=model_config.get('marker', 'o'), 
                       markersize=6, linewidth=2, alpha=0.8)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('Test R²')
        ax.set_title('Model Performance Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Highlight 2018 underperformance if using datetime x-axis
        if 'test_start' in self.yearly_results.columns:
            ax.axvspan(pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'), alpha=0.3, color='orange')
            ax.text(pd.Timestamp('2018-06-15'), ax.get_ylim()[1]*0.9, '2018\nIssue', ha='center', va='top', 
                    fontsize=10, color='darkorange', fontweight='bold')
        elif 'window_id' in self.yearly_results.columns:
            # Find 2018 windows
            windows_2018 = self.yearly_results[self.yearly_results['test_year'] == 2018]['window_id']
            if len(windows_2018) > 0:
                min_window = windows_2018.min() - 0.5
                max_window = windows_2018.max() + 0.5
                ax.axvspan(min_window, max_window, alpha=0.3, color='orange')
                ax.text((min_window + max_window) / 2, ax.get_ylim()[1]*0.9, '2018\nIssue', ha='center', va='top', 
                        fontsize=10, color='darkorange', fontweight='bold')
    
    def _plot_average_performance(self, ax):
        """Plot average performance comparison"""
        if not hasattr(self, 'model_performance'):
            return
        
        model_names = []
        avg_r2_values = []
        std_r2_values = []
        colors = []
        
        color_map = {
            'Benchmark_Last1': 'lightcoral',
            'Benchmark_4Avg': 'red',
            'Benchmark_6Avg': 'darkred',
            'IEVR_REVR_Basic': 'lightblue',
            'IEVR_REVR_Enhanced': 'darkgreen'
        }
        
        for model_name, perf in self.model_performance.items():
            model_names.append(perf['name'])
            avg_r2_values.append(perf['avg_r2'])
            std_r2_values.append(perf['std_r2'])
            colors.append(color_map.get(model_name, 'gray'))
        
        bars = ax.bar(range(len(model_names)), avg_r2_values, 
                     yerr=std_r2_values, capsize=5, 
                     color=colors, alpha=0.7)
        
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('Average R²')
        ax.set_title('Average Performance Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_r2_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    def _plot_performance_distribution(self, ax):
        """Plot distribution of R² values"""
        distributions = []
        labels = []
        
        key_models = ['Benchmark_4Avg', 'IEVR_REVR_Enhanced']
        
        for model_name in key_models:
            r2_col = f'{model_name}_r2'
            if r2_col in self.yearly_results.columns:
                r2_values = self.yearly_results[r2_col].dropna()
                if len(r2_values) > 0:
                    distributions.append(r2_values)
                    labels.append(self.models[model_name]['name'])
        
        if distributions:
            ax.boxplot(distributions, labels=labels)
            ax.set_ylabel('R² Distribution')
            ax.set_title('Performance Distribution')
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_improvement_analysis(self, ax):
        """Plot improvement over benchmark analysis"""
        if 'Benchmark_4Avg_r2' not in self.yearly_results.columns:
            return
        
        benchmark_r2 = self.yearly_results['Benchmark_4Avg_r2']
        
        models_to_compare = ['IEVR_REVR_Basic', 'IEVR_REVR_Enhanced']
        
        for model_name in models_to_compare:
            r2_col = f'{model_name}_r2'
            if r2_col in self.yearly_results.columns:
                model_r2 = self.yearly_results[r2_col]
                improvement = model_r2 - benchmark_r2
                
                ax.plot(self.yearly_results['year'], improvement, 
                       label=f'{self.models[model_name]["name"]} vs Benchmark',
                       marker='o', markersize=4)
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No Improvement')
        ax.set_xlabel('Year')
        ax.set_ylabel('R² Improvement over Benchmark')
        ax.set_title('Improvement Over 4-Period Average')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_model_stability(self, ax):
        """Plot model stability (coefficient of variation)"""
        if not hasattr(self, 'model_performance'):
            return
        
        model_names = []
        stability_scores = []
        
        for model_name, perf in self.model_performance.items():
            if perf['avg_r2'] > 0:
                cv = perf['std_r2'] / perf['avg_r2']  # Coefficient of variation
                model_names.append(perf['name'])
                stability_scores.append(cv)
        
        if model_names:
            bars = ax.bar(range(len(model_names)), stability_scores, alpha=0.7)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel('Coefficient of Variation (Lower = More Stable)')
            ax.set_title('Model Stability Analysis')
            ax.grid(True, alpha=0.3, axis='y')
    
    def _create_focused_comparison_plot(self):
        """Create a focused plot for slide presentation with BlackRock styling"""
        # Set BlackRock presentation styling
        plt.style.use('default')
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
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor('white')
        
        # Left plot: Performance over time for all three models
        # Create proper x-axis using test period dates or window sequence
        if 'test_start' in self.yearly_results.columns:
            # Convert test_start to datetime for proper chronological plotting
            x_axis = pd.to_datetime(self.yearly_results['test_start'], format='%Y-%m')
            x_label = 'Test Period Start'
        elif 'window_id' in self.yearly_results.columns:
            # Use window sequence if test_start not available
            x_axis = self.yearly_results['window_id']
            x_label = 'Window Number'
        else:
            # Fallback to test_year
            x_axis = self.yearly_results['test_year'] if 'test_year' in self.yearly_results.columns else self.yearly_results['year']
            x_label = 'Year'
        
        for model_name, model_config in self.models.items():
            r2_col = f'{model_name}_r2'
            if r2_col in self.yearly_results.columns:
                ax1.plot(x_axis, self.yearly_results[r2_col], 
                        label=model_config['name'], 
                        color=model_config['color'], 
                        linestyle=model_config.get('linestyle', '-'),
                        marker=model_config.get('marker', 'o'), 
                        markersize=7, linewidth=3, alpha=0.9, markeredgewidth=0.5, 
                        markeredgecolor='white')
        
        ax1.set_xlabel(x_label, fontsize=12, color='#003366', fontweight='semibold')
        ax1.set_ylabel('Test R²', fontsize=12, color='#003366', fontweight='semibold')
        ax1.set_title('Model Performance Comparison Over Time', 
                     fontsize=14, fontweight='bold', color='#003366', pad=20)
        ax1.legend(fontsize=10, loc='upper left', frameon=True, fancybox=True, 
                  shadow=True, framealpha=0.95)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=-0.2, top=0.3)
        
        # Highlight 2018 with BlackRock accent color
        if 'test_start' in self.yearly_results.columns:
            ax1.axvspan(pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'), 
                       alpha=0.2, color='#FF6633', label='2018 Market Stress')
            ax1.text(pd.Timestamp('2018-06-15'), ax1.get_ylim()[1]*0.85, '2018\nMarket Stress', 
                    ha='center', va='top', fontsize=10, color='#FF6633', fontweight='bold')
        elif 'window_id' in self.yearly_results.columns:
            # Find 2018 windows
            windows_2018 = self.yearly_results[self.yearly_results['test_year'] == 2018]['window_id']
            if len(windows_2018) > 0:
                min_window = windows_2018.min() - 0.5
                max_window = windows_2018.max() + 0.5
                ax1.axvspan(min_window, max_window, alpha=0.2, color='#FF6633', label='2018 Market Stress')
                ax1.text((min_window + max_window) / 2, ax1.get_ylim()[1]*0.85, '2018\nMarket Stress', 
                        ha='center', va='top', fontsize=10, color='#FF6633', fontweight='bold')
        else:
            # Fallback for year-based x-axis
            ax1.axvspan(2017.5, 2018.5, alpha=0.2, color='#FF6633', label='2018 Market Stress')
            ax1.text(2018, ax1.get_ylim()[1]*0.85, '2018\nMarket Stress', ha='center', va='top', 
                    fontsize=10, color='#FF6633', fontweight='bold')
        
        # Add horizontal line at 0 with better styling
        ax1.axhline(y=0, color='#8C8C8C', linestyle='-', alpha=0.7, linewidth=1.5)
        ax1.text(ax1.get_xlim()[0] + 0.1, 0.015, 'No Predictive Power', 
                fontsize=9, color='#8C8C8C', alpha=0.8, style='italic')
        
        # Right plot: Average performance comparison with BlackRock styling
        if hasattr(self, 'model_performance'):
            model_names = []
            performances = []
            colors = []
            
            # Order models for logical progression
            model_order = ['Benchmark_4Avg', 'Simple_IEVR', 'Multi_Factor']
            
            for model_name in model_order:
                if model_name in self.model_performance:
                    perf = self.model_performance[model_name]
                    model_names.append(self.models[model_name]['name'])
                    performances.append(perf['avg_r2'])
                    colors.append(self.models[model_name]['color'])
            
            if model_names:
                # Create bars with BlackRock styling
                bars = ax2.bar(range(len(model_names)), performances, 
                              color=colors, alpha=0.8, width=0.65, 
                              edgecolor='white', linewidth=1.5)
                
                # Add value labels on bars with better styling
                for i, (bar, value) in enumerate(zip(bars, performances)):
                    height = bar.get_height()
                    label_y = height + 0.008 if height >= 0 else height - 0.020
                    ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                            f'{value:.3f}', ha='center', 
                            va='bottom' if height >= 0 else 'top',
                            fontsize=11, fontweight='bold', color='#003366')
                
                # Add improvement annotations with BlackRock colors
                if len(performances) >= 3:
                    benchmark_r2 = performances[0]
                    simple_r2 = performances[1]
                    multi_r2 = performances[2]
                    
                    if benchmark_r2 != 0:
                        simple_improvement = simple_r2 - benchmark_r2
                        final_improvement = multi_r2 - simple_r2
                        
                        # Arrow from benchmark to simple with BlackRock light blue
                        ax2.annotate(f'+{simple_improvement:.3f}', 
                                   xy=(1, simple_r2), xytext=(0.5, simple_r2 + 0.06),
                                   arrowprops=dict(arrowstyle='->', color='#66CCFF', lw=2),
                                   fontsize=10, ha='center', va='bottom',
                                   color='#66CCFF', fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                           edgecolor='#66CCFF', alpha=0.9))
                        
                        # Arrow from simple to multi-factor with BlackRock dark blue
                        ax2.annotate(f'+{final_improvement:.3f}', 
                                   xy=(2, multi_r2), xytext=(1.5, multi_r2 + 0.06),
                                   arrowprops=dict(arrowstyle='->', color='#003366', lw=2),
                                   fontsize=10, ha='center', va='bottom',
                                   color='#003366', fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                           edgecolor='#003366', alpha=0.9))
                
                ax2.set_xticks(range(len(model_names)))
                ax2.set_xticklabels(model_names, fontsize=10, color='#003366')
                ax2.set_ylabel('Average Test R²', fontsize=12, color='#003366', fontweight='semibold')
                ax2.set_title('Average Performance Comparison', fontsize=14, fontweight='bold', 
                             color='#003366', pad=20)
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Set y-limits to show the progression clearly
                y_min = min(performances) - 0.06
                y_max = max(performances) + 0.12
                ax2.set_ylim(y_min, y_max)
                
                # Add horizontal line at 0 with BlackRock styling
                ax2.axhline(y=0, color='#8C8C8C', linestyle='-', alpha=0.7, linewidth=1.5)
                ax2.text(ax2.get_xlim()[0] + 0.05, 0.008, 'No Predictive Power', 
                        fontsize=9, color='#8C8C8C', alpha=0.8, style='italic')
        
        plt.tight_layout()
        
        # Save focused plot with high quality for presentations
        output_path = 'output_files/benchmark_focused_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        print(f"✅ Focused comparison plot saved: {output_path}")
        
        # Also save as SVG for perfect scaling in presentations
        output_path_svg = 'output_files/benchmark_focused_comparison.svg'
        plt.savefig(output_path_svg, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='svg')
        print(f"✅ SVG version saved: {output_path_svg}")
        
    def save_results(self):
        """Save detailed results to CSV files"""
        print(f"\n💾 SAVING BENCHMARK RESULTS")
        print("-" * 40)
        
        # Save yearly results
        if hasattr(self, 'yearly_results'):
            yearly_path = 'output_files/benchmark_yearly_results.csv'
            self.yearly_results.to_csv(yearly_path, index=False)
            print(f"✅ Yearly results saved: {yearly_path}")
        
        # Save summary statistics
        if hasattr(self, 'model_performance'):
            summary_data = []
            for model_name, perf in self.model_performance.items():
                summary_data.append({
                    'model_key': model_name,
                    'model_name': perf['name'],
                    'avg_r2': perf['avg_r2'],
                    'std_r2': perf['std_r2'],
                    'avg_rmse': perf['avg_rmse']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = 'output_files/benchmark_model_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"✅ Model summary saved: {summary_path}")
        
        print("Results saved successfully!")

def main():
    """
    Main function to run benchmark analysis
    """
    try:
        # Initialize analysis
        analyzer = BenchmarkAnalysis()
        
        # Create benchmark features
        analyzer.create_benchmark_features()
        
        # Run comprehensive comparison using proper rolling windows
        yearly_results = analyzer.run_benchmark_comparison(use_rolling_windows=True)
        
        # Create visualizations
        analyzer.create_benchmark_visualizations()
        
        # Save results
        analyzer.save_results()
        
        print(f"\n🎉 BENCHMARK ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"Key files created:")
        print(f"  • benchmark_comparison_analysis.png - Comprehensive comparison")
        print(f"  • benchmark_focused_comparison.png - Slide-ready comparison")
        print(f"  • benchmark_yearly_results.csv - Detailed yearly results")
        print(f"  • benchmark_model_summary.csv - Model performance summary")
        
    except Exception as e:
        print(f"❌ Error in benchmark analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
