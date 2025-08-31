#!/usr/bin/env python3
"""
Systematic Analysis of 2018 Model Underperformance
Analyzes correlation changes, coefficient stability, and market regime shifts during 2018 window
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class Systematic2018Analysis:
    """
    Comprehensive analysis framework for understanding 2018 model underperformance
    """
    
    def __init__(self, data_file_path='data_files/final_merged_dataset_with_momentum_final.csv'):
        """
        Initialize analysis with the final merged dataset
        """
        print("🔍 SYSTEMATIC 2018 ANALYSIS FRAMEWORK")
        print("="*60)
        
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
        
        # Key features for analysis - Model 3 + z_score_momentum
        self.base_features = ['ievr', 'normative_iv_rv_ratio', 'SKEW', 'KURT', 'IV_RATIO', 'SMIRK', 'vol_hl7', 'vol_hl10', 'vol_hl21']
        self.momentum_feature = 'z_score_momentum'
        self.features = self.base_features + [self.momentum_feature]
        self.target = 'revr'
        
        # Define time windows for comparison
        self.time_windows = {
            'pre_2018': (2015, 2017),
            '2018': (2018, 2018), 
            'post_2018': (2019, 2021),
            'recent': (2022, 2023)
        }
        
    def run_comprehensive_analysis(self):
        """
        Run the complete systematic analysis
        """
        print("\n🚀 RUNNING COMPREHENSIVE 2018 ANALYSIS")
        print("="*60)
        
        # 1. Temporal correlation analysis
        self.analyze_temporal_correlations()
        
        # 2. Feature distribution analysis
        self.analyze_feature_distributions()
        
        # 3. Coefficient stability analysis
        self.analyze_coefficient_stability()
        
        # 4. Rolling window regression analysis
        self.analyze_rolling_window_performance()
        
        # 5. Market regime analysis
        self.analyze_market_regimes()
        
        # 6. Structural break detection
        self.detect_structural_breaks()
        
        # 7. Create comprehensive visualization
        self.create_comprehensive_visualization()
        
        # 8. Deep dive into 2018 H2 underperformance
        self.analyze_2018_h2_breakdown()
        
        print("\n✅ COMPREHENSIVE ANALYSIS COMPLETE")
        
    def analyze_temporal_correlations(self):
        """
        Analyze how feature correlations change over time periods
        """
        print("\n📊 TEMPORAL CORRELATION ANALYSIS")
        print("-" * 40)
        
        correlation_results = {}
        
        for window_name, (start_year, end_year) in self.time_windows.items():
            # Filter data for time window
            window_data = self.df[
                (self.df['year'] >= start_year) & 
                (self.df['year'] <= end_year)
            ].copy()
            
            if len(window_data) < 50:  # Skip if insufficient data
                continue
                
            # Calculate correlation matrix
            corr_features = self.features + [self.target]
            window_corr = window_data[corr_features].corr()
            
            # Store target correlations
            target_corrs = window_corr[self.target].drop(self.target)
            correlation_results[window_name] = {
                'correlations': target_corrs,
                'sample_size': len(window_data),
                'corr_matrix': window_corr
            }
            
            print(f"\n{window_name.upper()} ({start_year}-{end_year}):")
            print(f"  Sample size: {len(window_data):,}")
            print("  Correlations with REVR:")
            for feature in self.features:
                if feature in target_corrs.index:
                    corr_val = target_corrs[feature]
                    print(f"    {feature:20s}: {corr_val:7.4f}")
        
        # Store results for later use
        self.correlation_results = correlation_results
        
        # Analyze correlation changes
        self._analyze_correlation_changes()
        
    def _analyze_correlation_changes(self):
        """
        Analyze how correlations changed specifically around 2018
        """
        print(f"\n🔍 CORRELATION CHANGE ANALYSIS:")
        print("-" * 40)
        
        if 'pre_2018' in self.correlation_results and '2018' in self.correlation_results:
            pre_2018_corrs = self.correlation_results['pre_2018']['correlations']
            corrs_2018 = self.correlation_results['2018']['correlations']
            
            print(f"Feature correlation changes (2018 vs pre-2018):")
            for feature in self.features:
                if feature in pre_2018_corrs.index and feature in corrs_2018.index:
                    pre_corr = pre_2018_corrs[feature]
                    curr_corr = corrs_2018[feature]
                    change = curr_corr - pre_corr
                    pct_change = (change / abs(pre_corr)) * 100 if abs(pre_corr) > 0.01 else np.inf
                    
                    status = "📈" if change > 0.05 else "📉" if change < -0.05 else "➡️"
                    print(f"  {status} {feature:20s}: {pre_corr:7.4f} → {curr_corr:7.4f} (Δ={change:+7.4f}, {pct_change:+6.1f}%)")
                    
    def analyze_feature_distributions(self):
        """
        Analyze how feature distributions changed in 2018
        """
        print(f"\n📈 FEATURE DISTRIBUTION ANALYSIS")
        print("-" * 40)
        
        distribution_results = {}
        
        for window_name, (start_year, end_year) in self.time_windows.items():
            window_data = self.df[
                (self.df['year'] >= start_year) & 
                (self.df['year'] <= end_year)
            ].copy()
            
            if len(window_data) < 50:
                continue
                
            window_stats = {}
            for feature in self.features + [self.target]:
                if feature in window_data.columns:
                    feature_data = window_data[feature].dropna()
                    if len(feature_data) > 0:
                        window_stats[feature] = {
                            'mean': feature_data.mean(),
                            'std': feature_data.std(),
                            'skew': feature_data.skew(),
                            'kurt': feature_data.kurtosis(),
                            'q25': feature_data.quantile(0.25),
                            'q75': feature_data.quantile(0.75)
                        }
            
            distribution_results[window_name] = window_stats
            
        self.distribution_results = distribution_results
        
        # Compare distributions
        self._compare_distributions()
        
    def _compare_distributions(self):
        """
        Compare distributions between 2018 and other periods
        """
        print(f"\nDistribution comparison (2018 vs pre-2018):")
        
        if 'pre_2018' in self.distribution_results and '2018' in self.distribution_results:
            pre_2018_stats = self.distribution_results['pre_2018']
            stats_2018 = self.distribution_results['2018']
            
            for feature in self.features + [self.target]:
                if feature in pre_2018_stats and feature in stats_2018:
                    pre_mean = pre_2018_stats[feature]['mean']
                    curr_mean = stats_2018[feature]['mean']
                    pre_std = pre_2018_stats[feature]['std']
                    curr_std = stats_2018[feature]['std']
                    
                    mean_change = (curr_mean - pre_mean) / abs(pre_mean) * 100 if abs(pre_mean) > 0.01 else 0
                    std_change = (curr_std - pre_std) / abs(pre_std) * 100 if abs(pre_std) > 0.01 else 0
                    
                    print(f"  {feature:20s}: Mean {mean_change:+6.1f}%, Std {std_change:+6.1f}%")
                    
    def analyze_coefficient_stability(self):
        """
        Analyze coefficient stability over rolling windows
        """
        print(f"\n🎯 COEFFICIENT STABILITY ANALYSIS")
        print("-" * 40)
        
        # Create rolling windows (1-year windows with 6-month overlap)
        window_size_months = 24
        step_size_months = 6
        
        min_date = self.df['earnings_date'].min()
        max_date = self.df['earnings_date'].max()
        
        # Generate date ranges
        current_date = min_date
        rolling_results = []
        
        while current_date + pd.DateOffset(months=window_size_months) <= max_date:
            window_start = current_date
            window_end = current_date + pd.DateOffset(months=window_size_months)
            
            # Filter data for window
            window_data = self.df[
                (self.df['earnings_date'] >= window_start) & 
                (self.df['earnings_date'] < window_end)
            ].copy()
            
            if len(window_data) >= 100:  # Minimum observations
                # Prepare features
                feature_cols = ['ievr', 'normative_iv_rv_ratio']
                X = window_data[feature_cols].values
                y = window_data[self.target].values
                
                # Remove NaN values
                valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X_clean = X[valid_mask]
                y_clean = y[valid_mask]
                
                if len(X_clean) >= 50:
                    # Fit regression
                    model = LinearRegression()
                    model.fit(X_clean, y_clean)
                    
                    # Calculate R²
                    y_pred = model.predict(X_clean)
                    r2 = r2_score(y_clean, y_pred)
                    
                    rolling_results.append({
                        'window_start': window_start,
                        'window_end': window_end,
                        'window_mid': window_start + (window_end - window_start) / 2,
                        'sample_size': len(X_clean),
                        'intercept': model.intercept_,
                        'ievr_coef': model.coef_[0],
                        'ratio_coef': model.coef_[1],
                        'r2': r2,
                        'year': window_start.year
                    })
            
            current_date += pd.DateOffset(months=step_size_months)
        
        self.rolling_results = pd.DataFrame(rolling_results)
        
        # Analyze coefficient stability
        if len(self.rolling_results) > 0:
            print(f"Rolling regression results:")
            print(f"  Windows analyzed: {len(self.rolling_results)}")
            print(f"  Date range: {self.rolling_results['window_start'].min().strftime('%Y-%m')} to {self.rolling_results['window_end'].max().strftime('%Y-%m')}")
            
            # Find 2018 windows
            windows_2018 = self.rolling_results[self.rolling_results['year'] == 2018]
            if len(windows_2018) > 0:
                print(f"\n2018 Performance Summary:")
                print(f"  Average R²: {windows_2018['r2'].mean():.4f}")
                print(f"  IEVR coefficient: {windows_2018['ievr_coef'].mean():.4f} ± {windows_2018['ievr_coef'].std():.4f}")
                print(f"  Ratio coefficient: {windows_2018['ratio_coef'].mean():.4f} ± {windows_2018['ratio_coef'].std():.4f}")
        
    def analyze_rolling_window_performance(self):
        """
        Analyze detailed rolling window performance focusing on 2018
        """
        print(f"\n🎪 ROLLING WINDOW PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        # Load existing rolling regression results
        try:
            rolling_data = pd.read_csv('/Users/joyceee_xby/blk_earnings_iv/analysis_scripts_2/output_files/momentum_rolling_regression(newest)/momentum_rolling_regression_time_series.csv')
            
            # Focus on Model3_plus_z_score_momentum (the newest model)
            model_data = rolling_data[rolling_data['combination'] == 'Model3_plus_z_score_momentum'].copy()
            
            if len(model_data) == 0:
                print("  ⚠️  No Model3_plus_z_score_momentum data found")
                return
            
            # Focus on 2018 windows
            windows_2018 = model_data[model_data['test_year'] == 2018]
            
            print(f"2018 Test Windows Performance (Model3_plus_z_score_momentum):")
            for _, row in windows_2018.iterrows():
                print(f"  {row['test_start']} to {row['test_end']}: R² = {row['test_r2']:.4f}, RMSE = {row['test_rmse']:.4f}")
            
            # Compare with adjacent years
            windows_2017 = model_data[model_data['test_year'] == 2017]
            windows_2019 = model_data[model_data['test_year'] == 2019]
            
            print(f"\nComparison with adjacent years (Model3_plus_z_score_momentum):")
            print(f"  2017 Average R²: {windows_2017['test_r2'].mean():.4f}")
            print(f"  2018 Average R²: {windows_2018['test_r2'].mean():.4f}")
            print(f"  2019 Average R²: {windows_2019['test_r2'].mean():.4f}")
            
            # Store detailed performance data
            self.rolling_performance_data = model_data
            self.performance_comparison = {
                '2017': windows_2017['test_r2'].tolist(),
                '2018': windows_2018['test_r2'].tolist(), 
                '2019': windows_2019['test_r2'].tolist()
            }
            
            # Analyze the dramatic 2018 H2 drop
            h1_2018 = windows_2018[windows_2018['test_start'].str.contains('2018-01')]['test_r2'].values
            h2_2018 = windows_2018[windows_2018['test_start'].str.contains('2018-07')]['test_r2'].values
            
            if len(h1_2018) > 0 and len(h2_2018) > 0:
                print(f"\n🔍 2018 Detailed Analysis:")
                print(f"  2018 H1 R²: {h1_2018[0]:.4f}")
                print(f"  2018 H2 R²: {h2_2018[0]:.4f}")
                print(f"  H2 vs H1 Drop: {((h2_2018[0] - h1_2018[0])/h1_2018[0]*100):+.1f}%")
            
        except FileNotFoundError:
            print("  ⚠️  Rolling regression results file not found")
            
    def analyze_market_regimes(self):
        """
        Analyze market regime characteristics during 2018
        """
        print(f"\n🏛️ MARKET REGIME ANALYSIS")
        print("-" * 40)
        
        # Calculate market-wide metrics for each year
        yearly_metrics = []
        
        for year in range(2015, 2024):
            year_data = self.df[self.df['year'] == year].copy()
            
            if len(year_data) > 50:
                metrics = {
                    'year': year,
                    'sample_size': len(year_data),
                    'avg_revr': year_data['revr'].mean(),
                    'avg_ievr': year_data['ievr'].mean(),
                    'avg_vol': year_data['vol_hl21'].mean() if 'vol_hl21' in year_data.columns else np.nan,
                    'avg_skew': year_data['SKEW'].mean() if 'SKEW' in year_data.columns else np.nan,
                    'revr_volatility': year_data['revr'].std(),
                    'ievr_volatility': year_data['ievr'].std()
                }
                yearly_metrics.append(metrics)
        
        self.yearly_metrics = pd.DataFrame(yearly_metrics)
        
        print("Market regime characteristics by year:")
        for _, row in self.yearly_metrics.iterrows():
            if row['year'] == 2018:
                marker = "🔴"
            else:
                marker = "  "
            print(f"{marker} {row['year']}: REVR={row['avg_revr']:.3f}, IEVR={row['avg_ievr']:.3f}, Vol={row['avg_vol']:.3f}")
            
    def detect_structural_breaks(self):
        """
        Detect structural breaks in the IEVR-REVR relationship
        """
        print(f"\n🔍 STRUCTURAL BREAK DETECTION")
        print("-" * 40)
        
        # Prepare data for structural break analysis
        analysis_data = self.df[['earnings_date', 'ievr', 'revr', 'year']].dropna()
        analysis_data = analysis_data.sort_values('earnings_date')
        
        # Calculate rolling correlations
        window_size = 500  # Approximately 1-2 years of data
        rolling_corrs = []
        
        for i in range(window_size, len(analysis_data)):
            window_data = analysis_data.iloc[i-window_size:i]
            corr = window_data['ievr'].corr(window_data['revr'])
            rolling_corrs.append({
                'date': window_data['earnings_date'].iloc[-1],
                'correlation': corr,
                'year': window_data['year'].iloc[-1]
            })
        
        rolling_corr_df = pd.DataFrame(rolling_corrs)
        
        # Find periods of low correlation
        low_corr_periods = rolling_corr_df[rolling_corr_df['correlation'] < 0.1]
        
        print(f"Periods of low IEVR-REVR correlation:")
        for _, row in low_corr_periods.iterrows():
            print(f"  {row['date'].strftime('%Y-%m')}: correlation = {row['correlation']:.4f}")
            
        self.rolling_corr_df = rolling_corr_df
        
    def analyze_2018_h2_breakdown(self):
        """
        Deep dive analysis of the 2018 H2 performance breakdown
        """
        print(f"\n🔬 2018 H2 BREAKDOWN ANALYSIS")
        print("-" * 40)
        
        # Define the periods for detailed comparison
        periods = {
            '2018_H1': ('2018-01-01', '2018-06-30'),
            '2018_H2': ('2018-07-01', '2018-12-31'),
            '2017_H2': ('2017-07-01', '2017-12-31'),
            '2019_H1': ('2019-01-01', '2019-06-30')
        }
        
        period_analysis = {}
        
        for period_name, (start_date, end_date) in periods.items():
            # Filter data for period
            period_mask = (
                (self.df['earnings_date'] >= start_date) & 
                (self.df['earnings_date'] <= end_date)
            )
            period_data = self.df[period_mask].copy()
            
            if len(period_data) < 10:
                continue
                
            # Analyze feature characteristics
            available_features = [f for f in self.features if f in period_data.columns]
            
            period_stats = {}
            for feature in available_features:
                feature_data = period_data[feature].dropna()
                if len(feature_data) > 0:
                    period_stats[feature] = {
                        'mean': feature_data.mean(),
                        'std': feature_data.std(),
                        'median': feature_data.median(),
                        'q25': feature_data.quantile(0.25),
                        'q75': feature_data.quantile(0.75),
                        'skew': feature_data.skew(),
                        'n_obs': len(feature_data)
                    }
            
            # Analyze correlations within period
            correlations = {}
            for feature in available_features:
                if feature in period_data.columns and self.target in period_data.columns:
                    clean_data = period_data[[feature, self.target]].dropna()
                    if len(clean_data) > 10:
                        correlations[feature] = clean_data[feature].corr(clean_data[self.target])
            
            # Fit a simple model for this period
            model_performance = None
            if len(available_features) >= 2:
                try:
                    # Use core features for model
                    core_features = ['ievr', 'normative_iv_rv_ratio']
                    available_core = [f for f in core_features if f in period_data.columns]
                    
                    if len(available_core) >= 2:
                        model_data = period_data[available_core + [self.target]].dropna()
                        
                        if len(model_data) >= 20:
                            X = model_data[available_core].values
                            y = model_data[self.target].values
                            
                            # Fit model
                            from sklearn.linear_model import LinearRegression
                            from sklearn.metrics import r2_score
                            
                            model = LinearRegression()
                            model.fit(X, y)
                            y_pred = model.predict(X)
                            r2 = r2_score(y, y_pred)
                            
                            model_performance = {
                                'r2': r2,
                                'coefficients': dict(zip(available_core, model.coef_)),
                                'intercept': model.intercept_,
                                'n_obs': len(model_data)
                            }
                except Exception as e:
                    print(f"    Warning: Model fitting failed for {period_name}: {e}")
            
            period_analysis[period_name] = {
                'feature_stats': period_stats,
                'correlations': correlations,
                'model_performance': model_performance,
                'sample_size': len(period_data)
            }
            
            print(f"\n{period_name.upper()}:")
            print(f"  Sample size: {len(period_data)}")
            if correlations:
                print(f"  Key correlations with REVR:")
                for feature, corr in correlations.items():
                    if feature in ['ievr', 'normative_iv_rv_ratio', 'z_score_momentum']:
                        print(f"    {feature:20s}: {corr:7.4f}")
            
            if model_performance:
                print(f"  Simple model R²: {model_performance['r2']:.4f}")
        
        self.period_analysis = period_analysis
        
        # Compare 2018 H2 with other periods
        self._compare_2018_h2_characteristics()
        
    def _compare_2018_h2_characteristics(self):
        """
        Compare 2018 H2 characteristics with other periods
        """
        print(f"\n📊 2018 H2 COMPARISON:")
        print("-" * 30)
        
        if '2018_H2' not in self.period_analysis:
            print("  No 2018 H2 data available for comparison")
            return
        
        h2_2018 = self.period_analysis['2018_H2']
        
        # Compare correlations
        print(f"Correlation Breakdown (2018 H2 vs others):")
        key_features = ['ievr', 'normative_iv_rv_ratio', 'z_score_momentum']
        
        for feature in key_features:
            if feature in h2_2018['correlations']:
                h2_corr = h2_2018['correlations'][feature]
                print(f"\n  {feature}:")
                print(f"    2018 H2: {h2_corr:7.4f}")
                
                for period_name, period_data in self.period_analysis.items():
                    if period_name != '2018_H2' and feature in period_data['correlations']:
                        other_corr = period_data['correlations'][feature]
                        change = h2_corr - other_corr
                        print(f"    {period_name:8s}: {other_corr:7.4f} (Δ={change:+7.4f})")
        
        # Compare feature distributions
        print(f"\nFeature Distribution Changes (2018 H2 vs 2018 H1):")
        if '2018_H1' in self.period_analysis:
            h1_2018 = self.period_analysis['2018_H1']
            
            for feature in key_features:
                if (feature in h2_2018['feature_stats'] and 
                    feature in h1_2018['feature_stats']):
                    
                    h1_mean = h1_2018['feature_stats'][feature]['mean']
                    h2_mean = h2_2018['feature_stats'][feature]['mean']
                    h1_std = h1_2018['feature_stats'][feature]['std']
                    h2_std = h2_2018['feature_stats'][feature]['std']
                    
                    mean_change = (h2_mean - h1_mean) / abs(h1_mean) * 100 if abs(h1_mean) > 0.01 else 0
                    std_change = (h2_std - h1_std) / abs(h1_std) * 100 if abs(h1_std) > 0.01 else 0
                    
                    print(f"  {feature:20s}: Mean {mean_change:+6.1f}%, Std {std_change:+6.1f}%")
        
        # Model performance comparison
        print(f"\nModel Performance Comparison:")
        for period_name, period_data in self.period_analysis.items():
            if period_data['model_performance']:
                r2 = period_data['model_performance']['r2']
                print(f"  {period_name:10s}: R² = {r2:.4f}")
                
                if period_name == '2018_H2':
                    coeffs = period_data['model_performance']['coefficients']
                    print(f"    Coefficients: {coeffs}")
        
        # Identify potential causes
        print(f"\n🔍 POTENTIAL CAUSES OF 2018 H2 BREAKDOWN:")
        print("-" * 45)
        
        causes = []
        
        # Check correlation breakdown
        for feature in key_features:
            if feature in h2_2018['correlations']:
                h2_corr = h2_2018['correlations'][feature]
                if abs(h2_corr) < 0.05:  # Very low correlation
                    causes.append(f"• {feature} correlation breakdown ({h2_corr:.4f})")
        
        # Check if there are systematic issues
        if h2_2018['model_performance'] and h2_2018['model_performance']['r2'] < 0.05:
            causes.append("• Overall model relationship breakdown")
        
        # Check for outliers or regime change
        if '2018_H1' in self.period_analysis:
            h1_stats = self.period_analysis['2018_H1']['feature_stats']
            h2_stats = h2_2018['feature_stats']
            
            for feature in key_features:
                if feature in h1_stats and feature in h2_stats:
                    h1_std = h1_stats[feature]['std']
                    h2_std = h2_stats[feature]['std']
                    
                    if h2_std > h1_std * 1.5:  # 50% increase in volatility
                        causes.append(f"• Increased volatility in {feature} (H2 vs H1: {h2_std/h1_std:.1f}x)")
        
        if causes:
            for cause in causes:
                print(f"  {cause}")
        else:
            print("  • No obvious systematic causes detected - may be data quality or external factors")
        
        # Store for visualization
        self.breakdown_analysis = {
            'causes': causes,
            'correlations': {period: data['correlations'] for period, data in self.period_analysis.items()},
            'performance': {period: data['model_performance'] for period, data in self.period_analysis.items() if data['model_performance']}
        }
        
    def create_comprehensive_visualization(self):
        """
        Create comprehensive visualization dashboard
        """
        print(f"\n📊 CREATING COMPREHENSIVE VISUALIZATION")
        print("-" * 40)
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Correlation heatmap comparison
        if hasattr(self, 'correlation_results'):
            ax1 = plt.subplot(3, 3, 1)
            self._plot_correlation_comparison(ax1)
        
        # 2. Feature distribution comparison
        if hasattr(self, 'distribution_results'):
            ax2 = plt.subplot(3, 3, 2)
            self._plot_distribution_comparison(ax2)
        
        # 3. Coefficient stability over time
        if hasattr(self, 'rolling_results'):
            ax3 = plt.subplot(3, 3, 3)
            self._plot_coefficient_stability(ax3)
        
        # 4. Performance over time
        if hasattr(self, 'performance_comparison'):
            ax4 = plt.subplot(3, 3, 4)
            self._plot_performance_comparison(ax4)
        
        # 5. Market regime analysis
        if hasattr(self, 'yearly_metrics'):
            ax5 = plt.subplot(3, 3, 5)
            self._plot_market_regimes(ax5)
        
        # 6. Rolling correlation
        if hasattr(self, 'rolling_corr_df'):
            ax6 = plt.subplot(3, 3, 6)
            self._plot_rolling_correlation(ax6)
        
        # 7. IEVR vs REVR scatter by period
        ax7 = plt.subplot(3, 3, 7)
        self._plot_ievr_revr_scatter(ax7)
        
        # 8. Residual analysis
        ax8 = plt.subplot(3, 3, 8)
        self._plot_residual_analysis(ax8)
        
        # 9. Feature importance over time
        ax9 = plt.subplot(3, 3, 9)
        self._plot_feature_importance_over_time(ax9)
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        output_path = 'output_files/systematic_2018_analysis_comprehensive.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comprehensive visualization saved: {output_path}")
        
        # Also create individual focused plots
        self._create_focused_plots()
        
    def _plot_correlation_comparison(self, ax):
        """Plot correlation comparison heatmap"""
        if hasattr(self, 'correlation_results'):
            # Create correlation comparison matrix
            periods = ['pre_2018', '2018', 'post_2018']
            features = ['ievr', 'normative_iv_rv_ratio']
            
            corr_matrix = np.zeros((len(features), len(periods)))
            for i, feature in enumerate(features):
                for j, period in enumerate(periods):
                    if period in self.correlation_results:
                        corr_matrix[i, j] = self.correlation_results[period]['correlations'].get(feature, 0)
            
            sns.heatmap(corr_matrix, 
                       xticklabels=periods,
                       yticklabels=features,
                       annot=True, fmt='.3f',
                       cmap='RdBu_r', center=0,
                       ax=ax)
            ax.set_title('Feature Correlations with REVR')
        
    def _plot_distribution_comparison(self, ax):
        """Plot feature distribution comparison"""
        if hasattr(self, 'distribution_results'):
            # Plot mean changes
            periods = ['pre_2018', '2018', 'post_2018']
            features = ['ievr', 'normative_iv_rv_ratio']
            
            means = []
            for feature in features:
                feature_means = []
                for period in periods:
                    if period in self.distribution_results and feature in self.distribution_results[period]:
                        feature_means.append(self.distribution_results[period][feature]['mean'])
                    else:
                        feature_means.append(0)
                means.append(feature_means)
            
            x = np.arange(len(periods))
            width = 0.35
            
            for i, (feature, feature_means) in enumerate(zip(features, means)):
                ax.bar(x + i*width, feature_means, width, label=feature)
            
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Mean Value')
            ax.set_title('Feature Means by Period')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(periods)
            ax.legend()
        
    def _plot_coefficient_stability(self, ax):
        """Plot coefficient stability over time"""
        if hasattr(self, 'rolling_results') and len(self.rolling_results) > 0:
            ax.plot(self.rolling_results['window_mid'], self.rolling_results['ievr_coef'], 
                   label='IEVR Coefficient', marker='o', markersize=3)
            ax.plot(self.rolling_results['window_mid'], self.rolling_results['ratio_coef'], 
                   label='Ratio Coefficient', marker='s', markersize=3)
            
            # Highlight 2018
            ax.axvspan(pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'), 
                      alpha=0.3, color='red', label='2018')
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Coefficient Value')
            ax.set_title('Coefficient Stability Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
    def _plot_performance_comparison(self, ax):
        """Plot performance comparison"""
        if hasattr(self, 'performance_comparison'):
            years = ['2017', '2018', '2019']
            r2_means = [np.mean(self.performance_comparison[year]) for year in years]
            r2_stds = [np.std(self.performance_comparison[year]) for year in years]
            
            colors = ['blue', 'red', 'green']
            bars = ax.bar(years, r2_means, yerr=r2_stds, capsize=5, color=colors, alpha=0.7)
            
            ax.set_ylabel('Test R²')
            ax.set_title('Model Performance by Year')
            ax.grid(True, alpha=0.3)
            
            # Highlight 2018
            bars[1].set_color('red')
        
    def _plot_market_regimes(self, ax):
        """Plot market regime characteristics"""
        if hasattr(self, 'yearly_metrics'):
            years = self.yearly_metrics['year']
            revr_values = self.yearly_metrics['avg_revr']
            
            colors = ['red' if year == 2018 else 'blue' for year in years]
            ax.scatter(years, revr_values, c=colors, s=50, alpha=0.7)
            
            # Highlight 2018
            year_2018_data = self.yearly_metrics[self.yearly_metrics['year'] == 2018]
            if len(year_2018_data) > 0:
                ax.scatter(2018, year_2018_data['avg_revr'].iloc[0], 
                          c='red', s=100, marker='*', label='2018')
            
            ax.set_xlabel('Year')
            ax.set_ylabel('Average REVR')
            ax.set_title('Market Regime: Average REVR by Year')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
    def _plot_rolling_correlation(self, ax):
        """Plot rolling correlation over time"""
        if hasattr(self, 'rolling_corr_df'):
            ax.plot(self.rolling_corr_df['date'], self.rolling_corr_df['correlation'], 
                   linewidth=2, alpha=0.8)
            
            # Highlight 2018
            ax.axvspan(pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'), 
                      alpha=0.3, color='red', label='2018')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Rolling Correlation')
            ax.set_title('Rolling IEVR-REVR Correlation')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
    def _plot_ievr_revr_scatter(self, ax):
        """Plot IEVR vs REVR scatter by period"""
        periods = {
            'pre_2018': (2015, 2017, 'blue'),
            '2018': (2018, 2018, 'red'),
            'post_2018': (2019, 2021, 'green')
        }
        
        for period_name, (start_year, end_year, color) in periods.items():
            period_data = self.df[
                (self.df['year'] >= start_year) & 
                (self.df['year'] <= end_year)
            ].copy()
            
            if len(period_data) > 0:
                ax.scatter(period_data['ievr'], period_data['revr'], 
                          alpha=0.5, label=period_name, color=color, s=10)
        
        ax.set_xlabel('IEVR')
        ax.set_ylabel('REVR')
        ax.set_title('IEVR vs REVR by Period')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_residual_analysis(self, ax):
        """Plot residual analysis for 2018"""
        # Fit model on pre-2018 data
        pre_2018_data = self.df[self.df['year'] < 2018].copy()
        data_2018 = self.df[self.df['year'] == 2018].copy()
        
        if len(pre_2018_data) > 0 and len(data_2018) > 0:
            # Prepare training data
            X_train = pre_2018_data[['ievr', 'normative_iv_rv_ratio']].values
            y_train = pre_2018_data['revr'].values
            
            # Remove NaN values
            train_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            X_train_clean = X_train[train_mask]
            y_train_clean = y_train[train_mask]
            
            # Fit model
            model = LinearRegression()
            model.fit(X_train_clean, y_train_clean)
            
            # Predict on 2018 data
            X_test = data_2018[['ievr', 'normative_iv_rv_ratio']].values
            y_test = data_2018['revr'].values
            
            test_mask = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
            X_test_clean = X_test[test_mask]
            y_test_clean = y_test[test_mask]
            
            y_pred = model.predict(X_test_clean)
            residuals = y_test_clean - y_pred
            
            ax.scatter(y_pred, residuals, alpha=0.6)
            ax.axhline(y=0, color='red', linestyle='--')
            ax.set_xlabel('Predicted REVR')
            ax.set_ylabel('Residuals')
            ax.set_title('2018 Residual Analysis')
            ax.grid(True, alpha=0.3)
        
    def _plot_feature_importance_over_time(self, ax):
        """Plot feature importance changes over time"""
        if hasattr(self, 'rolling_results') and len(self.rolling_results) > 0:
            # Calculate relative importance (absolute coefficient values)
            self.rolling_results['ievr_importance'] = np.abs(self.rolling_results['ievr_coef'])
            self.rolling_results['ratio_importance'] = np.abs(self.rolling_results['ratio_coef'])
            
            # Normalize to relative importance
            total_importance = self.rolling_results['ievr_importance'] + self.rolling_results['ratio_importance']
            self.rolling_results['ievr_rel_importance'] = self.rolling_results['ievr_importance'] / total_importance
            self.rolling_results['ratio_rel_importance'] = self.rolling_results['ratio_importance'] / total_importance
            
            ax.plot(self.rolling_results['window_mid'], self.rolling_results['ievr_rel_importance'], 
                   label='IEVR Relative Importance', marker='o', markersize=3)
            ax.plot(self.rolling_results['window_mid'], self.rolling_results['ratio_rel_importance'], 
                   label='Ratio Relative Importance', marker='s', markersize=3)
            
            # Highlight 2018
            ax.axvspan(pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'), 
                      alpha=0.3, color='red', label='2018')
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Relative Importance')
            ax.set_title('Feature Importance Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
    def _create_focused_plots(self):
        """Create individual focused plots for detailed analysis"""
        print("Creating focused individual plots...")
        
        # 1. Detailed 2018 analysis plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance comparison
        if hasattr(self, 'performance_comparison'):
            self._plot_performance_comparison(ax1)
        
        # Correlation changes
        if hasattr(self, 'correlation_results'):
            self._plot_correlation_comparison(ax2)
        
        # Market regime analysis
        if hasattr(self, 'yearly_metrics'):
            self._plot_market_regimes(ax3)
        
        # Rolling correlation
        if hasattr(self, 'rolling_corr_df'):
            self._plot_rolling_correlation(ax4)
        
        plt.suptitle('2018 Model Underperformance: Focused Analysis', fontsize=16)
        plt.tight_layout()
        
        focused_path = 'output_files/2018_focused_analysis.png'
        plt.savefig(focused_path, dpi=300, bbox_inches='tight')
        print(f"✅ Focused analysis saved: {focused_path}")
        
        plt.close()
        
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report
        """
        print(f"\n📋 GENERATING SUMMARY REPORT")
        print("="*60)
        
        report = []
        report.append("SYSTEMATIC 2018 MODEL UNDERPERFORMANCE ANALYSIS")
        report.append("="*60)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY:")
        report.append("-" * 20)
        
        if hasattr(self, 'performance_comparison'):
            avg_2017 = np.mean(self.performance_comparison.get('2017', [0]))
            avg_2018 = np.mean(self.performance_comparison.get('2018', [0]))
            avg_2019 = np.mean(self.performance_comparison.get('2019', [0]))
            
            report.append(f"• 2018 showed significant model underperformance:")
            report.append(f"  - 2017 Average R²: {avg_2017:.4f}")
            report.append(f"  - 2018 Average R²: {avg_2018:.4f} (↓{((avg_2018-avg_2017)/avg_2017*100):+.1f}%)")
            report.append(f"  - 2019 Average R²: {avg_2019:.4f} (Recovery: {((avg_2019-avg_2018)/avg_2018*100):+.1f}%)")
            report.append("")
        
        # Key Findings
        report.append("KEY FINDINGS:")
        report.append("-" * 15)
        
        # Correlation analysis
        if hasattr(self, 'correlation_results'):
            if 'pre_2018' in self.correlation_results and '2018' in self.correlation_results:
                pre_ievr_corr = self.correlation_results['pre_2018']['correlations'].get('ievr', 0)
                curr_ievr_corr = self.correlation_results['2018']['correlations'].get('ievr', 0)
                corr_change = curr_ievr_corr - pre_ievr_corr
                
                report.append(f"1. CORRELATION BREAKDOWN:")
                report.append(f"   • IEVR-REVR correlation: {pre_ievr_corr:.4f} → {curr_ievr_corr:.4f} (Δ={corr_change:+.4f})")
                
                if abs(corr_change) > 0.05:
                    report.append(f"   • ⚠️  Significant correlation change detected")
                report.append("")
        
        # Market regime analysis
        if hasattr(self, 'yearly_metrics'):
            metrics_2018 = self.yearly_metrics[self.yearly_metrics['year'] == 2018]
            if len(metrics_2018) > 0:
                revr_2018 = metrics_2018['avg_revr'].iloc[0]
                ievr_2018 = metrics_2018['avg_ievr'].iloc[0]
                vol_2018 = metrics_2018['avg_vol'].iloc[0]
                
                report.append(f"2. MARKET REGIME CHARACTERISTICS (2018):")
                report.append(f"   • Average REVR: {revr_2018:.4f}")
                report.append(f"   • Average IEVR: {ievr_2018:.4f}")
                report.append(f"   • Average Volatility: {vol_2018:.4f}")
                report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 17)
        report.append("1. FEATURE ENGINEERING:")
        report.append("   • Consider market regime indicators")
        report.append("   • Add volatility clustering features")
        report.append("   • Include macro-economic indicators")
        report.append("")
        report.append("2. MODEL IMPROVEMENTS:")
        report.append("   • Implement regime-switching models")
        report.append("   • Use time-varying coefficient models")
        report.append("   • Consider ensemble methods")
        report.append("")
        report.append("3. RISK MANAGEMENT:")
        report.append("   • Monitor correlation stability")
        report.append("   • Implement early warning systems")
        report.append("   • Use conservative confidence intervals during unstable periods")
        
        # Save report
        report_text = "\n".join(report)
        report_path = 'output_files/2018_analysis_summary_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Summary report saved: {report_path}")
        
        # Also print to console
        print("\n" + report_text)

def main():
    """
    Main function to run the systematic 2018 analysis
    """
    try:
        # Initialize analysis
        analyzer = Systematic2018Analysis()
        
        # Run comprehensive analysis
        analyzer.run_comprehensive_analysis()
        
        # Generate summary report
        analyzer.generate_summary_report()
        
        print(f"\n🎉 SYSTEMATIC 2018 ANALYSIS COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
