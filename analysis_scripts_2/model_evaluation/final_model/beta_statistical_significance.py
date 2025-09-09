#!/usr/bin/env python3
"""
Beta Statistical Significance Analysis
Enhanced analysis including p-values, t-statistics, and confidence intervals
for the multifactor linear regression coefficients
Features: IEVR + normative_iv_rv_ratio + IV_RATIO + SMIRK + vol_hl7 + vol_hl21 + z_score_momentum
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BetaStatisticalSignificance:
    """
    Analyze beta statistical significance with p-values and confidence intervals
    """
    
    def __init__(self):
        self.df = None
        self.features = [
            'ievr', 'normative_iv_rv_ratio', 'IV_RATIO', 
            'SMIRK', 'vol_hl21', 'z_score_momentum', 'dispersion_pct_ibes'
        ]
        self.target = 'revr'
        self.statistical_results = []
        self.windows = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("📊 LOADING DATASET FOR STATISTICAL SIGNIFICANCE ANALYSIS")
        print("="*65)
        
        try:
            # Use absolute paths to avoid directory issues
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, '..', '..', 'data_files')
            
            # Try model_df dataset first (same as algorithm comparison)
            try:
                self.df = pd.read_csv(os.path.join(data_dir, 'model_df.csv'))
                print(f"✅ Loaded model_df dataset: {len(self.df):,} observations")
            except FileNotFoundError:
                # Fallback to updated momentum dataset
                try:
                    self.df = pd.read_csv(os.path.join(data_dir, 'final_merged_dataset_with_momentum_updated.csv'))
                    print(f"✅ Loaded updated momentum dataset: {len(self.df):,} observations")
                except FileNotFoundError:
                    # Final fallback to original momentum dataset
                    self.df = pd.read_csv(os.path.join(data_dir, 'final_merged_dataset_with_momentum_final.csv'))
                    print(f"✅ Loaded original momentum dataset: {len(self.df):,} observations")
            
            # Convert earnings_date to datetime
            self.df['earnings_date'] = pd.to_datetime(self.df['earnings_date'])
            self.df['year'] = self.df['earnings_date'].dt.year
            
            print(f"📅 Date range: {self.df['year'].min()} - {self.df['year'].max()}")
            
            # Calculate normative_iv_rv_ratio if missing
            if 'normative_iv_rv_ratio' not in self.df.columns:
                if 'avg_pre' in self.df.columns and 'normative_realized_vol' in self.df.columns:
                    print("🔧 Calculating normative_iv_rv_ratio...")
                    self.df['normative_iv_rv_ratio'] = self.df['avg_pre'] / self.df['normative_realized_vol']
                    self.df['normative_iv_rv_ratio'] = self.df['normative_iv_rv_ratio'].replace([np.inf, -np.inf], np.nan)
                    print(f"✅ Created normative_iv_rv_ratio from avg_pre / normative_realized_vol")
            
            # Add z_score_momentum if missing
            if 'z_score_momentum' not in self.df.columns:
                print("🔧 Creating simple z_score_momentum feature...")
                self.df = self.df.sort_values(['ticker', 'earnings_date'])
                self.df['momentum_6m'] = self.df.groupby('ticker')['revr'].rolling(window=4, min_periods=2).mean().reset_index(0, drop=True)
                self.df['z_score_momentum'] = (
                    (self.df['momentum_6m'] - self.df.groupby('ticker')['momentum_6m'].transform('mean')) /
                    self.df.groupby('ticker')['momentum_6m'].transform('std')
                ).fillna(0)
                print(f"✅ Created z_score_momentum feature")
            
            # Check feature availability
            missing_features = [f for f in self.features if f not in self.df.columns]
            if missing_features:
                print(f"❌ Missing features: {missing_features}")
                return False
                
            print(f"✅ All {len(self.features)} features available")
            return True
            
        except FileNotFoundError:
            print("❌ Dataset file not found!")
            return False
    
    def create_rolling_windows(self, train_years=5, val_months=6, test_months=6):
        """Create rolling time windows for walk-forward validation"""
        print(f"\n🔄 CREATING ROLLING WINDOWS FOR STATISTICAL ANALYSIS")
        print("="*55)
        print(f"Training window: {train_years} years")
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
                
            # Skip validation for statistical analysis
            test_start = train_end + 1
            test_end = test_start + test_months - 1
            if test_end >= len(unique_dates):
                break
                
            # Create window
            window = {
                'train_start': unique_dates[current_idx],
                'train_end': unique_dates[train_end],
                'test_start': unique_dates[test_start],
                'test_end': unique_dates[test_end],
                'window_id': len(windows) + 1
            }
            
            windows.append(window)
            current_idx += test_months  # Move forward by test window size
        
        self.windows = windows
        print(f"Created {len(windows)} rolling windows")
        return windows
    
    def get_window_data(self, window):
        """Extract data for a specific rolling window"""
        # Convert periods to datetime for filtering
        train_start = window['train_start'].to_timestamp()
        train_end = window['train_end'].to_timestamp()
        test_start = window['test_start'].to_timestamp()
        test_end = window['test_end'].to_timestamp()
        
        # Filter data for each period
        train_data = self.df[(self.df['earnings_date'] >= train_start) & (self.df['earnings_date'] <= train_end)]
        test_data = self.df[(self.df['earnings_date'] >= test_start) & (self.df['earnings_date'] <= test_end)]
        
        return train_data, test_data
    
    def calculate_regression_statistics(self, X, y):
        """
        Calculate detailed regression statistics including p-values and confidence intervals
        """
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predictions
        y_pred = model.predict(X)
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Degrees of freedom
        n = X.shape[0]
        k = X.shape[1]
        df_resid = n - k - 1  # -1 for intercept
        
        # Mean squared error
        mse = np.sum(residuals**2) / df_resid
        
        # Calculate standard errors
        # X with intercept column
        X_with_intercept = np.column_stack([np.ones(n), X])
        
        try:
            # Covariance matrix
            cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            
            # Standard errors (diagonal of covariance matrix)
            std_errors = np.sqrt(np.diag(cov_matrix))
            
            # t-statistics
            coefficients = np.concatenate([[model.intercept_], model.coef_])
            t_stats = coefficients / std_errors
            
            # p-values (two-tailed test)
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))
            
            # 95% confidence intervals
            t_critical = stats.t.ppf(0.975, df_resid)  # 97.5th percentile for 95% CI
            ci_lower = coefficients - t_critical * std_errors
            ci_upper = coefficients + t_critical * std_errors
            
            # R-squared
            r2 = model.score(X, y)
            
            # Adjusted R-squared
            adj_r2 = 1 - (1 - r2) * (n - 1) / df_resid
            
            # F-statistic
            f_stat = (r2 / k) / ((1 - r2) / df_resid)
            f_p_value = 1 - stats.f.cdf(f_stat, k, df_resid)
            
            return {
                'model': model,
                'coefficients': coefficients[1:],  # Exclude intercept
                'intercept': coefficients[0],
                'std_errors': std_errors[1:],  # Exclude intercept std error
                't_stats': t_stats[1:],  # Exclude intercept t-stat
                'p_values': p_values[1:],  # Exclude intercept p-value
                'ci_lower': ci_lower[1:],  # Exclude intercept CI
                'ci_upper': ci_upper[1:],  # Exclude intercept CI
                'r2': r2,
                'adj_r2': adj_r2,
                'f_stat': f_stat,
                'f_p_value': f_p_value,
                'n_obs': n,
                'mse': mse
            }
            
        except np.linalg.LinAlgError:
            # Handle singular matrix (perfect multicollinearity)
            print("    ⚠️ Singular matrix detected - using approximate statistics")
            
            # Return basic statistics without detailed inference
            return {
                'model': model,
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'std_errors': np.full(len(model.coef_), np.nan),
                't_stats': np.full(len(model.coef_), np.nan),
                'p_values': np.full(len(model.coef_), np.nan),
                'ci_lower': np.full(len(model.coef_), np.nan),
                'ci_upper': np.full(len(model.coef_), np.nan),
                'r2': model.score(X, y),
                'adj_r2': np.nan,
                'f_stat': np.nan,
                'f_p_value': np.nan,
                'n_obs': n,
                'mse': np.nan
            }
    
    def analyze_statistical_significance(self):
        """Run comprehensive statistical significance analysis across all rolling windows"""
        print(f"\n📈 RUNNING STATISTICAL SIGNIFICANCE ANALYSIS")
        print("="*55)
        
        # Create rolling windows
        if self.windows is None:
            self.create_rolling_windows()
        
        statistical_results = []
        
        for window in self.windows:
            print(f"\n📊 Analyzing Window {window['window_id']} (Train: {window['train_start']} to {window['train_end']})...")
            
            # Get data for this window
            train_data, test_data = self.get_window_data(window)
            
            if len(train_data) < 100:
                print(f"  ⚠️ Insufficient data - skipping window")
                continue
            
            # Prepare clean training data
            all_required_cols = self.features + [self.target]
            train_clean = train_data[all_required_cols].dropna()
            
            if len(train_clean) < 50:
                print(f"  ⚠️ Insufficient clean training data - skipping window")
                continue
            
            # Prepare features and target
            X_train = train_clean[self.features].values
            y_train = train_clean[self.target].values
            
            # Standardize features for interpretability
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Calculate regression statistics
            stats_result = self.calculate_regression_statistics(X_train_scaled, y_train)
            
            # Store results
            window_result = {
                'window_id': window['window_id'],
                'train_start': str(window['train_start']),
                'train_end': str(window['train_end']),
                'test_start': str(window['test_start']),
                'test_end': str(window['test_end']),
                'test_year': int(str(window['test_start'])[:4]),
                'n_obs': stats_result['n_obs'],
                'r2': stats_result['r2'],
                'adj_r2': stats_result['adj_r2'],
                'f_stat': stats_result['f_stat'],
                'f_p_value': stats_result['f_p_value']
            }
            
            # Add individual coefficient statistics
            for i, feature in enumerate(self.features):
                window_result[f'beta_{feature}'] = stats_result['coefficients'][i]
                window_result[f'se_{feature}'] = stats_result['std_errors'][i]
                window_result[f't_stat_{feature}'] = stats_result['t_stats'][i]
                window_result[f'p_value_{feature}'] = stats_result['p_values'][i]
                window_result[f'ci_lower_{feature}'] = stats_result['ci_lower'][i]
                window_result[f'ci_upper_{feature}'] = stats_result['ci_upper'][i]
                
                # Significance indicators
                p_val = stats_result['p_values'][i]
                if np.isnan(p_val):
                    window_result[f'sig_{feature}'] = 'N/A'
                elif p_val < 0.001:
                    window_result[f'sig_{feature}'] = '***'
                elif p_val < 0.01:
                    window_result[f'sig_{feature}'] = '**'
                elif p_val < 0.05:
                    window_result[f'sig_{feature}'] = '*'
                elif p_val < 0.1:
                    window_result[f'sig_{feature}'] = '.'
                else:
                    window_result[f'sig_{feature}'] = ''
            
            statistical_results.append(window_result)
            
            # Print summary for this window
            significant_features = []
            for i, feature in enumerate(self.features):
                p_val = stats_result['p_values'][i]
                if not np.isnan(p_val) and p_val < 0.05:
                    significant_features.append(f"{feature}({p_val:.3f})")
            
            print(f"  ✅ R² = {stats_result['r2']:.4f}, N = {stats_result['n_obs']}")
            print(f"     Significant (p<0.05): {', '.join(significant_features) if significant_features else 'None'}")
        
        self.statistical_results = pd.DataFrame(statistical_results)
        
        if len(self.statistical_results) > 0:
            self.create_summary_statistics()
        
        return self.statistical_results
    
    def create_summary_statistics(self):
        """Create comprehensive summary statistics tables"""
        print(f"\n📊 STATISTICAL SIGNIFICANCE SUMMARY")
        print("="*50)
        
        # Overall significance summary
        significance_summary = {}
        
        for feature in self.features:
            p_col = f'p_value_{feature}'
            if p_col in self.statistical_results.columns:
                p_values = self.statistical_results[p_col].dropna()
                
                if len(p_values) > 0:
                    # Count significance levels
                    sig_001 = (p_values < 0.001).sum()
                    sig_01 = ((p_values >= 0.001) & (p_values < 0.01)).sum()
                    sig_05 = ((p_values >= 0.01) & (p_values < 0.05)).sum()
                    sig_10 = ((p_values >= 0.05) & (p_values < 0.1)).sum()
                    not_sig = (p_values >= 0.1).sum()
                    
                    significance_summary[feature] = {
                        'total_windows': len(p_values),
                        'sig_001': sig_001,
                        'sig_01': sig_01,
                        'sig_05': sig_05,
                        'sig_10': sig_10,
                        'not_sig': not_sig,
                        'pct_sig_05': (sig_001 + sig_01 + sig_05) / len(p_values) * 100,
                        'mean_p_value': p_values.mean(),
                        'median_p_value': p_values.median()
                    }
        
        # Print significance summary table
        print("FEATURE SIGNIFICANCE SUMMARY:")
        print("-" * 110)
        print(f"{'Feature':20s} {'N':>4s} {'***':>5s} {'**':>4s} {'*':>3s} {'.':>3s} {'ns':>4s} {'%Sig':>6s} {'Mean p':>8s} {'Med p':>7s}")
        print("-" * 110)
        
        # Sort by percentage significant
        sorted_features = sorted(significance_summary.keys(), 
                               key=lambda x: significance_summary[x]['pct_sig_05'], 
                               reverse=True)
        
        for feature in sorted_features:
            stats = significance_summary[feature]
            print(f"{feature:20s} {stats['total_windows']:4d} {stats['sig_001']:5d} {stats['sig_01']:4d} {stats['sig_05']:3d} {stats['sig_10']:3d} {stats['not_sig']:4d} {stats['pct_sig_05']:6.1f} {stats['mean_p_value']:8.4f} {stats['median_p_value']:7.4f}")
        
        print("\nLegend: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1, ns p≥0.1")
        
        # Coefficient stability and significance combined
        print(f"\nCOEFFICIENT STABILITY AND SIGNIFICANCE RANKING:")
        print("-" * 80)
        print(f"{'Feature':20s} {'Mean β':>8s} {'Std β':>8s} {'CV':>6s} {'%Sig':>6s} {'Rank':>6s}")
        print("-" * 80)
        
        combined_ranking = {}
        for feature in self.features:
            beta_col = f'beta_{feature}'
            if beta_col in self.statistical_results.columns and feature in significance_summary:
                betas = self.statistical_results[beta_col].dropna()
                
                if len(betas) > 0:
                    mean_beta = betas.mean()
                    std_beta = betas.std()
                    cv = std_beta / abs(mean_beta) if mean_beta != 0 else np.inf
                    pct_sig = significance_summary[feature]['pct_sig_05']
                    
                    # Combined score: high significance, low CV
                    score = pct_sig / (1 + cv)  # Higher is better
                    
                    combined_ranking[feature] = {
                        'mean_beta': mean_beta,
                        'std_beta': std_beta,
                        'cv': cv,
                        'pct_sig': pct_sig,
                        'score': score
                    }
        
        # Sort by combined score
        sorted_combined = sorted(combined_ranking.keys(), 
                               key=lambda x: combined_ranking[x]['score'], 
                               reverse=True)
        
        for i, feature in enumerate(sorted_combined):
            stats = combined_ranking[feature]
            cv_str = f"{stats['cv']:.2f}" if stats['cv'] != np.inf else "∞"
            print(f"{feature:20s} {stats['mean_beta']:8.4f} {stats['std_beta']:8.4f} {cv_str:>6s} {stats['pct_sig']:6.1f} {i+1:6d}")
        
        return significance_summary, combined_ranking
    
    def create_statistical_visualizations(self):
        """Create visualizations for statistical significance analysis"""
        print(f"\n📊 CREATING STATISTICAL SIGNIFICANCE VISUALIZATIONS")
        print("="*55)
        
        # Set BlackRock styling
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'grid.color': '#E5E5E5'
        })
        
        # Create comprehensive statistical plots
        self._create_pvalue_time_series()
        self._create_significance_heatmap()
        self._create_confidence_interval_plot()
        self._create_summary_table_plot()
    
    def _create_pvalue_time_series(self):
        """Create time series plot of p-values"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('white')
        fig.suptitle('Statistical Significance Analysis: P-Values Over Time\nMultifactor Linear Regression', 
                     fontsize=14, fontweight='bold', color='#003366')
        
        # Get test start dates for x-axis
        if 'test_start' in self.statistical_results.columns:
            x_axis = pd.to_datetime(self.statistical_results['test_start'], format='%Y-%m')
        else:
            x_axis = self.statistical_results['window_id']
        
        # Key features for different subplots
        feature_groups = [
            ['ievr', 'normative_iv_rv_ratio', 'z_score_momentum'],
            ['IV_RATIO', 'SMIRK', 'dispersion_pct_ibes'],
            ['vol_hl21'],
            ['ievr', 'normative_iv_rv_ratio', 'dispersion_pct_ibes']  # Combined key features
        ]
        
        titles = ['Key Economic Features', 'Option Features', 'Volatility Features', 'Core Features (Detailed)']
        colors = ['#003366', '#66CCFF', '#8C8C8C', '#FF6633', '#004d99', '#99d6ff', '#b3b3b3']
        
        for idx, (features_subset, title) in enumerate(zip(feature_groups, titles)):
            ax = axes[idx//2, idx%2]
            
            for i, feature in enumerate(features_subset):
                p_col = f'p_value_{feature}'
                if p_col in self.statistical_results.columns:
                    p_values = self.statistical_results[p_col]
                    
                    # Plot p-values
                    ax.plot(x_axis, p_values, label=feature.replace('_', ' ').title(), 
                           color=colors[i % len(colors)], linewidth=2, marker='o', markersize=4, alpha=0.8)
            
            # Add significance threshold lines
            ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
            ax.axhline(y=0.01, color='darkred', linestyle='--', alpha=0.7, label='p=0.01')
            
            ax.set_title(title, fontweight='bold', color='#003366')
            ax.set_ylabel('P-Value', color='#003366', fontweight='semibold')
            ax.legend(fontsize=8, loc='upper right')
            ax.set_ylim(0, 1)
            
            if idx >= 2:  # Bottom row
                ax.set_xlabel('Test Period Start' if 'test_start' in self.statistical_results.columns else 'Window ID', 
                             color='#003366', fontweight='semibold')
        
        plt.tight_layout()
        
        # Save plot
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output_files')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'statistical_significance_pvalues.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ P-value time series plot saved: {output_path}")
        
        plt.close()
    
    def _create_significance_heatmap(self):
        """Create heatmap showing significance levels across features and time"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.patch.set_facecolor('white')
        fig.suptitle('Statistical Significance Heatmap Analysis', 
                     fontsize=14, fontweight='bold', color='#003366')
        
        # Prepare significance matrix
        years = sorted(self.statistical_results['test_year'].unique())
        significance_matrix = []
        
        for year in years:
            year_data = self.statistical_results[self.statistical_results['test_year'] == year]
            if len(year_data) > 0:
                year_sig = []
                for feature in self.features:
                    p_col = f'p_value_{feature}'
                    if p_col in year_data.columns:
                        p_val = year_data[p_col].mean()  # Average if multiple windows per year
                        if np.isnan(p_val):
                            year_sig.append(1)  # Not significant
                        else:
                            year_sig.append(p_val)
                    else:
                        year_sig.append(1)
                significance_matrix.append(year_sig)
        
        # Convert to numpy array
        sig_array = np.array(significance_matrix).T
        feature_labels = [f.replace('_', ' ').title() for f in self.features]
        
        # Plot 1: P-value heatmap
        im1 = ax1.imshow(sig_array, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=0.2)
        ax1.set_title('P-Values by Feature and Year\n(Darker = More Significant)', fontweight='bold', color='#003366')
        ax1.set_xlabel('Year', color='#003366', fontweight='semibold')
        ax1.set_ylabel('Features', color='#003366', fontweight='semibold')
        ax1.set_xticks(range(len(years)))
        ax1.set_xticklabels(years, rotation=45)
        ax1.set_yticks(range(len(feature_labels)))
        ax1.set_yticklabels(feature_labels)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('P-Value', color='#003366')
        
        # Plot 2: Significance frequency
        sig_freq = []
        for feature in self.features:
            p_col = f'p_value_{feature}'
            if p_col in self.statistical_results.columns:
                p_values = self.statistical_results[p_col].dropna()
                if len(p_values) > 0:
                    sig_05_pct = (p_values < 0.05).mean() * 100
                    sig_freq.append(sig_05_pct)
                else:
                    sig_freq.append(0)
            else:
                sig_freq.append(0)
        
        # Create bar plot
        bars = ax2.barh(range(len(feature_labels)), sig_freq, 
                       color='#66CCFF', alpha=0.8, edgecolor='#003366')
        ax2.set_title('Frequency of Statistical Significance\n(% of Windows with p<0.05)', 
                     fontweight='bold', color='#003366')
        ax2.set_xlabel('% of Windows Significant', color='#003366', fontweight='semibold')
        ax2.set_yticks(range(len(feature_labels)))
        ax2.set_yticklabels(feature_labels)
        ax2.set_xlim(0, 100)
        
        # Add percentage labels
        for i, (bar, freq) in enumerate(zip(bars, sig_freq)):
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{freq:.1f}%', ha='left', va='center', fontsize=9, color='#003366')
        
        # Add significance threshold line
        ax2.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='50% threshold')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output_files')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'statistical_significance_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Significance heatmap saved: {output_path}")
        
        plt.close()
    
    def _create_confidence_interval_plot(self):
        """Create confidence interval plot for key features"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.patch.set_facecolor('white')
        fig.suptitle('95% Confidence Intervals for Key Features\nMultifactor Linear Regression Coefficients', 
                     fontsize=14, fontweight='bold', color='#003366')
        
        # Key features to plot
        key_features = ['ievr', 'normative_iv_rv_ratio', 'z_score_momentum', 'vol_hl21', 'dispersion_pct_ibes']
        
        # Get test start dates for x-axis
        if 'test_start' in self.statistical_results.columns:
            x_axis = pd.to_datetime(self.statistical_results['test_start'], format='%Y-%m')
            x_label = 'Test Period Start'
        else:
            x_axis = self.statistical_results['window_id']
            x_label = 'Window Number'
        
        for idx, feature in enumerate(key_features):
            ax = axes[idx//2, idx%2]
            
            beta_col = f'beta_{feature}'
            ci_lower_col = f'ci_lower_{feature}'
            ci_upper_col = f'ci_upper_{feature}'
            p_col = f'p_value_{feature}'
            
            if all(col in self.statistical_results.columns for col in [beta_col, ci_lower_col, ci_upper_col]):
                betas = self.statistical_results[beta_col]
                ci_lower = self.statistical_results[ci_lower_col]
                ci_upper = self.statistical_results[ci_upper_col]
                p_values = self.statistical_results[p_col] if p_col in self.statistical_results.columns else None
                
                # Color points by significance
                colors = []
                if p_values is not None:
                    for p in p_values:
                        if np.isnan(p):
                            colors.append('#8C8C8C')
                        elif p < 0.01:
                            colors.append('#003366')  # Highly significant
                        elif p < 0.05:
                            colors.append('#66CCFF')  # Significant
                        else:
                            colors.append('#FF6633')  # Not significant
                else:
                    colors = ['#003366'] * len(betas)
                
                # Plot confidence intervals
                for i in range(len(x_axis)):
                    ax.plot([x_axis.iloc[i], x_axis.iloc[i]], [ci_lower.iloc[i], ci_upper.iloc[i]], 
                           color=colors[i], alpha=0.6, linewidth=2)
                
                # Plot point estimates
                ax.scatter(x_axis, betas, c=colors, s=50, alpha=0.8, edgecolors='white', linewidth=1)
                
                # Add zero line
                ax.axhline(y=0, color='#8C8C8C', linestyle='-', alpha=0.5)
                
                ax.set_title(f'{feature.replace("_", " ").title()}', fontweight='bold', color='#003366')
                ax.set_ylabel('Coefficient Value', color='#003366', fontweight='semibold')
                
                if idx >= 4:  # Bottom row (adjusted for 3x2 grid)
                    ax.set_xlabel(x_label, color='#003366', fontweight='semibold')
        
        # Hide the unused 6th subplot
        axes[2, 1].set_visible(False)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#003366', markersize=8, label='p < 0.01'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#66CCFF', markersize=8, label='p < 0.05'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6633', markersize=8, label='p ≥ 0.05'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#8C8C8C', markersize=8, label='N/A')
        ]
        
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        
        # Save plot
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output_files')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'statistical_significance_confidence_intervals.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Confidence interval plot saved: {output_path}")
        
        plt.close()
    
    def _create_summary_table_plot(self):
        """Create publication-ready summary table as a plot"""
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor('white')
        
        # Calculate summary statistics
        summary_data = []
        
        for feature in self.features:
            beta_col = f'beta_{feature}'
            se_col = f'se_{feature}'
            p_col = f'p_value_{feature}'
            
            if all(col in self.statistical_results.columns for col in [beta_col, se_col, p_col]):
                betas = self.statistical_results[beta_col].dropna()
                ses = self.statistical_results[se_col].dropna()
                p_values = self.statistical_results[p_col].dropna()
                
                if len(betas) > 0:
                    mean_beta = betas.mean()
                    mean_se = ses.mean()
                    mean_p = p_values.mean()
                    pct_sig_05 = (p_values < 0.05).mean() * 100
                    pct_sig_01 = (p_values < 0.01).mean() * 100
                    
                    # Significance stars for mean p-value
                    if mean_p < 0.001:
                        sig_stars = '***'
                    elif mean_p < 0.01:
                        sig_stars = '**'
                    elif mean_p < 0.05:
                        sig_stars = '*'
                    elif mean_p < 0.1:
                        sig_stars = '.'
                    else:
                        sig_stars = ''
                    
                    summary_data.append([
                        feature.replace('_', ' ').title(),
                        f'{mean_beta:.4f}{sig_stars}',
                        f'({mean_se:.4f})',
                        f'{mean_p:.4f}',
                        f'{pct_sig_05:.1f}%',
                        f'{pct_sig_01:.1f}%',
                        f'{len(betas)}'
                    ])
        
        # Create table
        headers = ['Feature', 'Mean β', '(Std Err)', 'Mean p', '% Sig(0.05)', '% Sig(0.01)', 'N Windows']
        
        # Sort by percentage significant at 0.05 level
        summary_data.sort(key=lambda x: float(x[4].replace('%', '')), reverse=True)
        
        # Create the table
        table = ax.table(cellText=summary_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.12, 0.1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Header styling
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#003366')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows based on significance
        for i, row in enumerate(summary_data):
            pct_sig = float(row[4].replace('%', ''))
            if pct_sig >= 70:
                color = '#E6F3FF'  # Light blue for highly significant
            elif pct_sig >= 50:
                color = '#F0F8FF'  # Very light blue for moderately significant
            else:
                color = 'white'    # White for less significant
            
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor(color)
        
        ax.set_title('Statistical Significance Summary Table\nMultifactor Linear Regression (Rolling Windows)', 
                     fontsize=14, fontweight='bold', color='#003366', pad=20)
        ax.axis('off')
        
        # Add footnote
        footnote = "*** p<0.001, ** p<0.01, * p<0.05, . p<0.10\nβ coefficients from standardized features"
        ax.text(0.5, 0.02, footnote, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=9, style='italic', color='#666666')
        
        plt.tight_layout()
        
        # Save plot
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output_files')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'statistical_significance_summary_table.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Summary table plot saved: {output_path}")
        
        plt.close()
    
    def save_results(self):
        """Save statistical significance results to CSV files"""
        print(f"\n💾 SAVING STATISTICAL SIGNIFICANCE RESULTS")
        print("="*45)
        
        if len(self.statistical_results) > 0:
            # Create output directory
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, 'output_files')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save detailed results
            results_path = os.path.join(output_dir, 'statistical_significance_results.csv')
            self.statistical_results.to_csv(results_path, index=False)
            print(f"✅ Detailed results saved: {results_path}")
            
            # Create summary table for publication
            summary_table = []
            
            for feature in self.features:
                beta_col = f'beta_{feature}'
                se_col = f'se_{feature}'
                p_col = f'p_value_{feature}'
                
                if all(col in self.statistical_results.columns for col in [beta_col, se_col, p_col]):
                    betas = self.statistical_results[beta_col].dropna()
                    ses = self.statistical_results[se_col].dropna()
                    p_values = self.statistical_results[p_col].dropna()
                    
                    if len(betas) > 0:
                        summary_table.append({
                            'feature': feature,
                            'mean_beta': betas.mean(),
                            'std_beta': betas.std(),
                            'mean_se': ses.mean(),
                            'mean_p_value': p_values.mean(),
                            'median_p_value': p_values.median(),
                            'pct_sig_001': (p_values < 0.001).mean() * 100,
                            'pct_sig_01': (p_values < 0.01).mean() * 100,
                            'pct_sig_05': (p_values < 0.05).mean() * 100,
                            'pct_sig_10': (p_values < 0.1).mean() * 100,
                            'n_windows': len(betas)
                        })
            
            summary_df = pd.DataFrame(summary_table)
            summary_path = os.path.join(output_dir, 'statistical_significance_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"✅ Summary table saved: {summary_path}")
        
        print("\n🎉 STATISTICAL SIGNIFICANCE ANALYSIS COMPLETED!")
        print(f"Key outputs:")
        print(f"  • statistical_significance_pvalues.png - P-value evolution")
        print(f"  • statistical_significance_heatmap.png - Significance patterns")
        print(f"  • statistical_significance_confidence_intervals.png - CI analysis")
        print(f"  • statistical_significance_summary_table.png - Publication table")
        print(f"  • statistical_significance_results.csv - Detailed statistics")
        print(f"  • statistical_significance_summary.csv - Summary table")

def main():
    """
    Main function to run statistical significance analysis
    """
    try:
        print("🚀 STATISTICAL SIGNIFICANCE ANALYSIS FOR MULTIFACTOR LINEAR REGRESSION")
        print("="*85)
        print("Features: IEVR + normative_iv_rv_ratio + IV_RATIO + SMIRK + vol_hl21 + z_score_momentum + dispersion_pct_ibes")
        print("Analysis: P-values, t-statistics, confidence intervals, significance testing")
        print("="*85)
        
        # Initialize analyzer
        analyzer = BetaStatisticalSignificance()
        
        # Load data
        if not analyzer.load_data():
            return
        
        # Run statistical significance analysis
        results = analyzer.analyze_statistical_significance()
        
        if results is not None and len(results) > 0:
            # Create visualizations
            analyzer.create_statistical_visualizations()
            
            # Save results
            analyzer.save_results()
        else:
            print("❌ No results generated")
            
    except Exception as e:
        print(f"❌ Error in statistical significance analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

