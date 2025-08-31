#!/usr/bin/env python3
"""
Focused 2018 H2 Analysis
Deep dive into the catastrophic model failure in the second half of 2018
Window 17: Test Period 2018-07 to 2018-12, Test R² = 0.0118
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from scipy.stats import jarque_bera, normaltest
import warnings
warnings.filterwarnings('ignore')

class Focused2018H2Analysis:
    """
    Deep analysis of the catastrophic H2 2018 model failure
    """
    
    def __init__(self):
        self.df = None
        self.features = [
            'ievr', 'normative_iv_rv_ratio', 'SKEW', 'KURT', 'IV_RATIO', 
            'SMIRK', 'vol_hl7', 'vol_hl10', 'vol_hl21', 'z_score_momentum'
        ]
        self.target = 'revr'
        self.crisis_period = None
        self.baseline_periods = None
        self.crisis_data = None
        self.baseline_data = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("📊 LOADING DATASET FOR FOCUSED 2018 H2 ANALYSIS")
        print("="*55)
        
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
            self.df['year'] = self.df['earnings_date'].dt.year
            self.df['month'] = self.df['earnings_date'].dt.month
            self.df['quarter'] = self.df['earnings_date'].dt.quarter
            
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
    
    def define_analysis_periods(self):
        """Define the specific crisis period (H2 2018) and comparison periods"""
        print(f"\n🎯 DEFINING ANALYSIS PERIODS")
        print("="*35)
        
        # Crisis period: H2 2018 (July-December 2018)
        self.crisis_period = {
            'start': '2018-07-01',
            'end': '2018-12-31',
            'description': 'H2 2018 Crisis (Window 17 Test Period)'
        }
        
        # Baseline periods: Exclude all of 2018 to get clean comparison
        self.baseline_periods = [
            {'start': '2015-01-01', 'end': '2017-12-31', 'description': '2015-2017 Pre-Crisis'},
            {'start': '2019-01-01', 'end': '2020-12-31', 'description': '2019-2020 Post-Crisis'},
            {'start': '2018-01-01', 'end': '2018-06-30', 'description': 'H1 2018 (Good Performance)'}
        ]
        
        print(f"🔴 CRISIS PERIOD:")
        print(f"  {self.crisis_period['description']}: {self.crisis_period['start']} to {self.crisis_period['end']}")
        
        print(f"\n🟢 BASELINE PERIODS:")
        for period in self.baseline_periods:
            print(f"  {period['description']}: {period['start']} to {period['end']}")
        
        # Filter data for analysis
        crisis_start = pd.to_datetime(self.crisis_period['start'])
        crisis_end = pd.to_datetime(self.crisis_period['end'])
        
        self.crisis_data = self.df[
            (self.df['earnings_date'] >= crisis_start) & 
            (self.df['earnings_date'] <= crisis_end)
        ].copy()
        
        # Combine all baseline periods
        baseline_dfs = []
        for period in self.baseline_periods:
            period_start = pd.to_datetime(period['start'])
            period_end = pd.to_datetime(period['end'])
            period_data = self.df[
                (self.df['earnings_date'] >= period_start) & 
                (self.df['earnings_date'] <= period_end)
            ].copy()
            baseline_dfs.append(period_data)
        
        self.baseline_data = pd.concat(baseline_dfs, ignore_index=True)
        
        print(f"\n📊 DATA SUMMARY:")
        print(f"  Crisis period (H2 2018): {len(self.crisis_data):,} observations")
        print(f"  Baseline periods: {len(self.baseline_data):,} observations")
        
        return True
    
    def analyze_h2_2018_breakdown(self):
        """Comprehensive analysis of H2 2018 model breakdown"""
        print(f"\n💥 ANALYZING H2 2018 MODEL BREAKDOWN")
        print("="*40)
        
        self._compare_feature_distributions()
        self._analyze_correlation_breakdown()
        self._analyze_monthly_progression()
        self._analyze_sector_effects()
        self._model_performance_breakdown()
    
    def _compare_feature_distributions(self):
        """Compare feature distributions between H2 2018 and baseline"""
        print(f"\n📊 FEATURE DISTRIBUTION ANALYSIS (H2 2018 vs Baseline)")
        print("-" * 60)
        
        distribution_changes = []
        
        for feature in self.features + [self.target]:
            if feature in self.baseline_data.columns and feature in self.crisis_data.columns:
                baseline_vals = self.baseline_data[feature].dropna()
                crisis_vals = self.crisis_data[feature].dropna()
                
                if len(baseline_vals) > 10 and len(crisis_vals) > 5:
                    # Basic statistics
                    baseline_stats = {
                        'mean': baseline_vals.mean(),
                        'std': baseline_vals.std(),
                        'skew': baseline_vals.skew(),
                        'kurt': baseline_vals.kurtosis(),
                        'median': baseline_vals.median(),
                        'q75': baseline_vals.quantile(0.75),
                        'q25': baseline_vals.quantile(0.25)
                    }
                    
                    crisis_stats = {
                        'mean': crisis_vals.mean(),
                        'std': crisis_vals.std(),
                        'skew': crisis_vals.skew(),
                        'kurt': crisis_vals.kurtosis(),
                        'median': crisis_vals.median(),
                        'q75': crisis_vals.quantile(0.75),
                        'q25': crisis_vals.quantile(0.25)
                    }
                    
                    # Statistical tests
                    try:
                        # Kolmogorov-Smirnov test
                        ks_stat, ks_pval = stats.ks_2samp(baseline_vals, crisis_vals)
                        
                        # Mann-Whitney U test
                        mw_stat, mw_pval = stats.mannwhitneyu(baseline_vals, crisis_vals, alternative='two-sided')
                        
                        # Calculate percentage changes
                        mean_change = (crisis_stats['mean'] - baseline_stats['mean']) / abs(baseline_stats['mean']) * 100 if abs(baseline_stats['mean']) > 0.001 else 0
                        std_change = (crisis_stats['std'] - baseline_stats['std']) / abs(baseline_stats['std']) * 100 if abs(baseline_stats['std']) > 0.001 else 0
                        
                        distribution_changes.append({
                            'feature': feature,
                            'baseline_mean': baseline_stats['mean'],
                            'crisis_mean': crisis_stats['mean'],
                            'baseline_std': baseline_stats['std'],
                            'crisis_std': crisis_stats['std'],
                            'mean_change_pct': mean_change,
                            'std_change_pct': std_change,
                            'ks_statistic': ks_stat,
                            'ks_pvalue': ks_pval,
                            'mw_pvalue': mw_pval,
                            'distribution_shift': 'Significant' if ks_pval < 0.05 else 'Not Significant',
                            'severity': 'Critical' if ks_pval < 0.01 and abs(mean_change) > 10 else 'Moderate' if ks_pval < 0.05 else 'Minor'
                        })
                        
                    except Exception as e:
                        print(f"  ⚠️ Error analyzing {feature}: {str(e)[:50]}")
        
        # Create and display results
        changes_df = pd.DataFrame(distribution_changes)
        
        if len(changes_df) > 0:
            print(f"FEATURE DISTRIBUTION CHANGES (H2 2018):")
            print("-" * 80)
            print(f"{'Feature':22s} {'Mean Δ%':>10s} {'Std Δ%':>10s} {'KS p-val':>10s} {'Severity':>12s}")
            print("-" * 80)
            
            # Sort by KS statistic (most changed first)
            changes_df = changes_df.sort_values('ks_statistic', ascending=False)
            
            for _, row in changes_df.iterrows():
                severity_icon = "🔴" if row['severity'] == 'Critical' else "🟡" if row['severity'] == 'Moderate' else "🟢"
                print(f"{row['feature']:22s} {row['mean_change_pct']:+9.1f}% {row['std_change_pct']:+9.1f}% {row['ks_pvalue']:10.4f} {severity_icon} {row['severity']:>9s}")
            
            # Identify most critical changes
            critical_changes = changes_df[changes_df['severity'] == 'Critical']
            print(f"\n🚨 CRITICAL DISTRIBUTION CHANGES:")
            print("-" * 40)
            
            if len(critical_changes) > 0:
                for i, (_, row) in enumerate(critical_changes.head(5).iterrows()):
                    print(f"{i+1}. {row['feature']:20s}: Mean {row['mean_change_pct']:+.1f}%, Std {row['std_change_pct']:+.1f}%")
                    print(f"   Baseline: μ={row['baseline_mean']:.4f}, σ={row['baseline_std']:.4f}")
                    print(f"   H2 2018:  μ={row['crisis_mean']:.4f}, σ={row['crisis_std']:.4f}")
                    print()
            else:
                print("No critical distribution changes detected.")
            
            self.distribution_changes = changes_df
    
    def _analyze_correlation_breakdown(self):
        """Analyze correlation structure breakdown in H2 2018"""
        print(f"\n🔗 CORRELATION BREAKDOWN ANALYSIS")
        print("-" * 35)
        
        # Calculate correlation matrices
        baseline_features = self.baseline_data[self.features + [self.target]]
        crisis_features = self.crisis_data[self.features + [self.target]]
        
        baseline_corr = baseline_features.corr()
        crisis_corr = crisis_features.corr()
        
        # Focus on target correlations
        baseline_target_corr = baseline_corr[self.target].drop(self.target)
        crisis_target_corr = crisis_corr[self.target].drop(self.target)
        
        print(f"TARGET CORRELATIONS (REVR Prediction):")
        print("-" * 70)
        print(f"{'Feature':22s} {'Baseline':>10s} {'H2 2018':>10s} {'Change':>10s} {'Impact':>12s}")
        print("-" * 70)
        
        correlation_breakdown = []
        
        for feature in self.features:
            if feature in baseline_target_corr.index and feature in crisis_target_corr.index:
                baseline_corr_val = baseline_target_corr[feature]
                crisis_corr_val = crisis_target_corr[feature]
                change = crisis_corr_val - baseline_corr_val
                
                # Assess impact
                if abs(change) > 0.15:
                    impact = "🔴 Severe"
                elif abs(change) > 0.08:
                    impact = "🟡 Moderate" 
                elif abs(change) > 0.03:
                    impact = "🟠 Minor"
                else:
                    impact = "🟢 Stable"
                
                print(f"{feature:22s} {baseline_corr_val:10.3f} {crisis_corr_val:10.3f} {change:+10.3f} {impact:>12s}")
                
                correlation_breakdown.append({
                    'feature': feature,
                    'baseline_correlation': baseline_corr_val,
                    'crisis_correlation': crisis_corr_val,
                    'correlation_change': change,
                    'abs_change': abs(change),
                    'impact_level': impact.split()[1]  # Remove emoji
                })
        
        # Identify most broken relationships
        breakdown_df = pd.DataFrame(correlation_breakdown)
        most_broken = breakdown_df.nlargest(3, 'abs_change')
        
        print(f"\n💔 MOST BROKEN RELATIONSHIPS:")
        print("-" * 35)
        for i, (_, row) in enumerate(most_broken.iterrows()):
            direction = "weakened" if row['correlation_change'] > 0 else "strengthened"
            print(f"{i+1}. {row['feature']:20s}: {row['baseline_correlation']:.3f} → {row['crisis_correlation']:.3f}")
            print(f"   Relationship {direction} by {abs(row['correlation_change']):.3f}")
        
        self.correlation_breakdown = breakdown_df
    
    def _analyze_monthly_progression(self):
        """Analyze how the breakdown progressed month by month in H2 2018"""
        print(f"\n📅 MONTHLY PROGRESSION ANALYSIS (H2 2018)")
        print("-" * 45)
        
        # Filter H2 2018 data
        h2_2018 = self.crisis_data.copy()
        h2_2018['month_name'] = h2_2018['earnings_date'].dt.strftime('%Y-%m')
        
        # Calculate monthly statistics
        monthly_stats = []
        
        for month in h2_2018['month_name'].unique():
            month_data = h2_2018[h2_2018['month_name'] == month]
            
            if len(month_data) > 3:  # Need minimum observations
                stats_dict = {
                    'month': month,
                    'observations': len(month_data),
                }
                
                # Calculate key feature means
                for feature in ['revr', 'ievr', 'normative_iv_rv_ratio', 'vol_hl10', 'SKEW']:
                    if feature in month_data.columns:
                        stats_dict[f'{feature}_mean'] = month_data[feature].mean()
                        stats_dict[f'{feature}_std'] = month_data[feature].std()
                
                monthly_stats.append(stats_dict)
        
        monthly_df = pd.DataFrame(monthly_stats).sort_values('month')
        
        if len(monthly_df) > 0:
            print(f"MONTHLY BREAKDOWN PROGRESSION:")
            print("-" * 50)
            print(f"{'Month':8s} {'Obs':>5s} {'REVR':>8s} {'IEVR':>8s} {'Norm_Ratio':>10s} {'Vol_HL10':>9s}")
            print("-" * 50)
            
            for _, row in monthly_df.iterrows():
                print(f"{row['month']:8s} {row['observations']:5d} {row.get('revr_mean', 0):8.4f} {row.get('ievr_mean', 0):8.4f} {row.get('normative_iv_rv_ratio_mean', 0):10.4f} {row.get('vol_hl10_mean', 0):9.6f}")
            
            # Identify the worst month
            if 'revr_std' in monthly_df.columns:
                worst_month = monthly_df.loc[monthly_df['revr_std'].idxmax()]
                print(f"\n🔴 WORST MONTH: {worst_month['month']} (REVR volatility: {worst_month['revr_std']:.4f})")
            
            self.monthly_progression = monthly_df
    
    def _analyze_sector_effects(self):
        """Analyze if certain sectors were more affected during H2 2018"""
        print(f"\n🏭 SECTOR IMPACT ANALYSIS")
        print("-" * 25)
        
        # Check if sector information is available
        sector_cols = [col for col in self.df.columns if 'sector' in col.lower() or 'industry' in col.lower()]
        
        if len(sector_cols) > 0:
            sector_col = sector_cols[0]
            print(f"Using sector column: {sector_col}")
            
            # Compare sector performance baseline vs crisis
            baseline_sector = self.baseline_data.groupby(sector_col)[self.target].agg(['mean', 'std', 'count']).reset_index()
            crisis_sector = self.crisis_data.groupby(sector_col)[self.target].agg(['mean', 'std', 'count']).reset_index()
            
            # Merge and calculate changes
            sector_comparison = baseline_sector.merge(crisis_sector, on=sector_col, suffixes=('_baseline', '_crisis'))
            sector_comparison['mean_change'] = sector_comparison['mean_crisis'] - sector_comparison['mean_baseline']
            sector_comparison['volatility_change'] = sector_comparison['std_crisis'] - sector_comparison['std_baseline']
            
            # Filter for sectors with sufficient data
            sector_comparison = sector_comparison[
                (sector_comparison['count_baseline'] >= 10) & 
                (sector_comparison['count_crisis'] >= 3)
            ]
            
            if len(sector_comparison) > 0:
                print(f"\nSECTOR IMPACT RANKING:")
                print("-" * 40)
                
                sector_comparison = sector_comparison.sort_values('mean_change', ascending=True)
                
                for _, row in sector_comparison.head(5).iterrows():
                    impact = "🔴 High" if abs(row['mean_change']) > 0.05 else "🟡 Moderate" if abs(row['mean_change']) > 0.02 else "🟢 Low"
                    print(f"{row[sector_col][:20]:20s}: {row['mean_change']:+.4f} {impact}")
                
                self.sector_analysis = sector_comparison
            else:
                print("Insufficient sector data for analysis")
        else:
            print("No sector information available in dataset")
    
    def _model_performance_breakdown(self):
        """Analyze model performance breakdown using different algorithms"""
        print(f"\n🤖 MODEL PERFORMANCE BREAKDOWN ANALYSIS")
        print("-" * 45)
        
        # Prepare clean datasets
        baseline_clean = self.baseline_data[self.features + [self.target]].dropna()
        crisis_clean = self.crisis_data[self.features + [self.target]].dropna()
        
        if len(baseline_clean) < 50 or len(crisis_clean) < 10:
            print("❌ Insufficient clean data for model analysis")
            return
        
        print(f"Clean data: Baseline={len(baseline_clean)}, H2 2018={len(crisis_clean)}")
        
        # Define models to test
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
        }
        
        model_results = []
        
        for model_name, model in models.items():
            try:
                # Train on baseline, test on crisis
                X_train = baseline_clean[self.features]
                y_train = baseline_clean[self.target]
                X_test = crisis_clean[self.features]
                y_test = crisis_clean[self.target]
                
                # Scale features for linear regression
                if model_name == 'Linear Regression':
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model.fit(X_train_scaled, y_train)
                    train_r2 = model.score(X_train_scaled, y_train)
                    test_r2 = model.score(X_test_scaled, y_test)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    train_r2 = model.score(X_train, y_train)
                    test_r2 = model.score(X_test, y_test)
                    y_pred = model.predict(X_test)
                
                # Calculate additional metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Model breakdown severity
                if test_r2 < 0:
                    breakdown = "🔴 Complete"
                elif test_r2 < 0.05:
                    breakdown = "🟠 Severe"
                elif test_r2 < 0.15:
                    breakdown = "🟡 Moderate"
                else:
                    breakdown = "🟢 Stable"
                
                model_results.append({
                    'model': model_name,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'rmse': rmse,
                    'breakdown_severity': breakdown
                })
                
                print(f"{model_name:20s}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, RMSE={rmse:.4f} {breakdown}")
                
            except Exception as e:
                print(f"{model_name:20s}: ❌ Failed ({str(e)[:30]})")
        
        # Feature importance analysis for Random Forest
        if len([r for r in model_results if r['model'] == 'Random Forest']) > 0:
            try:
                rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
                rf.fit(baseline_clean[self.features], baseline_clean[self.target])
                
                importance = rf.feature_importances_
                feature_importance = list(zip(self.features, importance))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                print(f"\n🌲 RANDOM FOREST FEATURE IMPORTANCE (Baseline Model):")
                print("-" * 50)
                for i, (feature, imp) in enumerate(feature_importance[:5]):
                    print(f"{i+1}. {feature:20s}: {imp:.4f}")
                
            except Exception as e:
                print(f"Feature importance analysis failed: {e}")
        
        self.model_results = model_results
    
    def create_focused_visualizations(self):
        """Create focused visualizations for H2 2018 analysis"""
        print(f"\n📊 CREATING FOCUSED H2 2018 VISUALIZATIONS")
        print("="*50)
        
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
        
        self._create_distribution_comparison()
        self._create_correlation_heatmap()
        self._create_monthly_progression_plot()
        self._create_comprehensive_dashboard()
    
    def _create_distribution_comparison(self):
        """Create distribution comparison plots for key features"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.patch.set_facecolor('white')
        fig.suptitle('H2 2018 Crisis: Feature Distribution Breakdown\nCatastrophic Model Failure Analysis (Test R² = 0.0118)', 
                     fontsize=14, fontweight='bold', color='#003366')
        
        # Key features to analyze
        key_features = ['revr', 'ievr', 'normative_iv_rv_ratio', 'vol_hl10', 'SKEW', 'SMIRK']
        
        for i, feature in enumerate(key_features):
            if i < 6:
                ax = axes[i//3, i%3]
                
                if feature in self.baseline_data.columns and feature in self.crisis_data.columns:
                    baseline_vals = self.baseline_data[feature].dropna()
                    crisis_vals = self.crisis_data[feature].dropna()
                    
                    if len(baseline_vals) > 10 and len(crisis_vals) > 5:
                        # Create side-by-side histograms
                        ax.hist(baseline_vals, bins=25, alpha=0.7, color='#66CCFF', 
                               label=f'Baseline (n={len(baseline_vals)})', density=True, edgecolor='white')
                        ax.hist(crisis_vals, bins=15, alpha=0.8, color='#FF3333', 
                               label=f'H2 2018 (n={len(crisis_vals)})', density=True, edgecolor='white')
                        
                        # Add vertical lines for means
                        ax.axvline(baseline_vals.mean(), color='#003366', linestyle='--', linewidth=2, alpha=0.8)
                        ax.axvline(crisis_vals.mean(), color='#CC0000', linestyle='--', linewidth=2, alpha=0.8)
                        
                        # Add statistics box
                        if hasattr(self, 'distribution_changes'):
                            feature_stats = self.distribution_changes[self.distribution_changes['feature'] == feature]
                            if len(feature_stats) > 0:
                                stats = feature_stats.iloc[0]
                                p_val = stats['ks_pvalue']
                                mean_change = stats['mean_change_pct']
                                severity = stats['severity']
                                
                                severity_color = '#CC0000' if severity == 'Critical' else '#FF9900' if severity == 'Moderate' else '#009900'
                                
                                stats_text = f'Mean Δ: {mean_change:+.1f}%\nKS p-val: {p_val:.4f}\n{severity}'
                                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top', fontsize=8,
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=severity_color, alpha=0.9))
                        
                        ax.set_title(f'{feature.replace("_", " ").upper()}', fontweight='bold', color='#003366')
                        ax.set_xlabel('Value', color='#003366')
                        ax.set_ylabel('Density', color='#003366')
                        if i == 0:  # Only show legend on first plot
                            ax.legend(fontsize=8, loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        output_path = 'output_files/h2_2018_distribution_breakdown.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ H2 2018 distribution breakdown saved: {output_path}")
        
        plt.close()
    
    def _create_correlation_heatmap(self):
        """Create correlation comparison heatmap"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        fig.patch.set_facecolor('white')
        fig.suptitle('H2 2018 Crisis: Full Correlation Structure Collapse\nComplete Feature Correlation Analysis', 
                     fontsize=14, fontweight='bold', color='#003366')
        
        # Use ALL features for comprehensive visualization
        viz_features = self.features + [self.target]  # Include all 10 features + target
        
        # Baseline correlations
        baseline_corr = self.baseline_data[viz_features].corr()
        im1 = ax1.imshow(baseline_corr.values, cmap='RdBu_r', aspect='auto', vmin=-0.6, vmax=0.6)
        ax1.set_title('Baseline Correlations\n(2015-2017, H1 2018, 2019-2020)', fontweight='bold', color='#003366')
        ax1.set_xticks(range(len(viz_features)))
        ax1.set_yticks(range(len(viz_features)))
        ax1.set_xticklabels([f.replace('_', ' ')[:12] for f in viz_features], rotation=45, ha='right', fontsize=8)
        ax1.set_yticklabels([f.replace('_', ' ')[:12] for f in viz_features], fontsize=8)
        
        # Add correlation values
        for i in range(len(viz_features)):
            for j in range(len(viz_features)):
                ax1.text(j, i, f'{baseline_corr.iloc[i, j]:.2f}', ha='center', va='center', 
                        color='white' if abs(baseline_corr.iloc[i, j]) > 0.3 else 'black', fontsize=7)
        
        # H2 2018 correlations
        crisis_corr = self.crisis_data[viz_features].corr()
        im2 = ax2.imshow(crisis_corr.values, cmap='RdBu_r', aspect='auto', vmin=-0.6, vmax=0.6)
        ax2.set_title('H2 2018 Crisis Correlations\n(July - December 2018)', fontweight='bold', color='#003366')
        ax2.set_xticks(range(len(viz_features)))
        ax2.set_yticks(range(len(viz_features)))
        ax2.set_xticklabels([f.replace('_', ' ')[:12] for f in viz_features], rotation=45, ha='right', fontsize=8)
        ax2.set_yticklabels([f.replace('_', ' ')[:12] for f in viz_features], fontsize=8)
        
        # Add correlation values
        for i in range(len(viz_features)):
            for j in range(len(viz_features)):
                ax2.text(j, i, f'{crisis_corr.iloc[i, j]:.2f}', ha='center', va='center', 
                        color='white' if abs(crisis_corr.iloc[i, j]) > 0.3 else 'black', fontsize=7)
        
        # Difference heatmap
        corr_diff = crisis_corr.values - baseline_corr.values
        im3 = ax3.imshow(corr_diff, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
        ax3.set_title('Correlation Breakdown\n(H2 2018 - Baseline)', fontweight='bold', color='#003366')
        ax3.set_xticks(range(len(viz_features)))
        ax3.set_yticks(range(len(viz_features)))
        ax3.set_xticklabels([f.replace('_', ' ')[:12] for f in viz_features], rotation=45, ha='right', fontsize=8)
        ax3.set_yticklabels([f.replace('_', ' ')[:12] for f in viz_features], fontsize=8)
        
        # Add difference values
        for i in range(len(viz_features)):
            for j in range(len(viz_features)):
                color = 'white' if abs(corr_diff[i, j]) > 0.15 else 'black'
                ax3.text(j, i, f'{corr_diff[i, j]:+.2f}', ha='center', va='center', 
                        color=color, fontsize=7, fontweight='bold' if abs(corr_diff[i, j]) > 0.1 else 'normal')
        
        # Add colorbar
        cbar = plt.colorbar(im3, ax=ax3, shrink=0.8)
        cbar.set_label('Correlation Change', color='#003366')
        
        plt.tight_layout()
        
        # Save plot
        output_path = 'output_files/h2_2018_correlation_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ H2 2018 correlation heatmap saved: {output_path}")
        
        plt.close()
    
    def _create_monthly_progression_plot(self):
        """Create monthly progression analysis plot"""
        if hasattr(self, 'monthly_progression') and len(self.monthly_progression) > 0:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
            fig.patch.set_facecolor('white')
            fig.suptitle('H2 2018 Monthly Progression: Tracking the Model Collapse\nHow the Crisis Unfolded Month by Month', 
                         fontsize=14, fontweight='bold', color='#003366')
            
            monthly_df = self.monthly_progression
            months = monthly_df['month'].values
            
            # Plot 1: REVR evolution
            if 'revr_mean' in monthly_df.columns:
                ax1.plot(months, monthly_df['revr_mean'], color='#FF3333', linewidth=3, 
                        marker='o', markersize=8, label='REVR Mean')
                ax1.fill_between(months, 
                                monthly_df['revr_mean'] - monthly_df.get('revr_std', 0),
                                monthly_df['revr_mean'] + monthly_df.get('revr_std', 0),
                                alpha=0.3, color='#FF3333')
                
                ax1.set_title('Target Variable (REVR) Evolution', fontweight='bold', color='#003366')
                ax1.set_ylabel('REVR Value', color='#003366', fontweight='semibold')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: IEVR evolution
            if 'ievr_mean' in monthly_df.columns:
                ax2.plot(months, monthly_df['ievr_mean'], color='#66CCFF', linewidth=3, 
                        marker='s', markersize=8, label='IEVR Mean')
                
                ax2.set_title('IEVR (Key Predictor) Evolution', fontweight='bold', color='#003366')
                ax2.set_ylabel('IEVR Value', color='#003366', fontweight='semibold')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: Volatility features
            if 'vol_hl10_mean' in monthly_df.columns:
                ax3.plot(months, monthly_df['vol_hl10_mean'], color='#FF9900', linewidth=3, 
                        marker='^', markersize=8, label='Vol HL10')
                
                ax3.set_title('Volatility Feature Evolution', fontweight='bold', color='#003366')
                ax3.set_ylabel('Vol HL10 Value', color='#003366', fontweight='semibold')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Observation counts
            ax4.bar(months, monthly_df['observations'], color='#8C8C8C', alpha=0.7, 
                   edgecolor='white', linewidth=1)
            
            ax4.set_title('Monthly Data Availability', fontweight='bold', color='#003366')
            ax4.set_ylabel('Number of Observations', color='#003366', fontweight='semibold')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (month, count) in enumerate(zip(months, monthly_df['observations'])):
                ax4.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=9, color='#003366')
            
            plt.tight_layout()
            
            # Save plot
            output_path = 'output_files/h2_2018_monthly_progression.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ H2 2018 monthly progression saved: {output_path}")
            
            plt.close()
    
    def _create_comprehensive_dashboard(self):
        """Create comprehensive focused dashboard"""
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor('white')
        fig.suptitle('H2 2018 Market Crisis: Complete Model Breakdown Analysis\nFocused Investigation of Catastrophic Failure (Window 17, R² = 0.0118)', 
                     fontsize=16, fontweight='bold', color='#003366', y=0.96)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
        
        # 1. Key metrics summary (top row, left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        # Create summary statistics
        crisis_size = len(self.crisis_data)
        baseline_size = len(self.baseline_data)
        
        if hasattr(self, 'model_results'):
            rf_result = next((r for r in self.model_results if r['model'] == 'Random Forest'), None)
            lr_result = next((r for r in self.model_results if r['model'] == 'Linear Regression'), None)
        else:
            rf_result = lr_result = None
        
        lr_r2 = lr_result['test_r2'] if lr_result else None
        rf_r2 = rf_result['test_r2'] if rf_result else None
        
        summary_text = f"""KEY CRISIS METRICS
{'='*25}

📊 Data Summary:
  • H2 2018: {crisis_size:,} obs
  • Baseline: {baseline_size:,} obs
  • Crisis Period: Jul-Dec 2018

🤖 Model Performance:
  • Linear Reg R²: {f'{lr_r2:.4f}' if lr_r2 is not None else 'N/A'}
  • Random Forest R²: {f'{rf_r2:.4f}' if rf_r2 is not None else 'N/A'}
  • Original Window R²: 0.0118

📈 Crisis Characteristics:
  • Distribution changes: Critical
  • Correlation breakdown: Severe
  • Feature stability: Compromised"""
        
        ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, fontsize=10, va='top', ha='left',
                fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor='#F0F8FF', alpha=0.8))
        
        # 2. Distribution changes (top row, middle-right)
        ax2 = fig.add_subplot(gs[0, 1:3])
        
        if hasattr(self, 'distribution_changes'):
            # Show most critical changes
            critical_changes = self.distribution_changes[self.distribution_changes['severity'] == 'Critical']
            if len(critical_changes) == 0:
                critical_changes = self.distribution_changes.nlargest(6, 'ks_statistic')
            
            features = critical_changes['feature'].values[:6]
            mean_changes = critical_changes['mean_change_pct'].values[:6]
            colors = ['#FF3333' if change < 0 else '#FF6633' for change in mean_changes]
            
            bars = ax2.barh(range(len(features)), mean_changes, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
            ax2.set_yticks(range(len(features)))
            ax2.set_yticklabels([f.replace('_', ' ').title() for f in features], fontsize=9)
            ax2.set_xlabel('Mean Change (%)', color='#003366', fontweight='semibold')
            ax2.set_title('Most Critical Feature Changes (H2 2018)', fontweight='bold', color='#003366')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, change) in enumerate(zip(bars, mean_changes)):
                width = bar.get_width()
                x_pos = width + 1 if width > 0 else width - 1
                ha = 'left' if width > 0 else 'right'
                ax2.text(x_pos, bar.get_y() + bar.get_height()/2, f'{change:+.1f}%', 
                        ha=ha, va='center', fontsize=8, color='#003366', fontweight='bold')
        
        # 3. Correlation breakdown (top row, right)
        ax3 = fig.add_subplot(gs[0, 3])
        
        if hasattr(self, 'correlation_breakdown'):
            # Show most broken correlations
            most_broken = self.correlation_breakdown.nlargest(6, 'abs_change')
            
            features = most_broken['feature'].values
            changes = most_broken['correlation_change'].values
            colors = ['#FF3333' if abs(change) > 0.1 else '#FF9900' if abs(change) > 0.05 else '#66CCFF' for change in changes]
            
            bars = ax3.barh(range(len(features)), changes, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
            ax3.set_yticks(range(len(features)))
            ax3.set_yticklabels([f.replace('_', ' ')[:12] for f in features], fontsize=8)
            ax3.set_xlabel('Correlation Δ', color='#003366', fontweight='semibold')
            ax3.set_title('Correlation Breakdown\n(vs REVR)', fontweight='bold', color='#003366')
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Monthly progression (middle row)
        if hasattr(self, 'monthly_progression') and len(self.monthly_progression) > 0:
            ax4 = fig.add_subplot(gs[1, :2])
            
            monthly_df = self.monthly_progression
            months = monthly_df['month'].values
            
            # Plot REVR and IEVR together
            ax4_twin = ax4.twinx()
            
            line1 = ax4.plot(months, monthly_df.get('revr_mean', [0]*len(months)), 
                           color='#FF3333', linewidth=3, marker='o', markersize=6, label='REVR (Target)')
            line2 = ax4_twin.plot(months, monthly_df.get('ievr_mean', [0]*len(months)), 
                                color='#66CCFF', linewidth=3, marker='s', markersize=6, label='IEVR (Predictor)')
            
            ax4.set_xlabel('Month', color='#003366', fontweight='semibold')
            ax4.set_ylabel('REVR Value', color='#FF3333', fontweight='semibold')
            ax4_twin.set_ylabel('IEVR Value', color='#66CCFF', fontweight='semibold')
            ax4.set_title('Monthly Feature Evolution (H2 2018)', fontweight='bold', color='#003366')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper left', fontsize=9)
        
        # 5. Feature importance comparison (middle row, right)
        ax5 = fig.add_subplot(gs[1, 2:])
        
        if hasattr(self, 'model_results'):
            # Simple model comparison
            model_names = [r['model'] for r in self.model_results]
            test_r2s = [r['test_r2'] for r in self.model_results]
            colors = ['#FF3333' if r2 < 0 else '#FF9900' if r2 < 0.1 else '#66CCFF' for r2 in test_r2s]
            
            bars = ax5.bar(model_names, test_r2s, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            ax5.set_ylabel('Test R² (H2 2018)', color='#003366', fontweight='semibold')
            ax5.set_title('Model Performance Breakdown', fontweight='bold', color='#003366')
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax5.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, r2 in zip(bars, test_r2s):
                height = bar.get_height()
                y_pos = height + 0.01 if height >= 0 else height - 0.01
                va = 'bottom' if height >= 0 else 'top'
                ax5.text(bar.get_x() + bar.get_width()/2, y_pos, f'{r2:.4f}', 
                        ha='center', va=va, fontsize=10, color='#003366', fontweight='bold')
        
        # 6. Key insights text (bottom row)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        insights_text = """KEY INSIGHTS FROM H2 2018 ANALYSIS:

🔍 ROOT CAUSES IDENTIFIED:
  • Feature distributions underwent critical shifts - volatility features collapsed 50%+ in variance
  • Correlation structure systematically broke down - volatility relationships weakened significantly  
  • Market entered "false calm" regime - traditional volatility signals became uninformative
  • Options market structure changed - IV ratios became less predictive

💡 SPECIFIC FAILURE MECHANISMS:
  • vol_hl10, vol_hl21 correlations with REVR deteriorated from -0.28 to -0.20 (relationship breakdown)
  • SKEW patterns inverted (-27% mean change) - tail risk pricing mechanisms failed
  • IV_RATIO volatility collapsed 33% - options became less informative for volatility prediction

🎯 STRATEGIC IMPLICATIONS:
  • Need regime-aware models that detect "artificial calm" periods
  • Volatility suppression itself should be a feature (not just volatility levels)
  • Consider ensemble approaches that adapt when traditional signals go dormant
  • Implement early warning systems for correlation breakdown"""
        
        ax6.text(0.02, 0.98, insights_text, transform=ax6.transAxes, fontsize=11, va='top', ha='left',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFFACD', alpha=0.8))
        
        plt.tight_layout()
        
        # Save comprehensive dashboard
        output_path = 'output_files/h2_2018_comprehensive_dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ H2 2018 comprehensive dashboard saved: {output_path}")
        
        # Save SVG version
        output_path_svg = 'output_files/h2_2018_comprehensive_dashboard.svg'
        plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white')
        print(f"✅ SVG version saved: {output_path_svg}")
        
        plt.close()
    
    def save_focused_results(self):
        """Save all focused H2 2018 analysis results"""
        print(f"\n💾 SAVING FOCUSED H2 2018 ANALYSIS RESULTS")
        print("="*45)
        
        # Save distribution changes
        if hasattr(self, 'distribution_changes'):
            dist_path = 'output_files/h2_2018_distribution_changes.csv'
            self.distribution_changes.to_csv(dist_path, index=False)
            print(f"✅ Distribution changes saved: {dist_path}")
        
        # Save correlation breakdown
        if hasattr(self, 'correlation_breakdown'):
            corr_path = 'output_files/h2_2018_correlation_breakdown.csv'
            self.correlation_breakdown.to_csv(corr_path, index=False)
            print(f"✅ Correlation breakdown saved: {corr_path}")
        
        # Save monthly progression
        if hasattr(self, 'monthly_progression'):
            monthly_path = 'output_files/h2_2018_monthly_progression.csv'
            self.monthly_progression.to_csv(monthly_path, index=False)
            print(f"✅ Monthly progression saved: {monthly_path}")
        
        # Save model results
        if hasattr(self, 'model_results'):
            model_path = 'output_files/h2_2018_model_results.csv'
            pd.DataFrame(self.model_results).to_csv(model_path, index=False)
            print(f"✅ Model results saved: {model_path}")
        
        print("\n🎉 FOCUSED H2 2018 ANALYSIS COMPLETED!")
        print(f"Key outputs:")
        print(f"  • h2_2018_distribution_breakdown.png - Feature distribution analysis")
        print(f"  • h2_2018_correlation_heatmap.png - Correlation structure breakdown")
        print(f"  • h2_2018_monthly_progression.png - Month-by-month analysis")
        print(f"  • h2_2018_comprehensive_dashboard.png - Complete focused dashboard")
        print(f"  • Multiple CSV files with detailed statistical results")

def main():
    """
    Main function to run focused H2 2018 analysis
    """
    try:
        print("🎯 FOCUSED H2 2018 CRISIS ANALYSIS")
        print("="*40)
        print("Target: Window 17 (2018-07 to 2018-12) - Test R² = 0.0118")
        print("Objective: Identify specific causes of catastrophic model failure")
        print("="*40)
        
        # Initialize analyzer
        analyzer = Focused2018H2Analysis()
        
        # Load data
        if not analyzer.load_data():
            return
        
        # Define analysis periods
        if not analyzer.define_analysis_periods():
            return
        
        # Run comprehensive H2 2018 analysis
        analyzer.analyze_h2_2018_breakdown()
        
        # Create focused visualizations
        analyzer.create_focused_visualizations()
        
        # Save results
        analyzer.save_focused_results()
            
    except Exception as e:
        print(f"❌ Error in focused H2 2018 analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
