#!/usr/bin/env python3
"""
Beta Sensitivity Analysis for Multifactor Linear Regression
Analyze the stability and sensitivity of regression coefficients over rolling windows
Features: IEVR + normative_iv_rv_ratio + SKEW + KURT + IV_RATIO + SMIRK + vol_hl7 + vol_hl10 + vol_hl21 + z_score_momentum
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BetaSensitivityAnalysis:
    """
    Analyze beta sensitivity and stability for multifactor linear regression
    """
    
    def __init__(self):
        self.df = None
        self.features = [
            'ievr', 'normative_iv_rv_ratio', 'SKEW', 'KURT', 'IV_RATIO', 
            'SMIRK', 'vol_hl7', 'vol_hl10', 'vol_hl21', 'z_score_momentum'
        ]
        self.target = 'revr'
        self.beta_results = []
        self.windows = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("📊 LOADING DATASET FOR BETA SENSITIVITY ANALYSIS")
        print("="*60)
        
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
        print(f"\n🔄 CREATING ROLLING WINDOWS FOR BETA ANALYSIS")
        print("="*50)
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
                
            # Skip validation for beta analysis
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
        
        # Show first few windows
        for i, window in enumerate(windows[:3]):
            print(f"  Window {i+1}: Train {window['train_start']}-{window['train_end']}, "
                  f"Test {window['test_start']}-{window['test_end']}")
        
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
    
    def analyze_beta_sensitivity(self):
        """Run comprehensive beta sensitivity analysis across all rolling windows"""
        print(f"\n📈 RUNNING BETA SENSITIVITY ANALYSIS")
        print("="*50)
        
        # Create rolling windows
        if self.windows is None:
            self.create_rolling_windows()
        
        beta_results = []
        
        for window in self.windows:
            print(f"\n📊 Analyzing Window {window['window_id']} (Train: {window['train_start']} to {window['train_end']})...")
            
            # Get data for this window
            train_data, test_data = self.get_window_data(window)
            
            if len(train_data) < 100 or len(test_data) < 10:
                print(f"  ⚠️ Insufficient data - skipping window")
                continue
            
            # Prepare clean training data
            all_required_cols = self.features + [self.target]
            train_clean = train_data[all_required_cols].dropna()
            test_clean = test_data[all_required_cols].dropna()
            
            if len(train_clean) < 50:
                print(f"  ⚠️ Insufficient clean training data - skipping window")
                continue
            
            # Fit linear regression model
            X_train = train_clean[self.features]
            y_train = train_clean[self.target]
            X_test = test_clean[self.features]
            y_test = test_clean[self.target]
            
            # Standardize features for interpretability
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Fit model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Calculate performance metrics
            train_r2 = model.score(X_train_scaled, y_train)
            test_r2 = model.score(X_test_scaled, y_test)
            
            # Store beta coefficients and statistics
            window_result = {
                'window_id': window['window_id'],
                'train_start': str(window['train_start']),
                'train_end': str(window['train_end']),
                'test_start': str(window['test_start']),
                'test_end': str(window['test_end']),
                'test_year': int(str(window['test_start'])[:4]),
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_size': len(train_clean),
                'test_size': len(test_clean),
                'intercept': model.intercept_
            }
            
            # Add individual beta coefficients
            for i, feature in enumerate(self.features):
                window_result[f'beta_{feature}'] = model.coef_[i]
            
            # Calculate feature standard deviations for interpretation
            for i, feature in enumerate(self.features):
                window_result[f'feature_std_{feature}'] = X_train[feature].std()
            
            beta_results.append(window_result)
            
            # Print summary for this window
            print(f"  ✅ R² = {test_r2:.4f}, Samples = {len(train_clean)}")
            print(f"     Top 3 Betas: {sorted([(f, model.coef_[i]) for i, f in enumerate(self.features)], key=lambda x: abs(x[1]), reverse=True)[:3]}")
        
        self.beta_results = pd.DataFrame(beta_results)
        
        if len(self.beta_results) > 0:
            self.analyze_beta_statistics()
        
        return self.beta_results
    
    def analyze_beta_statistics(self):
        """Analyze beta coefficient statistics and stability"""
        print(f"\n📊 BETA STABILITY ANALYSIS")
        print("="*40)
        
        # Calculate statistics for each beta
        beta_stats = {}
        
        for feature in self.features:
            beta_col = f'beta_{feature}'
            if beta_col in self.beta_results.columns:
                betas = self.beta_results[beta_col].values
                
                beta_stats[feature] = {
                    'mean': np.mean(betas),
                    'std': np.std(betas),
                    'min': np.min(betas),
                    'max': np.max(betas),
                    'cv': np.std(betas) / np.abs(np.mean(betas)) if np.mean(betas) != 0 else np.inf,  # Coefficient of variation
                    'stability': 1 / (1 + np.std(betas))  # Higher = more stable
                }
        
        # Sort by stability (ascending coefficient of variation)
        sorted_features = sorted(beta_stats.keys(), key=lambda x: beta_stats[x]['cv'])
        
        print("BETA COEFFICIENT SUMMARY (sorted by stability):")
        print("-" * 80)
        print(f"{'Feature':20s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'CV':>8s} {'Stability':>10s}")
        print("-" * 80)
        
        for feature in sorted_features:
            stats = beta_stats[feature]
            cv_str = f"{stats['cv']:.3f}" if stats['cv'] != np.inf else "∞"
            print(f"{feature:20s} {stats['mean']:8.4f} {stats['std']:8.4f} {stats['min']:8.4f} {stats['max']:8.4f} {cv_str:>8s} {stats['stability']:10.4f}")
        
        # Identify most stable and volatile betas
        print(f"\nMOST STABLE BETAS (low coefficient of variation):")
        for feature in sorted_features[:3]:
            cv = beta_stats[feature]['cv']
            cv_str = f"{cv:.3f}" if cv != np.inf else "∞"
            print(f"  🟢 {feature}: CV = {cv_str}")
        
        print(f"\nMOST VOLATILE BETAS (high coefficient of variation):")
        for feature in sorted_features[-3:]:
            cv = beta_stats[feature]['cv']
            cv_str = f"{cv:.3f}" if cv != np.inf else "∞"
            print(f"  🔴 {feature}: CV = {cv_str}")
        
        # Analyze correlations between betas
        print(f"\nBETA CORRELATION ANALYSIS:")
        print("-" * 30)
        
        # Calculate correlation matrix for betas
        beta_columns = [f'beta_{feature}' for feature in self.features]
        available_beta_cols = [col for col in beta_columns if col in self.beta_results.columns]
        
        if len(available_beta_cols) > 1:
            beta_corr = self.beta_results[available_beta_cols].corr()
            
            # Find highest correlations
            corr_pairs = []
            for i in range(len(available_beta_cols)):
                for j in range(i+1, len(available_beta_cols)):
                    feat_i = available_beta_cols[i].replace('beta_', '')
                    feat_j = available_beta_cols[j].replace('beta_', '')
                    corr_val = beta_corr.iloc[i, j]
                    corr_pairs.append((feat_i, feat_j, corr_val))
            
            # Sort by absolute correlation
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            print("Top 5 beta correlations:")
            for feat_i, feat_j, corr in corr_pairs[:5]:
                print(f"  {feat_i} ↔ {feat_j}: {corr:+.3f}")
        
        return beta_stats
    
    def create_beta_visualizations(self):
        """Create comprehensive visualizations for beta sensitivity analysis"""
        print(f"\n📊 CREATING BETA SENSITIVITY VISUALIZATIONS")
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
        
        # Create main beta sensitivity plot
        self._create_beta_time_series_plot()
        
        # Create beta distribution and correlation plots
        self._create_beta_distribution_plot()
        
        # Create beta stability heatmap
        self._create_beta_stability_heatmap()
        
        # Create focused presentation plot
        self._create_focused_beta_plot()
    
    def _create_beta_time_series_plot(self):
        """Create time series plot of beta coefficients"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('white')
        fig.suptitle('Multifactor Linear Regression: Beta Sensitivity Analysis\nRolling Window Coefficient Evolution', 
                     fontsize=14, fontweight='bold', color='#003366')
        
        # Get test start dates for x-axis
        if 'test_start' in self.beta_results.columns:
            x_axis = pd.to_datetime(self.beta_results['test_start'], format='%Y-%m')
            x_label = 'Test Period Start'
        else:
            x_axis = self.beta_results['window_id']
            x_label = 'Window Number'
        
        # Define colors for different features (BlackRock palette)
        colors = ['#003366', '#66CCFF', '#8C8C8C', '#FF6633', '#004d99', '#99d6ff', '#b3b3b3', '#ff8c66', '#006bb3', '#cce6ff']
        
        # Plot 1: Key Economic Features
        ax1 = axes[0, 0]
        key_features = ['ievr', 'normative_iv_rv_ratio', 'z_score_momentum']
        for i, feature in enumerate(key_features):
            beta_col = f'beta_{feature}'
            if beta_col in self.beta_results.columns:
                ax1.plot(x_axis, self.beta_results[beta_col], 
                        label=feature.replace('_', ' ').title(), 
                        color=colors[i % len(colors)], linewidth=2, marker='o', markersize=4)
        
        ax1.set_title('Key Economic Features', fontweight='bold', color='#003366')
        ax1.set_ylabel('Beta Coefficient', color='#003366', fontweight='semibold')
        ax1.legend(fontsize=8)
        ax1.axhline(y=0, color='#8C8C8C', linestyle='-', alpha=0.5)
        
        # Plot 2: Option Features
        ax2 = axes[0, 1]
        option_features = ['SKEW', 'KURT', 'IV_RATIO', 'SMIRK']
        for i, feature in enumerate(option_features):
            beta_col = f'beta_{feature}'
            if beta_col in self.beta_results.columns:
                ax2.plot(x_axis, self.beta_results[beta_col], 
                        label=feature, 
                        color=colors[(i+3) % len(colors)], linewidth=2, marker='s', markersize=4)
        
        ax2.set_title('Option Features', fontweight='bold', color='#003366')
        ax2.set_ylabel('Beta Coefficient', color='#003366', fontweight='semibold')
        ax2.legend(fontsize=8)
        ax2.axhline(y=0, color='#8C8C8C', linestyle='-', alpha=0.5)
        
        # Plot 3: Volatility Features
        ax3 = axes[1, 0]
        vol_features = ['vol_hl7', 'vol_hl10', 'vol_hl21']
        for i, feature in enumerate(vol_features):
            beta_col = f'beta_{feature}'
            if beta_col in self.beta_results.columns:
                ax3.plot(x_axis, self.beta_results[beta_col], 
                        label=feature.replace('vol_hl', 'Vol HL'), 
                        color=colors[(i+7) % len(colors)], linewidth=2, marker='^', markersize=4)
        
        ax3.set_title('Volatility Features', fontweight='bold', color='#003366')
        ax3.set_xlabel(x_label, color='#003366', fontweight='semibold')
        ax3.set_ylabel('Beta Coefficient', color='#003366', fontweight='semibold')
        ax3.legend(fontsize=8)
        ax3.axhline(y=0, color='#8C8C8C', linestyle='-', alpha=0.5)
        
        # Plot 4: Model Performance
        ax4 = axes[1, 1]
        ax4.plot(x_axis, self.beta_results['test_r2'], 
                color='#003366', linewidth=3, marker='o', markersize=6, 
                label='Test R²', markeredgecolor='white', markeredgewidth=0.5)
        
        ax4.set_title('Model Performance', fontweight='bold', color='#003366')
        ax4.set_xlabel(x_label, color='#003366', fontweight='semibold')
        ax4.set_ylabel('Test R²', color='#003366', fontweight='semibold')
        ax4.axhline(y=0, color='#8C8C8C', linestyle='-', alpha=0.5)
        
        # Highlight 2018 market stress
        if 'test_start' in self.beta_results.columns:
            for ax in axes.flat:
                ax.axvspan(pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'), 
                          alpha=0.15, color='#FF6633', label='2018 Market Stress')
        
        plt.tight_layout()
        
        # Save plot
        output_path = 'output_files/beta_sensitivity_time_series.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Beta time series plot saved: {output_path}")
        
        plt.close()
    
    def _create_beta_distribution_plot(self):
        """Create distribution plots for beta coefficients"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.patch.set_facecolor('white')
        fig.suptitle('Beta Coefficient Distributions Across Rolling Windows', 
                     fontsize=14, fontweight='bold', color='#003366')
        
        # Select most important features for distribution analysis
        key_features = ['ievr', 'normative_iv_rv_ratio', 'z_score_momentum', 'SKEW', 'vol_hl21']
        
        for i, feature in enumerate(key_features):
            if i >= 6:  # Only plot first 6
                break
                
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            beta_col = f'beta_{feature}'
            if beta_col in self.beta_results.columns:
                betas = self.beta_results[beta_col].values
                
                # Create histogram
                ax.hist(betas, bins=15, alpha=0.7, color='#66CCFF', edgecolor='#003366', linewidth=1)
                
                # Add statistics
                mean_beta = np.mean(betas)
                std_beta = np.std(betas)
                cv = std_beta / abs(mean_beta) if mean_beta != 0 else np.inf
                
                ax.axvline(mean_beta, color='#FF6633', linestyle='--', linewidth=2, label=f'Mean: {mean_beta:.4f}')
                ax.axvline(0, color='#8C8C8C', linestyle='-', alpha=0.5)
                
                ax.set_title(f'{feature.replace("_", " ").title()}\n(CV: {cv:.3f})', 
                           fontweight='bold', color='#003366')
                ax.set_xlabel('Beta Coefficient', color='#003366')
                ax.set_ylabel('Frequency', color='#003366')
                ax.legend(fontsize=8)
        
        # Remove empty subplots
        for i in range(len(key_features), 6):
            row = i // 3
            col = i % 3
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        
        # Save plot
        output_path = 'output_files/beta_sensitivity_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Beta distribution plot saved: {output_path}")
        
        plt.close()
    
    def _create_beta_stability_heatmap(self):
        """Create heatmap showing beta stability across time periods"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor('white')
        fig.suptitle('Beta Coefficient Stability Analysis', 
                     fontsize=14, fontweight='bold', color='#003366')
        
        # Prepare data for heatmap - reshape betas by year
        beta_by_year = []
        years = []
        
        for year in sorted(self.beta_results['test_year'].unique()):
            year_data = self.beta_results[self.beta_results['test_year'] == year]
            if len(year_data) > 0:
                # Average betas for this year (in case multiple windows)
                year_betas = []
                for feature in self.features:
                    beta_col = f'beta_{feature}'
                    if beta_col in year_data.columns:
                        year_betas.append(year_data[beta_col].mean())
                    else:
                        year_betas.append(0)
                
                beta_by_year.append(year_betas)
                years.append(year)
        
        # Create heatmap data
        beta_matrix = np.array(beta_by_year).T
        feature_labels = [f.replace('_', ' ').title() for f in self.features]
        
        # Plot 1: Beta magnitude heatmap
        im1 = ax1.imshow(beta_matrix, cmap='RdBu_r', aspect='auto')
        ax1.set_title('Beta Coefficients by Year', fontweight='bold', color='#003366')
        ax1.set_xlabel('Year', color='#003366', fontweight='semibold')
        ax1.set_ylabel('Features', color='#003366', fontweight='semibold')
        ax1.set_xticks(range(len(years)))
        ax1.set_xticklabels(years, rotation=45)
        ax1.set_yticks(range(len(feature_labels)))
        ax1.set_yticklabels(feature_labels)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Beta Coefficient', color='#003366')
        
        # Plot 2: Beta volatility (coefficient of variation)
        cv_data = []
        for feature in self.features:
            beta_col = f'beta_{feature}'
            if beta_col in self.beta_results.columns:
                betas = self.beta_results[beta_col].values
                cv = np.std(betas) / abs(np.mean(betas)) if np.mean(betas) != 0 else 0
                cv_data.append(cv)
            else:
                cv_data.append(0)
        
        # Create bar plot for coefficient of variation
        bars = ax2.barh(range(len(feature_labels)), cv_data, color='#66CCFF', alpha=0.8, edgecolor='#003366')
        ax2.set_title('Beta Stability (Lower CV = More Stable)', fontweight='bold', color='#003366')
        ax2.set_xlabel('Coefficient of Variation', color='#003366', fontweight='semibold')
        ax2.set_yticks(range(len(feature_labels)))
        ax2.set_yticklabels(feature_labels)
        
        # Add value labels on bars
        for i, (bar, cv) in enumerate(zip(bars, cv_data)):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{cv:.3f}', ha='left', va='center', fontsize=8, color='#003366')
        
        plt.tight_layout()
        
        # Save plot
        output_path = 'output_files/beta_sensitivity_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Beta stability heatmap saved: {output_path}")
        
        plt.close()
    
    def _create_focused_beta_plot(self):
        """Create focused plot for presentations"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor('white')
        fig.suptitle('Multifactor REVR Model: Beta Coefficient Sensitivity\nRolling Window Analysis (2010-2023)', 
                     fontsize=16, fontweight='bold', color='#003366', y=0.95)
        
        # Get test start dates for x-axis
        if 'test_start' in self.beta_results.columns:
            x_axis = pd.to_datetime(self.beta_results['test_start'], format='%Y-%m')
            x_label = 'Test Period Start'
        else:
            x_axis = self.beta_results['window_id']
            x_label = 'Window Number'
        
        # Plot 1: Key feature betas over time
        key_features = ['ievr', 'normative_iv_rv_ratio', 'z_score_momentum']
        colors = ['#003366', '#66CCFF', '#FF6633']
        markers = ['o', 's', '^']
        
        for i, feature in enumerate(key_features):
            beta_col = f'beta_{feature}'
            if beta_col in self.beta_results.columns:
                ax1.plot(x_axis, self.beta_results[beta_col], 
                        label=feature.replace('_', ' ').replace('ievr', 'IEVR').title(), 
                        color=colors[i], linewidth=3, marker=markers[i], markersize=7,
                        markeredgecolor='white', markeredgewidth=1, alpha=0.9)
        
        ax1.set_title('Key Feature Coefficients Over Time', 
                     fontsize=14, fontweight='bold', color='#003366', pad=20)
        ax1.set_xlabel(x_label, fontsize=12, color='#003366', fontweight='semibold')
        ax1.set_ylabel('Beta Coefficient', fontsize=12, color='#003366', fontweight='semibold')
        ax1.legend(fontsize=11, loc='upper left', frameon=True, fancybox=True, 
                  shadow=True, framealpha=0.95)
        ax1.axhline(y=0, color='#8C8C8C', linestyle='-', alpha=0.7, linewidth=1.5)
        ax1.grid(True, alpha=0.3)
        
        # Highlight 2018 market stress
        if 'test_start' in self.beta_results.columns:
            ax1.axvspan(pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'), 
                       alpha=0.15, color='red', label='2018 Market Regime')
            ax1.text(pd.Timestamp('2018-06-15'), ax1.get_ylim()[1]*0.9, '2018\nMarket Stress', 
                    ha='center', va='top', fontsize=10, color='darkred', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.8))
        
        # Plot 2: Beta stability ranking
        cv_data = []
        feature_names = []
        
        for feature in self.features:
            beta_col = f'beta_{feature}'
            if beta_col in self.beta_results.columns:
                betas = self.beta_results[beta_col].values
                cv = np.std(betas) / abs(np.mean(betas)) if np.mean(betas) != 0 else 0
                cv_data.append(cv)
                feature_names.append(feature.replace('_', ' ').replace('ievr', 'IEVR').title())
        
        # Sort by stability (ascending CV)
        sorted_indices = np.argsort(cv_data)
        sorted_cv = [cv_data[i] for i in sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Color bars based on stability
        colors_bar = ['#003366' if cv < 0.5 else '#66CCFF' if cv < 1.0 else '#FF6633' for cv in sorted_cv]
        
        bars = ax2.barh(range(len(sorted_names)), sorted_cv, color=colors_bar, alpha=0.8, 
                       edgecolor='white', linewidth=1)
        ax2.set_title('Beta Stability Ranking\n(Lower = More Stable)', 
                     fontsize=14, fontweight='bold', color='#003366', pad=20)
        ax2.set_xlabel('Coefficient of Variation', fontsize=12, color='#003366', fontweight='semibold')
        ax2.set_yticks(range(len(sorted_names)))
        ax2.set_yticklabels(sorted_names, fontsize=10)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, cv) in enumerate(zip(bars, sorted_cv)):
            width = bar.get_width()
            ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{cv:.3f}', ha='left', va='center', fontsize=9, color='#003366', fontweight='bold')
        
        # Add stability categories
        ax2.axvline(x=0.5, color='#003366', linestyle='--', alpha=0.7)
        ax2.axvline(x=1.0, color='#66CCFF', linestyle='--', alpha=0.7)
        ax2.text(0.25, len(sorted_names)*0.9, 'Highly\nStable', ha='center', va='center', 
                fontsize=9, color='#003366', fontweight='bold', alpha=0.7)
        ax2.text(0.75, len(sorted_names)*0.9, 'Moderately\nStable', ha='center', va='center', 
                fontsize=9, color='#66CCFF', fontweight='bold', alpha=0.7)
        
        plt.tight_layout()
        
        # Save focused plot
        output_path = 'output_files/beta_sensitivity_focused.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Focused beta plot saved: {output_path}")
        
        # Save SVG version
        output_path_svg = 'output_files/beta_sensitivity_focused.svg'
        plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white')
        print(f"✅ SVG version saved: {output_path_svg}")
        
        plt.close()
    
    def save_results(self):
        """Save beta sensitivity results to CSV files"""
        print(f"\n💾 SAVING BETA SENSITIVITY RESULTS")
        print("="*40)
        
        if len(self.beta_results) > 0:
            # Save detailed results
            results_path = 'output_files/beta_sensitivity_results.csv'
            self.beta_results.to_csv(results_path, index=False)
            print(f"✅ Detailed results saved: {results_path}")
            
            # Create summary statistics
            summary_stats = {}
            
            for feature in self.features:
                beta_col = f'beta_{feature}'
                if beta_col in self.beta_results.columns:
                    betas = self.beta_results[beta_col].values
                    summary_stats[f'{feature}_mean_beta'] = np.mean(betas)
                    summary_stats[f'{feature}_std_beta'] = np.std(betas)
                    summary_stats[f'{feature}_min_beta'] = np.min(betas)
                    summary_stats[f'{feature}_max_beta'] = np.max(betas)
                    summary_stats[f'{feature}_cv'] = np.std(betas) / abs(np.mean(betas)) if np.mean(betas) != 0 else np.inf
            
            summary_df = pd.DataFrame([summary_stats])
            summary_path = 'output_files/beta_sensitivity_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"✅ Summary statistics saved: {summary_path}")
        
        print("\n🎉 BETA SENSITIVITY ANALYSIS COMPLETED!")
        print(f"Key outputs:")
        print(f"  • beta_sensitivity_time_series.png - Coefficient evolution")
        print(f"  • beta_sensitivity_distributions.png - Beta distributions")
        print(f"  • beta_sensitivity_heatmap.png - Stability analysis")
        print(f"  • beta_sensitivity_focused.png - Presentation-ready plot")
        print(f"  • beta_sensitivity_results.csv - Detailed coefficient data")
        print(f"  • beta_sensitivity_summary.csv - Summary statistics")

def main():
    """
    Main function to run beta sensitivity analysis
    """
    try:
        print("🚀 BETA SENSITIVITY ANALYSIS FOR MULTIFACTOR LINEAR REGRESSION")
        print("="*80)
        print("Features: IEVR + normative_iv_rv_ratio + SKEW + KURT + IV_RATIO + SMIRK + vol_hl7 + vol_hl10 + vol_hl21 + z_score_momentum")
        print("Methodology: 5-Year Training, 6-Month Testing Rolling Windows")
        print("="*80)
        
        # Initialize analyzer
        analyzer = BetaSensitivityAnalysis()
        
        # Load data
        if not analyzer.load_data():
            return
        
        # Run beta sensitivity analysis
        results = analyzer.analyze_beta_sensitivity()
        
        if results is not None and len(results) > 0:
            # Create visualizations
            analyzer.create_beta_visualizations()
            
            # Save results
            analyzer.save_results()
        else:
            print("❌ No results generated")
            
    except Exception as e:
        print(f"❌ Error in beta sensitivity analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

