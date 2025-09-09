#!/usr/bin/env python3
"""
Tree Models Feature Importance Analysis
Focused comparison between Random Forest and XGBoost feature importance
using rolling window methodology
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
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

class TreeModelsFeatureImportance:
    """
    Analyze feature importance for Random Forest and XGBoost using rolling windows
    """
    
    def __init__(self):
        self.df = None
        self.features = [
            'ievr', 'normative_iv_rv_ratio', 'IV_RATIO', 
            'SMIRK', 'vol_hl21', 'z_score_momentum', 'dispersion_pct_ibes'
        ]
        self.target = 'revr'
        self.importance_results = []
        self.windows = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("📊 LOADING DATASET FOR TREE MODELS FEATURE IMPORTANCE")
        print("="*60)
        
        try:
            # Use model_df.csv with proper path handling
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, '..', '..', 'data_files')
            data_file_path = os.path.join(data_dir, 'model_df.csv')
            
            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f"❌ Dataset file not found: {data_file_path}")
            
            self.df = pd.read_csv(data_file_path)
            print(f"✅ Loaded model dataset: {len(self.df):,} observations")
            
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
        print(f"\n🔄 CREATING ROLLING WINDOWS FOR TREE MODELS ANALYSIS")
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
                
            # Skip validation for feature importance analysis
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
    
    def analyze_tree_models_importance(self):
        """Run comprehensive feature importance analysis for Random Forest and XGBoost"""
        print(f"\n🌳 RUNNING TREE MODELS FEATURE IMPORTANCE ANALYSIS")
        print("="*55)
        
        # Create rolling windows
        if self.windows is None:
            self.create_rolling_windows()
        
        # Define tree-based algorithms only
        algorithms = {
            'RandomForest': {
                'model': RandomForestRegressor(
                    n_estimators=100, max_depth=10, min_samples_split=20,
                    min_samples_leaf=10, random_state=42, n_jobs=-1
                ),
                'color': '#66CCFF',  # BlackRock light blue
                'marker': 's',
                'description': 'Random Forest Regressor'
            },
            'XGBoost': {
                'model': XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    n_jobs=-1, verbosity=0
                ),
                'color': '#FF6633',  # BlackRock orange accent
                'marker': '^',
                'description': 'XGBoost Regressor'
            }
        }
        
        importance_results = []
        
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
            
            # Prepare features and target
            X_train = train_clean[self.features]
            y_train = train_clean[self.target]
            X_test = test_clean[self.features]
            y_test = test_clean[self.target]
            
            # Analyze each algorithm
            for algo_name, algo_config in algorithms.items():
                try:
                    # Create fresh model instance
                    if algo_name == 'RandomForest':
                        model = RandomForestRegressor(
                            n_estimators=100, max_depth=10, min_samples_split=20,
                            min_samples_leaf=10, random_state=42, n_jobs=-1
                        )
                    elif algo_name == 'XGBoost':
                        model = XGBRegressor(
                            n_estimators=100, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                            n_jobs=-1, verbosity=0
                        )
                    
                    # Fit model
                    model.fit(X_train, y_train)
                    
                    # Calculate performance
                    train_r2 = model.score(X_train, y_train)
                    test_r2 = model.score(X_test, y_test)
                    test_pred = model.predict(X_test)
                    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                    
                    # Extract feature importance
                    feature_importance = model.feature_importances_
                    
                    # Normalize importance to sum to 1
                    feature_importance_norm = feature_importance / feature_importance.sum()
                    
                    # Store results
                    window_result = {
                        'window_id': window['window_id'],
                        'algorithm': algo_name,
                        'train_start': str(window['train_start']),
                        'train_end': str(window['train_end']),
                        'test_start': str(window['test_start']),
                        'test_end': str(window['test_end']),
                        'test_year': int(str(window['test_start'])[:4]),
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'test_rmse': test_rmse,
                        'train_size': len(train_clean),
                        'test_size': len(test_clean)
                    }
                    
                    # Add individual feature importance
                    for i, feature in enumerate(self.features):
                        window_result[f'importance_{feature}'] = feature_importance_norm[i]
                        window_result[f'importance_raw_{feature}'] = feature_importance[i]
                    
                    importance_results.append(window_result)
                    
                    # Print top 3 features for this window/algorithm
                    top_features = sorted(zip(self.features, feature_importance_norm), 
                                        key=lambda x: x[1], reverse=True)[:3]
                    top_str = ', '.join([f"{feat}({imp:.3f})" for feat, imp in top_features])
                    print(f"  ✅ {algo_name:12s}: R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}, Top: {top_str}")
                    
                except Exception as e:
                    print(f"  ❌ {algo_name:12s}: Failed ({str(e)[:50]})")
        
        self.importance_results = pd.DataFrame(importance_results)
        
        if len(self.importance_results) > 0:
            self.analyze_importance_statistics()
        
        return self.importance_results
    
    def analyze_importance_statistics(self):
        """Analyze feature importance statistics across tree models"""
        print(f"\n📊 TREE MODELS FEATURE IMPORTANCE SUMMARY")
        print("="*50)
        
        # Calculate average importance by algorithm
        algorithms = ['RandomForest', 'XGBoost']
        
        print("AVERAGE FEATURE IMPORTANCE BY TREE MODEL:")
        print("-" * 70)
        print(f"{'Feature':25s}{'Random Forest':>15s}{'XGBoost':>12s}{'Difference':>12s}")
        print("-" * 70)
        
        feature_summary = {}
        
        for feature in self.features:
            importance_col = f'importance_{feature}'
            if importance_col in self.importance_results.columns:
                print(f"{feature:25s}", end="")
                
                rf_data = self.importance_results[self.importance_results['algorithm'] == 'RandomForest']
                xgb_data = self.importance_results[self.importance_results['algorithm'] == 'XGBoost']
                
                rf_avg = rf_data[importance_col].mean() if len(rf_data) > 0 else 0
                xgb_avg = xgb_data[importance_col].mean() if len(xgb_data) > 0 else 0
                difference = rf_avg - xgb_avg
                
                print(f"{rf_avg:15.4f}{xgb_avg:12.4f}{difference:+12.4f}")
                
                feature_summary[feature] = {
                    'rf_importance': rf_avg,
                    'xgb_importance': xgb_avg,
                    'difference': difference,
                    'avg_importance': (rf_avg + xgb_avg) / 2,
                    'agreement': 1 - abs(difference) / max(rf_avg, xgb_avg, 0.001)  # Agreement score
                }
        
        # Rank features by average importance across both models
        print(f"\nFEATURE RANKING BY AVERAGE TREE MODEL IMPORTANCE:")
        print("-" * 60)
        
        ranked_features = sorted(feature_summary.keys(), 
                               key=lambda x: feature_summary[x]['avg_importance'], 
                               reverse=True)
        
        for i, feature in enumerate(ranked_features):
            avg_imp = feature_summary[feature]['avg_importance']
            agreement = feature_summary[feature]['agreement']
            print(f"{i+1:2d}. {feature:25s}: {avg_imp:.4f} (Agreement: {agreement:.3f})")
        
        # Model agreement analysis (simplified)
        print(f"\nMODEL AGREEMENT ANALYSIS:")
        print("-" * 35)
        
        # Calculate correlation of feature importance
        rf_importance = []
        xgb_importance = []
        
        for feature in self.features:
            importance_col = f'importance_{feature}'
            
            rf_data = self.importance_results[self.importance_results['algorithm'] == 'RandomForest']
            xgb_data = self.importance_results[self.importance_results['algorithm'] == 'XGBoost']
            
            rf_avg = rf_data[importance_col].mean() if len(rf_data) > 0 else 0
            xgb_avg = xgb_data[importance_col].mean() if len(xgb_data) > 0 else 0
            
            rf_importance.append(rf_avg)
            xgb_importance.append(xgb_avg)
        
        correlation = np.corrcoef(rf_importance, xgb_importance)[0, 1]
        print(f"Random Forest ↔ XGBoost correlation: {correlation:.3f}")
        
        # Performance comparison
        print(f"\nMODEL PERFORMANCE COMPARISON:")
        print("-" * 35)
        
        rf_performance = self.importance_results[self.importance_results['algorithm'] == 'RandomForest']['test_r2']
        xgb_performance = self.importance_results[self.importance_results['algorithm'] == 'XGBoost']['test_r2']
        
        print(f"Random Forest: R² = {rf_performance.mean():.4f} (±{rf_performance.std():.4f})")
        print(f"XGBoost:       R² = {xgb_performance.mean():.4f} (±{xgb_performance.std():.4f})")
        
        if rf_performance.mean() > xgb_performance.mean():
            print(f"🏆 Random Forest outperforms by {rf_performance.mean() - xgb_performance.mean():.4f}")
        else:
            print(f"🏆 XGBoost outperforms by {xgb_performance.mean() - rf_performance.mean():.4f}")
        
        return feature_summary
    
    def create_tree_models_visualizations(self):
        """Create comprehensive visualizations for tree models feature importance"""
        print(f"\n📊 CREATING TREE MODELS VISUALIZATIONS")
        print("="*45)
        
        # Set BlackRock styling
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 10,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 10,
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'grid.color': '#E5E5E5'
        })
        
        # Create comprehensive tree models plots
        self._create_tree_comparison_plot()
        self._create_importance_evolution_plot()
        self._create_focused_tree_analysis()
        self._create_performance_vs_importance_plot()
    
    def _create_tree_comparison_plot(self):
        """Create side-by-side comparison of Random Forest vs XGBoost importance"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor('white')
        fig.suptitle('Machine Learning Models Feature Importance Analysis\nRandom Forest vs XGBoost Comparison', 
                     fontsize=14, fontweight='bold', color='#003366')
        
        # Calculate average importance for each algorithm
        rf_data = self.importance_results[self.importance_results['algorithm'] == 'RandomForest']
        xgb_data = self.importance_results[self.importance_results['algorithm'] == 'XGBoost']
        
        # Plot 1: Random Forest
        if len(rf_data) > 0:
            rf_importance = []
            feature_names = []
            
            for feature in self.features:
                importance_col = f'importance_{feature}'
                avg_imp = rf_data[importance_col].mean()
                rf_importance.append(avg_imp)
                feature_names.append(feature.replace('_', ' ').title())
            
            # Sort by importance
            sorted_data = sorted(zip(feature_names, rf_importance), key=lambda x: x[1], reverse=True)
            sorted_names, sorted_importance = zip(*sorted_data)
            
            bars1 = ax1.barh(range(len(sorted_names)), sorted_importance, 
                           color='#66CCFF', alpha=0.8, edgecolor='white', linewidth=1)
            
            ax1.set_yticks(range(len(sorted_names)))
            ax1.set_yticklabels(sorted_names, fontsize=9)
            ax1.set_xlabel('Average Feature Importance', color='#003366', fontweight='semibold')
            ax1.set_title('Random Forest', fontweight='bold', color='#003366')
            
            # Add value labels
            for i, (bar, imp) in enumerate(zip(bars1, sorted_importance)):
                width = bar.get_width()
                ax1.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                        f'{imp:.3f}', ha='left', va='center', fontsize=8, color='#003366')
        
        # Plot 2: XGBoost
        if len(xgb_data) > 0:
            xgb_importance = []
            feature_names = []
            
            for feature in self.features:
                importance_col = f'importance_{feature}'
                avg_imp = xgb_data[importance_col].mean()
                xgb_importance.append(avg_imp)
                feature_names.append(feature.replace('_', ' ').title())
            
            # Sort by importance
            sorted_data = sorted(zip(feature_names, xgb_importance), key=lambda x: x[1], reverse=True)
            sorted_names, sorted_importance = zip(*sorted_data)
            
            bars2 = ax2.barh(range(len(sorted_names)), sorted_importance, 
                           color='#FF6633', alpha=0.8, edgecolor='white', linewidth=1)
            
            ax2.set_yticks(range(len(sorted_names)))
            ax2.set_yticklabels(sorted_names, fontsize=9)
            ax2.set_xlabel('Average Feature Importance', color='#003366', fontweight='semibold')
            ax2.set_title('XGBoost', fontweight='bold', color='#003366')
            
            # Add value labels
            for i, (bar, imp) in enumerate(zip(bars2, sorted_importance)):
                width = bar.get_width()
                ax2.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                        f'{imp:.3f}', ha='left', va='center', fontsize=8, color='#003366')
        
        plt.tight_layout()
        
        # Save plot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output_files')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'tree_models_feature_importance_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Tree models comparison saved: {output_path}")
        
        plt.close()
    
    def _create_importance_evolution_plot(self):
        """Create time series showing feature importance evolution for both models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('white')
        fig.suptitle('Feature Importance Evolution Over Time\nRandom Forest vs XGBoost', 
                     fontsize=14, fontweight='bold', color='#003366')
        
        # Get test start dates for x-axis
        if 'test_start' in self.importance_results.columns:
            x_axis_rf = pd.to_datetime(
                self.importance_results[self.importance_results['algorithm'] == 'RandomForest']['test_start'], 
                format='%Y-%m'
            )
            x_axis_xgb = pd.to_datetime(
                self.importance_results[self.importance_results['algorithm'] == 'XGBoost']['test_start'], 
                format='%Y-%m'
            )
            x_label = 'Test Period Start'
        else:
            x_axis_rf = self.importance_results[self.importance_results['algorithm'] == 'RandomForest']['window_id']
            x_axis_xgb = self.importance_results[self.importance_results['algorithm'] == 'XGBoost']['window_id']
            x_label = 'Window Number'
        
        # Top features to track
        top_features = ['ievr', 'normative_iv_rv_ratio', 'vol_hl21', 'z_score_momentum']
        colors = ['#003366', '#66CCFF', '#8C8C8C', '#FF6633']
        
        # Plot Random Forest evolution
        ax1 = axes[0, 0]
        rf_data = self.importance_results[self.importance_results['algorithm'] == 'RandomForest']
        
        for i, feature in enumerate(top_features):
            importance_col = f'importance_{feature}'
            if importance_col in rf_data.columns:
                ax1.plot(x_axis_rf, rf_data[importance_col], 
                        label=feature.replace('_', ' ').title(), 
                        color=colors[i], linewidth=2, marker='o', markersize=4)
        
        ax1.set_title('Random Forest - Feature Importance Evolution', fontweight='bold', color='#003366')
        ax1.set_ylabel('Feature Importance', color='#003366', fontweight='semibold')
        ax1.legend(fontsize=9)
        ax1.set_ylim(0, None)
        
        # Plot XGBoost evolution
        ax2 = axes[0, 1]
        xgb_data = self.importance_results[self.importance_results['algorithm'] == 'XGBoost']
        
        for i, feature in enumerate(top_features):
            importance_col = f'importance_{feature}'
            if importance_col in xgb_data.columns:
                ax2.plot(x_axis_xgb, xgb_data[importance_col], 
                        label=feature.replace('_', ' ').title(), 
                        color=colors[i], linewidth=2, marker='^', markersize=4)
        
        ax2.set_title('XGBoost - Feature Importance Evolution', fontweight='bold', color='#003366')
        ax2.set_ylabel('Feature Importance', color='#003366', fontweight='semibold')
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, None)
        
        # Plot performance comparison
        ax3 = axes[1, 0]
        
        ax3.plot(x_axis_rf, rf_data['test_r2'], 
                label='Random Forest', color='#66CCFF', linewidth=3, marker='o', markersize=6)
        ax3.plot(x_axis_xgb, xgb_data['test_r2'], 
                label='XGBoost', color='#FF6633', linewidth=3, marker='^', markersize=6)
        
        ax3.set_title('Model Performance Comparison', fontweight='bold', color='#003366')
        ax3.set_xlabel(x_label, color='#003366', fontweight='semibold')
        ax3.set_ylabel('Test R²', color='#003366', fontweight='semibold')
        ax3.legend(fontsize=10)
        ax3.axhline(y=0, color='#8C8C8C', linestyle='-', alpha=0.5)
        
        # Plot top feature comparison (normative_iv_rv_ratio)
        ax4 = axes[1, 1]
        
        top_feature = 'normative_iv_rv_ratio'
        importance_col = f'importance_{top_feature}'
        
        ax4.plot(x_axis_rf, rf_data[importance_col], 
                label='Random Forest', color='#66CCFF', linewidth=3, marker='o', markersize=6)
        ax4.plot(x_axis_xgb, xgb_data[importance_col], 
                label='XGBoost', color='#FF6633', linewidth=3, marker='^', markersize=6)
        
        ax4.set_title(f'{top_feature.replace("_", " ").title()} Importance', fontweight='bold', color='#003366')
        ax4.set_xlabel(x_label, color='#003366', fontweight='semibold')
        ax4.set_ylabel('Feature Importance', color='#003366', fontweight='semibold')
        ax4.legend(fontsize=10)
        
        # Highlight 2018 market stress across all plots
        if 'test_start' in self.importance_results.columns:
            for ax in axes.flat:
                ax.axvspan(pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'), 
                          alpha=0.15, color='red')
        
        plt.tight_layout()
        
        # Save plot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output_files')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'tree_models_importance_evolution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Importance evolution plot saved: {output_path}")
        
        plt.close()
    
    def _create_focused_tree_analysis(self):
        """Create focused analysis for presentations"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('white')
        fig.suptitle('Tree Models Feature Importance: Comprehensive Analysis\nRandom Forest vs XGBoost for REVR Prediction', 
                     fontsize=16, fontweight='bold', color='#003366', y=0.96)
        
        # Plot 1: Feature importance ranking
        rf_data = self.importance_results[self.importance_results['algorithm'] == 'RandomForest']
        xgb_data = self.importance_results[self.importance_results['algorithm'] == 'XGBoost']
        
        # Calculate average importance
        feature_comparison = {}
        for feature in self.features:
            importance_col = f'importance_{feature}'
            rf_avg = rf_data[importance_col].mean() if len(rf_data) > 0 else 0
            xgb_avg = xgb_data[importance_col].mean() if len(xgb_data) > 0 else 0
            feature_comparison[feature] = {'rf': rf_avg, 'xgb': xgb_avg}
        
        # Sort by average importance
        sorted_features = sorted(feature_comparison.keys(), 
                               key=lambda x: (feature_comparison[x]['rf'] + feature_comparison[x]['xgb']) / 2, 
                               reverse=True)
        
        feature_labels = [f.replace('_', ' ').replace('ievr', 'IEVR').title() for f in sorted_features]
        rf_values = [feature_comparison[f]['rf'] for f in sorted_features]
        xgb_values = [feature_comparison[f]['xgb'] for f in sorted_features]
        
        x = np.arange(len(feature_labels))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, rf_values, width, label='Random Forest', 
                       color='#66CCFF', alpha=0.8, edgecolor='white', linewidth=1)
        bars2 = ax1.bar(x + width/2, xgb_values, width, label='XGBoost', 
                       color='#FF6633', alpha=0.8, edgecolor='white', linewidth=1)
        
        ax1.set_title('Feature Importance Ranking', fontsize=13, fontweight='bold', color='#003366')
        ax1.set_xlabel('Features', fontsize=11, color='#003366', fontweight='semibold')
        ax1.set_ylabel('Average Importance', fontsize=11, color='#003366', fontweight='semibold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=9)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Model agreement analysis
        agreement_scores = []
        differences = []
        
        for feature in sorted_features:
            rf_val = feature_comparison[feature]['rf']
            xgb_val = feature_comparison[feature]['xgb']
            diff = abs(rf_val - xgb_val)
            agreement = 1 - diff / max(rf_val, xgb_val, 0.001)
            agreement_scores.append(agreement)
            differences.append(rf_val - xgb_val)
        
        colors = ['green' if a > 0.8 else 'orange' if a > 0.6 else 'red' for a in agreement_scores]
        
        bars = ax2.barh(range(len(feature_labels)), agreement_scores, 
                       color=colors, alpha=0.7, edgecolor='white', linewidth=1)
        
        ax2.set_title('Model Agreement Score', fontsize=13, fontweight='bold', color='#003366')
        ax2.set_xlabel('Agreement Score (0-1)', fontsize=11, color='#003366', fontweight='semibold')
        ax2.set_yticks(range(len(feature_labels)))
        ax2.set_yticklabels(feature_labels, fontsize=9)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='High Agreement')
        ax2.axvline(x=0.6, color='orange', linestyle='--', alpha=0.7, label='Moderate Agreement')
        ax2.legend(fontsize=9)
        
        # Plot 3: Performance over time
        if 'test_start' in self.importance_results.columns:
            x_axis_rf = pd.to_datetime(rf_data['test_start'], format='%Y-%m')
            x_axis_xgb = pd.to_datetime(xgb_data['test_start'], format='%Y-%m')
            x_label = 'Test Period Start'
        else:
            x_axis_rf = rf_data['window_id']
            x_axis_xgb = xgb_data['window_id']
            x_label = 'Window Number'
        
        ax3.plot(x_axis_rf, rf_data['test_r2'], 
                label='Random Forest', color='#66CCFF', linewidth=3, 
                marker='o', markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        ax3.plot(x_axis_xgb, xgb_data['test_r2'], 
                label='XGBoost', color='#FF6633', linewidth=3, 
                marker='^', markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        
        ax3.set_title('Model Performance Over Time', fontsize=13, fontweight='bold', color='#003366')
        ax3.set_xlabel(x_label, fontsize=11, color='#003366', fontweight='semibold')
        ax3.set_ylabel('Test R²', fontsize=11, color='#003366', fontweight='semibold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='#8C8C8C', linestyle='-', alpha=0.7, linewidth=1)
        
        # Highlight 2018 market stress
        if 'test_start' in self.importance_results.columns:
            ax3.axvspan(pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'), 
                       alpha=0.15, color='red')
            ax3.text(pd.Timestamp('2018-06-15'), ax3.get_ylim()[1]*0.9, '2018\nMarket Stress', 
                    ha='center', va='top', fontsize=9, color='darkred', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.8))
        
        # Plot 4: Top feature importance distribution
        top_feature = 'normative_iv_rv_ratio'
        importance_col = f'importance_{top_feature}'
        
        # Create violin plot
        rf_importance_values = rf_data[importance_col].values
        xgb_importance_values = xgb_data[importance_col].values
        
        parts1 = ax4.violinplot([rf_importance_values], positions=[1], widths=0.6, 
                              showmeans=True, showmedians=True)
        parts2 = ax4.violinplot([xgb_importance_values], positions=[2], widths=0.6, 
                              showmeans=True, showmedians=True)
        
        # Color the violin plots
        for pc in parts1['bodies']:
            pc.set_facecolor('#66CCFF')
            pc.set_alpha(0.7)
        for pc in parts2['bodies']:
            pc.set_facecolor('#FF6633')
            pc.set_alpha(0.7)
        
        ax4.set_title(f'{top_feature.replace("_", " ").title()}\nImportance Distribution', 
                     fontsize=13, fontweight='bold', color='#003366')
        ax4.set_ylabel('Feature Importance', fontsize=11, color='#003366', fontweight='semibold')
        ax4.set_xticks([1, 2])
        ax4.set_xticklabels(['Random Forest', 'XGBoost'], fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save focused plot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output_files')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'tree_models_focused_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Focused tree analysis saved: {output_path}")
        
        # Save SVG version
        output_path_svg = os.path.join(output_dir, 'tree_models_focused_analysis.svg')
        plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white')
        print(f"✅ SVG version saved: {output_path_svg}")
        
        plt.close()
    
    def _create_performance_vs_importance_plot(self):
        """Create scatter plot showing relationship between feature importance and model performance"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor('white')
        fig.suptitle('Feature Importance vs Model Performance Analysis', 
                     fontsize=14, fontweight='bold', color='#003366')
        
        # Plot 1: IEVR importance vs performance
        rf_data = self.importance_results[self.importance_results['algorithm'] == 'RandomForest']
        xgb_data = self.importance_results[self.importance_results['algorithm'] == 'XGBoost']
        
        feature = 'ievr'
        importance_col = f'importance_{feature}'
        
        if len(rf_data) > 0:
            ax1.scatter(rf_data[importance_col], rf_data['test_r2'], 
                       color='#66CCFF', s=60, alpha=0.7, label='Random Forest', 
                       edgecolors='white', linewidth=1)
        
        if len(xgb_data) > 0:
            ax1.scatter(xgb_data[importance_col], xgb_data['test_r2'], 
                       color='#FF6633', s=60, alpha=0.7, label='XGBoost', 
                       marker='^', edgecolors='white', linewidth=1)
        
        ax1.set_xlabel(f'{feature.upper()} Importance', fontsize=11, color='#003366', fontweight='semibold')
        ax1.set_ylabel('Test R²', fontsize=11, color='#003366', fontweight='semibold')
        ax1.set_title(f'{feature.upper()} Importance vs Performance', fontweight='bold', color='#003366')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: normative_iv_rv_ratio importance vs performance
        feature = 'normative_iv_rv_ratio'
        importance_col = f'importance_{feature}'
        
        if len(rf_data) > 0:
            ax2.scatter(rf_data[importance_col], rf_data['test_r2'], 
                       color='#66CCFF', s=60, alpha=0.7, label='Random Forest', 
                       edgecolors='white', linewidth=1)
        
        if len(xgb_data) > 0:
            ax2.scatter(xgb_data[importance_col], xgb_data['test_r2'], 
                       color='#FF6633', s=60, alpha=0.7, label='XGBoost', 
                       marker='^', edgecolors='white', linewidth=1)
        
        ax2.set_xlabel('Normative IV/RV Ratio Importance', fontsize=11, color='#003366', fontweight='semibold')
        ax2.set_ylabel('Test R²', fontsize=11, color='#003366', fontweight='semibold')
        ax2.set_title('Normative IV/RV Ratio Importance vs Performance', fontweight='bold', color='#003366')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output_files')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'tree_models_performance_vs_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Performance vs importance plot saved: {output_path}")
        
        plt.close()
    
    def save_results(self):
        """Save tree models feature importance results to CSV files"""
        print(f"\n💾 SAVING TREE MODELS RESULTS")
        print("="*35)
        
        if len(self.importance_results) > 0:
            # Ensure output directory exists
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, 'output_files')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save detailed results
            results_path = os.path.join(output_dir, 'tree_models_feature_importance_results.csv')
            self.importance_results.to_csv(results_path, index=False)
            print(f"✅ Detailed results saved: {results_path}")
            
            # Create summary table
            summary_table = []
            
            for feature in self.features:
                importance_col = f'importance_{feature}'
                
                rf_data = self.importance_results[self.importance_results['algorithm'] == 'RandomForest']
                xgb_data = self.importance_results[self.importance_results['algorithm'] == 'XGBoost']
                
                rf_mean = rf_data[importance_col].mean() if len(rf_data) > 0 else 0
                rf_std = rf_data[importance_col].std() if len(rf_data) > 0 else 0
                xgb_mean = xgb_data[importance_col].mean() if len(xgb_data) > 0 else 0
                xgb_std = xgb_data[importance_col].std() if len(xgb_data) > 0 else 0
                
                summary_table.append({
                    'feature': feature,
                    'random_forest_mean': rf_mean,
                    'random_forest_std': rf_std,
                    'xgboost_mean': xgb_mean,
                    'xgboost_std': xgb_std,
                    'average_importance': (rf_mean + xgb_mean) / 2,
                    'difference': rf_mean - xgb_mean,
                    'agreement': 1 - abs(rf_mean - xgb_mean) / max(rf_mean, xgb_mean, 0.001)
                })
            
            summary_df = pd.DataFrame(summary_table)
            summary_df = summary_df.sort_values('average_importance', ascending=False)
            
            summary_path = os.path.join(output_dir, 'tree_models_feature_importance_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"✅ Summary table saved: {summary_path}")
        
        print("\n🎉 TREE MODELS FEATURE IMPORTANCE ANALYSIS COMPLETED!")
        print(f"Key outputs:")
        print(f"  • tree_models_feature_importance_comparison.png - Side-by-side comparison")
        print(f"  • tree_models_importance_evolution.png - Evolution over time")
        print(f"  • tree_models_focused_analysis.png - Comprehensive presentation plot")
        print(f"  • tree_models_performance_vs_importance.png - Performance correlation")
        print(f"  • tree_models_feature_importance_results.csv - Detailed importance data")
        print(f"  • tree_models_feature_importance_summary.csv - Summary statistics")

def main():
    """
    Main function to run tree models feature importance analysis
    """
    try:
        print("🌳 TREE MODELS FEATURE IMPORTANCE ANALYSIS")
        print("="*55)
        print("Algorithms: Random Forest vs XGBoost")
        print("Features: IEVR + normative_iv_rv_ratio + IV_RATIO + SMIRK + vol_hl21 + z_score_momentum + dispersion_pct_ibes")
        print("Methodology: 5-Year Training, 6-Month Testing Rolling Windows")
        print("="*55)
        
        # Initialize analyzer
        analyzer = TreeModelsFeatureImportance()
        
        # Load data
        if not analyzer.load_data():
            return
        
        # Run feature importance analysis
        results = analyzer.analyze_tree_models_importance()
        
        if results is not None and len(results) > 0:
            # Create visualizations
            analyzer.create_tree_models_visualizations()
            
            # Save results
            analyzer.save_results()
        else:
            print("❌ No results generated")
            
    except Exception as e:
        print(f"❌ Error in tree models feature importance analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

