#!/usr/bin/env python3
"""
Data Coverage Analysis by Quarter
Analyze data coverage for each feature and target variable organized by quarter
Reference universe: top500_liquidity_2005_2023.csv
Features: IEVR + normative_iv_rv_ratio + SKEW + KURT + IV_RATIO + SMIRK + vol_hl7 + vol_hl10 + vol_hl21 + z_score_momentum
Target: REVR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class DataCoverageAnalysis:
    """
    Analyze data coverage by quarter for all features
    """
    
    def __init__(self):
        self.liquidity_df = None
        self.original_df = None
        self.updated_df = None
        self.features = [
            'ievr', 'normative_iv_rv_ratio', 'SKEW', 'KURT', 'IV_RATIO', 
            'SMIRK', 'vol_hl7', 'vol_hl10', 'vol_hl21', 'z_score_momentum'
        ]
        self.target = 'revr'
        self.all_vars = self.features + [self.target]
        
    def load_datasets(self):
        """Load all datasets"""
        print("📊 LOADING DATASETS FOR COVERAGE ANALYSIS")
        print("="*60)
        
        try:
            # Load top500 liquidity reference
            self.liquidity_df = pd.read_csv('../../data_files/top500_liquidity_2005_2023.csv')
            print(f"✅ Loaded liquidity reference: {len(self.liquidity_df):,} observations")
            
            # Load original dataset
            self.original_df = pd.read_csv('../../data_files/final_merged_dataset_with_momentum_final.csv')
            print(f"✅ Loaded original dataset: {len(self.original_df):,} observations")
            
            # Load updated dataset
            self.updated_df = pd.read_csv('../../data_files/final_merged_dataset_with_momentum_updated.csv')
            print(f"✅ Loaded updated dataset: {len(self.updated_df):,} observations")
            
            # Process dates and quarters for all datasets
            for df_name, df in [("Liquidity", self.liquidity_df), ("Original", self.original_df), ("Updated", self.updated_df)]:
                print(f"🔧 Processing dates for {df_name} dataset...")
                
                # Handle different date column names
                if 'earnings_date' in df.columns:
                    date_col = 'earnings_date'
                elif 'date' in df.columns:
                    date_col = 'date'
                else:
                    # Try to find any date-like column
                    date_cols = [col for col in df.columns if 'date' in col.lower()]
                    if date_cols:
                        date_col = date_cols[0]
                    else:
                        print(f"❌ No date column found in {df_name} dataset")
                        continue
                
                df[date_col] = pd.to_datetime(df[date_col])
                df['year'] = df[date_col].dt.year
                df['quarter'] = df[date_col].dt.quarter
                df['year_quarter'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)
                
                print(f"   Date range: {df['year'].min()} - {df['year'].max()}")
            
            # Calculate normative_iv_rv_ratio if missing
            for df_name, df in [("Original", self.original_df), ("Updated", self.updated_df)]:
                if 'normative_iv_rv_ratio' not in df.columns:
                    if 'avg_pre' in df.columns and 'normative_realized_vol' in df.columns:
                        print(f"🔧 Calculating normative_iv_rv_ratio for {df_name} dataset...")
                        df['normative_iv_rv_ratio'] = df['avg_pre'] / df['normative_realized_vol']
                        df['normative_iv_rv_ratio'] = df['normative_iv_rv_ratio'].replace([np.inf, -np.inf], np.nan)
                        print(f"✅ Created normative_iv_rv_ratio for {df_name} dataset")
            
            # Add z_score_momentum if missing
            for df_name, df in [("Original", self.original_df), ("Updated", self.updated_df)]:
                if 'z_score_momentum' not in df.columns:
                    print(f"🔧 Creating z_score_momentum for {df_name} dataset...")
                    df_sorted = df.sort_values(['ticker', 'earnings_date'])
                    df_sorted['momentum_6m'] = df_sorted.groupby('ticker')['revr'].rolling(window=4, min_periods=2).mean().reset_index(0, drop=True)
                    df_sorted['z_score_momentum'] = (
                        (df_sorted['momentum_6m'] - df_sorted.groupby('ticker')['momentum_6m'].transform('mean')) /
                        df_sorted.groupby('ticker')['momentum_6m'].transform('std')
                    ).fillna(0)
                    df['z_score_momentum'] = df_sorted['z_score_momentum']
                    print(f"✅ Created z_score_momentum for {df_name} dataset")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return False
    
    def calculate_quarterly_coverage(self):
        """Calculate data coverage by quarter"""
        print(f"\n📈 CALCULATING QUARTERLY COVERAGE")
        print("="*50)
        
        coverage_results = {}
        
        # Get reference universe from liquidity data
        if 'year_quarter' in self.liquidity_df.columns:
            reference_quarters = sorted(self.liquidity_df['year_quarter'].unique())
            print(f"Reference quarters: {len(reference_quarters)} quarters from {reference_quarters[0]} to {reference_quarters[-1]}")
        else:
            print("❌ Could not determine quarters from liquidity data")
            return None
        
        # Calculate coverage for each dataset
        for dataset_name, df in [("Original", self.original_df), ("Updated", self.updated_df)]:
            print(f"\nCalculating coverage for {dataset_name} dataset...")
            
            quarterly_coverage = []
            
            for quarter in reference_quarters:
                quarter_data = df[df['year_quarter'] == quarter]
                
                if len(quarter_data) == 0:
                    # No data for this quarter
                    quarter_stats = {'year_quarter': quarter, 'total_observations': 0}
                    for var in self.all_vars:
                        quarter_stats[f'{var}_count'] = 0
                        quarter_stats[f'{var}_coverage'] = 0.0
                else:
                    quarter_stats = {'year_quarter': quarter, 'total_observations': len(quarter_data)}
                    
                    for var in self.all_vars:
                        if var in quarter_data.columns:
                            available = quarter_data[var].notna().sum()
                            coverage = (available / len(quarter_data)) * 100 if len(quarter_data) > 0 else 0
                        else:
                            available = 0
                            coverage = 0.0
                        
                        quarter_stats[f'{var}_count'] = available
                        quarter_stats[f'{var}_coverage'] = coverage
                
                quarterly_coverage.append(quarter_stats)
            
            coverage_df = pd.DataFrame(quarterly_coverage)
            coverage_df['year'] = coverage_df['year_quarter'].str.extract('(\d{4})').astype(int)
            coverage_df['quarter'] = coverage_df['year_quarter'].str.extract('Q(\d)').astype(int)
            
            coverage_results[dataset_name] = coverage_df
            
            print(f"   Processed {len(coverage_df)} quarters")
            print(f"   Average observations per quarter: {coverage_df['total_observations'].mean():.1f}")
        
        return coverage_results
    
    def create_coverage_visualizations(self, coverage_results):
        """Create comprehensive coverage visualizations"""
        print(f"\n📊 CREATING COVERAGE VISUALIZATIONS")
        print("="*50)
        
        # Set up the plotting style
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 10,
            'figure.facecolor': 'white'
        })
        
        self._create_coverage_heatmap(coverage_results)
        self._create_coverage_time_series(coverage_results)
        self._create_coverage_comparison(coverage_results)
        self._create_option_features_focus(coverage_results)
    
    def _create_coverage_heatmap(self, coverage_results):
        """Create coverage heatmap for each dataset"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle('Data Coverage by Quarter: Original vs Updated Dataset', 
                    fontsize=16, fontweight='bold')
        
        for idx, (dataset_name, coverage_df) in enumerate(coverage_results.items()):
            ax = axes[idx]
            
            # Prepare data for heatmap
            coverage_cols = [f'{var}_coverage' for var in self.all_vars]
            heatmap_data = coverage_df.set_index('year_quarter')[coverage_cols]
            
            # Rename columns for better display
            heatmap_data.columns = self.all_vars
            
            # Create heatmap
            sns.heatmap(heatmap_data.T, 
                       annot=False, 
                       cmap='RdYlGn', 
                       vmin=0, vmax=100,
                       cbar_kws={'label': 'Coverage (%)'},
                       ax=ax)
            
            ax.set_title(f'{dataset_name} Dataset Coverage', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Quarter', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
            
            # Show only every 4th quarter label to avoid crowding
            xticks = ax.get_xticks()
            xlabels = [label.get_text() for label in ax.get_xticklabels()]
            ax.set_xticks(xticks[::4])
            ax.set_xticklabels([xlabels[i] for i in range(0, len(xlabels), 4)])
        
        plt.tight_layout()
        
        output_path = 'output_files/data_coverage_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Coverage heatmap saved: {output_path}")
        plt.close()
    
    def _create_coverage_time_series(self, coverage_results):
        """Create time series plot of coverage"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Data Coverage Evolution Over Time', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        # Plot 1: Overall coverage comparison
        ax1 = axes[0]
        for dataset_name, coverage_df in coverage_results.items():
            # Calculate average coverage across all features
            coverage_cols = [f'{var}_coverage' for var in self.all_vars]
            coverage_df['avg_coverage'] = coverage_df[coverage_cols].mean(axis=1)
            
            ax1.plot(range(len(coverage_df)), coverage_df['avg_coverage'], 
                    marker='o', linewidth=2, label=f'{dataset_name} Dataset', alpha=0.8)
        
        ax1.set_title('Average Coverage Across All Features', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Quarter Index', fontsize=11)
        ax1.set_ylabel('Average Coverage (%)', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Plot 2: Option features coverage
        ax2 = axes[1]
        option_features = ['SKEW', 'KURT', 'IV_RATIO', 'SMIRK']
        
        for dataset_name, coverage_df in coverage_results.items():
            option_coverage_cols = [f'{var}_coverage' for var in option_features]
            coverage_df['option_avg_coverage'] = coverage_df[option_coverage_cols].mean(axis=1)
            
            ax2.plot(range(len(coverage_df)), coverage_df['option_avg_coverage'], 
                    marker='s', linewidth=2, label=f'{dataset_name} Dataset', alpha=0.8)
        
        ax2.set_title('Option Surface Features Coverage', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Quarter Index', fontsize=11)
        ax2.set_ylabel('Option Features Coverage (%)', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot 3: Total observations over time
        ax3 = axes[2]
        for dataset_name, coverage_df in coverage_results.items():
            ax3.plot(range(len(coverage_df)), coverage_df['total_observations'], 
                    marker='^', linewidth=2, label=f'{dataset_name} Dataset', alpha=0.8)
        
        ax3.set_title('Total Observations per Quarter', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Quarter Index', fontsize=11)
        ax3.set_ylabel('Number of Observations', fontsize=11)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Coverage improvement (Updated - Original)
        ax4 = axes[3]
        if 'Original' in coverage_results and 'Updated' in coverage_results:
            original_df = coverage_results['Original']
            updated_df = coverage_results['Updated']
            
            for feature in self.all_vars:
                orig_col = f'{feature}_coverage'
                if orig_col in original_df.columns and orig_col in updated_df.columns:
                    improvement = updated_df[orig_col].values - original_df[orig_col].values
                    ax4.plot(range(len(improvement)), improvement, 
                            label=feature, alpha=0.7, linewidth=1.5)
            
            ax4.set_title('Coverage Improvement: Updated vs Original', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Quarter Index', fontsize=11)
            ax4.set_ylabel('Coverage Improvement (%)', fontsize=11)
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        output_path = 'output_files/data_coverage_time_series.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Coverage time series saved: {output_path}")
        plt.close()
    
    def _create_coverage_comparison(self, coverage_results):
        """Create detailed coverage comparison"""
        if 'Original' not in coverage_results or 'Updated' not in coverage_results:
            print("⚠️ Cannot create comparison - missing dataset")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Coverage Analysis: Original vs Updated', 
                    fontsize=16, fontweight='bold')
        
        original_df = coverage_results['Original']
        updated_df = coverage_results['Updated']
        
        # 1. Average coverage by feature
        ax1.set_title('Average Coverage by Feature', fontsize=12, fontweight='bold')
        
        feature_avg_original = []
        feature_avg_updated = []
        feature_names = []
        
        for feature in self.all_vars:
            orig_col = f'{feature}_coverage'
            if orig_col in original_df.columns and orig_col in updated_df.columns:
                feature_avg_original.append(original_df[orig_col].mean())
                feature_avg_updated.append(updated_df[orig_col].mean())
                feature_names.append(feature)
        
        x_pos = np.arange(len(feature_names))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, feature_avg_original, width, 
                       label='Original', color='#1f77b4', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, feature_avg_updated, width, 
                       label='Updated', color='#ff7f0e', alpha=0.8)
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(feature_names, rotation=45, ha='right')
        ax1.set_ylabel('Average Coverage (%)')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 2. Coverage improvement histogram
        ax2.set_title('Coverage Improvement Distribution', fontsize=12, fontweight='bold')
        
        improvements = []
        for feature in self.all_vars:
            orig_col = f'{feature}_coverage'
            if orig_col in original_df.columns and orig_col in updated_df.columns:
                improvement = updated_df[orig_col].mean() - original_df[orig_col].mean()
                improvements.append(improvement)
        
        ax2.hist(improvements, bins=10, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Coverage Improvement (%)')
        ax2.set_ylabel('Number of Features')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No change')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Quarterly trend comparison
        ax3.set_title('Coverage Trends Over Time', fontsize=12, fontweight='bold')
        
        quarters = original_df['year_quarter']
        x_quarters = range(len(quarters))
        
        # Calculate overall coverage for each quarter
        orig_overall = []
        upd_overall = []
        
        for idx in range(len(original_df)):
            orig_avg = np.mean([original_df.iloc[idx][f'{var}_coverage'] for var in self.all_vars 
                               if f'{var}_coverage' in original_df.columns])
            upd_avg = np.mean([updated_df.iloc[idx][f'{var}_coverage'] for var in self.all_vars 
                              if f'{var}_coverage' in updated_df.columns])
            orig_overall.append(orig_avg)
            upd_overall.append(upd_avg)
        
        ax3.plot(x_quarters, orig_overall, label='Original', linewidth=2, alpha=0.8)
        ax3.plot(x_quarters, upd_overall, label='Updated', linewidth=2, alpha=0.8)
        ax3.fill_between(x_quarters, orig_overall, upd_overall, alpha=0.3, color='green')
        
        ax3.set_xlabel('Quarter Index')
        ax3.set_ylabel('Overall Coverage (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics table
        ax4.axis('off')
        ax4.set_title('Coverage Summary Statistics', fontsize=12, fontweight='bold', pad=20)
        
        # Create summary table
        summary_data = []
        for feature in self.all_vars:
            orig_col = f'{feature}_coverage'
            if orig_col in original_df.columns and orig_col in updated_df.columns:
                orig_avg = original_df[orig_col].mean()
                upd_avg = updated_df[orig_col].mean()
                improvement = upd_avg - orig_avg
                
                summary_data.append([
                    feature, 
                    f'{orig_avg:.1f}%', 
                    f'{upd_avg:.1f}%', 
                    f'{improvement:+.1f}%'
                ])
        
        # Sort by improvement
        summary_data.sort(key=lambda x: float(x[3].replace('%', '').replace('+', '')), reverse=True)
        
        table = ax4.table(cellText=summary_data,
                         colLabels=['Feature', 'Original', 'Updated', 'Improvement'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style the table
        for i in range(len(summary_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                    # Color improvement column
                    if j == 3 and i > 0:
                        improvement_val = float(summary_data[i-1][3].replace('%', '').replace('+', ''))
                        if improvement_val > 0:
                            cell.set_facecolor('#e8f5e8')
                        elif improvement_val < 0:
                            cell.set_facecolor('#ffeaea')
        
        plt.tight_layout()
        
        output_path = 'output_files/data_coverage_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Coverage comparison saved: {output_path}")
        plt.close()
    
    def _create_option_features_focus(self, coverage_results):
        """Create focused analysis of option surface features"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Option Surface Features Coverage Analysis', 
                    fontsize=16, fontweight='bold')
        
        option_features = ['SKEW', 'KURT', 'IV_RATIO', 'SMIRK']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # 1. Option features coverage over time
        ax1.set_title('Option Features Coverage Evolution', fontsize=12, fontweight='bold')
        
        for dataset_name, coverage_df in coverage_results.items():
            quarters = range(len(coverage_df))
            
            for i, feature in enumerate(option_features):
                coverage_col = f'{feature}_coverage'
                if coverage_col in coverage_df.columns:
                    linestyle = '-' if dataset_name == 'Updated' else '--'
                    alpha = 0.8 if dataset_name == 'Updated' else 0.6
                    ax1.plot(quarters, coverage_df[coverage_col], 
                            color=colors[i], linestyle=linestyle, alpha=alpha,
                            label=f'{feature} ({dataset_name})', linewidth=2)
        
        ax1.set_xlabel('Quarter Index')
        ax1.set_ylabel('Coverage (%)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # 2. Average coverage comparison for option features
        ax2.set_title('Average Option Features Coverage', fontsize=12, fontweight='bold')
        
        if 'Original' in coverage_results and 'Updated' in coverage_results:
            original_means = []
            updated_means = []
            
            for feature in option_features:
                coverage_col = f'{feature}_coverage'
                orig_mean = coverage_results['Original'][coverage_col].mean()
                upd_mean = coverage_results['Updated'][coverage_col].mean()
                original_means.append(orig_mean)
                updated_means.append(upd_mean)
            
            x_pos = np.arange(len(option_features))
            width = 0.35
            
            bars1 = ax2.bar(x_pos - width/2, original_means, width, 
                           label='Original', color='lightcoral', alpha=0.8)
            bars2 = ax2.bar(x_pos + width/2, updated_means, width, 
                           label='Updated', color='lightgreen', alpha=0.8)
            
            # Add value labels
            for bars, values in [(bars1, original_means), (bars2, updated_means)]:
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(option_features)
            ax2.set_ylabel('Average Coverage (%)')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            ax2.set_ylim(0, 105)
        
        # 3. Coverage improvement for option features
        ax3.set_title('Coverage Improvement by Option Feature', fontsize=12, fontweight='bold')
        
        if 'Original' in coverage_results and 'Updated' in coverage_results:
            improvements = []
            for feature in option_features:
                coverage_col = f'{feature}_coverage'
                improvement = (coverage_results['Updated'][coverage_col].mean() - 
                              coverage_results['Original'][coverage_col].mean())
                improvements.append(improvement)
            
            colors_imp = ['green' if x > 0 else 'red' for x in improvements]
            bars = ax3.bar(option_features, improvements, color=colors_imp, alpha=0.7)
            
            # Add value labels
            for bar, value in zip(bars, improvements):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., 
                        height + 0.1 if height >= 0 else height - 0.1,
                        f'{value:+.1f}%', ha='center', 
                        va='bottom' if height >= 0 else 'top', 
                        fontsize=10, fontweight='bold')
            
            ax3.set_ylabel('Coverage Improvement (%)')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.grid(axis='y', alpha=0.3)
        
        # 4. Option features correlation with total observations
        ax4.set_title('Coverage vs Total Observations', fontsize=12, fontweight='bold')
        
        for dataset_name, coverage_df in coverage_results.items():
            # Calculate average option coverage for each quarter
            option_coverage_cols = [f'{feature}_coverage' for feature in option_features]
            avg_option_coverage = coverage_df[option_coverage_cols].mean(axis=1)
            
            scatter = ax4.scatter(coverage_df['total_observations'], avg_option_coverage, 
                                 label=f'{dataset_name} Dataset', alpha=0.7, s=50)
        
        ax4.set_xlabel('Total Observations per Quarter')
        ax4.set_ylabel('Average Option Features Coverage (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = 'output_files/option_features_coverage_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Option features coverage analysis saved: {output_path}")
        plt.close()
    
    def save_coverage_data(self, coverage_results):
        """Save coverage data as CSV files"""
        print(f"\n💾 SAVING COVERAGE DATA")
        print("="*40)
        
        for dataset_name, coverage_df in coverage_results.items():
            filename = f'output_files/data_coverage_{dataset_name.lower()}.csv'
            coverage_df.to_csv(filename, index=False)
            print(f"✅ {dataset_name} coverage data saved: {filename}")
        
        # Save coverage summary
        if 'Original' in coverage_results and 'Updated' in coverage_results:
            summary_data = []
            
            for feature in self.all_vars:
                orig_col = f'{feature}_coverage'
                if (orig_col in coverage_results['Original'].columns and 
                    orig_col in coverage_results['Updated'].columns):
                    
                    orig_avg = coverage_results['Original'][orig_col].mean()
                    orig_std = coverage_results['Original'][orig_col].std()
                    upd_avg = coverage_results['Updated'][orig_col].mean()
                    upd_std = coverage_results['Updated'][orig_col].std()
                    improvement = upd_avg - orig_avg
                    
                    summary_data.append({
                        'feature': feature,
                        'original_avg_coverage': orig_avg,
                        'original_std_coverage': orig_std,
                        'updated_avg_coverage': upd_avg,
                        'updated_std_coverage': upd_std,
                        'coverage_improvement': improvement,
                        'improvement_percentage': (improvement / orig_avg) * 100 if orig_avg > 0 else 0
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_filename = 'output_files/coverage_improvement_summary.csv'
            summary_df.to_csv(summary_filename, index=False)
            print(f"✅ Coverage improvement summary saved: {summary_filename}")

def main():
    """
    Main function to run data coverage analysis
    """
    try:
        print("🚀 DATA COVERAGE ANALYSIS BY QUARTER")
        print("="*80)
        print("Analyzing data coverage for features and target variable")
        print("Reference universe: top500_liquidity_2005_2023.csv")
        print("Features: IEVR + normative_iv_rv_ratio + SKEW + KURT + IV_RATIO + SMIRK + vol_hl7 + vol_hl10 + vol_hl21 + z_score_momentum")
        print("Target: REVR")
        print("="*80)
        
        # Initialize analyzer
        analyzer = DataCoverageAnalysis()
        
        # Load datasets
        if not analyzer.load_datasets():
            print("❌ Failed to load datasets")
            return
        
        # Calculate quarterly coverage
        coverage_results = analyzer.calculate_quarterly_coverage()
        
        if coverage_results:
            # Create visualizations
            analyzer.create_coverage_visualizations(coverage_results)
            
            # Save coverage data
            analyzer.save_coverage_data(coverage_results)
            
            print(f"\n🎉 DATA COVERAGE ANALYSIS COMPLETED!")
            print("Key outputs:")
            print("  • data_coverage_heatmap.png - Coverage heatmap by quarter")
            print("  • data_coverage_time_series.png - Coverage evolution over time")
            print("  • data_coverage_comparison.png - Detailed comparison analysis")
            print("  • option_features_coverage_analysis.png - Option features focus")
            print("  • data_coverage_*.csv - Coverage data by dataset")
            print("  • coverage_improvement_summary.csv - Summary statistics")
        else:
            print("❌ No coverage data calculated")
            
    except Exception as e:
        print(f"❌ Error in data coverage analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

