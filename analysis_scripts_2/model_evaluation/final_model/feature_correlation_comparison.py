#!/usr/bin/env python3
"""
Feature Correlation Matrix Comparison
Compare correlation matrices between original and updated datasets
Features: IEVR + normative_iv_rv_ratio + SKEW + KURT + IV_RATIO + SMIRK + vol_hl7 + vol_hl10 + vol_hl21 + z_score_momentum
Target: REVR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FeatureCorrelationComparison:
    """
    Compare feature correlations between original and updated datasets
    """
    
    def __init__(self):
        self.original_df = None
        self.updated_df = None
        self.features = [
            'ievr', 'normative_iv_rv_ratio', 'SKEW', 'KURT', 'IV_RATIO', 
            'SMIRK', 'vol_hl7', 'vol_hl10', 'vol_hl21', 'z_score_momentum'
        ]
        self.target = 'revr'
        self.all_vars = self.features + [self.target]
        
    def load_datasets(self):
        """Load both original and updated datasets"""
        print("📊 LOADING DATASETS FOR CORRELATION COMPARISON")
        print("="*60)
        
        try:
            # Load original dataset
            self.original_df = pd.read_csv('../../data_files/final_merged_dataset_with_momentum_final.csv')
            print(f"✅ Loaded original dataset: {len(self.original_df):,} observations")
            
            # Load updated dataset
            self.updated_df = pd.read_csv('../../data_files/final_merged_dataset_with_momentum_updated.csv')
            print(f"✅ Loaded updated dataset: {len(self.updated_df):,} observations")
            
            # Convert earnings_date to datetime for both
            for df in [self.original_df, self.updated_df]:
                df['earnings_date'] = pd.to_datetime(df['earnings_date'])
                df['year'] = df['earnings_date'].dt.year
            
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
            
            print(f"📅 Date range - Original: {self.original_df['year'].min()} - {self.original_df['year'].max()}")
            print(f"📅 Date range - Updated: {self.updated_df['year'].min()} - {self.updated_df['year'].max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return False
    
    def check_data_availability(self):
        """Check feature availability in both datasets"""
        print(f"\n📋 FEATURE AVAILABILITY CHECK")
        print("="*50)
        
        for df_name, df in [("Original", self.original_df), ("Updated", self.updated_df)]:
            print(f"\n{df_name} Dataset:")
            for var in self.all_vars:
                if var in df.columns:
                    available = df[var].notna().sum()
                    total = len(df)
                    coverage = (available / total) * 100
                    print(f"  {var:25s}: {available:,} ({coverage:.1f}% coverage)")
                else:
                    print(f"  {var:25s}: NOT FOUND")
    
    def calculate_correlations(self):
        """Calculate correlation matrices for both datasets"""
        print(f"\n📊 CALCULATING CORRELATION MATRICES")
        print("="*50)
        
        correlations = {}
        
        for df_name, df in [("Original", self.original_df), ("Updated", self.updated_df)]:
            print(f"\nCalculating correlations for {df_name} dataset...")
            
            # Select only available variables
            available_vars = [var for var in self.all_vars if var in df.columns]
            df_subset = df[available_vars].copy()
            
            # Remove rows with any missing values
            df_clean = df_subset.dropna()
            print(f"  Using {len(df_clean):,} complete observations")
            
            # Calculate correlation matrix
            corr_matrix = df_clean.corr()
            correlations[df_name] = corr_matrix
            
            # Show correlations with target variable
            target_corrs = corr_matrix[self.target].drop(self.target).sort_values(key=abs, ascending=False)
            print(f"  Top correlations with {self.target}:")
            for feature, corr in target_corrs.head(5).items():
                print(f"    {feature:25s}: {corr:+.4f}")
        
        return correlations
    
    def create_correlation_visualizations(self, correlations):
        """Create comprehensive correlation visualizations"""
        print(f"\n📊 CREATING CORRELATION VISUALIZATIONS")
        print("="*50)
        
        # Set up the plotting style
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 10,
            'figure.facecolor': 'white'
        })
        
        # Create comparison visualization
        self._create_side_by_side_heatmaps(correlations)
        self._create_correlation_difference_analysis(correlations)
        self._create_target_correlation_comparison(correlations)
        self._create_feature_correlation_changes(correlations)
    
    def _create_side_by_side_heatmaps(self, correlations):
        """Create side-by-side correlation heatmaps"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Feature Correlation Matrix Comparison', fontsize=16, fontweight='bold', y=0.95)
        
        # Common settings for both heatmaps
        mask_lower = np.triu(np.ones_like(correlations['Original'], dtype=bool))
        
        # Original dataset heatmap
        sns.heatmap(correlations['Original'], 
                   mask=mask_lower,
                   annot=True, 
                   fmt='.3f',
                   cmap='RdBu_r', 
                   center=0,
                   vmin=-1, vmax=1,
                   square=True,
                   ax=ax1,
                   cbar_kws={'label': 'Correlation Coefficient'})
        ax1.set_title('Original Dataset\n(final_merged_dataset_with_momentum_final.csv)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        
        # Updated dataset heatmap
        sns.heatmap(correlations['Updated'], 
                   mask=mask_lower,
                   annot=True, 
                   fmt='.3f',
                   cmap='RdBu_r', 
                   center=0,
                   vmin=-1, vmax=1,
                   square=True,
                   ax=ax2,
                   cbar_kws={'label': 'Correlation Coefficient'})
        ax2.set_title('Updated Dataset\n(final_merged_dataset_with_momentum_updated.csv)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = 'output_files/correlation_matrix_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Side-by-side correlation heatmaps saved: {output_path}")
        plt.close()
    
    def _create_correlation_difference_analysis(self, correlations):
        """Create correlation difference heatmap"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Calculate difference matrix
        diff_matrix = correlations['Updated'] - correlations['Original']
        
        # Create mask for lower triangle
        mask_lower = np.triu(np.ones_like(diff_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(diff_matrix, 
                   mask=mask_lower,
                   annot=True, 
                   fmt='+.4f',
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   ax=ax,
                   cbar_kws={'label': 'Correlation Difference (Updated - Original)'})
        
        ax.set_title('Correlation Changes: Updated vs Original Dataset\n' + 
                    'Positive = Stronger correlation in updated dataset', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = 'output_files/correlation_difference_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Correlation difference heatmap saved: {output_path}")
        plt.close()
    
    def _create_target_correlation_comparison(self, correlations):
        """Create bar chart comparing correlations with target variable"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Extract correlations with target variable
        original_target_corrs = correlations['Original'][self.target].drop(self.target)
        updated_target_corrs = correlations['Updated'][self.target].drop(self.target)
        
        # Create DataFrame for easier plotting
        comparison_df = pd.DataFrame({
            'Original': original_target_corrs,
            'Updated': updated_target_corrs
        }).fillna(0)
        
        # Sort by absolute value of updated correlations
        comparison_df = comparison_df.reindex(
            comparison_df['Updated'].abs().sort_values(ascending=True).index
        )
        
        # Create horizontal bar chart
        x_pos = np.arange(len(comparison_df))
        width = 0.35
        
        bars1 = ax.barh(x_pos - width/2, comparison_df['Original'], width, 
                       label='Original Dataset', color='#1f77b4', alpha=0.8)
        bars2 = ax.barh(x_pos + width/2, comparison_df['Updated'], width, 
                       label='Updated Dataset', color='#ff7f0e', alpha=0.8)
        
        # Add value labels on bars
        for i, (orig, upd) in enumerate(zip(comparison_df['Original'], comparison_df['Updated'])):
            ax.text(orig + 0.005 if orig >= 0 else orig - 0.005, i - width/2, f'{orig:.3f}', 
                   va='center', ha='left' if orig >= 0 else 'right', fontsize=9)
            ax.text(upd + 0.005 if upd >= 0 else upd - 0.005, i + width/2, f'{upd:.3f}', 
                   va='center', ha='left' if upd >= 0 else 'right', fontsize=9)
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(comparison_df.index)
        ax.set_xlabel('Correlation with REVR', fontsize=12, fontweight='semibold')
        ax.set_title('Feature Correlations with REVR: Dataset Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = 'output_files/target_correlation_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Target correlation comparison saved: {output_path}")
        plt.close()
    
    def _create_feature_correlation_changes(self, correlations):
        """Create detailed analysis of correlation changes"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Correlation Analysis: Original vs Updated', 
                    fontsize=16, fontweight='bold')
        
        # 1. Scatter plot of correlation changes
        original_target = correlations['Original'][self.target].drop(self.target)
        updated_target = correlations['Updated'][self.target].drop(self.target)
        
        ax1.scatter(original_target, updated_target, s=100, alpha=0.7, color='steelblue')
        ax1.plot([-1, 1], [-1, 1], 'r--', alpha=0.5, label='No change line')
        
        # Add feature labels
        for feature, orig, upd in zip(original_target.index, original_target, updated_target):
            ax1.annotate(feature, (orig, upd), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
        
        ax1.set_xlabel('Original Dataset Correlation', fontsize=11)
        ax1.set_ylabel('Updated Dataset Correlation', fontsize=11)
        ax1.set_title('REVR Correlations: Original vs Updated', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Correlation change magnitude
        changes = updated_target - original_target
        changes_sorted = changes.abs().sort_values(ascending=True)
        
        colors = ['green' if x > 0 else 'red' for x in changes[changes_sorted.index]]
        bars = ax2.barh(range(len(changes_sorted)), changes[changes_sorted.index], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(changes_sorted)))
        ax2.set_yticklabels(changes_sorted.index)
        ax2.set_xlabel('Correlation Change (Updated - Original)', fontsize=11)
        ax2.set_title('Correlation Changes with REVR', fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Option surface features focus
        option_features = ['SKEW', 'KURT', 'IV_RATIO', 'SMIRK']
        option_changes = changes[option_features]
        
        bars = ax3.bar(option_features, option_changes, 
                      color=['green' if x > 0 else 'red' for x in option_changes], alpha=0.7)
        ax3.set_ylabel('Correlation Change', fontsize=11)
        ax3.set_title('Option Surface Feature Changes', fontsize=12, fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, option_changes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001 if height >= 0 else height - 0.001,
                    f'{value:+.4f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        # 4. Summary statistics table
        ax4.axis('off')
        
        # Create summary table
        summary_data = []
        for feature in self.features:
            if feature in original_target.index and feature in updated_target.index:
                orig_corr = original_target[feature]
                upd_corr = updated_target[feature]
                change = upd_corr - orig_corr
                summary_data.append([feature, f'{orig_corr:.4f}', f'{upd_corr:.4f}', f'{change:+.4f}'])
        
        # Sort by absolute change
        summary_data.sort(key=lambda x: abs(float(x[3])), reverse=True)
        
        # Create table
        table = ax4.table(cellText=summary_data,
                         colLabels=['Feature', 'Original', 'Updated', 'Change'],
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
        
        ax4.set_title('Correlation Changes Summary', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = 'output_files/correlation_detailed_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Detailed correlation analysis saved: {output_path}")
        plt.close()
    
    def save_correlation_tables(self, correlations):
        """Save correlation matrices as CSV files"""
        print(f"\n💾 SAVING CORRELATION TABLES")
        print("="*40)
        
        for dataset_name, corr_matrix in correlations.items():
            filename = f'output_files/correlation_matrix_{dataset_name.lower()}.csv'
            corr_matrix.to_csv(filename)
            print(f"✅ {dataset_name} correlation matrix saved: {filename}")
        
        # Save correlation differences
        if 'Original' in correlations and 'Updated' in correlations:
            diff_matrix = correlations['Updated'] - correlations['Original']
            diff_filename = 'output_files/correlation_differences.csv'
            diff_matrix.to_csv(diff_filename)
            print(f"✅ Correlation differences saved: {diff_filename}")
        
        # Save target correlations comparison
        if 'Original' in correlations and 'Updated' in correlations:
            target_comparison = pd.DataFrame({
                'Original': correlations['Original'][self.target].drop(self.target),
                'Updated': correlations['Updated'][self.target].drop(self.target)
            })
            target_comparison['Change'] = target_comparison['Updated'] - target_comparison['Original']
            target_comparison['Abs_Change'] = target_comparison['Change'].abs()
            target_comparison = target_comparison.sort_values('Abs_Change', ascending=False)
            
            target_filename = 'output_files/target_correlation_comparison.csv'
            target_comparison.to_csv(target_filename)
            print(f"✅ Target correlation comparison saved: {target_filename}")

def main():
    """
    Main function to run correlation comparison analysis
    """
    try:
        print("🚀 FEATURE CORRELATION MATRIX COMPARISON")
        print("="*80)
        print("Comparing correlations between original and updated datasets")
        print("Features: IEVR + normative_iv_rv_ratio + SKEW + KURT + IV_RATIO + SMIRK + vol_hl7 + vol_hl10 + vol_hl21 + z_score_momentum")
        print("Target: REVR")
        print("="*80)
        
        # Initialize analyzer
        analyzer = FeatureCorrelationComparison()
        
        # Load datasets
        if not analyzer.load_datasets():
            print("❌ Failed to load datasets")
            return
        
        # Check data availability
        analyzer.check_data_availability()
        
        # Calculate correlations
        correlations = analyzer.calculate_correlations()
        
        if correlations:
            # Create visualizations
            analyzer.create_correlation_visualizations(correlations)
            
            # Save correlation tables
            analyzer.save_correlation_tables(correlations)
            
            print(f"\n🎉 CORRELATION COMPARISON ANALYSIS COMPLETED!")
            print("Key outputs:")
            print("  • correlation_matrix_comparison.png - Side-by-side heatmaps")
            print("  • correlation_difference_matrix.png - Difference analysis")
            print("  • target_correlation_comparison.png - REVR correlations")
            print("  • correlation_detailed_analysis.png - Comprehensive analysis")
            print("  • correlation_matrix_*.csv - Correlation matrices")
            print("  • target_correlation_comparison.csv - Target correlation changes")
        else:
            print("❌ No correlations calculated")
            
    except Exception as e:
        print(f"❌ Error in correlation comparison analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

