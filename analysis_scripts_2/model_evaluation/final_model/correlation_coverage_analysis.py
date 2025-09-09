#!/usr/bin/env python3
"""
Correlation Matrix and Data Coverage Analysis
Generate correlation matrix for 6 features + target variable
and data coverage time series across quarters
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CorrelationCoverageAnalysis:
    """
    Analyze correlations and data coverage for the 7-feature model
    """
    
    def __init__(self):
        self.df = None
        self.features = [
            'ievr', 'normative_iv_rv_ratio', 'IV_RATIO', 
            'SMIRK', 'vol_hl21', 'z_score_momentum', 'dispersion_pct_ibes'
        ]
        self.target = 'revr'
        self.all_variables = self.features + [self.target]
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("📊 LOADING DATASET FOR CORRELATION & COVERAGE ANALYSIS")
        print("="*60)
        
        try:
            # Use model_df.csv with proper path handling
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up two levels to reach the project root, then to data_files
            project_root = os.path.dirname(os.path.dirname(script_dir))
            data_file_path = os.path.join(project_root, 'data_files', 'model_df.csv')
            
            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f"❌ Dataset file not found: {data_file_path}")
            
            self.df = pd.read_csv(data_file_path)
            print(f"✅ Loaded model dataset: {len(self.df):,} observations")
            
            # Convert earnings_date to datetime
            self.df['earnings_date'] = pd.to_datetime(self.df['earnings_date'])
            self.df['year'] = self.df['earnings_date'].dt.year
            self.df['quarter'] = self.df['earnings_date'].dt.quarter
            
            # Create year-quarter column for time series analysis
            self.df['year_quarter'] = self.df['year'].astype(str) + 'Q' + self.df['quarter'].astype(str)
            self.df['quarter_date'] = pd.to_datetime(self.df['year'].astype(str) + '-' + 
                                                   (self.df['quarter'] * 3).astype(str) + '-01')
            
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
            
            # Add z_score_momentum if missing
            if 'z_score_momentum' not in self.df.columns:
                print("🔧 Creating simple z_score_momentum feature...")
                self.df = self.df.sort_values(['ticker', 'earnings_date'])
                # Simple momentum calculation
                self.df['momentum_3m'] = self.df.groupby('ticker')['revr'].rolling(window=3, min_periods=2).mean().reset_index(0, drop=True).shift(1)
                self.df['z_score_momentum'] = self.df.groupby('ticker')['momentum_3m'].transform(lambda x: (x - x.mean()) / x.std()).fillna(0)
                print(f"✅ Created z_score_momentum feature")
            
            # Check feature availability
            missing_features = [f for f in self.all_variables if f not in self.df.columns]
            if missing_features:
                print(f"❌ Missing features: {missing_features}")
                return False
                
            print(f"✅ All {len(self.all_variables)} variables available")
            return True
            
        except FileNotFoundError:
            print("❌ Dataset file not found!")
            return False
    
    def analyze_correlations(self):
        """Analyze correlations between all variables"""
        print(f"\n📊 CORRELATION ANALYSIS")
        print("="*40)
        
        # Get clean data for correlation analysis
        correlation_data = self.df[self.all_variables].dropna()
        print(f"Clean data for correlation: {len(correlation_data):,} observations")
        
        # Calculate correlation matrix
        correlation_matrix = correlation_data.corr()
        
        # Print correlation summary
        print(f"\nCORRELATION WITH TARGET VARIABLE ({self.target.upper()}):")
        print("-" * 50)
        target_correlations = correlation_matrix[self.target].drop(self.target).sort_values(key=abs, ascending=False)
        
        for feature, corr in target_correlations.items():
            print(f"{feature:25s}: {corr:+7.4f}")
        
        # Find highest correlations between features
        print(f"\nHIGHEST FEATURE-TO-FEATURE CORRELATIONS:")
        print("-" * 45)
        
        feature_correlations = []
        for i, feature1 in enumerate(self.features):
            for j, feature2 in enumerate(self.features):
                if i < j:  # Avoid duplicates
                    corr = correlation_matrix.loc[feature1, feature2]
                    feature_correlations.append((feature1, feature2, corr))
        
        # Sort by absolute correlation
        feature_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for feature1, feature2, corr in feature_correlations[:5]:
            print(f"{feature1:15s} ↔ {feature2:15s}: {corr:+7.4f}")
        
        return correlation_matrix
    
    def analyze_data_coverage(self):
        """Analyze data coverage across time"""
        print(f"\n📊 DATA COVERAGE ANALYSIS")
        print("="*35)
        
        # Calculate coverage by quarter
        coverage_by_quarter = []
        
        # Get all unique quarters
        quarters = self.df.groupby(['year', 'quarter']).size().index
        
        for year, quarter in quarters:
            quarter_data = self.df[(self.df['year'] == year) & (self.df['quarter'] == quarter)]
            quarter_date = pd.to_datetime(f"{year}-{quarter * 3:02d}-01")
            year_quarter = f"{year}Q{quarter}"
            
            coverage_info = {
                'year': year,
                'quarter': quarter,
                'year_quarter': year_quarter,
                'quarter_date': quarter_date,
                'total_observations': len(quarter_data)
            }
            
            # Calculate coverage for each variable
            for var in self.all_variables:
                valid_count = quarter_data[var].notna().sum()
                coverage_pct = (valid_count / len(quarter_data)) * 100 if len(quarter_data) > 0 else 0
                coverage_info[f'{var}_count'] = valid_count
                coverage_info[f'{var}_coverage'] = coverage_pct
            
            # Calculate complete case coverage (all variables available)
            complete_cases = quarter_data[self.all_variables].dropna()
            coverage_info['complete_cases_count'] = len(complete_cases)
            coverage_info['complete_cases_coverage'] = (len(complete_cases) / len(quarter_data)) * 100 if len(quarter_data) > 0 else 0
            
            coverage_by_quarter.append(coverage_info)
        
        coverage_df = pd.DataFrame(coverage_by_quarter)
        
        # Print summary statistics
        print(f"OVERALL DATA COVERAGE SUMMARY:")
        print("-" * 35)
        print(f"Total quarters analyzed: {len(coverage_df)}")
        print(f"Date range: {coverage_df['year_quarter'].iloc[0]} to {coverage_df['year_quarter'].iloc[-1]}")
        print(f"Average observations per quarter: {coverage_df['total_observations'].mean():.1f}")
        print(f"Average complete cases per quarter: {coverage_df['complete_cases_count'].mean():.1f}")
        print(f"Average complete case coverage: {coverage_df['complete_cases_coverage'].mean():.1f}%")
        
        print(f"\nAVERAGE COVERAGE BY VARIABLE:")
        print("-" * 30)
        for var in self.all_variables:
            avg_coverage = coverage_df[f'{var}_coverage'].mean()
            print(f"{var:25s}: {avg_coverage:6.1f}%")
        
        return coverage_df
    
    def create_correlation_visualization(self, correlation_matrix):
        """Create correlation matrix heatmap"""
        print(f"\n📊 CREATING CORRELATION VISUALIZATION")
        print("="*40)
        
        # Set professional styling
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 13,
            'axes.titlesize': 14,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': False
        })
        
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('white')
        
        # Create custom colormap (BlackRock colors)
        colors = ['#003366', '#66CCFF', 'white', '#FF6633', '#8C0000']
        n_bins = 100
        cmap = sns.blend_palette(colors, n_colors=n_bins, as_cmap=True)
        
        # Create heatmap - full matrix (no masking)
        mask = None  # No masking for full correlation matrix
        
        heatmap = sns.heatmap(
            correlation_matrix, 
            mask=mask,
            annot=True, 
            fmt='.3f',
            cmap=cmap,
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            annot_kws={"size": 12, "weight": "bold"}
        )
        
        # Customize labels
        feature_labels = [f.replace('_', ' ').replace('ievr', 'IEVR').replace('revr', 'REVR').title() 
                         for f in self.all_variables]
        
        ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(feature_labels, rotation=0, fontsize=12)
        
        # Adjust colorbar font size
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Correlation Coefficient', fontsize=12)
        
        # Add title and styling
        ax.set_title('Feature Correlation Matrix\n7-Feature Model + Target Variable', 
                    fontsize=16, fontweight='bold', color='#003366', pad=20)
        
        # Highlight target variable
        target_idx = self.all_variables.index(self.target)
        
        # Add border around target row/column
        ax.add_patch(plt.Rectangle((0, target_idx), len(self.all_variables), 1, 
                                  fill=False, edgecolor='#003366', lw=3))
        ax.add_patch(plt.Rectangle((target_idx, 0), 1, len(self.all_variables), 
                                  fill=False, edgecolor='#003366', lw=3))
        
        plt.tight_layout()
        
        # Save plot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output_files', 'correlation_coverage_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'feature_correlation_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Correlation matrix saved: {output_path}")
        
        # Save SVG version
        output_path_svg = os.path.join(output_dir, 'feature_correlation_matrix.svg')
        plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white')
        print(f"✅ SVG version saved: {output_path_svg}")
        
        plt.close()
    
    def create_coverage_visualization(self, coverage_df):
        """Create data coverage time series plots"""
        print(f"\n📊 CREATING COVERAGE VISUALIZATION")
        print("="*38)
        
        # Create comprehensive coverage plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.patch.set_facecolor('white')
        fig.suptitle('Data Coverage Analysis Across Time\n7-Feature Model + Target Variable', 
                     fontsize=16, fontweight='bold', color='#003366')
        
        # Plot 1: Total observations per quarter
        ax1.plot(coverage_df['quarter_date'], coverage_df['total_observations'], 
                color='#003366', linewidth=3, marker='o', markersize=6, 
                markeredgecolor='white', markeredgewidth=1)
        
        ax1.set_title('Total Observations per Quarter', fontsize=13, fontweight='bold', color='#003366')
        ax1.set_xlabel('Quarter', fontsize=11, color='#003366', fontweight='semibold')
        ax1.set_ylabel('Number of Observations', fontsize=11, color='#003366', fontweight='semibold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add average line
        avg_obs = coverage_df['total_observations'].mean()
        ax1.axhline(y=avg_obs, color='#FF6633', linestyle='--', alpha=0.7, linewidth=2)
        ax1.text(coverage_df['quarter_date'].iloc[-5], avg_obs + 20, f'Avg: {avg_obs:.0f}', 
                color='#FF6633', fontweight='bold')
        
        # Plot 2: Complete cases coverage percentage
        ax2.plot(coverage_df['quarter_date'], coverage_df['complete_cases_coverage'], 
                color='#66CCFF', linewidth=3, marker='s', markersize=6, 
                markeredgecolor='white', markeredgewidth=1)
        
        ax2.set_title('Complete Cases Coverage (%)', fontsize=13, fontweight='bold', color='#003366')
        ax2.set_xlabel('Quarter', fontsize=11, color='#003366', fontweight='semibold')
        ax2.set_ylabel('Coverage Percentage (%)', fontsize=11, color='#003366', fontweight='semibold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 100)
        
        # Add average line
        avg_coverage = coverage_df['complete_cases_coverage'].mean()
        ax2.axhline(y=avg_coverage, color='#FF6633', linestyle='--', alpha=0.7, linewidth=2)
        ax2.text(coverage_df['quarter_date'].iloc[-5], avg_coverage + 3, f'Avg: {avg_coverage:.1f}%', 
                color='#FF6633', fontweight='bold')
        
        # Plot 3: Individual feature coverage
        colors = ['#003366', '#66CCFF', '#8C8C8C', '#FF6633', '#00CC66', '#9933FF']
        
        for i, feature in enumerate(self.features):
            coverage_col = f'{feature}_coverage'
            ax3.plot(coverage_df['quarter_date'], coverage_df[coverage_col], 
                    label=feature.replace('_', ' ').replace('ievr', 'IEVR').title(),
                    color=colors[i % len(colors)], linewidth=2, marker='o', markersize=4,
                    alpha=0.8)
        
        ax3.set_title('Individual Feature Coverage (%)', fontsize=13, fontweight='bold', color='#003366')
        ax3.set_xlabel('Quarter', fontsize=11, color='#003366', fontweight='semibold')
        ax3.set_ylabel('Coverage Percentage (%)', fontsize=11, color='#003366', fontweight='semibold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(fontsize=9, loc='lower right')
        ax3.set_ylim(0, 100)
        
        # Plot 4: Target variable coverage with complete cases comparison
        ax4.plot(coverage_df['quarter_date'], coverage_df[f'{self.target}_coverage'], 
                label=f'{self.target.upper()} Coverage', color='#003366', linewidth=3, 
                marker='o', markersize=6, markeredgecolor='white', markeredgewidth=1)
        
        ax4.plot(coverage_df['quarter_date'], coverage_df['complete_cases_coverage'], 
                label='Complete Cases', color='#66CCFF', linewidth=3, 
                marker='s', markersize=6, markeredgecolor='white', markeredgewidth=1, alpha=0.7)
        
        ax4.set_title(f'{self.target.upper()} vs Complete Cases Coverage', fontsize=13, fontweight='bold', color='#003366')
        ax4.set_xlabel('Quarter', fontsize=11, color='#003366', fontweight='semibold')
        ax4.set_ylabel('Coverage Percentage (%)', fontsize=11, color='#003366', fontweight='semibold')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(fontsize=10)
        ax4.set_ylim(0, 100)
        
        # Highlight periods with low coverage
        low_coverage_threshold = 50
        low_coverage_periods = coverage_df[coverage_df['complete_cases_coverage'] < low_coverage_threshold]
        
        if len(low_coverage_periods) > 0:
            for _, period in low_coverage_periods.iterrows():
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.axvline(x=period['quarter_date'], color='red', alpha=0.3, linestyle=':', linewidth=2)
        
        plt.tight_layout()
        
        # Save plot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output_files', 'correlation_coverage_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'data_coverage_time_series.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Coverage time series saved: {output_path}")
        
        # Save SVG version
        output_path_svg = os.path.join(output_dir, 'data_coverage_time_series.svg')
        plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white')
        print(f"✅ SVG version saved: {output_path_svg}")
        
        plt.close()
    
    def save_results(self, correlation_matrix, coverage_df):
        """Save detailed results to CSV files"""
        print(f"\n💾 SAVING RESULTS")
        print("="*20)
        
        # Ensure output directory exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output_files', 'correlation_coverage_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save correlation matrix
        correlation_path = os.path.join(output_dir, 'feature_correlation_matrix.csv')
        correlation_matrix.to_csv(correlation_path)
        print(f"✅ Correlation matrix saved: {correlation_path}")
        
        # Save coverage data
        coverage_path = os.path.join(output_dir, 'data_coverage_by_quarter.csv')
        coverage_df.to_csv(coverage_path, index=False)
        print(f"✅ Coverage data saved: {coverage_path}")
        
        # Create summary statistics
        summary_stats = {
            'total_quarters': len(coverage_df),
            'date_range_start': coverage_df['year_quarter'].iloc[0],
            'date_range_end': coverage_df['year_quarter'].iloc[-1],
            'avg_observations_per_quarter': coverage_df['total_observations'].mean(),
            'avg_complete_cases_per_quarter': coverage_df['complete_cases_count'].mean(),
            'avg_complete_case_coverage_pct': coverage_df['complete_cases_coverage'].mean(),
        }
        
        # Add average coverage for each variable
        for var in self.all_variables:
            summary_stats[f'avg_{var}_coverage_pct'] = coverage_df[f'{var}_coverage'].mean()
        
        # Add correlation with target
        target_correlations = correlation_matrix[self.target].drop(self.target)
        for feature, corr in target_correlations.items():
            summary_stats[f'{feature}_target_correlation'] = corr
        
        summary_df = pd.DataFrame([summary_stats])
        summary_path = os.path.join(output_dir, 'analysis_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Summary statistics saved: {summary_path}")

def main():
    """
    Main function to run correlation and coverage analysis
    """
    try:
        print("📊 CORRELATION & COVERAGE ANALYSIS")
        print("="*50)
        print("Features: IEVR + normative_iv_rv_ratio + IV_RATIO + SMIRK + vol_hl21 + z_score_momentum + dispersion_pct_ibes")
        print("Target: REVR")
        print("="*50)
        
        # Initialize analyzer
        analyzer = CorrelationCoverageAnalysis()
        
        # Load data
        if not analyzer.load_data():
            return
        
        # Analyze correlations
        correlation_matrix = analyzer.analyze_correlations()
        
        # Analyze data coverage
        coverage_df = analyzer.analyze_data_coverage()
        
        # Create visualizations
        analyzer.create_correlation_visualization(correlation_matrix)
        analyzer.create_coverage_visualization(coverage_df)
        
        # Save results
        analyzer.save_results(correlation_matrix, coverage_df)
        
        print(f"\n🎉 CORRELATION & COVERAGE ANALYSIS COMPLETED!")
        print(f"Key outputs:")
        print(f"  • feature_correlation_matrix.png - Correlation heatmap")
        print(f"  • data_coverage_time_series.png - Coverage time series")
        print(f"  • feature_correlation_matrix.csv - Correlation data")
        print(f"  • data_coverage_by_quarter.csv - Coverage data")
        print(f"  • analysis_summary.csv - Summary statistics")
        
    except Exception as e:
        print(f"❌ Error in correlation & coverage analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
