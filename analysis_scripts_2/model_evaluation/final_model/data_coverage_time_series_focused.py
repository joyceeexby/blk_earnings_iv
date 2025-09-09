#!/usr/bin/env python3
"""
Data Coverage Time Series Analysis - Focused Version
Generate data coverage time series for 7 features relative to top500 liquidity universe
Focus only on individual feature coverage percentage plot
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataCoverageAnalysis:
    """
    Analyze data coverage for the 7-feature model relative to top500 liquidity universe
    """
    
    def __init__(self):
        self.model_df = None
        self.top500_df = None
        self.features = [
            'ievr', 'normative_iv_rv_ratio', 'IV_RATIO', 
            'SMIRK', 'vol_hl21', 'z_score_momentum', 'dispersion_pct_ibes'
        ]
        
    def load_data(self):
        """Load and prepare both datasets"""
        print("📊 LOADING DATASETS FOR COVERAGE ANALYSIS")
        print("="*50)
        
        try:
            # Load model dataset
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, 'data_files')
            
            model_file_path = os.path.join(data_dir, 'model_df.csv')
            top500_file_path = os.path.join(data_dir, 'top500_liquidity_2005_2023.csv')
            
            if not os.path.exists(model_file_path):
                raise FileNotFoundError(f"❌ Model dataset file not found: {model_file_path}")
            if not os.path.exists(top500_file_path):
                raise FileNotFoundError(f"❌ Top500 liquidity file not found: {top500_file_path}")
            
            # Load model data
            self.model_df = pd.read_csv(model_file_path)
            print(f"✅ Loaded model dataset: {len(self.model_df):,} observations")
            
            # Load top500 liquidity data
            self.top500_df = pd.read_csv(top500_file_path)
            print(f"✅ Loaded top500 liquidity dataset: {len(self.top500_df):,} observations")
            
            # Convert earnings_date to datetime for model data
            self.model_df['earnings_date'] = pd.to_datetime(self.model_df['earnings_date'])
            self.model_df['year'] = self.model_df['earnings_date'].dt.year
            self.model_df['quarter'] = self.model_df['earnings_date'].dt.quarter
            
            # Create year-quarter column for model data
            self.model_df['year_quarter'] = self.model_df['year'].astype(str) + 'Q' + self.model_df['quarter'].astype(str)
            self.model_df['quarter_date'] = pd.to_datetime(self.model_df['year'].astype(str) + '-' + 
                                                         (self.model_df['quarter'] * 3).astype(str) + '-01')
            
            # Prepare top500 data
            self.top500_df['year_quarter'] = self.top500_df['year'].astype(str) + 'Q' + self.top500_df['quarter'].astype(str)
            self.top500_df['quarter_date'] = pd.to_datetime(self.top500_df['year'].astype(str) + '-' + 
                                                          (self.top500_df['quarter'] * 3).astype(str) + '-01')
            
            print(f"📅 Model data range: {self.model_df['year'].min()} - {self.model_df['year'].max()}")
            print(f"📅 Top500 data range: {self.top500_df['year'].min()} - {self.top500_df['year'].max()}")
            
            # Calculate normative_iv_rv_ratio if missing
            if 'normative_iv_rv_ratio' not in self.model_df.columns:
                if 'avg_pre' in self.model_df.columns and 'normative_realized_vol' in self.model_df.columns:
                    print("🔧 Calculating normative_iv_rv_ratio...")
                    self.model_df['normative_iv_rv_ratio'] = self.model_df['avg_pre'] / self.model_df['normative_realized_vol']
                    self.model_df['normative_iv_rv_ratio'] = self.model_df['normative_iv_rv_ratio'].replace([np.inf, -np.inf], np.nan)
                    print(f"✅ Created normative_iv_rv_ratio from avg_pre / normative_realized_vol")
                else:
                    print("❌ Cannot calculate normative_iv_rv_ratio - missing avg_pre or normative_realized_vol")
            
            # Check feature availability
            missing_features = [f for f in self.features if f not in self.model_df.columns]
            if missing_features:
                print(f"❌ Missing features: {missing_features}")
                # Remove missing features from analysis
                self.features = [f for f in self.features if f in self.model_df.columns]
                print(f"✅ Analyzing {len(self.features)} available features: {self.features}")
            else:
                print(f"✅ All {len(self.features)} features available")
                
            return True
            
        except FileNotFoundError as e:
            print(str(e))
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def calculate_coverage_by_quarter(self):
        """Calculate coverage for each feature by quarter relative to top500 universe"""
        print(f"\n📊 CALCULATING COVERAGE BY QUARTER")
        print("="*40)
        
        # Filter data to 2005 Q1 - 2022 Q4 time range (remove the final dip)
        print("🕒 Filtering data to 2005 Q1 - 2022 Q4 time range...")
        
        # Filter top500 data
        top500_filtered = self.top500_df[
            ((self.top500_df['year'] == 2005) & (self.top500_df['quarter'] >= 1)) |
            ((self.top500_df['year'] > 2005) & (self.top500_df['year'] < 2022)) |
            ((self.top500_df['year'] == 2022) & (self.top500_df['quarter'] <= 4))
        ].copy()
        
        # Filter model data
        self.model_df = self.model_df[
            ((self.model_df['year'] == 2005) & (self.model_df['quarter'] >= 1)) |
            ((self.model_df['year'] > 2005) & (self.model_df['year'] < 2022)) |
            ((self.model_df['year'] == 2022) & (self.model_df['quarter'] <= 4))
        ].copy()
        
        print(f"✅ Filtered top500 data: {len(top500_filtered):,} observations")
        print(f"✅ Filtered model data: {len(self.model_df):,} observations")
        
        # Get all quarters that exist in filtered top500 data
        top500_quarters = top500_filtered.groupby(['year', 'quarter']).size().reset_index(name='top500_count')
        
        coverage_results = []
        
        for _, quarter_info in top500_quarters.iterrows():
            year = quarter_info['year']
            quarter = quarter_info['quarter']
            top500_count = quarter_info['top500_count']
            
            # Create quarter identifiers
            year_quarter = f"{year}Q{quarter}"
            quarter_date = pd.to_datetime(f"{year}-{quarter * 3:02d}-01")
            
            # Get model data for this quarter
            model_quarter_data = self.model_df[
                (self.model_df['year'] == year) & 
                (self.model_df['quarter'] == quarter)
            ]
            
            # Initialize coverage info
            coverage_info = {
                'year': year,
                'quarter': quarter,
                'year_quarter': year_quarter,
                'quarter_date': quarter_date,
                'top500_observations': top500_count,
                'model_observations': len(model_quarter_data)
            }
            
            # Calculate coverage for each feature
            for feature in self.features:
                if feature in model_quarter_data.columns:
                    valid_count = model_quarter_data[feature].notna().sum()
                    coverage_pct = (valid_count / top500_count) * 100 if top500_count > 0 else 0
                    coverage_info[f'{feature}_count'] = valid_count
                    coverage_info[f'{feature}_coverage'] = coverage_pct
                else:
                    coverage_info[f'{feature}_count'] = 0
                    coverage_info[f'{feature}_coverage'] = 0.0
            
            # Calculate complete case coverage (all features available)
            if len(self.features) > 0:
                complete_cases = model_quarter_data[self.features].dropna()
                coverage_info['complete_cases_count'] = len(complete_cases)
                coverage_info['complete_cases_coverage'] = (len(complete_cases) / top500_count) * 100 if top500_count > 0 else 0
            else:
                coverage_info['complete_cases_count'] = 0
                coverage_info['complete_cases_coverage'] = 0.0
            
            coverage_results.append(coverage_info)
        
        coverage_df = pd.DataFrame(coverage_results)
        coverage_df = coverage_df.sort_values('quarter_date').reset_index(drop=True)
        
        # Print summary statistics
        print(f"COVERAGE ANALYSIS SUMMARY:")
        print("-" * 30)
        print(f"Total quarters analyzed: {len(coverage_df)}")
        print(f"Date range: {coverage_df['year_quarter'].iloc[0]} to {coverage_df['year_quarter'].iloc[-1]}")
        print(f"Average top500 observations per quarter: {coverage_df['top500_observations'].mean():.1f}")
        print(f"Average model observations per quarter: {coverage_df['model_observations'].mean():.1f}")
        print(f"Average complete case coverage: {coverage_df['complete_cases_coverage'].mean():.1f}%")
        
        print(f"\nAVERAGE COVERAGE BY FEATURE:")
        print("-" * 35)
        for feature in self.features:
            if f'{feature}_coverage' in coverage_df.columns:
                avg_coverage = coverage_df[f'{feature}_coverage'].mean()
                print(f"{feature:25s}: {avg_coverage:6.1f}%")
        
        return coverage_df
    
    def create_focused_coverage_plot(self, coverage_df):
        """Create focused individual feature coverage plot"""
        print(f"\n📊 CREATING FOCUSED COVERAGE VISUALIZATION")
        print("="*45)
        
        # Set professional styling
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True
        })
        
        # Create single focused plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.patch.set_facecolor('white')
        
        # Since most features have identical coverage, plot just one representative line
        # Use IEVR as the representative feature (first in the list)
        representative_feature = 'ievr'
        coverage_col = f'{representative_feature}_coverage'
        
        ax.plot(coverage_df['quarter_date'], coverage_df[coverage_col], 
               label='Feature Coverage',
               color='#003366', 
               linewidth=3, 
               marker='o', 
               markersize=6,
               alpha=0.9)
        
        # Customize plot
        ax.set_title('Feature Coverage Over Time\n(Relative to Top 500 Liquidity Universe)', 
                    fontsize=18, fontweight='bold', color='#003366', pad=20)
        ax.set_xlabel('Quarter', fontsize=14, color='#003366', fontweight='semibold')
        ax.set_ylabel('Coverage Percentage (%)', fontsize=14, color='#003366', fontweight='semibold')
        
        # Set grid and styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_ylim(0, 105)  # Give some room at the top
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
        
        # Position legend (simpler since we only have one line)
        ax.legend(fontsize=12, loc='lower right', frameon=True, 
                 fancybox=True, shadow=True, framealpha=0.95)
        
        # Add horizontal reference lines
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=75, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add text annotations for reference lines
        ax.text(coverage_df['quarter_date'].iloc[-1], 52, '50%', 
               color='red', fontweight='bold', fontsize=10, ha='right')
        ax.text(coverage_df['quarter_date'].iloc[-1], 77, '75%', 
               color='orange', fontweight='bold', fontsize=10, ha='right')
        ax.text(coverage_df['quarter_date'].iloc[-1], 92, '90%', 
               color='green', fontweight='bold', fontsize=10, ha='right')
        
        # Highlight periods with very low coverage (< 25%)
        low_coverage_threshold = 25
        representative_feature = 'ievr'
        coverage_col = f'{representative_feature}_coverage'
        low_coverage_periods = coverage_df[coverage_df[coverage_col] < low_coverage_threshold]
        if len(low_coverage_periods) > 0:
            for _, period in low_coverage_periods.iterrows():
                ax.axvline(x=period['quarter_date'], color='red', alpha=0.2, linestyle=':', linewidth=1)
        
        plt.tight_layout()
        
        # Save plot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output_files', 'correlation_coverage_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'feature_coverage_time_series.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Feature coverage plot saved: {output_path}")
        
        # Save SVG version
        output_path_svg = os.path.join(output_dir, 'feature_coverage_time_series.svg')
        plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white')
        print(f"✅ SVG version saved: {output_path_svg}")
        
        plt.close()
    
    def save_coverage_results(self, coverage_df):
        """Save detailed coverage results to CSV"""
        print(f"\n💾 SAVING COVERAGE RESULTS")
        print("="*30)
        
        # Ensure output directory exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output_files', 'correlation_coverage_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed coverage data
        coverage_path = os.path.join(output_dir, 'feature_coverage_by_quarter_top500.csv')
        coverage_df.to_csv(coverage_path, index=False)
        print(f"✅ Coverage data saved: {coverage_path}")
        
        # Create summary statistics
        summary_stats = {
            'total_quarters': len(coverage_df),
            'date_range_start': coverage_df['year_quarter'].iloc[0],
            'date_range_end': coverage_df['year_quarter'].iloc[-1],
            'avg_top500_observations_per_quarter': coverage_df['top500_observations'].mean(),
            'avg_model_observations_per_quarter': coverage_df['model_observations'].mean(),
            'avg_complete_case_coverage_pct': coverage_df['complete_cases_coverage'].mean(),
        }
        
        # Add average coverage for each feature
        for feature in self.features:
            coverage_col = f'{feature}_coverage'
            if coverage_col in coverage_df.columns:
                summary_stats[f'avg_{feature}_coverage_pct'] = coverage_df[coverage_col].mean()
        
        summary_df = pd.DataFrame([summary_stats])
        summary_path = os.path.join(output_dir, 'coverage_analysis_summary_top500.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Summary statistics saved: {summary_path}")

def main():
    """
    Main function to run focused data coverage analysis
    """
    try:
        print("📊 DATA COVERAGE ANALYSIS - FOCUSED VERSION")
        print("="*55)
        print("Features: IEVR + normative_iv_rv_ratio + IV_RATIO + SMIRK + vol_hl21 + z_score_momentum + dispersion_pct_ibes")
        print("Universe: Top 500 Liquidity (2005 Q1 - 2022 Q4)")
        print("="*55)
        
        # Initialize analyzer
        analyzer = DataCoverageAnalysis()
        
        # Load data
        if not analyzer.load_data():
            return
        
        # Calculate coverage by quarter
        coverage_df = analyzer.calculate_coverage_by_quarter()
        
        # Create focused visualization
        analyzer.create_focused_coverage_plot(coverage_df)
        
        # Save results
        analyzer.save_coverage_results(coverage_df)
        
        print(f"\n🎉 FOCUSED COVERAGE ANALYSIS COMPLETED!")
        print(f"Key outputs:")
        print(f"  • feature_coverage_time_series.png - Feature coverage time series plot")
        print(f"  • feature_coverage_by_quarter_top500.csv - Detailed coverage data")
        print(f"  • coverage_analysis_summary_top500.csv - Summary statistics")
        
    except Exception as e:
        print(f"❌ Error in coverage analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
