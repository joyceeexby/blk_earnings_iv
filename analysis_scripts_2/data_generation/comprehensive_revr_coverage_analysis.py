#!/usr/bin/env python3
"""
Comprehensive REVR Coverage Analysis
Analyze REVR coverage using the bulk_revr_comprehensive_st_mt_static_cusip_comparison.csv dataset
Track quarterly REVR observations out of 500 possible
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveREVRCoverageAnalysis:
    """
    Analyze comprehensive REVR coverage calculating observations per quarter divided by 500
    """
    
    def __init__(self, universe_size=500):
        self.df = None
        self.universe_size = universe_size
        self.quarterly_coverage = None
        
    def load_comprehensive_data(self):
        """Load and prepare the comprehensive REVR dataset"""
        print("📊 LOADING COMPREHENSIVE REVR DATASET")
        print("="*45)
        
        try:
            # Load the comprehensive dataset
            self.df = pd.read_csv('data_files/bulk_revr_comprehensive_st_mt_static_cusip_comparison.csv')
            print(f"✅ Loaded comprehensive REVR dataset: {len(self.df):,} observations")
            
            # Convert earnings_date to datetime
            self.df['earnings_date'] = pd.to_datetime(self.df['earnings_date'])
            
            # Create year_quarter string for grouping (quarter is already Q1, Q2, etc.)
            self.df['year_quarter'] = self.df['year'].astype(str) + '-' + self.df['quarter'].astype(str)
            
            print(f"📅 Date range: {self.df['year'].min()} - {self.df['year'].max()}")
            print(f"📈 Total REVR observations: {len(self.df):,}")
            print(f"🏢 Unique companies: {self.df['ticker'].nunique():,}")
            
            # Show data quality info
            valid_revr = self.df['revr'].notna().sum()
            print(f"📊 Valid REVR values: {valid_revr:,} ({valid_revr/len(self.df)*100:.1f}%)")
            
            # Show quarters coverage
            quarters_covered = len(self.df['year_quarter'].unique())
            print(f"📅 Quarters covered: {quarters_covered}")
            
            return True
            
        except FileNotFoundError:
            print("❌ Comprehensive REVR dataset file not found!")
            return False
    
    def analyze_quarterly_coverage(self):
        """Analyze quarterly REVR coverage - count observations per quarter / 500"""
        print(f"\n📈 ANALYZING COMPREHENSIVE REVR QUARTERLY COVERAGE")
        print("="*55)
        
        # Analyze by quarter - simple count of valid REVR observations
        quarterly_coverage = []
        
        # Get all unique quarters in the data, sorted
        all_quarters = sorted(self.df['year_quarter'].unique())
        
        for quarter in all_quarters:
            quarter_data = self.df[self.df['year_quarter'] == quarter]
            
            # Count valid REVR observations in this quarter
            valid_revr_count = quarter_data['revr'].notna().sum()
            
            # Calculate coverage rate as observations / 500
            coverage_rate = (valid_revr_count / self.universe_size) * 100
            
            # Extract year and quarter for plotting
            year = int(quarter.split('-')[0])
            q_str = quarter.split('-')[1]  # This will be 'Q1', 'Q2', etc.
            q = int(q_str[1])  # Extract the number from 'Q1' -> 1
            
            # Count unique companies in this quarter
            unique_companies = quarter_data['ticker'].nunique()
            
            quarterly_coverage.append({
                'year_quarter': quarter,
                'year': year,
                'quarter': q,
                'quarter_decimal': year + (q - 1) * 0.25,  # For smooth plotting
                'valid_revr_count': valid_revr_count,
                'coverage_rate': coverage_rate,
                'unique_companies': unique_companies,
                'total_observations': len(quarter_data)
            })
        
        self.quarterly_coverage = pd.DataFrame(quarterly_coverage)
        
        # Print summary statistics
        self._print_coverage_summary()
        
        return self.quarterly_coverage
    
    def _print_coverage_summary(self):
        """Print coverage summary statistics"""
        print(f"\nCOMPREHENSIVE REVR COVERAGE SUMMARY:")
        print("-" * 50)
        
        if self.quarterly_coverage is None or len(self.quarterly_coverage) == 0:
            print("❌ No coverage data available")
            return
        
        # Overall statistics
        avg_coverage = self.quarterly_coverage['coverage_rate'].mean()
        max_coverage = self.quarterly_coverage['coverage_rate'].max()
        min_coverage = self.quarterly_coverage['coverage_rate'].min()
        final_coverage = self.quarterly_coverage['coverage_rate'].iloc[-1]
        final_count = self.quarterly_coverage['valid_revr_count'].iloc[-1]
        
        print(f"📊 Overall Statistics:")
        print(f"  • Reference size: {self.universe_size}")
        print(f"  • Quarters analyzed: {len(self.quarterly_coverage)}")
        print(f"  • Average coverage: {avg_coverage:.1f}%")
        print(f"  • Coverage range: {min_coverage:.1f}% - {max_coverage:.1f}%")
        print(f"  • Latest quarter coverage: {final_coverage:.1f}% ({final_count} observations)")
        
        # Growth analysis
        early_period = self.quarterly_coverage[self.quarterly_coverage['year'] <= 2010]
        recent_period = self.quarterly_coverage[self.quarterly_coverage['year'] >= 2020]
        
        if len(early_period) > 0 and len(recent_period) > 0:
            early_avg_count = early_period['valid_revr_count'].mean()
            recent_avg_count = recent_period['valid_revr_count'].mean()
            early_avg_rate = early_period['coverage_rate'].mean()
            recent_avg_rate = recent_period['coverage_rate'].mean()
            
            print(f"\n📈 Growth Analysis:")
            print(f"  • Early period (≤2010): {early_avg_count:.0f} observations ({early_avg_rate:.1f}% coverage)")
            print(f"  • Recent period (≥2020): {recent_avg_count:.0f} observations ({recent_avg_rate:.1f}% coverage)")
            print(f"  • Observation count growth: {(recent_avg_count/early_avg_count-1)*100:+.1f}%")
            print(f"  • Coverage improvement: {recent_avg_rate-early_avg_rate:+.1f} percentage points")
        
        # Peak performance
        peak_quarter = self.quarterly_coverage.loc[self.quarterly_coverage['coverage_rate'].idxmax()]
        print(f"\n🏆 Peak Performance:")
        print(f"  • Best quarter: {peak_quarter['year_quarter']}")
        print(f"  • Peak coverage: {peak_quarter['coverage_rate']:.1f}% ({peak_quarter['valid_revr_count']} observations)")
        print(f"  • Companies covered: {peak_quarter['unique_companies']}")
    
    def create_comprehensive_coverage_visualization(self):
        """Create comprehensive REVR coverage visualization matching IEVR chart style"""
        print(f"\n📊 CREATING COMPREHENSIVE REVR COVERAGE VISUALIZATION")
        print("="*60)
        
        if self.quarterly_coverage is None or len(self.quarterly_coverage) == 0:
            print("❌ No coverage data available")
            return
        
        # Set professional styling to match the reference chart exactly
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'axes.linewidth': 1.0,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': False,  # Turn off grid to match reference
            'figure.facecolor': 'white'
        })
        
        # Create the main visualization
        fig, ax1 = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        
        # Prepare data for plotting
        years = self.quarterly_coverage['quarter_decimal'].values
        valid_counts = self.quarterly_coverage['valid_revr_count'].values
        coverage_rates = self.quarterly_coverage['coverage_rate'].values
        
        # Primary y-axis: Valid REVR Count (blue line, matching reference)
        color1 = '#1f4e79'  # Deep blue similar to reference
        line1 = ax1.plot(years, valid_counts, 
                        color=color1, linewidth=2.5, 
                        label='Valid REVR Count', zorder=3)
        
        ax1.set_xlabel('')  # Remove x-axis label to match reference
        ax1.set_ylabel('Valid REVR Count', fontsize=12, fontweight='normal', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1, colors=color1)
        ax1.set_ylim(0, max(valid_counts) * 1.1)
        
        # Secondary y-axis: Coverage Rate (orange line, matching reference)
        ax2 = ax1.twinx()
        color2 = '#c55a11'  # Orange similar to reference
        line2 = ax2.plot(years, coverage_rates, 
                        color=color2, linewidth=2.5, 
                        label='Coverage Rate', zorder=3)
        
        ax2.set_ylabel('Coverage Rate', fontsize=12, fontweight='normal', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2, colors=color2)
        ax2.set_ylim(0, 100)
        
        # Format y-axis for percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # Set x-axis range and formatting to match reference style
        ax1.set_xlim(years.min() - 0.5, years.max() + 0.5)
        ax1.set_xticks(range(int(years.min()), int(years.max()) + 1, 2))
        
        # Add final values as text annotations (matching reference style)
        final_count = valid_counts[-1]
        final_rate = coverage_rates[-1]
        final_year = years[-1]
        
        # Annotate final count (positioned like reference)
        ax1.text(final_year + 0.3, final_count + max(valid_counts)*0.02, f'{int(final_count)}', 
                fontsize=12, fontweight='bold', color=color1,
                ha='left', va='bottom')
        
        # Annotate final rate (positioned like reference)
        ax2.text(final_year + 0.3, final_rate + 2, f'{final_rate:.0f}%', 
                fontsize=12, fontweight='bold', color=color2,
                ha='left', va='bottom')
        
        # Create legend (positioned like reference)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, 
                  loc='upper left', frameon=False, fontsize=12)
        
        # Remove all spines and grids to match clean reference style
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_linewidth(0.8)
        ax1.spines['left'].set_linewidth(0.8)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_linewidth(0.8)
        
        # Set background color
        ax1.set_facecolor('white')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = 'output_files/comprehensive_revr_coverage.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✅ Comprehensive REVR coverage visualization saved: {output_path}")
        
        # Save SVG version
        output_path_svg = 'output_files/comprehensive_revr_coverage.svg'
        plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✅ SVG version saved: {output_path_svg}")
        
        plt.close()
    
    def create_detailed_analysis_dashboard(self):
        """Create detailed dashboard for comprehensive REVR analysis"""
        print(f"\n📊 CREATING DETAILED ANALYSIS DASHBOARD")
        print("="*40)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('white')
        fig.suptitle('Comprehensive REVR Coverage Analysis Dashboard\nQuarterly REVR Observations Tracking & Growth Analysis', 
                     fontsize=16, fontweight='bold', color='#333333', y=0.95)
        
        # Plot 1: Coverage Over Time (Main Chart)
        years = self.quarterly_coverage['quarter_decimal'].values
        coverage_rates = self.quarterly_coverage['coverage_rate'].values
        observation_counts = self.quarterly_coverage['valid_revr_count'].values
        
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(years, observation_counts, color='#1f4e79', linewidth=3, marker='o', markersize=3, label='REVR Count')
        line2 = ax1_twin.plot(years, coverage_rates, color='#c55a11', linewidth=3, marker='s', markersize=3, label='Coverage Rate')
        
        ax1.set_title('Comprehensive REVR Coverage Growth', fontweight='bold', color='#333333')
        ax1.set_ylabel('Valid REVR Count', fontweight='semibold', color='#1f4e79')
        ax1_twin.set_ylabel('Coverage Rate (%)', fontweight='semibold', color='#c55a11')
        ax1.tick_params(axis='y', labelcolor='#1f4e79')
        ax1_twin.tick_params(axis='y', labelcolor='#c55a11')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coverage Distribution
        ax2.hist(coverage_rates, bins=15, alpha=0.7, color='#17becf', edgecolor='white', linewidth=1)
        ax2.axvline(coverage_rates.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {coverage_rates.mean():.1f}%')
        ax2.axvline(np.median(coverage_rates), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(coverage_rates):.1f}%')
        
        ax2.set_title('Quarterly Coverage Distribution', fontweight='bold', color='#333333')
        ax2.set_xlabel('Coverage Rate (%)', fontweight='semibold')
        ax2.set_ylabel('Frequency', fontweight='semibold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Annual Trends
        annual_stats = self.quarterly_coverage.groupby('year').agg({
            'valid_revr_count': 'mean',
            'coverage_rate': 'mean',
            'unique_companies': 'mean'
        }).reset_index()
        
        bars = ax3.bar(annual_stats['year'], annual_stats['valid_revr_count'], 
                      alpha=0.7, color='#2ca02c', label='Avg REVR per Quarter')
        
        ax3.set_title('Annual Coverage Trends', fontweight='bold', color='#333333')
        ax3.set_ylabel('Average REVR Observations', fontweight='semibold')
        ax3.set_xlabel('Year', fontweight='semibold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add trendline
        z = np.polyfit(annual_stats['year'], annual_stats['valid_revr_count'], 1)
        p = np.poly1d(z)
        ax3.plot(annual_stats['year'], p(annual_stats['year']), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:+.1f} obs/year')
        ax3.legend()
        
        # Plot 4: Company Coverage
        ax4.plot(annual_stats['year'], annual_stats['unique_companies'], color='#9467bd', linewidth=3, marker='o', markersize=5)
        ax4.set_title('Unique Companies Covered per Year', fontweight='bold', color='#333333')
        ax4.set_ylabel('Average Unique Companies', fontweight='semibold')
        ax4.set_xlabel('Year', fontweight='semibold')
        ax4.grid(True, alpha=0.3)
        
        # Add data labels for recent years
        for i, row in annual_stats.tail(3).iterrows():
            ax4.annotate(f'{row["unique_companies"]:.0f}', 
                        xy=(row['year'], row['unique_companies']), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, color='#333333')
        
        plt.tight_layout()
        
        # Save dashboard
        output_path = 'output_files/comprehensive_revr_dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Comprehensive analysis dashboard saved: {output_path}")
        
        plt.close()
    
    def save_comprehensive_results(self):
        """Save comprehensive coverage analysis results"""
        print(f"\n💾 SAVING COMPREHENSIVE COVERAGE RESULTS")
        print("="*45)
        
        if self.quarterly_coverage is not None:
            # Save quarterly coverage data
            quarterly_path = 'output_files/comprehensive_revr_quarterly_coverage.csv'
            self.quarterly_coverage.to_csv(quarterly_path, index=False)
            print(f"✅ Quarterly coverage data saved: {quarterly_path}")
            
            # Create annual summary
            annual_summary = self.quarterly_coverage.groupby('year').agg({
                'valid_revr_count': ['mean', 'min', 'max', 'sum'],
                'coverage_rate': ['mean', 'min', 'max'],
                'unique_companies': ['mean', 'min', 'max'],
                'total_observations': 'sum'
            }).reset_index()
            
            # Flatten column names
            annual_summary.columns = ['year', 'avg_revr_count', 'min_revr_count', 'max_revr_count', 'total_revr_count',
                                    'avg_coverage', 'min_coverage', 'max_coverage', 
                                    'avg_companies', 'min_companies', 'max_companies', 'total_observations']
            annual_summary['reference_size'] = self.universe_size
            
            annual_path = 'output_files/comprehensive_revr_annual_summary.csv'
            annual_summary.to_csv(annual_path, index=False)
            print(f"✅ Annual summary saved: {annual_path}")
            
            # Print final statistics
            print(f"\n📊 FINAL COMPREHENSIVE COVERAGE STATISTICS:")
            print("-" * 55)
            
            latest_quarter = self.quarterly_coverage.iloc[-1]
            peak_quarter = self.quarterly_coverage.loc[self.quarterly_coverage['coverage_rate'].idxmax()]
            
            print(f"Latest quarter ({latest_quarter['year_quarter']}):")
            print(f"  • Valid REVR observations: {latest_quarter['valid_revr_count']}")
            print(f"  • Coverage rate: {latest_quarter['coverage_rate']:.1f}%")
            print(f"  • Unique companies: {latest_quarter['unique_companies']}")
            
            print(f"\nPeak performance ({peak_quarter['year_quarter']}):")
            print(f"  • Peak REVR observations: {peak_quarter['valid_revr_count']}")
            print(f"  • Peak coverage rate: {peak_quarter['coverage_rate']:.1f}%")
            print(f"  • Companies at peak: {peak_quarter['unique_companies']}")
        
        print("\n🎉 COMPREHENSIVE REVR COVERAGE ANALYSIS COMPLETED!")
        print("Key outputs:")
        print("  • comprehensive_revr_coverage.png - Main coverage chart (matching IEVR style)")
        print("  • comprehensive_revr_dashboard.png - Detailed analysis dashboard")
        print("  • comprehensive_revr_quarterly_coverage.csv - Quarterly coverage data")
        print("  • comprehensive_revr_annual_summary.csv - Annual summary statistics")

def main():
    """
    Main function to run comprehensive REVR coverage analysis
    """
    try:
        print("📊 COMPREHENSIVE REVR COVERAGE ANALYSIS")
        print("="*50)
        print("Dataset: bulk_revr_comprehensive_st_mt_static_cusip_comparison.csv")
        print("Objective: Track comprehensive REVR observations per quarter / 500")
        print("Output: Professional visualization matching IEVR coverage style")
        print("="*50)
        
        # Initialize analyzer
        analyzer = ComprehensiveREVRCoverageAnalysis(universe_size=500)
        
        # Load comprehensive data
        if not analyzer.load_comprehensive_data():
            return
        
        # Analyze quarterly coverage
        coverage_stats = analyzer.analyze_quarterly_coverage()
        
        if coverage_stats is not None and len(coverage_stats) > 0:
            # Create main visualization
            analyzer.create_comprehensive_coverage_visualization()
            
            # Create detailed dashboard
            analyzer.create_detailed_analysis_dashboard()
            
            # Save results
            analyzer.save_comprehensive_results()
        else:
            print("❌ No coverage statistics generated")
            
    except Exception as e:
        print(f"❌ Error in comprehensive REVR coverage analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
