"""
EPS Estimate Momentum Analysis
Calculates and plots momentum of mean EPS estimates across the universe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
import wrds
import warnings
warnings.filterwarnings('ignore')

def calculate_eps_momentum():
    """
    Calculate momentum of mean EPS estimates across the universe
    """
    print("EPS ESTIMATE MOMENTUM ANALYSIS")
    print("="*50)
    
    # Connect to WRDS
    try:
        db = wrds.Connection()
        print("✓ Connected to WRDS")
    except Exception as e:
        print(f"✗ Error connecting to WRDS: {e}")
        return
    
    # Get S&P 500 constituents
    print("Getting S&P 500 constituents...")
    sp500_query = """
        SELECT *
        FROM comp_na_daily_all.wrds_idx_cst_current t
        WHERE indexname = 'S&P 500'
    """
    sp500_constituents = db.raw_sql(sp500_query)
    print(f"Retrieved {len(sp500_constituents)} S&P 500 constituents")
    
    # Get CUSIPs for IBES
    cusip_list = sp500_constituents['cusip'].dropna().unique().tolist()
    cusip_list = [cusip[:8] for cusip in cusip_list if len(cusip) >= 8]
    
    print(f"Processing {len(cusip_list)} CUSIPs for IBES data")
    
    # Get IBES estimates for the universe
    print("Getting IBES estimates...")
    cusip_list_str = ', '.join(f"'{cusip}'" for cusip in cusip_list[:100])  # Limit for performance
    
    ibes_query = f"""
        SELECT ticker, cusip, statpers, fpedats, anndats_act,
               meanest, stdev, numest, fpi
        FROM tr_ibes.statsum_epsus
        WHERE cusip IN ({cusip_list_str})
          AND statpers BETWEEN '2015-01-01' AND '2023-12-31'
          AND measure = 'EPS'
          AND fiscalp = 'QTR'
          AND meanest IS NOT NULL
    """
    
    ibes_estimates = db.raw_sql(ibes_query, date_cols=['statpers', 'fpedats', 'anndats_act'])
    print(f"Retrieved {len(ibes_estimates)} IBES estimates")
    
    # Process the data
    ibes_estimates['statpers'] = pd.to_datetime(ibes_estimates['statpers'])
    ibes_estimates['fpedats'] = pd.to_datetime(ibes_estimates['fpedats'])
    
    # Filter for one-quarter-ahead estimates
    mask_future = (ibes_estimates['fpedats'] > ibes_estimates['statpers'])
    ibes_future = ibes_estimates[mask_future].copy()
    
    # Get the first estimate for each CUSIP-date combination
    ibes_filtered = (
        ibes_future.sort_values(['cusip', 'statpers', 'fpedats'])
        .groupby(['cusip', 'statpers'], as_index=False)
        .first()
    )
    
    print(f"Filtered to {len(ibes_filtered)} one-quarter-ahead estimates")
    
    # Calculate universe-wide mean EPS estimates by date
    print("Calculating universe-wide mean EPS estimates...")
    universe_means = ibes_filtered.groupby('statpers')['meanest'].agg(['mean', 'std', 'count']).reset_index()
    universe_means.columns = ['date', 'mean_eps', 'std_eps', 'num_estimates']
    
    # Calculate momentum measures
    print("Calculating momentum measures...")
    
    # 1. Simple momentum (change over time)
    universe_means['momentum_1m'] = universe_means['mean_eps'].pct_change(periods=20)  # ~1 month
    universe_means['momentum_3m'] = universe_means['mean_eps'].pct_change(periods=60)  # ~3 months
    universe_means['momentum_6m'] = universe_means['mean_eps'].pct_change(periods=120)  # ~6 months
    
    # 2. Rolling momentum (smoothed)
    universe_means['rolling_momentum_3m'] = universe_means['mean_eps'].rolling(window=60).mean().pct_change(periods=20)
    
    # 3. Z-score momentum (relative to recent history)
    universe_means['z_score_momentum'] = (
        (universe_means['mean_eps'] - universe_means['mean_eps'].rolling(window=252).mean()) / 
        universe_means['mean_eps'].rolling(window=252).std()
    )
    
    # Remove NaN values
    universe_means = universe_means.dropna()
    
    print(f"Final dataset: {len(universe_means)} observations")
    print(f"Date range: {universe_means['date'].min()} to {universe_means['date'].max()}")
    
    return universe_means, ibes_filtered

def create_momentum_plots(universe_means):
    """
    Create comprehensive plots of EPS momentum
    """
    print("\nCreating momentum plots...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('EPS Estimate Momentum Analysis: S&P 500 Universe', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean EPS estimates over time
    ax1 = axes[0, 0]
    ax1.plot(universe_means['date'], universe_means['mean_eps'], linewidth=2, color='blue', alpha=0.8)
    ax1.fill_between(universe_means['date'], 
                     universe_means['mean_eps'] - universe_means['std_eps'],
                     universe_means['mean_eps'] + universe_means['std_eps'],
                     alpha=0.3, color='blue')
    ax1.set_title('Mean EPS Estimates Over Time', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Mean EPS Estimate')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Momentum measures
    ax2 = axes[0, 1]
    ax2.plot(universe_means['date'], universe_means['momentum_1m'] * 100, 
             label='1-Month Momentum', linewidth=2, alpha=0.8)
    ax2.plot(universe_means['date'], universe_means['momentum_3m'] * 100, 
             label='3-Month Momentum', linewidth=2, alpha=0.8)
    ax2.plot(universe_means['date'], universe_means['momentum_6m'] * 100, 
             label='6-Month Momentum', linewidth=2, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('EPS Estimate Momentum (% Change)', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Momentum (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling momentum
    ax3 = axes[1, 0]
    ax3.plot(universe_means['date'], universe_means['rolling_momentum_3m'] * 100, 
             color='green', linewidth=2, alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Rolling 3-Month Momentum (Smoothed)', fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Rolling Momentum (%)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Z-score momentum
    ax4 = axes[1, 1]
    ax4.plot(universe_means['date'], universe_means['z_score_momentum'], 
             color='red', linewidth=2, alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax4.axhline(y=-1, color='gray', linestyle=':', alpha=0.5)
    ax4.set_title('Z-Score Momentum (Relative to 1-Year History)', fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Z-Score')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'analysis_scripts/output_files/eps_momentum_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_file}")
    
    # Show the plot
    plt.show()
    
    return fig

def create_summary_statistics(universe_means):
    """
    Create summary statistics for the momentum measures
    """
    print("\n" + "="*50)
    print("EPS MOMENTUM SUMMARY STATISTICS")
    print("="*50)
    
    # Calculate summary stats
    summary_stats = universe_means[['momentum_1m', 'momentum_3m', 'momentum_6m', 
                                   'rolling_momentum_3m', 'z_score_momentum']].describe()
    
    print("\nMomentum Statistics:")
    print(summary_stats)
    
    # Calculate correlation matrix
    print("\nCorrelation Matrix:")
    correlation_matrix = universe_means[['momentum_1m', 'momentum_3m', 'momentum_6m', 
                                        'rolling_momentum_3m', 'z_score_momentum']].corr()
    print(correlation_matrix)
    
    # Save summary statistics
    summary_file = 'analysis_scripts/data_files/eps_momentum_summary.csv'
    summary_stats.to_csv(summary_file)
    print(f"\n✓ Summary statistics saved to {summary_file}")
    
    return summary_stats, correlation_matrix

def main():
    """
    Main function to run the EPS momentum analysis
    """
    print("Starting EPS Estimate Momentum Analysis...")
    
    # Calculate momentum
    universe_means, ibes_data = calculate_eps_momentum()
    
    if universe_means is not None and len(universe_means) > 0:
        # Create plots
        fig = create_momentum_plots(universe_means)
        
        # Create summary statistics
        summary_stats, correlation_matrix = create_summary_statistics(universe_means)
        
        # Save the momentum data
        momentum_file = 'analysis_scripts/data_files/eps_momentum_data.csv'
        universe_means.to_csv(momentum_file, index=False)
        print(f"✓ Momentum data saved to {momentum_file}")
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print("✓ EPS momentum calculated and plotted")
        print("✓ Summary statistics generated")
        print("✓ Data files saved")
        
    else:
        print("✗ No data available for analysis")

if __name__ == "__main__":
    main() 