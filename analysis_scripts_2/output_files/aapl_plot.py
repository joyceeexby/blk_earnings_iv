#!/usr/bin/env python3
"""
AAPL 2023 Q3 Volatility Analysis: ST vs MT Volatility Over Time
Creates the exact plot shown in the image with ST/MT volatility and ratio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AAPL2023Q3VolatilityAnalysis:
    """
    Creates the exact 2023 Q3 AAPL volatility visualization
    """
    
    def __init__(self):
        self.stock_data = None
        self.earnings_date = datetime(2023, 8, 3)  # 2023 Q3 earnings
        self.analysis_date = datetime(2023, 8, 4)  # Day after earnings
        self.start_date = datetime(2023, 5, 1)     # Start of Q3
        self.end_date = datetime(2023, 9, 30)      # End of Q3
        
    def download_aapl_data(self):
        """Download AAPL stock data for 2023 Q3"""
        print("📊 DOWNLOADING AAPL DATA FOR 2023 Q3")
        print("="*60)
        
        try:
            # Download AAPL data
            aapl = yf.download('AAPL', start=self.start_date, end=self.end_date, progress=False)
            
            if aapl.empty:
                print("❌ Failed to download AAPL data")
                return False
            
            print(f"✅ Downloaded raw data: {len(aapl)} rows")
            print(f"📊 Columns: {list(aapl.columns)}")
            
            # Flatten column names if they are multi-level
            if isinstance(aapl.columns, pd.MultiIndex):
                aapl.columns = [col[0] if col[1] == 'AAPL' else f"{col[0]}_{col[1]}" for col in aapl.columns]
                print(f"📊 Flattened columns: {list(aapl.columns)}")
            
            # Reset index to make date a column
            aapl = aapl.reset_index()
            
            # Ensure date column is properly formatted
            aapl['date'] = pd.to_datetime(aapl['Date'])
            
            print(f"📊 Available columns after reset: {list(aapl.columns)}")
            
            # Calculate returns - use 'Close' column
            aapl['returns'] = aapl['Close'].pct_change()
            
            # For simplicity, use returns directly as excess returns
            # (you could subtract SPY returns if you want true excess returns)
            aapl['excess_returns'] = aapl['returns'].copy()
            
            print(f"✅ Calculated returns and excess returns")
            print(f"📊 Data shape before cleaning: {aapl.shape}")
            print(f"📊 Column names: {list(aapl.columns)}")
            
            # Remove NaN values (first row will be NaN due to pct_change)
            aapl = aapl.dropna(subset=['returns', 'excess_returns'])
            
            print(f"📊 Data shape after cleaning: {aapl.shape}")
            
            # Keep only the columns we need
            self.stock_data = aapl[['date', 'Close', 'returns', 'excess_returns']].copy()
            
            print(f"✅ Downloaded {len(self.stock_data)} days of AAPL data")
            print(f"📅 Date range: {self.stock_data['date'].min().strftime('%Y-%m-%d')} to {self.stock_data['date'].max().strftime('%Y-%m-%d')}")
            print(f"📊 Final columns: {list(self.stock_data.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_st_mt_volatility(self):
        """Calculate ST and MT volatility using the REVR methodology"""
        print("\n🔢 CALCULATING ST/MT VOLATILITY")
        print("="*60)
        
        if self.stock_data is None:
            print("❌ No stock data available")
            return False
        
        try:
            # Calculate expanding EWM volatility with two half-lives
            # ST (Short-term): 5-day half-life - captures recent volatility dynamics
            self.stock_data['vol_st'] = self.stock_data['excess_returns'].ewm(
                halflife=5, min_periods=20
            ).std() * np.sqrt(252)  # Annualize
            
            # MT (Medium-term): 21-day half-life - provides stable baseline
            self.stock_data['vol_mt'] = self.stock_data['excess_returns'].ewm(
                halflife=21, min_periods=20
            ).std() * np.sqrt(252)  # Annualize
            
            # Calculate ST/MT ratio
            self.stock_data['st_mt_ratio'] = self.stock_data['vol_st'] / self.stock_data['vol_mt']
            
            # Convert volatility to percentage for better visualization
            self.stock_data['vol_st_pct'] = self.stock_data['vol_st'] * 100
            self.stock_data['vol_mt_pct'] = self.stock_data['vol_mt'] * 100
            
            print(f"✅ Calculated ST volatility (5-day half-life)")
            print(f"✅ Calculated MT volatility (21-day half-life)")
            print(f"✅ Calculated ST/MT ratio")
            
            return True
            
        except Exception as e:
            print(f"❌ Error calculating volatility: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_visualization(self):
        """Create the exact volatility visualization from the image"""
        print("\n🎨 CREATING 2023 Q3 AAPL VOLATILITY VISUALIZATION")
        print("="*60)
        
        if self.stock_data is None:
            print("❌ No data available for visualization")
            return False
        
        try:
            # Set up BlackRock-style plotting
            plt.style.use('default')
            
            # Configure matplotlib for professional appearance
            plt.rcParams.update({
                'font.family': 'Arial',
                'font.size': 11,
                'axes.titlesize': 16,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 18,
                'axes.grid': True,
                'grid.alpha': 0.3,
                'grid.linewidth': 0.5,
                'axes.axisbelow': True,
                'axes.edgecolor': '#CCCCCC',
                'axes.linewidth': 1,
                'xtick.color': '#666666',
                'ytick.color': '#666666'
            })
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            fig.patch.set_facecolor('white')
            
            # BlackRock official color palette
            colors = {
                'st_vol': '#003366',      # BlackRock Navy Blue (ST Volatility)
                'mt_vol': '#DC143C',      # Deep Red (MT Volatility)
                'ratio': '#228B22',       # Forest Green (ST/MT Ratio)
                'earnings_line': '#000000', # Black (Earnings Date)
                'analysis_line': '#4169E1', # Royal Blue (Analysis Date)
                'grid': '#E5E5E5',        # Light gray grid
                'text': '#333333'         # Dark gray text
            }
            
            # Top plot: ST vs MT Volatility with professional styling
            ax1.plot(self.stock_data['date'], self.stock_data['vol_st_pct'], 
                    color=colors['st_vol'], linewidth=2.5, label='ST Volatility (5-day half-life)', 
                    alpha=0.9)
            ax1.plot(self.stock_data['date'], self.stock_data['vol_mt_pct'], 
                    color=colors['mt_vol'], linewidth=2.5, label='MT Volatility (21-day half-life)', 
                    alpha=0.9)
            
            # Add vertical lines for key dates with professional styling
            ax1.axvline(x=self.earnings_date, color=colors['earnings_line'], linestyle='--', 
                       linewidth=2, alpha=0.8, label='Earnings Date')
            ax1.axvline(x=self.analysis_date, color=colors['analysis_line'], linestyle=':', 
                       linewidth=2, alpha=0.8, label='Analysis Date')
            
            # Customize top plot with BlackRock styling
            ax1.set_title('AAPL ST vs MT Volatility Over Time', fontsize=16, fontweight='bold', 
                         pad=20, color=colors['text'])
            ax1.set_ylabel('Volatility (%)', fontsize=12, color=colors['text'])
            ax1.grid(True, alpha=0.3, color=colors['grid'], linewidth=0.5)
            ax1.set_facecolor('#FAFAFA')
            
            # Professional legend styling
            legend1 = ax1.legend(loc='upper left', framealpha=0.95, fontsize=10, 
                               fancybox=True, shadow=True, borderpad=1)
            legend1.get_frame().set_facecolor('white')
            legend1.get_frame().set_edgecolor('#CCCCCC')
            
            # Set y-axis limits for better visualization
            vol_min = min(self.stock_data['vol_st_pct'].min(), self.stock_data['vol_mt_pct'].min())
            vol_max = max(self.stock_data['vol_st_pct'].max(), self.stock_data['vol_mt_pct'].max())
            ax1.set_ylim(vol_min * 0.9, vol_max * 1.1)
            
            # Bottom plot: ST/MT Ratio with professional styling
            ax2.plot(self.stock_data['date'], self.stock_data['st_mt_ratio'], 
                    color=colors['ratio'], linewidth=2.5, label='ST/MT Ratio', alpha=0.9)
            
            # Add horizontal line at ratio = 1 with professional styling
            ax2.axhline(y=1.0, color=colors['mt_vol'], linestyle='--', 
                       linewidth=1.5, alpha=0.7, label='No Change (Ratio=1)')
            
            # Add vertical lines for key dates with professional styling
            ax2.axvline(x=self.earnings_date, color=colors['earnings_line'], linestyle='--', 
                       linewidth=2, alpha=0.8, label='Earnings Date')
            ax2.axvline(x=self.analysis_date, color=colors['analysis_line'], linestyle=':', 
                       linewidth=2, alpha=0.8, label='Analysis Date')
            
            # Mark the analysis point with the REVR value - enhanced styling
            analysis_data = self.stock_data[self.stock_data['date'] == self.analysis_date]
            if not analysis_data.empty:
                analysis_ratio = analysis_data['st_mt_ratio'].iloc[0]
                ax2.scatter(self.analysis_date, analysis_ratio, 
                           color=colors['mt_vol'], s=200, marker='*', 
                           label=f'Analysis Point (REVR={analysis_ratio:.3f})', 
                           zorder=5, edgecolors='white', linewidth=1.5)
            
            # Customize bottom plot with BlackRock styling
            ax2.set_title('AAPL ST/MT Volatility Ratio Over Time', fontsize=16, fontweight='bold', 
                         pad=20, color=colors['text'])
            ax2.set_ylabel('ST/MT Ratio', fontsize=12, color=colors['text'])
            ax2.set_xlabel('Date', fontsize=12, color=colors['text'])
            ax2.grid(True, alpha=0.3, color=colors['grid'], linewidth=0.5)
            ax2.set_facecolor('#FAFAFA')
            
            # Professional legend styling
            legend2 = ax2.legend(loc='upper left', framealpha=0.95, fontsize=10, 
                               fancybox=True, shadow=True, borderpad=1)
            legend2.get_frame().set_facecolor('white')
            legend2.get_frame().set_edgecolor('#CCCCCC')
            
            # Set y-axis limits for ratio plot
            ratio_min = self.stock_data['st_mt_ratio'].min()
            ratio_max = self.stock_data['st_mt_ratio'].max()
            ax2.set_ylim(max(0.5, ratio_min * 0.9), ratio_max * 1.1)
            
            # Format x-axis with BlackRock professional styling
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=10))
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                ax.tick_params(axis='x', rotation=45, labelsize=10, color=colors['text'])
                ax.tick_params(axis='y', labelsize=10, color=colors['text'])
                
                # Style the spines (borders)
                for spine in ax.spines.values():
                    spine.set_color('#CCCCCC')
                    spine.set_linewidth(1)
                
                # Remove top and right spines for cleaner look
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            
            
            plt.subplots_adjust(hspace=0.45, top=0.93)
            
            # Save the plot with high quality for professional presentation
            output_path = 'aapl_2023_q3_volatility_visualization.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', transparent=False)
            print(f"✅ Professional BlackRock-style visualization saved to: {output_path}")
            
            # Show the plot
            plt.show()
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_analysis_summary(self):
        """Print analysis summary"""
        print("\n📊 2023 Q3 AAPL ANALYSIS SUMMARY")
        print("="*60)
        
        try:
            # Get data around key dates
            earnings_data = self.stock_data[self.stock_data['date'] == self.earnings_date]
            analysis_data = self.stock_data[self.stock_data['date'] == self.analysis_date]
            
            print(f"🎯 Key Dates:")
            print(f"  Earnings Date: {self.earnings_date.strftime('%Y-%m-%d')}")
            print(f"  Analysis Date: {self.analysis_date.strftime('%Y-%m-%d')}")
            
            if not earnings_data.empty:
                print(f"\n�� Earnings Date Data:")
                print(f"  ST Volatility (vol_st): {earnings_data['vol_st_pct'].iloc[0]:.2f}%")
                print(f"  MT Volatility (vol_mt): {earnings_data['vol_mt_pct'].iloc[0]:.2f}%")
                print(f"  ST/MT Ratio: {earnings_data['st_mt_ratio'].iloc[0]:.3f}")
            
            if not analysis_data.empty:
                print(f"\n�� Analysis Date Data:")
                print(f"  ST Volatility (vol_st): {analysis_data['vol_st_pct'].iloc[0]:.2f}%")
                print(f"  MT Volatility (vol_mt): {analysis_data['vol_mt_pct'].iloc[0]:.2f}%")
                print(f"  ST/MT Ratio (REVR): {analysis_data['st_mt_ratio'].iloc[0]:.3f}")
            
            # Overall statistics
            print(f"\n�� Overall Statistics:")
            print(f"  Data Points: {len(self.stock_data)}")
            print(f"  Date Range: {self.stock_data['date'].min().strftime('%Y-%m-%d')} to {self.stock_data['date'].max().strftime('%Y-%m-%d')}")
            print(f"  ST Volatility Range: {self.stock_data['vol_st_pct'].min():.2f}% to {self.stock_data['vol_st_pct'].max():.2f}%")
            print(f"  MT Volatility Range: {self.stock_data['vol_mt_pct'].min():.2f}% to {self.stock_data['vol_mt_pct'].max():.2f}%")
            print(f"  ST/MT Ratio Range: {self.stock_data['st_mt_ratio'].min():.3f} to {self.stock_data['st_mt_ratio'].max():.3f}")
            
            # Calculate average ratios before and after earnings
            before_earnings = self.stock_data[self.stock_data['date'] < self.earnings_date]
            after_earnings = self.stock_data[self.stock_data['date'] > self.earnings_date]
            
            if not before_earnings.empty:
                print(f"\n📊 Before Earnings (Avg ST/MT Ratio): {before_earnings['st_mt_ratio'].mean():.3f}")
            if not after_earnings.empty:
                print(f"📊 After Earnings (Avg ST/MT Ratio): {after_earnings['st_mt_ratio'].mean():.3f}")
                
        except Exception as e:
            print(f"❌ Error printing summary: {e}")
            import traceback
            traceback.print_exc()
    
    def run_analysis(self):
        """Run the complete 2023 Q3 analysis"""
        print("�� AAPL 2023 Q3 VOLATILITY ANALYSIS")
        print("="*80)
        
        # Download data
        if not self.download_aapl_data():
            return False
        
        # Calculate volatility
        if not self.calculate_st_mt_volatility():
            return False
        
        # Create visualization
        if not self.create_visualization():
            return False
        
        # Print summary
        self.print_analysis_summary()
        
        print("\n✅ 2023 Q3 Analysis complete!")
        return True

def main():
    """Main function"""
    analyzer = AAPL2023Q3VolatilityAnalysis()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()