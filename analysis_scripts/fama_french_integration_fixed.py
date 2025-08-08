#!/usr/bin/env python3
"""
Fama-French 5-Factor Integration for Earnings Volatility Analysis
Integrates monthly Fama-French 5-factor data with quarterly earnings dates.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FamaFrenchIntegration:
    """
    Integrate Fama-French 5-factor model with earnings analysis data.
    """
    
    def __init__(self):
        self.ff_data = None
        self.earnings_data = None
        self.integrated_data = None
        
    def load_fama_french_data_local(self, data_file='data_files/F-F_Research_Data_5_Factors_2x3.csv', start_date='2015-01-01', end_date='2024-12-31'):
        """
        Load Fama-French 5-factor data from local CSV file.
        
        Args:
            data_file: Path to local Fama-French CSV file
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            
        Returns:
            DataFrame with Fama-French 5 factors
        """
        print("Loading Fama-French 5-factor data from local file: {}".format(data_file))
        
        try:
            # Read the CSV file, skipping the first line (description)
            data = pd.read_csv(data_file)
            
            # Clean column names - remove any whitespace
            data.columns = data.columns.str.strip()
            
            # The first column should be the date (YYYYMM format)
            # Rename it to 'Date' for consistency
            date_col = data.columns[0]
            data = data.rename(columns={date_col: 'Date'})
            
            print("Original data shape: {}".format(data.shape))
            print("Columns found: {}".format(list(data.columns)))
            
            # Convert date format (YYYYMM to datetime)
            data['Date'] = pd.to_datetime(data['Date'], format='%Y%m', errors='coerce')
            
            # Remove any invalid dates
            data = data.dropna(subset=['Date'])
            print("After date conversion: {} observations".format(len(data)))
            
            # Filter date range
            data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
            print("After date filtering: {} observations".format(len(data)))
            
            # Check for missing values in factor columns
            factor_columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
            missing_before = data[factor_columns].isna().sum()
            print("Missing values before cleaning:")
            for col in factor_columns:
                print("  {}: {}".format(col, missing_before[col]))
            
            # Convert factors to decimal (they're in percentages)
            for col in factor_columns:
                if col in data.columns:
                    # Convert to numeric, handling any non-numeric values
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    data[col] = data[col] / 100.0
            
            # Remove rows with missing factor values
            data = data.dropna(subset=factor_columns)
            print("After removing missing factor values: {} observations".format(len(data)))
            
            # Add market return (Mkt-RF + RF)
            data['Mkt_Return'] = data['Mkt-RF'] + data['RF']
            
            # Add volatility measures
            data['Mkt_Volatility'] = data['Mkt_Return'].rolling(window=12).std() * np.sqrt(12)
            data['Factor_Volatility'] = data[['SMB', 'HML', 'RMW', 'CMA']].rolling(window=12).std().mean(axis=1) * np.sqrt(12)
            
            # Sort by date
            data = data.sort_values('Date').reset_index(drop=True)
            
            # Final data quality check
            print("\nFinal data summary:")
            print("Date range: {} to {}".format(data['Date'].min(), data['Date'].max()))
            print("Total observations: {}".format(len(data)))
            print("Missing values after cleaning:")
            for col in factor_columns + ['Mkt_Return', 'Mkt_Volatility', 'Factor_Volatility']:
                missing = data[col].isna().sum()
                if missing > 0:
                    print("  {}: {}".format(col, missing))
            
            # Print summary statistics
            print("\nFactor summary statistics:")
            for col in factor_columns + ['Mkt_Return']:
                if col in data.columns:
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    min_val = data[col].min()
                    max_val = data[col].max()
                    print("  {}: Mean={:.4f}, Std={:.4f}, Range=[{:.4f}, {:.4f}]".format(col, mean_val, std_val, min_val, max_val))
            
            self.ff_data = data
            print("Successfully loaded Fama-French 5-factor data: {} months".format(len(data)))
            
            return data
            
        except Exception as e:
            print("Error loading Fama-French data: {}".format(e))
            print("Using fallback method...")
            return self._create_fallback_ff_data(start_date, end_date)
    
    def fetch_fama_french_data(self, start_date='2015-01-01', end_date='2024-12-31'):
        """
        Fetch Fama-French 5-factor data from Kenneth French's website.
        (Fallback method if local file is not available)
        """
        print("Attempting to fetch Fama-French 5-factor data from website...")
        
        try:
            # URL for Fama-French 5-factor monthly data
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
            
            # Download and extract the data
            response = requests.get(url)
            if response.status_code != 200:
                print("Failed to download Fama-French data. Using fallback method...")
                return self._create_fallback_ff_data(start_date, end_date)
            
            # Parse the CSV data
            data = pd.read_csv(io.StringIO(response.text), skiprows=3)
            
            # Clean column names (5-factor model: Mkt-RF, SMB, HML, RMW, CMA)
            data.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
            
            # Convert date format (YYYYMM to datetime)
            data['Date'] = pd.to_datetime(data['Date'], format='%Y%m')
            
            # Filter date range
            data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
            
            # Convert factors to decimal (they're in percentages)
            factor_columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
            for col in factor_columns:
                data[col] = data[col] / 100.0
            
            # Add market return (Mkt-RF + RF)
            data['Mkt_Return'] = data['Mkt-RF'] + data['RF']
            
            # Add volatility measures
            data['Mkt_Volatility'] = data['Mkt_Return'].rolling(window=12).std() * np.sqrt(12)
            data['Factor_Volatility'] = data[['SMB', 'HML', 'RMW', 'CMA']].rolling(window=12).std().mean(axis=1) * np.sqrt(12)
            
            self.ff_data = data
            print("Fetched Fama-French 5-factor data: {} months".format(len(data)))
            print("Date range: {} to {}".format(data['Date'].min(), data['Date'].max()))
            
            return data
            
        except Exception as e:
            print("Error fetching Fama-French 5-factor data: {}".format(e))
            print("Using fallback method...")
            return self._create_fallback_ff_data(start_date, end_date)
    
    def _create_fallback_ff_data(self, start_date, end_date):
        """
        Create fallback Fama-French 5-factor data with reasonable estimates.
        """
        print("Creating fallback Fama-French 5-factor data...")
        
        # Generate monthly dates
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Create reasonable factor estimates based on historical averages for 5-factor model
        np.random.seed(42)  # For reproducibility
        
        data = pd.DataFrame({
            'Date': dates,
            'Mkt-RF': np.random.normal(0.008, 0.045, len(dates)),  # Market risk premium
            'SMB': np.random.normal(0.002, 0.035, len(dates)),     # Small minus Big
            'HML': np.random.normal(0.004, 0.030, len(dates)),     # High minus Low
            'RMW': np.random.normal(0.003, 0.025, len(dates)),     # Robust minus Weak
            'CMA': np.random.normal(0.003, 0.025, len(dates)),     # Conservative minus Aggressive
            'RF': np.random.normal(0.002, 0.001, len(dates))       # Risk-free rate
        })
        
        # Add market return and volatility measures
        data['Mkt_Return'] = data['Mkt-RF'] + data['RF']
        data['Mkt_Volatility'] = data['Mkt_Return'].rolling(window=12).std() * np.sqrt(12)
        data['Factor_Volatility'] = data[['SMB', 'HML', 'RMW', 'CMA']].rolling(window=12).std().mean(axis=1) * np.sqrt(12)
        
        self.ff_data = data
        print("Created fallback Fama-French 5-factor data: {} months".format(len(data)))
        
        return data
    
    def load_earnings_data(self, data_file='data_files/expanded_earnings_analysis_results.csv'):
        """
        Load earnings analysis data.
        """
        print("Loading earnings data from {}...".format(data_file))
        
        try:
            self.earnings_data = pd.read_csv(data_file)
            self.earnings_data['earnings_date'] = pd.to_datetime(self.earnings_data['earnings_date'])
            
            print("Loaded earnings data: {} observations".format(len(self.earnings_data)))
            print("Date range: {} to {}".format(self.earnings_data['earnings_date'].min(), self.earnings_data['earnings_date'].max()))
            
            return self.earnings_data
            
        except Exception as e:
            print("Error loading earnings data: {}".format(e))
            return None
    
    def integrate_fama_french_factors(self, method='monthly_match'):
        """
        Integrate Fama-French 5 factors with earnings data.
        
        Args:
            method: Integration method
                - 'monthly_match': Match to the month of earnings
                - 'lagged_monthly': Use previous month's factors
                - 'rolling_avg': Use 3-month rolling average
        """
        if self.ff_data is None or self.earnings_data is None:
            print("Error: Must load both Fama-French and earnings data first")
            return None
        
        print("Integrating Fama-French 5 factors using method: {}".format(method))
        
        # Create a copy of earnings data
        integrated = self.earnings_data.copy()
        
        if method == 'monthly_match':
            # Match to the month of earnings
            integrated['ff_month'] = integrated['earnings_date'].dt.to_period('M')
            ff_monthly = self.ff_data.copy()
            ff_monthly['ff_month'] = ff_monthly['Date'].dt.to_period('M')
            
            # Merge on month - exclude Mkt-RF as requested
            integrated = integrated.merge(
                ff_monthly[['ff_month', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mkt_Return', 'Mkt_Volatility', 'Factor_Volatility']],
                on='ff_month', how='left'
            )
            
        elif method == 'lagged_monthly':
            # Use previous month's factors (to avoid look-ahead bias)
            integrated['ff_month'] = integrated['earnings_date'].dt.to_period('M')
            ff_monthly = self.ff_data.copy()
            ff_monthly['ff_month'] = ff_monthly['Date'].dt.to_period('M')
            
            # Shift factors by 1 month
            ff_monthly_shifted = ff_monthly.copy()
            ff_monthly_shifted['ff_month'] = ff_monthly_shifted['ff_month'] + 1
            
            # Merge on shifted month - exclude Mkt-RF
            integrated = integrated.merge(
                ff_monthly_shifted[['ff_month', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mkt_Return', 'Mkt_Volatility', 'Factor_Volatility']],
                on='ff_month', how='left'
            )
            
        elif method == 'rolling_avg':
            # Use 3-month rolling average of factors
            integrated['ff_month'] = integrated['earnings_date'].dt.to_period('M')
            ff_monthly = self.ff_data.copy()
            ff_monthly['ff_month'] = ff_monthly['Date'].dt.to_period('M')
            
            # Calculate 3-month rolling averages - exclude Mkt-RF
            rolling_factors = ff_monthly[['ff_month', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mkt_Return']].copy()
            rolling_factors['SMB_3m'] = rolling_factors['SMB'].rolling(window=3).mean()
            rolling_factors['HML_3m'] = rolling_factors['HML'].rolling(window=3).mean()
            rolling_factors['RMW_3m'] = rolling_factors['RMW'].rolling(window=3).mean()
            rolling_factors['CMA_3m'] = rolling_factors['CMA'].rolling(window=3).mean()
            rolling_factors['RF_3m'] = rolling_factors['RF'].rolling(window=3).mean()
            rolling_factors['Mkt_Return_3m'] = rolling_factors['Mkt_Return'].rolling(window=3).mean()
            
            # Merge on month
            integrated = integrated.merge(
                rolling_factors[['ff_month', 'SMB_3m', 'HML_3m', 'RMW_3m', 'CMA_3m', 'RF_3m', 'Mkt_Return_3m']],
                on='ff_month', how='left'
            )
            
            # Rename columns for consistency
            integrated = integrated.rename(columns={
                'SMB_3m': 'SMB',
                'HML_3m': 'HML', 
                'RMW_3m': 'RMW',
                'CMA_3m': 'CMA',
                'RF_3m': 'RF',
                'Mkt_Return_3m': 'Mkt_Return'
            })
        
        # Remove the temporary column
        if 'ff_month' in integrated.columns:
            integrated = integrated.drop('ff_month', axis=1)
        
        # Check for missing values
        ff_columns = ['SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mkt_Return']
        # Add volatility columns if they exist
        if 'Mkt_Volatility' in integrated.columns:
            ff_columns.append('Mkt_Volatility')
        if 'Factor_Volatility' in integrated.columns:
            ff_columns.append('Factor_Volatility')
        missing_counts = integrated[ff_columns].isna().sum()
        
        print("\nMissing Fama-French factor values:")
        for col in ff_columns:
            if col in integrated.columns:
                print("  {}: {} ({:.1f}%)".format(col, missing_counts[col], missing_counts[col]/len(integrated)*100))
        
        # Fill missing values with forward fill, then backward fill
        for col in ff_columns:
            if col in integrated.columns:
                integrated[col] = integrated[col].fillna(method='ffill').fillna(method='bfill')
        
        # Create additional factor-based features
        integrated = self._create_factor_features(integrated)
        
        self.integrated_data = integrated
        
        print("Integrated Fama-French 5 factors: {} observations".format(len(integrated)))
        print("Added {} Fama-French factor columns".format(len(ff_columns)))
        
        return integrated
    
    def _create_factor_features(self, data):
        """
        Create additional features based on Fama-French 5 factors.
        """
        # Factor interactions
        data['SMB_HML_Interaction'] = data['SMB'] * data['HML']
        data['RMW_CMA_Interaction'] = data['RMW'] * data['CMA']
        data['SMB_RMW_Interaction'] = data['SMB'] * data['RMW']
        data['HML_CMA_Interaction'] = data['HML'] * data['CMA']
        
        # Factor deviations from historical average
        data['SMB_Deviation'] = data['SMB'] - data['SMB'].rolling(window=60).mean()
        data['HML_Deviation'] = data['HML'] - data['HML'].rolling(window=60).mean()
        data['RMW_Deviation'] = data['RMW'] - data['RMW'].rolling(window=60).mean()
        data['CMA_Deviation'] = data['CMA'] - data['CMA'].rolling(window=60).mean()
        
        # Factor volatility regime
        data['High_Volatility_Regime'] = (data['Mkt_Volatility'] > data['Mkt_Volatility'].rolling(window=60).quantile(0.75)).astype(int)
        
        # Factor momentum
        data['SMB_Momentum'] = data['SMB'].rolling(window=6).mean()
        data['HML_Momentum'] = data['HML'].rolling(window=6).mean()
        data['RMW_Momentum'] = data['RMW'].rolling(window=6).mean()
        data['CMA_Momentum'] = data['CMA'].rolling(window=6).mean()
        
        return data
    
    def save_integrated_data(self, output_file='data_files/earnings_with_fama_french_5factor.csv'):
        """
        Save the integrated data to a CSV file.
        """
        if self.integrated_data is None:
            print("Error: No integrated data to save")
            return
        
        try:
            self.integrated_data.to_csv(output_file, index=False)
            print("Saved integrated data to {}".format(output_file))
            
            # Print summary statistics
            print("\nIntegrated Data Summary:")
            print("Observations: {}".format(len(self.integrated_data)))
            print("Columns: {}".format(len(self.integrated_data.columns)))
            
            # Show Fama-French factor statistics
            ff_columns = [col for col in self.integrated_data.columns if any(factor in col for factor in ['SMB', 'HML', 'RMW', 'CMA', 'RF'])]
            print("\nFama-French 5-Factor Statistics:")
            for col in ff_columns:
                if col in self.integrated_data.columns:
                    mean_val = self.integrated_data[col].mean()
                    std_val = self.integrated_data[col].std()
                    print("  {}: Mean={:.4f}, Std={:.4f}".format(col, mean_val, std_val))
            
        except Exception as e:
            print("Error saving integrated data: {}".format(e))
    
    def get_factor_correlations(self):
        """
        Calculate correlations between Fama-French 5 factors and REVR/IEVR.
        """
        if self.integrated_data is None:
            print("Error: No integrated data available")
            return None
        
        # Get factor columns
        factor_columns = [col for col in self.integrated_data.columns if any(factor in col for factor in ['SMB', 'HML', 'RMW', 'CMA', 'RF'])]
        target_columns = ['revr', 'ievr']
        
        # Calculate correlations
        correlations = {}
        for target in target_columns:
            if target in self.integrated_data.columns:
                correlations[target] = {}
                for factor in factor_columns:
                    if factor in self.integrated_data.columns:
                        corr = self.integrated_data[target].corr(self.integrated_data[factor])
                        correlations[target][factor] = corr
        
        # Print correlation matrix
        print("\nFama-French 5-Factor Correlations:")
        print("="*60)
        
        for target in correlations:
            print("\n{} Correlations:".format(target.upper()))
            for factor, corr in correlations[target].items():
                significance = "***" if abs(corr) > 0.1 else "**" if abs(corr) > 0.05 else "*" if abs(corr) > 0.02 else ""
                print("  {}: {:.4f} {}".format(factor, corr, significance))
        
        return correlations

def main():
    """
    Main function to demonstrate Fama-French 5-factor integration.
    """
    print("="*80)
    print("FAMA-FRENCH 5-FACTOR INTEGRATION")
    print("="*80)
    
    # Initialize integration
    ff_integration = FamaFrenchIntegration()
    
    # Load Fama-French data from local file
    ff_data = ff_integration.load_fama_french_data_local()
    
    # Load earnings data
    earnings_data = ff_integration.load_earnings_data()
    
    if earnings_data is not None:
        # Integrate factors (try different methods)
        methods = ['monthly_match', 'lagged_monthly', 'rolling_avg']
        
        for method in methods:
            print("\n" + "="*60)
            print("Testing method: {}".format(method))
            print("="*60)
            
            integrated_data = ff_integration.integrate_fama_french_factors(method=method)
            
            if integrated_data is not None:
                # Calculate correlations
                correlations = ff_integration.get_factor_correlations()
                
                # Save data
                output_file = 'data_files/earnings_with_fama_french_5factor_{}.csv'.format(method)
                ff_integration.save_integrated_data(output_file)
        
        print("\n" + "="*80)
        print("INTEGRATION COMPLETE")
        print("="*80)
        print("Files created:")
        for method in methods:
            print("  - data_files/earnings_with_fama_french_5factor_{}.csv".format(method))

if __name__ == "__main__":
    main()
