#!/usr/bin/env python3
"""
Sector Classification Script
Add sector classification to any dataset with ticker symbols
"""

import pandas as pd
import numpy as np

class SectorClassifier:
    """
    Classify stocks into sectors based on ticker symbols.
    """
    
    def __init__(self):
        self.sector_mapping = self.create_sector_mapping()
    
    def create_sector_mapping(self):
        """
        Create comprehensive sector mapping for all stocks.
        """
        return {
            # Technology (30 stocks)
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
            'AMZN': 'Technology', 'TSLA': 'Technology', 'NVDA': 'Technology', 'META': 'Technology',
            'NFLX': 'Technology', 'ADBE': 'Technology', 'CRM': 'Technology', 'ORCL': 'Technology',
            'INTC': 'Technology', 'AMD': 'Technology', 'QCOM': 'Technology', 'AVGO': 'Technology',
            'TXN': 'Technology', 'MU': 'Technology', 'ADI': 'Technology', 'KLAC': 'Technology',
            'SNPS': 'Technology', 'CDNS': 'Technology', 'MCHP': 'Technology', 'LRCX': 'Technology',
            'TER': 'Technology', 'WDC': 'Technology', 'STX': 'Technology', 'HPQ': 'Technology',
            'DELL': 'Technology', 'CSCO': 'Technology',
            
            # Financial Services (25 stocks)
            'JPM': 'Financial Services', 'BAC': 'Financial Services', 'WFC': 'Financial Services',
            'GS': 'Financial Services', 'MS': 'Financial Services', 'C': 'Financial Services',
            'USB': 'Financial Services', 'PNC': 'Financial Services', 'TFC': 'Financial Services',
            'COF': 'Financial Services', 'AXP': 'Financial Services', 'BLK': 'Financial Services',
            'SCHW': 'Financial Services', 'CME': 'Financial Services', 'ICE': 'Financial Services',
            'SPGI': 'Financial Services', 'MCO': 'Financial Services', 'FIS': 'Financial Services',
            'FISV': 'Financial Services', 'V': 'Financial Services', 'MA': 'Financial Services',
            'PYPL': 'Financial Services', 'SQ': 'Financial Services', 'COIN': 'Financial Services',
            'HOOD': 'Financial Services',
            
            # Healthcare (25 stocks)
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
            'MRK': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare', 'DHR': 'Healthcare',
            'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare', 'CVS': 'Healthcare',
            'CI': 'Healthcare', 'ANTM': 'Healthcare', 'HUM': 'Healthcare', 'REGN': 'Healthcare',
            'VRTX': 'Healthcare', 'LLY': 'Healthcare', 'DVA': 'Healthcare', 'HCA': 'Healthcare',
            'ISRG': 'Healthcare', 'DXCM': 'Healthcare', 'ILMN': 'Healthcare', 'BIIB': 'Healthcare',
            'ALGN': 'Healthcare',
            
            # Consumer Discretionary (25 stocks)
            'HD': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary',
            'SBUX': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary',
            'BKNG': 'Consumer Discretionary', 'MAR': 'Consumer Discretionary', 'YUM': 'Consumer Discretionary',
            'CMG': 'Consumer Discretionary', 'TGT': 'Consumer Discretionary', 'COST': 'Consumer Discretionary',
            'WMT': 'Consumer Discretionary', 'PM': 'Consumer Discretionary', 'MO': 'Consumer Discretionary',
            'DIS': 'Consumer Discretionary', 'CMCSA': 'Consumer Discretionary', 'VZ': 'Consumer Discretionary',
            'T': 'Consumer Discretionary', 'TMUS': 'Consumer Discretionary', 'CHTR': 'Consumer Discretionary',
            'PARA': 'Consumer Discretionary', 'NWSA': 'Consumer Discretionary', 'FOX': 'Consumer Discretionary',
            'FOXA': 'Consumer Discretionary', 'SNAP': 'Consumer Discretionary',
            
            # Consumer Staples (20 stocks)
            'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
            'EL': 'Consumer Staples', 'CL': 'Consumer Staples', 'GIS': 'Consumer Staples',
            'KMB': 'Consumer Staples', 'HSY': 'Consumer Staples', 'SJM': 'Consumer Staples',
            'CPB': 'Consumer Staples', 'KR': 'Consumer Staples', 'WBA': 'Consumer Staples',
            'RAD': 'Consumer Staples', 'WAG': 'Consumer Staples', 'DLTR': 'Consumer Discretionary',
            'DG': 'Consumer Discretionary',
            
            # Energy (20 stocks)
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy', 'SLB': 'Energy',
            'PSX': 'Energy', 'VLO': 'Energy', 'MPC': 'Energy', 'OXY': 'Energy', 'HAL': 'Energy',
            'KMI': 'Energy', 'WMB': 'Energy', 'OKE': 'Energy', 'PXD': 'Energy', 'DVN': 'Energy',
            'FANG': 'Energy', 'MRO': 'Energy', 'HES': 'Energy',
            
            # Industrials (20 stocks)
            'BA': 'Industrials', 'CAT': 'Industrials', 'MMM': 'Industrials', 'GE': 'Industrials',
            'HON': 'Industrials', 'UPS': 'Industrials', 'FDX': 'Industrials', 'RTX': 'Industrials',
            'LMT': 'Industrials', 'NOC': 'Industrials', 'DE': 'Industrials', 'EMR': 'Industrials',
            'ETN': 'Industrials', 'ITW': 'Industrials', 'PH': 'Industrials', 'ROK': 'Industrials',
            'SWK': 'Industrials', 'TXT': 'Industrials', 'WM': 'Industrials', 'WMG': 'Industrials',
            
            # Materials (15 stocks)
            'LIN': 'Materials', 'APD': 'Materials', 'FCX': 'Materials', 'NEM': 'Materials',
            'AA': 'Materials', 'BLL': 'Materials', 'IP': 'Materials', 'PKG': 'Materials',
            'SEE': 'Materials', 'WRK': 'Materials', 'ALB': 'Materials', 'LVS': 'Consumer Discretionary',
            'VMC': 'Materials', 'MLM': 'Materials', 'NUE': 'Materials',
            
            # Communication Services (15 stocks)
            # Note: Some stocks are already classified above
            
            # Real Estate (15 stocks)
            'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate',
            'DLR': 'Real Estate', 'PSA': 'Real Estate', 'SPG': 'Real Estate', 'O': 'Real Estate',
            'EQR': 'Real Estate', 'AVB': 'Real Estate', 'MAA': 'Real Estate', 'UDR': 'Real Estate',
            'ESS': 'Real Estate', 'ARE': 'Real Estate', 'BXP': 'Real Estate',
            
            # Utilities (15 stocks)
            'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
            'AEP': 'Utilities', 'XEL': 'Utilities', 'ED': 'Utilities', 'EIX': 'Utilities',
            'PCG': 'Utilities', 'SRE': 'Utilities', 'WEC': 'Utilities', 'DTE': 'Utilities',
            'CMS': 'Utilities', 'AEE': 'Utilities', 'LNT': 'Utilities'
        }
    
    def determine_sector_for_ticker(self, ticker):
        """
        Determine sector for a ticker using various heuristics.
        This is a fallback method for stocks not in the predefined mapping.
        """
        ticker_upper = ticker.upper()
        
        # Technology indicators
        tech_keywords = ['TECH', 'SOFT', 'SYS', 'NET', 'DATA', 'AI', 'ML', 'BIO', 'GEN', 'PHARMA']
        if any(keyword in ticker_upper for keyword in tech_keywords):
            return 'Technology'
        
        # Financial indicators
        fin_keywords = ['BANK', 'FIN', 'INS', 'CAP', 'TRUST', 'MORT', 'LOAN', 'CREDIT']
        if any(keyword in ticker_upper for keyword in fin_keywords):
            return 'Financial Services'
        
        # Healthcare indicators
        health_keywords = ['HEALTH', 'MED', 'CARE', 'BIO', 'GEN', 'PHARMA', 'THERA', 'MEDI']
        if any(keyword in ticker_upper for keyword in health_keywords):
            return 'Healthcare'
        
        # Energy indicators
        energy_keywords = ['OIL', 'GAS', 'ENERGY', 'POWER', 'FUEL', 'PETRO', 'RENEW']
        if any(keyword in ticker_upper for keyword in energy_keywords):
            return 'Energy'
        
        # Consumer indicators
        consumer_keywords = ['FOOD', 'RETAIL', 'STORE', 'SHOP', 'REST', 'HOTEL', 'TRAVEL']
        if any(keyword in ticker_upper for keyword in consumer_keywords):
            return 'Consumer Discretionary'
        
        # Industrial indicators
        industrial_keywords = ['IND', 'MANU', 'AUTO', 'STEEL', 'CHEM', 'ENG', 'CONST']
        if any(keyword in ticker_upper for keyword in industrial_keywords):
            return 'Industrials'
        
        # Default to Technology for unknown stocks (most likely to be tech in modern markets)
        return 'Technology'
    
    def add_sector_column(self, df, ticker_column='ticker'):
        """
        Add sector column to a DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing ticker symbols
        ticker_column : str
            Name of the column containing ticker symbols
        
        Returns:
        --------
        pd.DataFrame : DataFrame with added sector column
        """
        if ticker_column not in df.columns:
            print(f"❌ Column '{ticker_column}' not found in DataFrame")
            return df
        
        # Create a copy to avoid modifying the original
        df_with_sector = df.copy()
        
        # Add sector column using predefined mapping first
        df_with_sector['sector'] = df_with_sector[ticker_column].map(self.sector_mapping)
        
        # For unmapped tickers, use heuristic determination
        unmapped_mask = df_with_sector['sector'].isna()
        if unmapped_mask.any():
            unmapped_tickers = df_with_sector.loc[unmapped_mask, ticker_column].unique()
            print(f"  ⚠️  {len(unmapped_tickers)} tickers not in predefined mapping, using heuristics...")
            
            for ticker in unmapped_tickers:
                sector = self.determine_sector_for_ticker(ticker)
                df_with_sector.loc[df_with_sector[ticker_column] == ticker, 'sector'] = sector
                print(f"    {ticker} → {sector}")
        
        # Check for any remaining unmapped tickers
        unmapped_tickers = df_with_sector[df_with_sector['sector'].isna()][ticker_column].unique()
        if len(unmapped_tickers) > 0:
            print(f"  ⚠️  Warning: {len(unmapped_tickers)} tickers still unmapped:")
            for ticker in unmapped_tickers[:10]:  # Show first 10
                print(f"    {ticker}")
            if len(unmapped_tickers) > 10:
                print(f"    ... and {len(unmapped_tickers) - 10} more")
        
        # Summary statistics
        sector_counts = df_with_sector['sector'].value_counts()
        print(f"\n✅ Sector classification added:")
        print(f"  Total observations: {len(df_with_sector)}")
        print(f"  Stocks with sectors: {df_with_sector['sector'].notna().sum()}")
        print(f"  Stocks without sectors: {df_with_sector['sector'].isna().sum()}")
        print(f"\nSector distribution:")
        for sector, count in sector_counts.items():
            if pd.notna(sector):
                print(f"  {sector}: {count}")
        
        return df_with_sector
    
    def get_sector_summary(self, df, ticker_column='ticker'):
        """
        Get summary statistics by sector.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with sector column
        ticker_column : str
            Name of the column containing ticker symbols
        
        Returns:
        --------
        pd.DataFrame : Summary statistics by sector
        """
        if 'sector' not in df.columns:
            print("❌ No sector column found. Run add_sector_column() first.")
            return None
        
        # Group by sector and calculate summary statistics
        sector_summary = df.groupby('sector').agg({
            ticker_column: ['count', 'nunique'],
            'sector': 'count'
        }).round(2)
        
        # Flatten column names
        sector_summary.columns = ['total_observations', 'unique_stocks', 'sector_count']
        
        # Sort by number of stocks
        sector_summary = sector_summary.sort_values('unique_stocks', ascending=False)
        
        return sector_summary
    
    def save_with_sectors(self, df, filename, ticker_column='ticker'):
        """
        Add sector classification and save to CSV.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to process
        filename : str
            Output filename
        ticker_column : str
            Name of the column containing ticker symbols
        """
        # Add sector classification
        df_with_sectors = self.add_sector_column(df, ticker_column)
        
        # Save to CSV
        df_with_sectors.to_csv(filename, index=False)
        print(f"✅ Data with sectors saved to: {filename}")
        
        return df_with_sectors

def main():
    """
    Example usage of the sector classifier.
    """
    print("Sector Classification Tool")
    print("="*50)
    
    # Example: Load a dataset and add sectors
    try:
        # Try to load an existing dataset
        import os
        data_files = [f for f in os.listdir('data_files') if f.endswith('.csv')]
        
        if data_files:
            print(f"Found data files: {data_files}")
            
            # Load the first available file
            sample_file = f"data_files/{data_files[0]}"
            print(f"\nLoading sample file: {sample_file}")
            
            df = pd.read_csv(sample_file)
            print(f"Loaded {len(df)} observations")
            
            # Initialize classifier
            classifier = SectorClassifier()
            
            # Add sector classification
            df_with_sectors = classifier.add_sector_column(df)
            
            # Show sector summary
            sector_summary = classifier.get_sector_summary(df_with_sectors)
            if sector_summary is not None:
                print(f"\nSector Summary:")
                print(sector_summary)
            
            # Save with sectors
            output_file = sample_file.replace('.csv', '_with_sectors.csv')
            classifier.save_with_sectors(df, output_file)
            
        else:
            print("No data files found in data_files/ directory")
            print("Run one of the analysis scripts first to generate data")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTo use this script with your data:")
        print("1. Run one of the analysis scripts (REVR, IEVR, or Beta)")
        print("2. Then run this script to add sector classification")

if __name__ == "__main__":
    main()
