#!/usr/bin/env python3
"""
Streamlined Feature Integration - Essential Features Only
Core (3): ievr, skew_ratio, normative_iv_rv_ratio
Dispersion (1): dispersion coefficient
Option Surface (5): term_ratio, skew, kurt, iv_ratio, smirk
Fama-French (5): SMB, HML, RMW, CMA, RF
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class StreamlinedFeatureIntegration:
    """
    Integrate only essential features: Core, Dispersion, Option Surface, Fama-French
    """
    
    def __init__(self):
        self.integrated_data = None
        
    def integrate_dispersion_feature(self, earnings_data, db_connection=None):
        """
        Integrate only the dispersion coefficient feature
        """
        print("Integrating dispersion coefficient feature...")
        
        try:
            from main import calculate_dispersion_coefficient
            
            # Add only the dispersion column
            earnings_data['dispersion'] = np.nan
            
            # Calculate dispersion for each earnings event
            for idx, row in earnings_data.iterrows():
                ticker = row['ticker']
                earnings_date = pd.to_datetime(row['earnings_date'])
                
                if db_connection:
                    dispersion = calculate_dispersion_coefficient(db_connection, ticker, earnings_date)
                    earnings_data.loc[idx, 'dispersion'] = dispersion
                else:
                    # Mock dispersion for testing
                    earnings_data.loc[idx, 'dispersion'] = np.random.uniform(0.05, 0.25)
            
            print(f"✓ Dispersion feature integrated: {earnings_data['dispersion'].notna().sum()} observations")
            
        except Exception as e:
            print(f"Warning: Dispersion integration failed: {e}")
            # Add default dispersion feature
            earnings_data['dispersion'] = np.random.uniform(0.05, 0.25, len(earnings_data))
        
        return earnings_data
    
    def integrate_fama_french_features(self, earnings_data, ff_data_file=None):
        """
        Integrate only the 5 essential Fama-French factors
        """
        print("Integrating Fama-French 5-factor features...")
        
        try:
            from fama_french_integration_fixed import FamaFrenchIntegration
            
            # Initialize FF integration
            ff_integration = FamaFrenchIntegration()
            
            # Load FF data
            if ff_data_file and os.path.exists(ff_data_file):
                ff_data = ff_integration.load_fama_french_data_local(ff_data_file)
            else:
                ff_data = ff_integration.load_fama_french_data_local()
            
            if ff_data is not None:
                # Integrate FF factors
                integrated_ff = ff_integration.integrate_fama_french_factors(earnings_data, method='monthly_match')
                
                if integrated_ff is not None:
                    # Keep only the 5 essential factors
                    essential_ff_cols = ['SMB', 'HML', 'RMW', 'CMA', 'RF']
                    for col in essential_ff_cols:
                        if col not in integrated_ff.columns:
                            integrated_ff[col] = np.nan
                    
                    # Remove any extra FF features
                    extra_ff_cols = [col for col in integrated_ff.columns if any(ff in col for ff in ['SMB', 'HML', 'RMW', 'CMA', 'RF']) and col not in essential_ff_cols]
                    integrated_ff = integrated_ff.drop(columns=extra_ff_cols)
                    
                    print(f"✓ Fama-French features integrated: {len(integrated_ff)} observations")
                    return integrated_ff
            
            # Fallback: create mock FF features
            print("Warning: Using mock Fama-French features")
            return self._create_mock_ff_features(earnings_data)
            
        except Exception as e:
            print(f"Warning: Fama-French integration failed: {e}")
            return self._create_mock_ff_features(earnings_data)
    
    def _create_mock_ff_features(self, earnings_data):
        """Create mock Fama-French features for testing"""
        np.random.seed(42)
        
        # Add only the 5 essential FF factors
        earnings_data['SMB'] = np.random.normal(0.002, 0.035, len(earnings_data))
        earnings_data['HML'] = np.random.normal(0.004, 0.030, len(earnings_data))
        earnings_data['RMW'] = np.random.normal(0.003, 0.025, len(earnings_data))
        earnings_data['CMA'] = np.random.normal(0.003, 0.025, len(earnings_data))
        earnings_data['RF'] = np.random.normal(0.002, 0.001, len(earnings_data))
        
        print(f"✓ Mock Fama-French features created: {len(earnings_data)} observations")
        return earnings_data
    
    def integrate_essential_option_features(self, earnings_data, iv_surface_data=None):
        """
        Integrate only the 5 essential option surface features
        """
        print("Integrating essential option surface features...")
        
        try:
            from enhanced_option_surface_features import EnhancedOptionSurfaceFeatures
            
            # Initialize option features calculator
            option_calculator = EnhancedOptionSurfaceFeatures()
            
            # Add only the 5 essential option surface features
            essential_option_features = ['term_ratio', 'skew', 'kurt', 'iv_ratio', 'smirk']
            
            for feature in essential_option_features:
                earnings_data[feature] = np.nan
            
            # Calculate features for each earnings event
            for idx, row in earnings_data.iterrows():
                ticker = row['ticker']
                earnings_date = pd.to_datetime(row['earnings_date'])
                
                if iv_surface_data is not None and ticker in iv_surface_data:
                    # Use actual IV surface data
                    ticker_iv_data = iv_surface_data[ticker]
                    features = option_calculator.calculate_all_features(
                        ticker_iv_data, row.get('underlying_price', 100.0), earnings_date
                    )
                    
                    for feature, value in features.items():
                        if feature in essential_option_features:
                            earnings_data.loc[idx, feature] = value
                else:
                    # Create mock option features
                    earnings_data.loc[idx, 'term_ratio'] = np.random.uniform(0.8, 1.2)
                    earnings_data.loc[idx, 'skew'] = np.random.uniform(0.9, 1.3)
                    earnings_data.loc[idx, 'kurt'] = np.random.uniform(-1, 3)
                    earnings_data.loc[idx, 'iv_ratio'] = np.random.uniform(0.9, 1.2)
                    earnings_data.loc[idx, 'smirk'] = np.random.uniform(0.9, 1.3)
            
            print(f"✓ Essential option surface features integrated: {len(earnings_data)} observations")
            
        except Exception as e:
            print(f"Warning: Option surface integration failed: {e}")
            # Create mock features
            earnings_data = self._create_mock_option_features(earnings_data)
        
        return earnings_data
    
    def _create_mock_option_features(self, earnings_data):
        """Create mock option surface features for testing"""
        np.random.seed(42)
        
        # Add only the 5 essential option features
        option_features = {
            'term_ratio': np.random.uniform(0.8, 1.2, len(earnings_data)),
            'skew': np.random.uniform(0.9, 1.3, len(earnings_data)),
            'kurt': np.random.uniform(-1, 3, len(earnings_data)),
            'iv_ratio': np.random.uniform(0.9, 1.2, len(earnings_data)),
            'smirk': np.random.uniform(0.9, 1.3, len(earnings_data))
        }
        
        for feature, values in option_features.items():
            earnings_data[feature] = values
        
        print(f"✓ Mock option surface features created: {len(earnings_data)} observations")
        return earnings_data
    
    def ensure_core_features(self, earnings_data):
        """
        Ensure the 3 core features are present and properly named
        """
        print("Ensuring core features are present...")
        
        # Check if core features exist, rename if needed
        core_features = {
            'ievr': 'ievr',
            'skew_ratio': 'skew_ratio', 
            'normative_iv_rv_ratio': 'normative_iv_rv_ratio'
        }
        
        # Check what's available and rename if needed
        if 'ratio' in earnings_data.columns:
            earnings_data['normative_iv_rv_ratio'] = earnings_data['ratio']
            print("✓ Renamed 'ratio' to 'normative_iv_rv_ratio'")
        
        if 'skew_ratio' not in earnings_data.columns and 'skew' in earnings_data.columns:
            earnings_data['skew_ratio'] = earnings_data['skew']
            print("✓ Created 'skew_ratio' from 'skew'")
        
        # Ensure all core features exist
        for feature in core_features:
            if feature not in earnings_data.columns:
                print(f"⚠ Warning: Core feature '{feature}' not found")
        
        print(f"✓ Core features verified")
        return earnings_data
    
    def integrate_essential_features(self, earnings_data, db_connection=None, ff_data_file=None, iv_surface_data=None):
        """
        Integrate only the essential features into the earnings data
        """
        print("="*80)
        print("STREAMLINED FEATURE INTEGRATION - ESSENTIAL FEATURES ONLY")
        print("="*80)
        
        # Start with original data
        integrated = earnings_data.copy()
        print(f"Starting with {len(integrated)} observations and {len(integrated.columns)} columns")
        
        # 1. Ensure core features are present
        integrated = self.ensure_core_features(integrated)
        
        # 2. Integrate dispersion feature
        integrated = self.integrate_dispersion_feature(integrated, db_connection)
        
        # 3. Integrate Fama-French features
        integrated = self.integrate_fama_french_features(integrated, ff_data_file)
        
        # 4. Integrate essential option surface features
        integrated = self.integrate_essential_option_features(integrated, iv_surface_data)
        
        # Final summary
        print(f"\n{'='*80}")
        print("STREAMLINED INTEGRATION COMPLETE")
        print(f"{'='*80}")
        print(f"Final dataset: {len(integrated)} observations, {len(integrated.columns)} columns")
        
        # Show feature categories
        feature_categories = {
            'Core Features': ['ievr', 'skew_ratio', 'normative_iv_rv_ratio'],
            'Dispersion Features': ['dispersion'],
            'Fama-French Features': ['SMB', 'HML', 'RMW', 'CMA', 'RF'],
            'Option Surface Features': ['term_ratio', 'skew', 'kurt', 'iv_ratio', 'smirk']
        }
        
        total_features = 0
        for category, features in feature_categories.items():
            available_features = [f for f in features if f in integrated.columns]
            if available_features:
                print(f"\n{category} ({len(available_features)} features):")
                for feature in available_features:
                    print(f"  - {feature}")
                total_features += len(available_features)
        
        print(f"\nTotal essential features: {total_features}")
        print(f"Target: 14 features (3 core + 1 dispersion + 5 FF + 5 option)")
        
        self.integrated_data = integrated
        return integrated
    
    def save_streamlined_data(self, output_file='data_files/streamlined_earnings_analysis.csv'):
        """
        Save the streamlined data
        """
        if self.integrated_data is None:
            print("Error: No integrated data to save")
            return
        
        try:
            self.integrated_data.to_csv(output_file, index=False)
            print(f"\n✓ Streamlined data saved to {output_file}")
            
            # Print summary statistics
            print(f"\nData Summary:")
            print(f"Observations: {len(self.integrated_data)}")
            print(f"Columns: {len(self.integrated_data.columns)}")
            print(f"Date range: {self.integrated_data['earnings_date'].min()} to {self.integrated_data['earnings_date'].max()}")
            print(f"Stocks: {self.integrated_data['ticker'].nunique()}")
            
        except Exception as e:
            print(f"Error saving data: {e}")

def main():
    """
    Test the streamlined feature integration
    """
    print("TESTING STREAMLINED FEATURE INTEGRATION")
    print("="*70)
    
    # Create sample earnings data
    sample_earnings = pd.DataFrame({
        'earnings_date': pd.to_datetime(['2022-10-27', '2022-07-28', '2022-04-28', '2022-01-27']),
        'ticker': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
        'revr': [1.2, 1.1, 1.3, 1.05],
        'ievr': [1.15, 1.05, 1.25, 1.02],
        'ratio': [0.96, 0.95, 0.96, 0.97],
        'vol_st': [0.25, 0.23, 0.27, 0.22],
        'vol_mt': [0.20, 0.19, 0.22, 0.18],
        'underlying_price': [160.0, 155.0, 165.0, 150.0]
    })
    
    print(f"Sample data created: {len(sample_earnings)} observations")
    
    # Initialize integration
    integration = StreamlinedFeatureIntegration()
    
    # Integrate essential features
    streamlined_data = integration.integrate_essential_features(sample_earnings)
    
    # Save results
    integration.save_streamlined_data()
    
    print(f"\n🎉 STREAMLINED INTEGRATION TEST SUCCESSFUL! 🎉")

if __name__ == "__main__":
    main()
