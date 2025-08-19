#!/usr/bin/env python3
"""
Enhanced Option Surface Features for Earnings Volatility Analysis
Includes: term_ratio, skew, kurt, iv_ratio, and smirk calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedOptionSurfaceFeatures:
    """
    Calculate enhanced option surface features for earnings analysis.
    """
    
    def __init__(self, db_connection=None):
        self.db = db_connection
        self.features = {}
        
    def calculate_term_ratio(self, short_term_iv, long_term_iv, short_tte=30, long_tte=90):
        """
        Calculate term ratio: short-term IV / long-term IV
        
        Args:
            short_term_iv: Short-term implied volatility
            long_term_iv: Long-term implied volatility
            short_tte: Short-term time to expiry (days)
            long_tte: Long-term time to expiry (days)
            
        Returns:
            term_ratio: Ratio of short-term to long-term IV
        """
        try:
            if pd.isna(short_term_iv) or pd.isna(long_term_iv) or long_term_iv == 0:
                return np.nan
            
            # Calculate term ratio
            term_ratio = short_term_iv / long_term_iv
            
            # Validate ratio (should typically be between 0.5 and 2.0)
            if 0.5 <= term_ratio <= 2.0:
                return term_ratio
            else:
                # Log unusual values but still return them
                print(f"Warning: Unusual term ratio: {term_ratio:.3f}")
                return term_ratio
                
        except Exception as e:
            print(f"Error calculating term ratio: {e}")
            return np.nan
    
    def calculate_skew(self, put_iv, call_iv, put_moneyness=0.9, call_moneyness=1.1):
        """
        Calculate volatility skew: put IV / call IV
        
        Args:
            put_iv: Put option implied volatility
            call_iv: Call option implied volatility
            put_moneyness: Put option moneyness (typically 0.9)
            call_moneyness: Call option moneyness (typically 1.1)
            
        Returns:
            skew: Volatility skew ratio
        """
        try:
            if pd.isna(put_iv) or pd.isna(call_iv) or call_iv == 0:
                return np.nan
            
            # Calculate skew
            skew = put_iv / call_iv
            
            # Validate skew (should typically be between 0.8 and 1.5)
            if 0.8 <= skew <= 1.5:
                return skew
            else:
                print(f"Warning: Unusual skew: {skew:.3f}")
                return skew
                
        except Exception as e:
            print(f"Error calculating skew: {e}")
            return np.nan
    
    def calculate_kurtosis(self, iv_surface_data, moneyness_range=(0.8, 1.2), tte_range=(30, 90)):
        """
        Calculate volatility kurtosis from IV surface
        
        Args:
            iv_surface_data: DataFrame with IV surface data
            moneyness_range: Moneyness range to consider
            tte_range: Time to expiry range to consider
            
        Returns:
            kurt: Volatility kurtosis
        """
        try:
            # Filter data for specified ranges
            filtered_data = iv_surface_data[
                (iv_surface_data['moneyness'].between(moneyness_range[0], moneyness_range[1])) &
                (iv_surface_data['tte'].between(tte_range[0], tte_range[1]))
            ]
            
            if len(filtered_data) < 10:
                return np.nan
            
            # Calculate kurtosis of IV values
            iv_values = filtered_data['impl_volatility'].dropna()
            if len(iv_values) < 4:
                return np.nan
            
            kurt = iv_values.kurtosis()
            
            # Validate kurtosis (should typically be between -3 and 10)
            if -3 <= kurt <= 10:
                return kurt
            else:
                print(f"Warning: Unusual kurtosis: {kurt:.3f}")
                return kurt
                
        except Exception as e:
            print(f"Error calculating kurtosis: {e}")
            return np.nan
    
    def calculate_iv_ratio(self, atm_iv, otm_iv, atm_moneyness=1.0, otm_moneyness=1.1):
        """
        Calculate IV ratio: OTM IV / ATM IV
        
        Args:
            atm_iv: At-the-money implied volatility
            otm_iv: Out-of-the-money implied volatility
            atm_moneyness: ATM moneyness (typically 1.0)
            otm_moneyness: OTM moneyness (typically 1.1)
            
        Returns:
            iv_ratio: IV ratio
        """
        try:
            if pd.isna(atm_iv) or pd.isna(otm_iv) or atm_iv == 0:
                return np.nan
            
            # Calculate IV ratio
            iv_ratio = otm_iv / atm_iv
            
            # Validate ratio (should typically be between 0.8 and 1.3)
            if 0.8 <= iv_ratio <= 1.3:
                return iv_ratio
            else:
                print(f"Warning: Unusual IV ratio: {iv_ratio:.3f}")
                return iv_ratio
                
        except Exception as e:
            print(f"Error calculating IV ratio: {e}")
            return np.nan
    
    def calculate_smirk(self, put_iv_90, put_iv_100, call_iv_100, call_iv_110):
        """
        Calculate volatility smirk: (Put_90 + Call_110) / (2 * ATM_100)
        
        Args:
            put_iv_90: Put IV at 90% moneyness
            put_iv_100: Put IV at 100% moneyness (ATM)
            call_iv_100: Call IV at 100% moneyness (ATM)
            call_iv_110: Call IV at 110% moneyness
            
        Returns:
            smirk: Volatility smirk measure
        """
        try:
            # Check for valid data
            if any(pd.isna(iv) for iv in [put_iv_90, put_iv_100, call_iv_100, call_iv_110]):
                return np.nan
            
            # Calculate ATM IV (average of put and call at 100% moneyness)
            atm_iv = (put_iv_100 + call_iv_100) / 2
            
            if atm_iv == 0:
                return np.nan
            
            # Calculate smirk
            smirk = (put_iv_90 + call_iv_110) / (2 * atm_iv)
            
            # Validate smirk (should typically be between 0.8 and 1.4)
            if 0.8 <= smirk <= 1.4:
                return smirk
            else:
                print(f"Warning: Unusual smirk: {smirk:.3f}")
                return smirk
                
        except Exception as e:
            print(f"Error calculating smirk: {e}")
            return np.nan
    
    def calculate_all_features(self, iv_surface_data, underlying_price, earnings_date):
        """
        Calculate all enhanced option surface features
        
        Args:
            iv_surface_data: DataFrame with IV surface data
            underlying_price: Current underlying price
            earnings_date: Earnings announcement date
            
        Returns:
            features_dict: Dictionary with all calculated features
        """
        print(f"Calculating enhanced option surface features...")
        
        features = {}
        
        try:
            # Filter for reasonable data
            valid_data = iv_surface_data[
                (iv_surface_data['moneyness'].between(0.7, 1.3)) &
                (iv_surface_data['tte'].between(10, 120)) &
                (iv_surface_data['impl_volatility'] > 0)
            ].copy()
            
            if len(valid_data) < 5:
                print(f"Warning: Insufficient IV data for feature calculation ({len(valid_data)} points)")
                return self._create_default_features()
            
            # Calculate term ratio (30-day vs 90-day)
            short_term = valid_data[valid_data['tte'].between(25, 35)]['impl_volatility'].mean()
            long_term = valid_data[valid_data['tte'].between(85, 95)]['impl_volatility'].mean()
            features['term_ratio'] = self.calculate_term_ratio(short_term, long_term)
            
            # Calculate skew (90% put vs 110% call)
            put_90 = valid_data[
                (valid_data['moneyness'].between(0.88, 0.92)) & 
                (valid_data['cp_flag'] == 'P')
            ]['impl_volatility'].mean()
            
            call_110 = valid_data[
                (valid_data['moneyness'].between(1.08, 1.12)) & 
                (valid_data['cp_flag'] == 'C')
            ]['impl_volatility'].mean()
            
            features['skew'] = self.calculate_skew(put_90, call_110)
            
            # Calculate kurtosis
            features['kurt'] = self.calculate_kurtosis(valid_data)
            
            # Calculate IV ratio (OTM vs ATM)
            atm_iv = valid_data[
                valid_data['moneyness'].between(0.98, 1.02)
            ]['impl_volatility'].mean()
            
            otm_iv = valid_data[
                valid_data['moneyness'].between(1.08, 1.12)
            ]['impl_volatility'].mean()
            
            features['iv_ratio'] = self.calculate_iv_ratio(atm_iv, otm_iv)
            
            # Calculate smirk
            put_100 = valid_data[
                (valid_data['moneyness'].between(0.98, 1.02)) & 
                (valid_data['cp_flag'] == 'P')
            ]['impl_volatility'].mean()
            
            call_100 = valid_data[
                (valid_data['moneyness'].between(0.98, 1.02)) & 
                (valid_data['cp_flag'] == 'C')
            ]['impl_volatility'].mean()
            
            features['smirk'] = self.calculate_smirk(put_90, put_100, call_100, call_110)
            
            # Add additional features
            features['iv_surface_richness'] = self._calculate_iv_richness(valid_data)
            features['term_structure_slope'] = self._calculate_term_slope(valid_data)
            features['moneyness_curvature'] = self._calculate_moneyness_curvature(valid_data)
            
            print(f"✓ Calculated {len(features)} option surface features")
            
        except Exception as e:
            print(f"Error calculating option surface features: {e}")
            features = self._create_default_features()
        
        self.features = features
        return features
    
    def _calculate_iv_richness(self, iv_data):
        """Calculate IV surface richness relative to historical average"""
        try:
            if len(iv_data) < 10:
                return np.nan
            
            current_iv = iv_data['impl_volatility'].mean()
            # Use rolling average as proxy for historical
            historical_iv = iv_data['impl_volatility'].rolling(window=min(20, len(iv_data))).mean().iloc[-1]
            
            if pd.isna(historical_iv) or historical_iv == 0:
                return np.nan
            
            richness = current_iv / historical_iv
            return richness
            
        except Exception as e:
            return np.nan
    
    def _calculate_term_slope(self, iv_data):
        """Calculate term structure slope"""
        try:
            # Group by TTE and calculate average IV
            term_structure = iv_data.groupby('tte')['impl_volatility'].mean().reset_index()
            
            if len(term_structure) < 3:
                return np.nan
            
            # Fit linear trend
            slope = np.polyfit(term_structure['tte'], term_structure['impl_volatility'], 1)[0]
            return slope
            
        except Exception as e:
            return np.nan
    
    def _calculate_moneyness_curvature(self, iv_data):
        """Calculate moneyness curvature"""
        try:
            # Group by moneyness and calculate average IV
            moneyness_structure = iv_data.groupby('moneyness')['impl_volatility'].mean().reset_index()
            
            if len(moneyness_structure) < 3:
                return np.nan
            
            # Fit quadratic trend
            curvature = np.polyfit(moneyness_structure['moneyness'], moneyness_structure['impl_volatility'], 2)[0]
            return curvature
            
        except Exception as e:
            return np.nan
    
    def _create_default_features(self):
        """Create default features when calculation fails"""
        return {
            'term_ratio': np.nan,
            'skew': np.nan,
            'kurt': np.nan,
            'iv_ratio': np.nan,
            'smirk': np.nan,
            'iv_surface_richness': np.nan,
            'term_structure_slope': np.nan,
            'moneyness_curvature': np.nan
        }
    
    def get_feature_summary(self):
        """Get summary of calculated features"""
        if not self.features:
            return "No features calculated yet"
        
        summary = "Option Surface Features Summary:\n"
        summary += "="*40 + "\n"
        
        for feature, value in self.features.items():
            if pd.isna(value):
                summary += f"{feature}: N/A\n"
            else:
                summary += f"{feature}: {value:.4f}\n"
        
        return summary

def main():
    """
    Test the enhanced option surface features
    """
    print("TESTING ENHANCED OPTION SURFACE FEATURES")
    print("="*60)
    
    # Create test data
    test_data = pd.DataFrame({
        'tte': [30, 30, 30, 60, 60, 60, 90, 90, 90],
        'moneyness': [0.9, 1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0, 1.1],
        'cp_flag': ['P', 'P', 'C', 'P', 'P', 'C', 'P', 'P', 'C'],
        'impl_volatility': [0.25, 0.22, 0.20, 0.23, 0.20, 0.18, 0.21, 0.19, 0.17]
    })
    
    print("Test data created:")
    print(test_data)
    
    # Initialize calculator
    calculator = EnhancedOptionSurfaceFeatures()
    
    # Calculate features
    features = calculator.calculate_all_features(test_data, 100.0, '2022-10-27')
    
    # Display results
    print("\n" + calculator.get_feature_summary())

if __name__ == "__main__":
    main()
