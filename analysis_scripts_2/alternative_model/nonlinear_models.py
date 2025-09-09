"""
Non-linear Machine Learning Models for IEVR-REVR Analysis

This module implements various non-linear models to explore the relationship between
Implied Earnings Volatility Ratio (IEVR) and Realized Earnings Volatility Ratio (REVR).

Models included:
- Random Forest Regression
- XGBoost Regression
- Linear Regression
- Model comparison and evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.inspection import permutation_importance
import warnings
from regression_analysis import FixedRegressionAnalysis
warnings.filterwarnings('ignore')

class NonlinearModelAnalysis:
    """
    Class for implementing and comparing non-linear models for IEVR-REVR analysis.
    """
    
    def __init__(self, data_file='data_files/expanded_earnings_analysis_results_with_vix.csv'):
        """
        Initialize the analysis with data.
        
        Parameters:
        -----------
        data_file : str
            Path to the CSV file containing the analysis results
        """
        self.data_file = data_file
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
        # Load and prepare data
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """
        Load data and prepare features for modeling.
        """
        print("Loading and preparing data for non-linear modeling...")
        
        # Load data
        self.data = pd.read_csv(self.data_file)
        
        # Ensure earnings_date is available for temporal splitting
        if 'earnings_date' not in self.data.columns:
            print("Warning: 'earnings_date' not found. Creating dummy dates for temporal split.")
            self.data['earnings_date'] = pd.date_range(start='2020-01-01', periods=len(self.data), freq='D')
        
        # Convert earnings_date to datetime
        self.data['earnings_date'] = pd.to_datetime(self.data['earnings_date'])
        
        # Clean data - remove NaN and infinite values
        self.data = self.data.dropna(subset=['revr', 'ievr'])
        self.data = self.data[np.isfinite(self.data['revr']) & np.isfinite(self.data['ievr'])]
        
        # Create additional features
        self.create_features()
        
        # Prepare features and target - only use truly independent features
        # Include sector peer leader REVR as exogenous information (earlier in time)
        # Include rolling beta features for systematic risk factors
        # Include VIX features for market volatility controls
        feature_columns = ['ievr', 'normative_iv_rv_ratio', 'skew_ratio', 'spx_ievr', 'sector_leader_revr', 
                          'beta_market', 'beta_smb', 'beta_hml', 'vix_momentum_5d']
        # Remove columns that might not exist
        available_features = [col for col in feature_columns if col in self.data.columns]
        
        if len(available_features) < 2:
            # Fallback to basic features
            available_features = ['ievr']
            print("Warning: Limited features available, using only IEVR")
        
        X = self.data[available_features].copy()
        y = self.data['revr'].copy()
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print(f"Final dataset: {len(X)} observations, {len(available_features)} features")
        print(f"Features: {available_features}")
        
        # Summary of placeholder usage
        print(f"\n" + "="*60)
        print("PLACEHOLDER FEATURE SUMMARY")
        print("="*60)
        
        placeholder_features = []
        
        # Check for placeholder patterns in each feature
        for feature in available_features:
            if feature in self.data.columns:
                feature_data = self.data[feature]
                
                # Check for placeholder patterns
                if feature.startswith('beta_'):
                    # Beta features are always placeholders (sector-based)
                    placeholder_features.append(f"{feature}: {len(feature_data)} observations (sector-based placeholders)")
                
                elif feature == 'sector_leader_revr':
                    # Check if this feature has many NaN values (expected for this feature)
                    nan_count = feature_data.isna().sum()
                    if nan_count > 0:
                        placeholder_features.append(f"{feature}: {nan_count} NaN values (expected for sector peer feature)")
                
                elif feature_data.nunique() == 1:  # All values are the same
                    unique_val = feature_data.iloc[0]
                    if unique_val == 1.0:
                        placeholder_features.append(f"{feature}: {len(feature_data)} observations (all = 1.0 - likely placeholder)")
                
                # Check for suspicious patterns
                elif feature_data.std() < 0.01:  # Very low variance
                    placeholder_features.append(f"{feature}: {len(feature_data)} observations (very low variance - suspicious)")
        
        if placeholder_features:
            print("⚠️  PLACEHOLDER FEATURES DETECTED:")
            for feature in placeholder_features:
                print(f"  - {feature}")
        else:
            print("✅ No placeholder features detected - all features appear to be real data")
        
        print("="*60)
        
        # Sort by date for temporal splitting
        self.data = self.data.sort_values('earnings_date').reset_index(drop=True)
        
        # Create temporal split (80% train, 20% test)
        split_idx = int(len(self.data) * 0.8)
        
        # Split data temporally
        train_data = self.data.iloc[:split_idx]
        test_data = self.data.iloc[split_idx:]
        
        self.X_train = train_data[available_features].copy()
        self.y_train = train_data['revr'].copy()
        self.X_test = test_data[available_features].copy()
        self.y_test = test_data['revr'].copy()
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {len(self.X_train)} observations")
        print(f"Test set: {len(self.X_test)} observations")
        print(f"Training period: {train_data['earnings_date'].min().strftime('%Y-%m-%d')} to {train_data['earnings_date'].max().strftime('%Y-%m-%d')}")
        print(f"Test period: {test_data['earnings_date'].min().strftime('%Y-%m-%d')} to {test_data['earnings_date'].max().strftime('%Y-%m-%d')}")
    
    def create_features(self):
        """
        Create additional features for modeling.
        """
        # Note: Removed vol_st, vol_mt, and volatility_spread features to avoid circular dependency
        # These are components of the REVR target variable (REVR = vol_st / vol_mt)
        # Only IEVR is truly independent of the target
        
        # Create normative IV/RV ratio feature
        self.create_normative_iv_rv_ratio()
        
        # Create skew ratio feature
        self.create_skew_ratio()
        
        # Create S&P 500 IEVR feature
        self.create_spx_ievr_feature()

        # Ensure sector column exists for peer feature
        if 'sector' not in self.data.columns and 'ticker' in self.data.columns:
            try:
                self.data['sector'] = self.data['ticker'].map(FixedRegressionAnalysis.ticker_to_sector())
            except Exception:
                pass

        # Create sector leader REVR feature (peer's earlier REVR within same quarter)
        self.create_sector_leader_revr_feature(min_days_gap=30)
        
        # Check if rolling beta features exist, if not create placeholders
        self.check_and_create_beta_features()
        
        # Log transformations (for positive values)
        mask_positive = (self.data['revr'] > 0) & (self.data['ievr'] > 0)
        if mask_positive.sum() > 0:
            self.data.loc[mask_positive, 'log_revr'] = np.log(self.data.loc[mask_positive, 'revr'])
            self.data.loc[mask_positive, 'log_ievr'] = np.log(self.data.loc[mask_positive, 'ievr'])
        
        # Squared terms
        self.data['ievr_squared'] = self.data['ievr'] ** 2

    def create_sector_leader_revr_feature(self, min_days_gap: int = 30):
        """
        Create `sector_leader_revr` by assigning, for each event, the average of all
        sector peers' REVR from the same year-quarter whose earnings date is at
        least `min_days_gap` days earlier.
        """
        try:
            required = {'sector', 'earnings_date', 'revr'}
            if not required.issubset(set(self.data.columns)):
                return

            df = self.data.copy()
            df['earnings_date'] = pd.to_datetime(df['earnings_date'])
            df['_year'] = df['earnings_date'].dt.year
            df['_quarter'] = df['earnings_date'].dt.quarter

            # For each event, find all qualifying peers and take their average REVR
            sector_leader_revr_list = []
            
            for idx, row in df.iterrows():
                # Find all peers in same sector, year, quarter
                peer_mask = (
                    (df['sector'] == row['sector']) &
                    (df['_year'] == row['_year']) &
                    (df['_quarter'] == row['_quarter']) &
                    (df['earnings_date'] <= row['earnings_date'] - pd.Timedelta(days=min_days_gap)) &
                    (df['ticker'] != row['ticker'])  # Exclude self
                )
                
                qualifying_peers = df[peer_mask]
                
                if len(qualifying_peers) > 0:
                    # Take average of all qualifying peers' REVR
                    avg_peer_revr = qualifying_peers['revr'].mean()
                    sector_leader_revr_list.append(avg_peer_revr)
                else:
                    sector_leader_revr_list.append(np.nan)
            
            # Assign back to main dataframe
            self.data['sector_leader_revr'] = sector_leader_revr_list
            self.data['sector_leader_revr'] = self.data['sector_leader_revr'].replace([np.inf, -np.inf], np.nan)

            # Clean temp
            self.data.drop(columns=[c for c in ['_year', '_quarter'] if c in self.data.columns], inplace=True)
            print("Created feature: sector_leader_revr (average peer REVR from same quarter, >=30 days earlier)")
        except Exception as e:
            print(f"Error creating sector_leader_revr feature: {e}")
    
    def check_and_create_beta_features(self):
        """
        Check if rolling beta features exist in the data, if not create placeholders.
        """
        print("Checking for rolling beta features...")
        
        # Check if beta features already exist
        beta_features_exist = all(feature in self.data.columns for feature in ['beta_market', 'beta_smb', 'beta_hml'])
        
        if beta_features_exist:
            # Check if they have real data (not all NaN)
            has_real_data = (
                self.data['beta_market'].notna().sum() > 0 and
                self.data['beta_smb'].notna().sum() > 0 and
                self.data['beta_hml'].notna().sum() > 0
            )
            
            if has_real_data:
                print("✓ Real rolling beta features found in data")
                print(f"  beta_market - Mean: {self.data['beta_market'].mean():.3f}, Std: {self.data['beta_market'].std():.3f}")
                print(f"  beta_smb - Mean: {self.data['beta_smb'].mean():.3f}, Std: {self.data['beta_smb'].std():.3f}")
                print(f"  beta_hml - Mean: {self.data['beta_hml'].mean():.3f}, Std: {self.data['beta_hml'].std():.3f}")
                return
            else:
                print("⚠ Beta features exist but contain only NaN values")
        
        # Create placeholder betas if real data is not available
        print("Creating placeholder beta features (sector-based)...")
        
        try:
            # Initialize beta columns
            self.data['beta_market'] = np.nan
            self.data['beta_smb'] = np.nan
            self.data['beta_hml'] = np.nan
            
            # Sector-based placeholder betas
            sector_betas = {
                'Technology': {'market': 1.2, 'smb': -0.3, 'hml': -0.2},
                'Healthcare': {'market': 0.8, 'smb': 0.1, 'hml': -0.1},
                'Financial Services': {'market': 1.1, 'smb': -0.2, 'hml': 0.3},
                'Consumer Cyclical': {'market': 1.0, 'smb': 0.0, 'hml': 0.0},
                'Communication Services': {'market': 0.9, 'smb': -0.1, 'hml': -0.1},
                'Industrials': {'market': 1.1, 'smb': 0.1, 'hml': 0.1},
                'Consumer Defensive': {'market': 0.7, 'smb': -0.1, 'hml': 0.2},
                'Energy': {'market': 1.0, 'smb': 0.0, 'hml': 0.1},
                'Basic Materials': {'market': 1.1, 'smb': 0.1, 'hml': 0.2},
                'Real Estate': {'market': 0.8, 'smb': 0.2, 'hml': 0.1},
                'Utilities': {'market': 0.6, 'smb': -0.1, 'hml': 0.3}
            }
            
            # Assign sector-based betas
            for idx, row in self.data.iterrows():
                sector = row.get('sector', 'Technology')  # Default to Technology
                if sector in sector_betas:
                    self.data.loc[idx, 'beta_market'] = sector_betas[sector]['market']
                    self.data.loc[idx, 'beta_smb'] = sector_betas[sector]['smb']
                    self.data.loc[idx, 'beta_hml'] = sector_betas[sector]['hml']
                else:
                    # Default values for unknown sectors
                    self.data.loc[idx, 'beta_market'] = 1.0
                    self.data.loc[idx, 'beta_smb'] = 0.0
                    self.data.loc[idx, 'beta_hml'] = 0.0
            
            print(f"Created placeholder beta features:")
            print(f"  beta_market - Mean: {self.data['beta_market'].mean():.3f}, Std: {self.data['beta_market'].std():.3f}")
            print(f"  beta_smb - Mean: {self.data['beta_smb'].mean():.3f}, Std: {self.data['beta_smb'].std():.3f}")
            print(f"  beta_hml - Mean: {self.data['beta_hml'].mean():.3f}, Std: {self.data['beta_hml'].std():.3f}")
            print(f"  ⚠ PLACEHOLDER VALUES: {len(self.data)} observations using sector-based placeholder betas")
            
        except Exception as e:
            print(f"Error creating placeholder beta features: {e}")
            # Create default placeholder columns if there's an error
            self.data['beta_market'] = 1.0
            self.data['beta_smb'] = 0.0
            self.data['beta_hml'] = 0.0
    
    def create_spx_ievr_feature(self):
        """
        Create S&P 500 IEVR feature.
        This captures market-level volatility expectations for comparison with individual stock IEVR.
        """
        print("Creating S&P 500 IEVR feature...")
        
        placeholder_count = 0
        
        # Check if we have the necessary data
        if 'spx_ievr' not in self.data.columns:
            print("Warning: 'spx_ievr' not found in data. Creating placeholder.")
            # Create a placeholder - in practice, this should come from your IEVR calculation
            self.data['spx_ievr'] = 1.0  # Placeholder (no market effect)
            placeholder_count = len(self.data)
        else:
            # Check if the column exists but is empty
            if self.data['spx_ievr'].isna().all():
                print("Warning: 'spx_ievr' column exists but is empty. Creating placeholder.")
                self.data['spx_ievr'] = 1.0  # Placeholder (no market effect)
                placeholder_count = len(self.data)
            else:
                # Count existing placeholder values (if all values are 1.0, they're likely placeholders)
                if self.data['spx_ievr'].nunique() == 1 and self.data['spx_ievr'].iloc[0] == 1.0:
                    placeholder_count = len(self.data)
                    print("Warning: 'spx_ievr' appears to be placeholder data (all values = 1.0)")
        
        # Handle infinite values
        self.data['spx_ievr'] = self.data['spx_ievr'].replace([np.inf, -np.inf], np.nan)
        
        print(f"Created spx_ievr feature. Non-null values: {self.data['spx_ievr'].notna().sum()}")
        if placeholder_count > 0:
            print(f"  ⚠ PLACEHOLDER VALUES: {placeholder_count} observations using placeholder (1.0)")
        
        # Print summary statistics
        if self.data['spx_ievr'].notna().sum() > 0:
            print(f"  Mean: {self.data['spx_ievr'].mean():.4f}")
            print(f"  Std: {self.data['spx_ievr'].std():.4f}")
            print(f"  Min: {self.data['spx_ievr'].min():.4f}")
            print(f"  Max: {self.data['spx_ievr'].max():.4f}")
            
            # Check for reasonable values
            if 0.5 <= self.data['spx_ievr'].mean() <= 2.0:
                print(f"  ✓ S&P 500 IEVR is in reasonable range")
            else:
                print(f"  ⚠ S&P 500 IEVR mean ({self.data['spx_ievr'].mean():.3f}) seems unusual")
            
            # Note: Removed relative_ievr feature to avoid multicollinearity with individual ievr and spx_ievr
            print(f"  ✓ Using individual ievr and spx_ievr features (no ratio to avoid redundancy)")
        else:
            print("  ⚠ No valid S&P 500 IEVR data available")
    
    def create_normative_iv_rv_ratio(self):
        """
        Create normative IV/RV ratio feature.
        This calculates the ratio of medium-term implied vol to medium-term realized vol
        at 30 days before earnings (same time point as normative implied vol in IEVR).
        """
        print("Creating normative IV/RV ratio feature...")
        
        placeholder_count_implied = 0
        placeholder_count_realized = 0
        
        # Check if we have the necessary data
        if 'normative_implied_vol' not in self.data.columns:
            print("Warning: 'normative_implied_vol' not found in data. Creating placeholder.")
            # Create a placeholder - in practice, this should come from your IEVR calculation
            self.data['normative_implied_vol'] = self.data['ievr'] * 1.0  # Placeholder
            placeholder_count_implied = len(self.data)
        
        if 'normative_realized_vol' not in self.data.columns:
            print("Warning: 'normative_realized_vol' not found in data. Creating placeholder.")
            # Create a placeholder - in practice, this should come from your REVR calculation
            self.data['normative_realized_vol'] = 1.0  # Placeholder
            placeholder_count_realized = len(self.data)
        
        # Calculate the ratio
        mask = (self.data['normative_implied_vol'] > 0) & (self.data['normative_realized_vol'] > 0)
        self.data.loc[mask, 'normative_iv_rv_ratio'] = (
            self.data.loc[mask, 'normative_implied_vol'] / 
            self.data.loc[mask, 'normative_realized_vol']
        )
        
        # Handle infinite values
        self.data['normative_iv_rv_ratio'] = self.data['normative_iv_rv_ratio'].replace([np.inf, -np.inf], np.nan)
        
        print(f"Created normative_iv_rv_ratio feature. Non-null values: {self.data['normative_iv_rv_ratio'].notna().sum()}")
        
        # Report placeholder usage
        if placeholder_count_implied > 0:
            print(f"  ⚠ PLACEHOLDER VALUES: {placeholder_count_implied} observations using placeholder for normative_implied_vol")
        if placeholder_count_realized > 0:
            print(f"  ⚠ PLACEHOLDER VALUES: {placeholder_count_realized} observations using placeholder for normative_realized_vol")
        
        # Print summary statistics
        if self.data['normative_iv_rv_ratio'].notna().sum() > 0:
            print(f"  Mean: {self.data['normative_iv_rv_ratio'].mean():.4f}")
            print(f"  Std: {self.data['normative_iv_rv_ratio'].std():.4f}")
            print(f"  Min: {self.data['normative_iv_rv_ratio'].min():.4f}")
            print(f"  Max: {self.data['normative_iv_rv_ratio'].max():.4f}")
            
            # Print additional diagnostics
            print(f"\nNormative Values Summary:")
            print(f"  Normative Implied Vol - Mean: {self.data['normative_implied_vol'].mean():.4f}")
            print(f"  Normative Realized Vol - Mean: {self.data['normative_realized_vol'].mean():.4f}")
            print(f"  IV/RV Ratio - Mean: {self.data['normative_iv_rv_ratio'].mean():.4f}")
            
            # Check for reasonable values
            if self.data['normative_iv_rv_ratio'].mean() > 1.0:
                print(f"  ✓ IV > RV on average (typical volatility risk premium)")
            else:
                print(f"  ⚠ RV > IV on average (unusual)")
    

    
    def create_skew_ratio(self):
        """
        Create skew ratio feature (95Put IV / 105Call IV).
        This captures the directional bias in volatility expectations.
        """
        print("Creating skew ratio feature (90Put / 110Call)...")
        
        placeholder_count = 0
        
        # Check if we have the necessary data
        if 'skew_ratio' not in self.data.columns:
            print("Warning: 'skew_ratio' not found in data. Creating placeholder.")
            # Create a placeholder - in practice, this should come from your IEVR calculation
            self.data['skew_ratio'] = 1.0  # Placeholder (no skew)
            placeholder_count = len(self.data)
        else:
            # Check if the column exists but is empty
            if self.data['skew_ratio'].isna().all():
                print("Warning: 'skew_ratio' column exists but is empty. Creating placeholder.")
                self.data['skew_ratio'] = 1.0  # Placeholder (no skew)
                placeholder_count = len(self.data)
            else:
                # Count existing placeholder values (if all values are 1.0, they're likely placeholders)
                if self.data['skew_ratio'].nunique() == 1 and self.data['skew_ratio'].iloc[0] == 1.0:
                    placeholder_count = len(self.data)
                    print("Warning: 'skew_ratio' appears to be placeholder data (all values = 1.0)")
        
        # Handle infinite values
        self.data['skew_ratio'] = self.data['skew_ratio'].replace([np.inf, -np.inf], np.nan)
        
        print(f"Created skew_ratio feature. Non-null values: {self.data['skew_ratio'].notna().sum()}")
        if placeholder_count > 0:
            print(f"  ⚠ PLACEHOLDER VALUES: {placeholder_count} observations using placeholder (1.0)")
        
        # Print summary statistics
        if self.data['skew_ratio'].notna().sum() > 0:
            print(f"  Mean: {self.data['skew_ratio'].mean():.4f}")
            print(f"  Std: {self.data['skew_ratio'].std():.4f}")
            print(f"  Min: {self.data['skew_ratio'].min():.4f}")
            print(f"  Max: {self.data['skew_ratio'].max():.4f}")
            
            # Check for reasonable values
            if self.data['skew_ratio'].mean() > 1.0:
                print(f"  ✓ Put skew > Call skew on average (typical for earnings)")
            else:
                print(f"  ⚠ Call skew > Put skew on average (unusual)")
            
            # Check correlation with REVR
            correlation = self.data['revr'].corr(self.data['skew_ratio'])
            print(f"  Correlation with REVR: {correlation:.4f}")
            
            if abs(correlation) > 0.1:
                print(f"  ✓ Skew ratio shows meaningful correlation with REVR")
            else:
                print(f"  ⚠ Skew ratio shows weak correlation with REVR")
    
    def train_random_forest(self, optimize_hyperparameters=True):
        """
        Train Random Forest model.
        
        Parameters:
        -----------
        optimize_hyperparameters : bool
            Whether to perform hyperparameter optimization
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*60)
        
        if optimize_hyperparameters:
            # Hyperparameter grid for optimization
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=42)
            # Use TimeSeriesSplit for temporal cross-validation
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            
            grid_search = GridSearchCV(
                rf, param_grid, cv=tscv, scoring='r2', n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            best_rf = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Use default parameters
            best_rf = RandomForestRegressor(
                n_estimators=100, 
                max_depth=5, 
                random_state=42
            )
            best_rf.fit(self.X_train_scaled, self.y_train)
        
        # Store model
        self.models['random_forest'] = best_rf
        
        # Evaluate model
        self.evaluate_model('random_forest', 'Random Forest')
        
        return best_rf
    
    def train_xgboost(self, optimize_hyperparameters=True):
        """
        Train XGBoost model.
        
        Parameters:
        -----------
        optimize_hyperparameters : bool
            Whether to perform hyperparameter optimization
        """
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)
        
        if optimize_hyperparameters:
            # Hyperparameter grid for optimization
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=42)
            # Use TimeSeriesSplit for temporal cross-validation
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=tscv, scoring='r2', n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            best_xgb = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Use default parameters
            best_xgb = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            best_xgb.fit(self.X_train_scaled, self.y_train)
        
        # Store model
        self.models['xgboost'] = best_xgb
        
        # Evaluate model
        self.evaluate_model('xgboost', 'XGBoost')
        
        return best_xgb
    
    def train_linear_regression(self):
        """
        Train Linear Regression model with proper NaN handling.
        """
        print("\n" + "="*60)
        print("TRAINING LINEAR REGRESSION MODEL")
        print("="*60)
        
        # Handle NaN values for linear regression - impute sector_leader_revr only
        X_train_lr = self.X_train.copy()
        X_test_lr = self.X_test.copy()
        
        # Impute sector_leader_revr NaN values with sector median or global median
        if 'sector_leader_revr' in X_train_lr.columns:
            # Calculate sector medians for imputation
            sector_medians = {}
            for sector in self.data['sector'].unique():
                sector_data = self.data[self.data['sector'] == sector]
                sector_median = sector_data['sector_leader_revr'].median()
                if pd.notna(sector_median):
                    sector_medians[sector] = sector_median
            
            # Global median as fallback
            global_median = self.data['sector_leader_revr'].median()
            
            # Impute training data
            for idx in X_train_lr.index:
                if pd.isna(X_train_lr.loc[idx, 'sector_leader_revr']):
                    sector = self.data.loc[idx, 'sector']
                    if sector in sector_medians:
                        X_train_lr.loc[idx, 'sector_leader_revr'] = sector_medians[sector]
                    else:
                        X_train_lr.loc[idx, 'sector_leader_revr'] = global_median
            
            # Impute test data
            for idx in X_test_lr.index:
                if pd.isna(X_test_lr.loc[idx, 'sector_leader_revr']):
                    sector = self.data.loc[idx, 'sector']
                    if sector in sector_medians:
                        X_test_lr.loc[idx, 'sector_leader_revr'] = sector_medians[sector]
                    else:
                        X_test_lr.loc[idx, 'sector_leader_revr'] = global_median
            
            print(f"Imputed {X_train_lr['sector_leader_revr'].isna().sum()} NaN values in sector_leader_revr")
        
        # Remove any remaining NaN values (should be minimal now)
        train_mask = ~(X_train_lr.isna().any(axis=1) | self.y_train.isna())
        test_mask = ~(X_test_lr.isna().any(axis=1) | self.y_test.isna())
        
        X_train_clean = X_train_lr[train_mask]
        y_train_clean = self.y_train[train_mask]
        X_test_clean = X_test_lr[test_mask]
        y_test_clean = self.y_test[test_mask]
        
        # Scale the data
        X_train_scaled_clean = self.scaler.transform(X_train_clean)
        X_test_scaled_clean = self.scaler.transform(X_test_clean)
        
        print(f"Linear Regression using {len(X_train_clean)} observations (removed {len(self.X_train) - len(X_train_clean)} rows with other NaN values)")
        
        lr = LinearRegression()
        lr.fit(X_train_scaled_clean, y_train_clean)
        
        # Store model
        self.models['linear_regression'] = lr
        
        # Store clean versions for evaluation
        self.X_train_clean = X_train_clean
        self.y_train_clean = y_train_clean
        self.X_train_scaled_clean = X_train_scaled_clean
        self.X_test_clean = X_test_clean
        self.y_test_clean = y_test_clean
        self.X_test_scaled_clean = X_test_scaled_clean
        
        # Evaluate model
        self.evaluate_model('linear_regression', 'Linear Regression')
        
        return lr
    
    def evaluate_model(self, model_name, model_display_name):
        """
        Evaluate a trained model.
        
        Parameters:
        -----------
        model_name : str
            Key name of the model in self.models
        model_display_name : str
            Display name for output
        """
        model = self.models[model_name]
        
        # Use clean data for linear regression, regular data for others
        if model_name == 'linear_regression':
            X_train_eval = self.X_train_scaled_clean
            y_train_eval = self.y_train_clean
            X_test_eval = self.X_test_scaled_clean
            y_test_eval = self.y_test_clean
        else:
            X_train_eval = self.X_train_scaled
            y_train_eval = self.y_train
            X_test_eval = self.X_test_scaled
            y_test_eval = self.y_test
        
        # Predictions
        y_train_pred = model.predict(X_train_eval)
        y_test_pred = model.predict(X_test_eval)
        
        # Metrics
        train_r2 = r2_score(y_train_eval, y_train_pred)
        test_r2 = r2_score(y_test_eval, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_eval, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_eval, y_test_pred))
        train_mae = mean_absolute_error(y_train_eval, y_train_pred)
        test_mae = mean_absolute_error(y_test_eval, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_eval, y_train_eval, cv=5, scoring='r2')
        
        # Store results
        self.results[model_name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        # Print results
        print(f"\n{model_display_name} Results:")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Training RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Training MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    def analyze_feature_importance(self):
        """
        Analyze feature importance for all models.
        """
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        feature_names = self.X_train.columns.tolist()
        
        for model_name, model in self.models.items():
            print(f"\n{model_name.upper()} Feature Importance:")
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models have feature_importances_
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                for i, idx in enumerate(indices):
                    print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
            
            elif hasattr(model, 'coef_'):
                # Linear models have coefficients
                importances = np.abs(model.coef_)
                indices = np.argsort(importances)[::-1]
                
                for i, idx in enumerate(indices):
                    print(f"  {feature_names[idx]}: {model.coef_[idx]:.4f} (abs: {importances[idx]:.4f})")
            
            # Permutation importance (more robust)
            try:
                # Use clean data for linear regression
                if model_name == 'linear_regression':
                    X_test_perm = self.X_test_scaled_clean
                    y_test_perm = self.y_test_clean
                else:
                    X_test_perm = self.X_test_scaled
                    y_test_perm = self.y_test
                
                perm_importance = permutation_importance(
                    model, X_test_perm, y_test_perm, 
                    n_repeats=10, random_state=42
                )
                
                print(f"\n{model_name.upper()} Permutation Importance:")
                perm_indices = np.argsort(perm_importance.importances_mean)[::-1]
                
                for i, idx in enumerate(perm_indices):
                    print(f"  {feature_names[idx]}: {perm_importance.importances_mean[idx]:.4f} "
                          f"(±{perm_importance.importances_std[idx]:.4f})")
            except Exception as e:
                print(f"  Permutation importance failed: {e}")
    
    def plot_model_comparison(self):
        """
        Create comparison plots for all models.
        """
        print("\n" + "="*60)
        print("CREATING MODEL COMPARISON PLOTS")
        print("="*60)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Non-linear Model Comparison: IEVR vs REVR', fontsize=16)
        
        # Plot 1: Actual vs Predicted (Test Set)
        ax1 = axes[0, 0]
        colors = ['blue', 'red', 'green']
        
        for i, (model_name, results) in enumerate(self.results.items()):
            # Use clean test data for linear regression
            if model_name == 'linear_regression':
                y_test_plot = self.y_test_clean
            else:
                y_test_plot = self.y_test
            
            ax1.scatter(y_test_plot, results['y_test_pred'], 
                       alpha=0.6, label=model_name.replace('_', ' ').title(), 
                       color=colors[i % len(colors)])
        
        # Perfect prediction line
        all_y_test = []
        all_y_pred = []
        for model_name, results in self.results.items():
            if model_name == 'linear_regression':
                all_y_test.append(self.y_test_clean)
            else:
                all_y_test.append(self.y_test)
            all_y_pred.append(results['y_test_pred'])
        
        min_val = min(min(y.min() for y in all_y_test), min(r['y_test_pred'].min() for r in self.results.values()))
        max_val = max(max(y.max() for y in all_y_test), max(r['y_test_pred'].max() for r in self.results.values()))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax1.set_xlabel('Actual REVR')
        ax1.set_ylabel('Predicted REVR')
        ax1.set_title('Actual vs Predicted (Test Set)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        ax2 = axes[0, 1]
        for i, (model_name, results) in enumerate(self.results.items()):
            # Use clean test data for linear regression
            if model_name == 'linear_regression':
                y_test_resid = self.y_test_clean
            else:
                y_test_resid = self.y_test
            
            residuals = y_test_resid - results['y_test_pred']
            ax2.scatter(results['y_test_pred'], residuals, 
                       alpha=0.6, label=model_name.replace('_', ' ').title(),
                       color=colors[i % len(colors)])
        
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Predicted REVR')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Model Performance Comparison
        ax3 = axes[1, 0]
        model_names = list(self.results.keys())
        test_r2_scores = [self.results[name]['test_r2'] for name in model_names]
        cv_scores = [self.results[name]['cv_mean'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax3.bar(x - width/2, test_r2_scores, width, label='Test R²', alpha=0.8)
        ax3.bar(x + width/2, cv_scores, width, label='CV R²', alpha=0.8)
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('R² Score')
        ax3.set_title('Model Performance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels([name.replace('_', ' ').title() for name in model_names])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Feature Importance (Random Forest or Linear Regression)
        ax4 = axes[1, 1]
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feature_names = self.X_train.columns.tolist()
            
            ax4.bar(range(len(importances)), importances[indices])
            ax4.set_xlabel('Features')
            ax4.set_ylabel('Importance')
            ax4.set_title('Random Forest Feature Importance')
            ax4.set_xticks(range(len(importances)))
            ax4.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
        elif 'linear_regression' in self.models:
            lr_model = self.models['linear_regression']
            importances = np.abs(lr_model.coef_)
            indices = np.argsort(importances)[::-1]
            feature_names = self.X_train.columns.tolist()
            
            ax4.bar(range(len(importances)), importances[indices])
            ax4.set_xlabel('Features')
            ax4.set_ylabel('|Coefficient|')
            ax4.set_title('Linear Regression Feature Importance')
            ax4.set_xticks(range(len(importances)))
            ax4.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output_files/nonlinear_model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Model comparison plots saved to output_files/nonlinear_model_comparison.png")
        plt.show()
    
    def print_summary_table(self):
        """
        Print a summary table comparing all models.
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Test R²': f"{results['test_r2']:.4f}",
                'CV R²': f"{results['cv_mean']:.4f} (±{results['cv_std']:.4f})",
                'Test RMSE': f"{results['test_rmse']:.4f}",
                'Test MAE': f"{results['test_mae']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_df.to_csv('data_files/nonlinear_model_summary.csv', index=False)
        print("\n✓ Model summary saved to data_files/nonlinear_model_summary.csv")
    
    def run_complete_analysis(self, optimize_hyperparameters=True):
        """
        Run complete non-linear analysis.
        
        Parameters:
        -----------
        optimize_hyperparameters : bool
            Whether to perform hyperparameter optimization
        """
        print("="*80)
        print("NON-LINEAR MACHINE LEARNING ANALYSIS")
        print("="*80)
        
        # Train models
        self.train_random_forest(optimize_hyperparameters)
        self.train_xgboost(optimize_hyperparameters)
        self.train_linear_regression() # Added Linear Regression training
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        # Create plots
        self.plot_model_comparison()
        
        # Print summary
        self.print_summary_table()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)


def main():
    """
    Main function to run the non-linear analysis.
    """
    # Run analysis with hyperparameter optimization
    analysis = NonlinearModelAnalysis()
    analysis.run_complete_analysis(optimize_hyperparameters=True)


if __name__ == "__main__":
    main() 