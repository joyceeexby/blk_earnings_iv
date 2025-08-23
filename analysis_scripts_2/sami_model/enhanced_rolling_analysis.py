"""
Enhanced Rolling Walk-Forward Analysis for IEVR-REVR Relationship

This module implements an enhanced rolling walk-forward analysis with:
- Adaptive window sizes based on market regimes
- Regime-specific models and features
- Ensemble methods across multiple window sizes
- Enhanced monitoring and early warning systems
- Dynamic feature selection based on market conditions

Key Enhancements:
- Market regime detection and classification
- Adaptive training windows (larger for high volatility)
- Ensemble predictions from multiple models
- Performance degradation monitoring
- Regime-specific feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Detects market regimes based on VIX levels and volatility patterns.
    """
    
    def __init__(self, vix_threshold_low=15, vix_threshold_high=25):
        self.vix_threshold_low = vix_threshold_low
        self.vix_threshold_high = vix_threshold_high
    
    def detect_regime(self, data: pd.DataFrame) -> str:
        """
        Detect market regime based on VIX levels and volatility patterns.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data containing VIX and volatility information
        
        Returns:
        --------
        str : Market regime ('low_volatility', 'normal', 'high_volatility', 'crisis')
        """
        if 'vix_analysis' not in data.columns:
            return 'normal'
        
        vix_data = data['vix_analysis'].dropna()
        if len(vix_data) == 0:
            return 'normal'
        
        vix_mean = vix_data.mean()
        vix_std = vix_data.std()
        
        # Regime classification based on VIX levels
        if vix_mean < self.vix_threshold_low:
            return 'low_volatility'
        elif vix_mean > self.vix_threshold_high:
            return 'high_volatility'
        elif vix_mean > 35:  # Crisis threshold
            return 'crisis'
        else:
            return 'normal'

class AdaptiveWindowManager:
    """
    Manages adaptive window sizes based on market regimes.
    """
    
    def __init__(self):
        self.regime_windows = {
            'low_volatility': 18,    # 18 months - standard
            'normal': 24,            # 24 months - more data
            'high_volatility': 30,   # 30 months - maximum stability
            'crisis': 36             # 36 months - crisis stability
        }
    
    def get_window_size(self, regime: str) -> int:
        """
        Get appropriate window size for market regime.
        
        Parameters:
        -----------
        regime : str
            Market regime
        
        Returns:
        --------
        int : Window size in months
        """
        return self.regime_windows.get(regime, 24)

class EnsembleModel:
    """
    Ensemble model combining multiple algorithms and window sizes.
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = {}
        self.weights = {}
        
    def create_models(self, feature_columns: List[str]) -> None:
        """
        Create ensemble of models.
        
        Parameters:
        -----------
        feature_columns : List[str]
            List of feature column names
        """
        # Linear models
        self.models['linear'] = LinearRegression()
        self.models['ridge'] = Ridge(alpha=1.0, random_state=self.random_state)
        self.models['lasso'] = Lasso(alpha=0.1, random_state=self.random_state)
        
        # Tree-based models
        self.models['rf'] = RandomForestRegressor(
            n_estimators=self.n_estimators, 
            max_depth=5, 
            random_state=self.random_state
        )
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=3,
            learning_rate=0.1,
            random_state=self.random_state
        )
        
        # Initialize equal weights
        for model_name in self.models.keys():
            self.weights[model_name] = 1.0 / len(self.models)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit all models in the ensemble.
        
        Parameters:
        -----------
        X_train, y_train : array-like
            Training data
        """
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                logger.info(f"✓ Fitted {name} model")
            except Exception as e:
                logger.error(f"✗ Failed to fit {name} model: {str(e)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble prediction.
        
        Parameters:
        -----------
        X : array-like
            Input features
        
        Returns:
        --------
        np.ndarray : Ensemble predictions
        """
        predictions = []
        valid_weights = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred * self.weights[name])
                valid_weights.append(self.weights[name])
            except Exception as e:
                logger.warning(f"✗ {name} prediction failed: {str(e)}")
        
        if not predictions:
            raise ValueError("No valid predictions from ensemble models")
        
        # Weighted average
        total_weight = sum(valid_weights)
        ensemble_pred = sum(predictions) / total_weight
        
        return ensemble_pred
    
    def update_weights(self, performance_scores: Dict[str, float]) -> None:
        """
        Update model weights based on recent performance.
        
        Parameters:
        -----------
        performance_scores : Dict[str, float]
            Recent performance scores for each model
        """
        total_score = sum(performance_scores.values())
        if total_score > 0:
            for name in self.weights.keys():
                if name in performance_scores:
                    self.weights[name] = performance_scores[name] / total_score
                else:
                    self.weights[name] = 0.01  # Small weight for missing models

class EnhancedRollingWalkForwardAnalysis:
    """
    Enhanced rolling walk-forward analysis with adaptive features.
    """
    
    def __init__(self, 
                 data_file: str = 'data_files/expanded_earnings_analysis_results_with_vix.csv',
                 output_dir: str = 'output_files',
                 data_dir: str = 'data_files'):
        """
        Initialize the enhanced rolling walk-forward analysis.
        
        Parameters:
        -----------
        data_file : str
            Path to the CSV file containing the analysis results
        output_dir : str
            Directory for output files
        data_dir : str
            Directory for data files
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.data = None
        self.results = []
        self.feature_columns = []
        
        # Enhanced components
        self.regime_detector = MarketRegimeDetector()
        self.window_manager = AdaptiveWindowManager()
        self.ensemble_model = EnsembleModel()
        
        # Performance monitoring
        self.performance_history = []
        self.degradation_warnings = []
        
        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load and prepare data
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self) -> None:
        """
        Load data and prepare features for enhanced rolling analysis.
        """
        logger.info("Loading and preparing data for enhanced rolling walk-forward analysis...")
        
        try:
            # Load data
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"Data file not found: {self.data_file}")
            
            self.data = pd.read_csv(self.data_file)
            logger.info(f"Loaded {len(self.data)} rows from {self.data_file}")
            
            # Validate required columns
            required_columns = ['earnings_date', 'revr', 'ievr']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert earnings_date to datetime
            self.data['earnings_date'] = pd.to_datetime(self.data['earnings_date'])
            
            # Sort by date to ensure temporal order
            self.data = self.data.sort_values('earnings_date').reset_index(drop=True)
            
            # Clean data - remove NaN and infinite values for target and main predictor
            initial_size = len(self.data)
            self.data = self.data[
                (self.data['revr'].notna()) & 
                (self.data['ievr'].notna()) &
                (np.isfinite(self.data['revr'])) & 
                (np.isfinite(self.data['ievr']))
            ].reset_index(drop=True)
            
            logger.info(f"Cleaned data: {initial_size} -> {len(self.data)} rows")
            
            # Prepare enhanced features
            self.prepare_enhanced_features()
            
            # Set up feature columns
            self.setup_feature_columns()
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_enhanced_features(self) -> None:
        """
        Prepare enhanced features including regime-specific features.
        """
        logger.info("Preparing enhanced features...")
        
        # Add market regime features
        self.add_regime_features()
        
        # Add volatility regime features
        self.add_volatility_features()
        
        # Add interaction features
        self.add_interaction_features()
        
        # Add lagged features
        self.add_lagged_features()
        
        logger.info("✓ Enhanced features prepared")
    
    def add_regime_features(self) -> None:
        """
        Add market regime features.
        """
        # Detect market regimes
        self.data['market_regime'] = self.data.groupby(
            self.data['earnings_date'].dt.to_period('M')
        )['vix_analysis'].transform(
            lambda x: self.regime_detector.detect_regime(pd.DataFrame({'vix_analysis': x}))
        )
        
        # Create regime dummies
        regime_dummies = pd.get_dummies(self.data['market_regime'], prefix='regime')
        self.data = pd.concat([self.data, regime_dummies], axis=1)
        
        # Add regime-specific VIX features
        self.data['vix_regime_normal'] = self.data['vix_analysis'] * (self.data['market_regime'] == 'normal')
        self.data['vix_regime_high'] = self.data['vix_analysis'] * (self.data['market_regime'] == 'high_volatility')
        self.data['vix_regime_crisis'] = self.data['vix_analysis'] * (self.data['market_regime'] == 'crisis')
    
    def add_volatility_features(self) -> None:
        """
        Add volatility regime features.
        """
        # Rolling volatility measures
        self.data['revr_volatility_30d'] = self.data['revr'].rolling(30).std()
        self.data['ievr_volatility_30d'] = self.data['ievr'].rolling(30).std()
        
        # Volatility regime
        self.data['volatility_regime'] = pd.cut(
            self.data['revr_volatility_30d'], 
            bins=[0, 0.1, 0.2, 0.3, np.inf], 
            labels=['low', 'medium', 'high', 'extreme']
        )
        
        # Volatility regime dummies
        vol_dummies = pd.get_dummies(self.data['volatility_regime'], prefix='vol_regime')
        self.data = pd.concat([self.data, vol_dummies], axis=1)
    
    def add_interaction_features(self) -> None:
        """
        Add interaction features between key variables.
        """
        # IEVR-VIX interactions
        self.data['ievr_vix_interaction'] = self.data['ievr'] * self.data['vix_analysis']
        self.data['ievr_vix_momentum'] = self.data['ievr'] * self.data['vix_momentum_5d']
        
        # Beta interactions
        self.data['ievr_beta_market'] = self.data['ievr'] * self.data['beta_market']
        self.data['ievr_beta_smb'] = self.data['ievr'] * self.data['beta_smb']
        self.data['ievr_beta_hml'] = self.data['ievr'] * self.data['beta_hml']
        
        # Skew interactions
        self.data['ievr_skew_interaction'] = self.data['ievr'] * self.data['skew_ratio']
    
    def add_lagged_features(self) -> None:
        """
        Add lagged features for time series patterns.
        """
        # Lagged REVR (previous earnings)
        self.data['revr_lag1'] = self.data.groupby('ticker')['revr'].shift(1)
        self.data['revr_lag2'] = self.data.groupby('ticker')['revr'].shift(2)
        
        # Lagged IEVR
        self.data['ievr_lag1'] = self.data.groupby('ticker')['ievr'].shift(1)
        self.data['ievr_lag2'] = self.data.groupby('ticker')['ievr'].shift(2)
        
        # Change features
        self.data['revr_change'] = self.data.groupby('ticker')['revr'].pct_change()
        self.data['ievr_change'] = self.data.groupby('ticker')['ievr'].pct_change()
    
    def setup_feature_columns(self) -> None:
        """
        Set up feature columns for the enhanced analysis.
        """
        # Base features
        base_features = [
            'ievr', 'normative_iv_rv_ratio', 'skew_ratio', 'spx_ievr', 'sector_leader_revr',
            'beta_market', 'beta_smb', 'beta_hml', 'vix_momentum_5d'
        ]
        
        # Enhanced features
        enhanced_features = [
            'vix_regime_normal', 'vix_regime_high', 'vix_regime_crisis',
            'revr_volatility_30d', 'ievr_volatility_30d',
            'vol_regime_medium', 'vol_regime_high', 'vol_regime_extreme',
            'ievr_vix_interaction', 'ievr_vix_momentum',
            'ievr_beta_market', 'ievr_beta_smb', 'ievr_beta_hml',
            'ievr_skew_interaction', 'revr_lag1', 'revr_lag2',
            'ievr_lag1', 'ievr_lag2', 'revr_change', 'ievr_change'
        ]
        
        # Combine and filter available features
        all_features = base_features + enhanced_features
        self.feature_columns = [col for col in all_features if col in self.data.columns]
        
        logger.info(f"✓ Using {len(self.feature_columns)} features: {self.feature_columns}")
    
    def detect_performance_degradation(self, recent_performance: List[float], 
                                     threshold: float = 0.1) -> bool:
        """
        Detect performance degradation.
        
        Parameters:
        -----------
        recent_performance : List[float]
            Recent performance scores
        threshold : float
            Degradation threshold
        
        Returns:
        --------
        bool : True if degradation detected
        """
        if len(recent_performance) < 3:
            return False
        
        # Calculate trend
        x = np.arange(len(recent_performance))
        slope, _, r_value, _, _ = stats.linregress(x, recent_performance)
        
        # Check if trend is significantly negative
        if slope < -threshold and r_value**2 > 0.3:
            return True
        
        return False
    
    def run_enhanced_analysis(self, 
                            start_date: str = '2013-01-01',
                            end_date: str = '2024-12-31',
                            initial_months: int = 36,
                            step_months: int = 6,
                            min_test_size: int = 10,
                            optimize_hyperparameters: bool = True) -> None:
        """
        Run enhanced rolling walk-forward analysis.
        
        Parameters:
        -----------
        start_date : str
            Start date for analysis
        end_date : str
            End date for analysis
        initial_months : int
            Initial training window size in months
        step_months : int
            Step size in months
        min_test_size : int
            Minimum test set size
        optimize_hyperparameters : bool
            Whether to optimize hyperparameters
        """
        logger.info("="*80)
        logger.info("ENHANCED ROLLING WALK-FORWARD ANALYSIS")
        logger.info("="*80)
        
        # Convert dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Initialize
        self.results = []
        current_train_start = start_date
        current_train_end = start_date + timedelta(days=initial_months * 30)
        
        iteration = 0
        while current_train_end < end_date - timedelta(days=step_months * 30):
            iteration += 1
            
            # Define test period
            test_start = current_train_end
            test_end = current_train_end + timedelta(days=step_months * 30)
            
            # Detect market regime for adaptive window
            train_data = self.data[
                (self.data['earnings_date'] >= current_train_start) & 
                (self.data['earnings_date'] <= current_train_end)
            ]
            
            regime = self.regime_detector.detect_regime(train_data)
            adaptive_window = self.window_manager.get_window_size(regime)
            
            # Adjust training window based on regime
            if adaptive_window != initial_months:
                adjusted_train_start = current_train_end - timedelta(days=adaptive_window * 30)
                train_data = self.data[
                    (self.data['earnings_date'] >= adjusted_train_start) & 
                    (self.data['earnings_date'] <= current_train_end)
                ]
            
            test_data = self.data[
                (self.data['earnings_date'] > test_start) & 
                (self.data['earnings_date'] <= test_end)
            ]
            
            # Skip if insufficient data
            if len(test_data) < min_test_size:
                logger.warning(f"Insufficient test data in iteration {iteration}: {len(test_data)} < {min_test_size}")
                current_train_start += timedelta(days=step_months * 30)
                current_train_end += timedelta(days=step_months * 30)
                continue
            
            # Prepare features
            X_train = train_data[self.feature_columns].fillna(0)
            y_train = train_data['revr']
            X_test = test_data[self.feature_columns].fillna(0)
            y_test = test_data['revr']
            
            # Scale features
            scaler = RobustScaler()  # More robust to outliers
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble model
            try:
                self.ensemble_model.create_models(self.feature_columns)
                self.ensemble_model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred_train = self.ensemble_model.predict(X_train_scaled)
                y_pred_test = self.ensemble_model.predict(X_test_scaled)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                # Cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = cross_val_score(
                    self.ensemble_model.models['rf'], 
                    X_train_scaled, y_train, 
                    cv=tscv, scoring='r2'
                )
                
                # Store results
                period_results = {
                    'iteration': iteration,
                    'train_start': current_train_start,
                    'train_end': current_train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_observations': len(train_data),
                    'test_observations': len(test_data),
                    'market_regime': regime,
                    'adaptive_window': adaptive_window,
                    'ensemble_results': {
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'train_mae': train_mae,
                        'test_mae': test_mae,
                        'cv_r2_mean': cv_scores.mean(),
                        'cv_r2_std': cv_scores.std()
                    },
                    'feature_columns': self.feature_columns.copy()
                }
                
                self.results.append(period_results)
                
                # Performance monitoring
                self.performance_history.append(test_r2)
                if len(self.performance_history) >= 3:
                    if self.detect_performance_degradation(self.performance_history[-3:]):
                        warning_msg = f"Performance degradation detected in iteration {iteration}"
                        self.degradation_warnings.append(warning_msg)
                        logger.warning(warning_msg)
                
                # Log results
                logger.info(f"\n{'='*60}")
                logger.info(f"ITERATION {iteration}")
                logger.info(f"Training period: {current_train_start.strftime('%Y-%m-%d')} to {current_train_end.strftime('%Y-%m-%d')}")
                logger.info(f"Test period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
                logger.info(f"Market regime: {regime} (window: {adaptive_window} months)")
                logger.info(f"{'='*60}")
                logger.info(f"Training observations: {len(train_data)}")
                logger.info(f"Test observations: {len(test_data)}")
                logger.info(f"Ensemble Results:")
                logger.info(f"  Training R²: {train_r2:.4f}")
                logger.info(f"  Test R²: {test_r2:.4f}")
                logger.info(f"  Training RMSE: {train_rmse:.4f}")
                logger.info(f"  Test RMSE: {test_rmse:.4f}")
                logger.info(f"  CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {str(e)}")
                continue
            
            # Move to next period
            current_train_start += timedelta(days=step_months * 30)
            current_train_end += timedelta(days=step_months * 30)
        
        logger.info(f"\n{'='*80}")
        logger.info("ENHANCED ANALYSIS COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total periods analyzed: {len(self.results)}")
        logger.info(f"Performance warnings: {len(self.degradation_warnings)}")
    
    def create_enhanced_plots(self) -> plt.Figure:
        """
        Create enhanced performance plots.
        """
        if not self.results:
            logger.error("No results to plot. Run enhanced analysis first.")
            return None
        
        logger.info("Creating enhanced performance plots...")
        
        # Prepare data for plotting
        periods = []
        ensemble_test_r2 = []
        regimes = []
        window_sizes = []
        train_sizes = []
        test_sizes = []
        
        for result in self.results:
            periods.append(result['test_start'].strftime('%Y-%m'))
            ensemble_test_r2.append(result['ensemble_results']['test_r2'])
            regimes.append(result['market_regime'])
            window_sizes.append(result['adaptive_window'])
            train_sizes.append(result['train_observations'])
            test_sizes.append(result['test_observations'])
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Enhanced Rolling Walk-Forward Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Ensemble R² over time with regime colors
        regime_colors = {'low_volatility': 'green', 'normal': 'blue', 'high_volatility': 'orange', 'crisis': 'red'}
        colors = [regime_colors.get(regime, 'gray') for regime in regimes]
        
        axes[0, 0].scatter(range(len(periods)), ensemble_test_r2, c=colors, s=50, alpha=0.7)
        axes[0, 0].plot(range(len(periods)), ensemble_test_r2, 'k-', alpha=0.3)
        axes[0, 0].set_title('Ensemble Test R² Over Time (Colored by Regime)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Time Period')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add legend for regimes
        for regime, color in regime_colors.items():
            axes[0, 0].scatter([], [], c=color, label=regime, s=50)
        axes[0, 0].legend()
        
        # Plot 2: Adaptive window sizes
        axes[0, 1].plot(range(len(periods)), window_sizes, 'o-', color='purple', linewidth=2, markersize=6)
        axes[0, 1].set_title('Adaptive Window Sizes Over Time', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Time Period')
        axes[0, 1].set_ylabel('Window Size (Months)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Sample sizes
        axes[1, 0].plot(range(len(periods)), train_sizes, 'o-', color='green', 
                       linewidth=2, markersize=4, alpha=0.8, label='Training')
        axes[1, 0].plot(range(len(periods)), test_sizes, 's-', color='orange', 
                       linewidth=2, markersize=4, alpha=0.8, label='Test')
        axes[1, 0].set_title('Sample Sizes Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Time Period')
        axes[1, 0].set_ylabel('Number of Observations')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Performance distribution by regime
        regime_performance = {}
        for regime, r2 in zip(regimes, ensemble_test_r2):
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(r2)
        
        regime_names = list(regime_performance.keys())
        regime_means = [np.mean(regime_performance[regime]) for regime in regime_names]
        regime_stds = [np.std(regime_performance[regime]) for regime in regime_names]
        
        x = np.arange(len(regime_names))
        axes[1, 1].bar(x, regime_means, yerr=regime_stds, capsize=5, 
                      color=[regime_colors.get(regime, 'gray') for regime in regime_names], alpha=0.7)
        axes[1, 1].set_title('Performance by Market Regime', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Market Regime')
        axes[1, 1].set_ylabel('Average R²')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(regime_names, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'enhanced_rolling_walk_forward_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Enhanced performance plots saved to {plot_path}")
        
        return fig
    
    def create_summary_report(self) -> pd.DataFrame:
        """
        Create comprehensive summary report.
        """
        if not self.results:
            logger.error("No results to summarize. Run enhanced analysis first.")
            return pd.DataFrame()
        
        logger.info("Creating enhanced summary report...")
        
        # Prepare summary data
        summary_data = []
        
        for result in self.results:
            summary_data.append({
                'Iteration': result['iteration'],
                'Train_Start': result['train_start'].strftime('%Y-%m-%d'),
                'Train_End': result['train_end'].strftime('%Y-%m-%d'),
                'Test_Start': result['test_start'].strftime('%Y-%m-%d'),
                'Test_End': result['test_end'].strftime('%Y-%m-%d'),
                'Market_Regime': result['market_regime'],
                'Adaptive_Window': result['adaptive_window'],
                'Train_Obs': result['train_observations'],
                'Test_Obs': result['test_observations'],
                'Ensemble_R2': result['ensemble_results']['test_r2'],
                'Ensemble_RMSE': result['ensemble_results']['test_rmse'],
                'Ensemble_MAE': result['ensemble_results']['test_mae'],
                'CV_R2_Mean': result['ensemble_results']['cv_r2_mean'],
                'CV_R2_Std': result['ensemble_results']['cv_r2_std']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_path = os.path.join(self.data_dir, 'enhanced_rolling_walk_forward_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"✓ Enhanced summary report saved to {summary_path}")
        
        # Print enhanced statistics
        self.print_enhanced_statistics(summary_df)
        
        return summary_df
    
    def print_enhanced_statistics(self, summary_df: pd.DataFrame) -> None:
        """
        Print enhanced summary statistics.
        """
        logger.info("\n" + "="*80)
        logger.info("ENHANCED ANALYSIS SUMMARY STATISTICS")
        logger.info("="*80)
        
        # Overall performance
        logger.info(f"Overall Performance:")
        logger.info(f"  Mean Ensemble R²: {summary_df['Ensemble_R2'].mean():.4f}")
        logger.info(f"  Std Ensemble R²: {summary_df['Ensemble_R2'].std():.4f}")
        logger.info(f"  Best R²: {summary_df['Ensemble_R2'].max():.4f}")
        logger.info(f"  Worst R²: {summary_df['Ensemble_R2'].min():.4f}")
        
        # Performance by regime
        logger.info(f"\nPerformance by Market Regime:")
        for regime in summary_df['Market_Regime'].unique():
            regime_data = summary_df[summary_df['Market_Regime'] == regime]
            logger.info(f"  {regime}:")
            logger.info(f"    Count: {len(regime_data)}")
            logger.info(f"    Mean R²: {regime_data['Ensemble_R2'].mean():.4f}")
            logger.info(f"    Std R²: {regime_data['Ensemble_R2'].std():.4f}")
        
        # Adaptive window analysis
        logger.info(f"\nAdaptive Window Analysis:")
        for window_size in summary_df['Adaptive_Window'].unique():
            window_data = summary_df[summary_df['Adaptive_Window'] == window_size]
            logger.info(f"  {window_size}-month window:")
            logger.info(f"    Count: {len(window_data)}")
            logger.info(f"    Mean R²: {window_data['Ensemble_R2'].mean():.4f}")
        
        # Performance warnings
        if self.degradation_warnings:
            logger.info(f"\nPerformance Warnings ({len(self.degradation_warnings)}):")
            for warning in self.degradation_warnings:
                logger.info(f"  ⚠ {warning}")
        else:
            logger.info(f"\n✓ No performance degradation warnings")
        
        logger.info(f"\n" + "="*80)

def main():
    """
    Main function to run enhanced rolling walk-forward analysis.
    """
    try:
        # Initialize enhanced analysis
        analysis = EnhancedRollingWalkForwardAnalysis()
        
        # Run enhanced analysis
        analysis.run_enhanced_analysis(
            start_date='2013-01-01',
            end_date='2024-12-31',
            initial_months=36,  # 3 years initial training
            step_months=6,      # 6-month steps
            min_test_size=10    # Minimum 10 observations for testing
        )
        
        # Create enhanced plots
        analysis.create_enhanced_plots()
        
        # Create summary report
        analysis.create_summary_report()
        
        logger.info("\n" + "="*80)
        logger.info("ENHANCED ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info("✓ Enhanced rolling walk-forward analysis finished")
        logger.info("✓ Adaptive windows implemented")
        logger.info("✓ Ensemble models trained")
        logger.info("✓ Performance monitoring active")
        logger.info("✓ Enhanced features engineered")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

