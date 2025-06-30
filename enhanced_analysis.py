"""
Enhanced Analysis Functions for Earnings IV Project
Based on meeting requirements for regression and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

def calculate_realized_volatility_estimators(stock_prices_df, windows=[5, 10, 21, 30]):
    """
    Calculate multiple realized volatility estimators for analysis
    
    Args:
        stock_prices_df: DataFrame with 'date' and 'close' columns
        windows: List of window sizes for rolling volatility
    """
    if stock_prices_df is None or len(stock_prices_df) == 0:
        print("No stock price data available for realized volatility calculation")
        return None
    
    df = stock_prices_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate log returns
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Initialize realized volatility columns
    for window in windows:
        # Standard rolling volatility
        df[f'realized_vol_{window}d'] = df['log_returns'].rolling(window=window).std() * np.sqrt(252)
        
        # Exponentially weighted volatility
        df[f'ewm_vol_{window}d'] = df['log_returns'].ewm(span=window).std() * np.sqrt(252)
        
        # Parkinson estimator (if OHLC available)
        if all(col in df.columns for col in ['high', 'low']):
            hl_ratio = np.log(df['high'] / df['low'])
            df[f'parkinson_vol_{window}d'] = np.sqrt(
                hl_ratio.rolling(window=window).apply(lambda x: np.sum(x**2) / len(x)) * 252
            )
    
    print(f"Calculated realized volatility estimators for {len(df)} observations")
    return df

def analyze_option_volume_vs_stock_adv(options_df, stock_df, ticker=None):
    """
    Analyze option volume relative to stock average daily volume
    
    Args:
        options_df: Options data DataFrame
        stock_df: Stock price data DataFrame
        ticker: Specific ticker to analyze (if None, analyze all)
    """
    if options_df is None or stock_df is None:
        print("Need both options and stock price data for volume analysis")
        return None
    
    options = options_df.copy()
    stock = stock_df.copy()
    
    # Filter by ticker if specified
    if ticker and 'ticker' in options.columns:
        options = options[options['ticker'] == ticker]
        stock = stock[stock['ticker'] == ticker]
    
    if len(options) == 0:
        print(f"No options data found for {ticker if ticker else 'specified tickers'}")
        return None
    
    # Calculate stock ADV
    if 'volume' in stock.columns:
        stock_adv = stock['volume'].mean()
    else:
        print("No stock volume data available")
        return None
    
    # Calculate daily option metrics
    daily_option_metrics = options.groupby('date').agg({
        'volume': 'sum',
        'open_interest': 'sum',
        'impl_volatility': 'mean',
        'strike_price': 'count'  # Number of contracts
    }).reset_index()
    
    # Calculate option notional (assuming 100 shares per contract)
    daily_option_metrics['option_notional'] = daily_option_metrics['volume'] * 100
    
    # Volume ratios
    daily_option_metrics['volume_ratio'] = daily_option_metrics['volume'] / stock_adv * 100
    daily_option_metrics['notional_ratio'] = daily_option_metrics['option_notional'] / (stock_adv * stock['close'].mean()) * 100
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Daily option volume
    axes[0,0].plot(daily_option_metrics['date'], daily_option_metrics['volume'], 'b-', alpha=0.7)
    axes[0,0].set_title('Daily Option Volume')
    axes[0,0].set_ylabel('Contracts')
    
    # Volume ratio
    axes[0,1].plot(daily_option_metrics['date'], daily_option_metrics['volume_ratio'], 'g-', alpha=0.7)
    axes[0,1].set_title('Option Volume as % of Stock ADV')
    axes[0,1].set_ylabel('Percentage')
    
    # Notional ratio
    axes[1,0].plot(daily_option_metrics['date'], daily_option_metrics['notional_ratio'], 'r-', alpha=0.7)
    axes[1,0].set_title('Option Notional as % of Stock Notional')
    axes[1,0].set_ylabel('Percentage')
    
    # Open Interest
    axes[1,1].plot(daily_option_metrics['date'], daily_option_metrics['open_interest'], 'purple', alpha=0.7)
    axes[1,1].set_title('Daily Open Interest')
    axes[1,1].set_ylabel('Contracts')
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print(f"\n📊 Volume Analysis Summary:")
    print(f"Stock ADV: {stock_adv:,.0f}")
    print(f"Average Daily Option Volume: {daily_option_metrics['volume'].mean():,.0f}")
    print(f"Average Volume Ratio: {daily_option_metrics['volume_ratio'].mean():.1f}%")
    print(f"Average Notional Ratio: {daily_option_metrics['notional_ratio'].mean():.1f}%")
    
    return daily_option_metrics

def enhanced_kernel_regression_analysis(X, y, gamma=0.1, alpha=1.0, test_size=0.3):
    """
    Enhanced kernel regression analysis (from Wolfe paper approach)
    
    Args:
        X: Feature matrix
        y: Target variable
        gamma: RBF kernel parameter
        alpha: Regularization parameter
        test_size: Fraction of data for testing
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit kernel ridge regression
    model = KernelRidge(kernel='rbf', gamma=gamma, alpha=alpha)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    
    print(f"\n🔬 Enhanced Kernel Regression Results:")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training set
    axes[0].scatter(y_train, y_train_pred, alpha=0.6, s=20)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'Training Set (R² = {train_r2:.3f})')
    
    # Test set
    axes[1].scatter(y_test, y_test_pred, alpha=0.6, s=20)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title(f'Test Set (R² = {test_r2:.3f})')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'model': model,
        'scaler': scaler,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

def build_regression_dataset(earnings_options_df, realized_vol_df, target_window=21, feature_window=10):
    """
    Build dataset for regression analysis
    
    Args:
        earnings_options_df: Earnings-options merged data
        realized_vol_df: Realized volatility data
        target_window: Window for realized volatility target (days)
        feature_window: Window for feature calculation (days)
    """
    if earnings_options_df is None or realized_vol_df is None:
        print("Need both earnings_options and realized_volatility data")
        return None
    
    earnings_df = earnings_options_df.copy()
    rv_df = realized_vol_df.copy()
    
    # Print available columns for debugging
    print(f"Available columns in earnings_options_df: {list(earnings_df.columns)}")
    print(f"Available columns in realized_vol_df: {list(rv_df.columns)}")
    
    # Merge realized volatility with earnings options
    earnings_df['date'] = pd.to_datetime(earnings_df['date'])
    rv_df['date'] = pd.to_datetime(rv_df['date'])
    
    merged_df = earnings_df.merge(rv_df, on='date', how='left')
    
    # Check which columns are actually available
    required_columns = ['impl_volatility', 'volume', 'bid_ask_spread', 'tte', 'moneyness']
    available_columns = []
    missing_columns = []
    
    for col in required_columns:
        if col in merged_df.columns:
            available_columns.append(col)
        else:
            missing_columns.append(col)
    
    print(f"Available feature columns: {available_columns}")
    print(f"Missing columns: {missing_columns}")
    
    # Create features
    features = []
    targets = []
    dates = []
    tickers = []
    
    for ticker in merged_df['ticker'].unique():
        ticker_data = merged_df[merged_df['ticker'] == ticker].sort_values('date')
        
        for i in range(feature_window, len(ticker_data) - target_window):
            # Features (pre-earnings implied volatility characteristics)
            feature_window_data = ticker_data.iloc[i-feature_window:i]
            
            # Target (post-earnings realized volatility)
            target_window_data = ticker_data.iloc[i:i+target_window]
            
            if len(feature_window_data) > 0 and len(target_window_data) > 0:
                # Build feature vector dynamically based on available columns
                feature_vector = []
                feature_names = []
                
                # Always include implied volatility if available
                if 'impl_volatility' in available_columns:
                    feature_vector.extend([
                        feature_window_data['impl_volatility'].mean(),  # Average IV
                        feature_window_data['impl_volatility'].std()    # IV dispersion
                    ])
                    feature_names.extend(['avg_iv', 'iv_std'])
                
                # Add other features if available
                if 'volume' in available_columns:
                    feature_vector.append(feature_window_data['volume'].mean())
                    feature_names.append('avg_volume')
                else:
                    feature_vector.append(0)  # Default value
                    feature_names.append('avg_volume')
                
                if 'bid_ask_spread' in available_columns:
                    feature_vector.append(feature_window_data['bid_ask_spread'].mean())
                    feature_names.append('avg_spread')
                else:
                    feature_vector.append(0)  # Default value
                    feature_names.append('avg_spread')
                
                if 'tte' in available_columns:
                    feature_vector.append(feature_window_data['tte'].mean())
                    feature_names.append('avg_tte')
                else:
                    feature_vector.append(30)  # Default TTE
                    feature_names.append('avg_tte')
                
                if 'moneyness' in available_columns:
                    feature_vector.append(feature_window_data['moneyness'].mean())
                    feature_names.append('avg_moneyness')
                else:
                    feature_vector.append(1.0)  # Default moneyness
                    feature_names.append('avg_moneyness')
                
                # Target: realized volatility in the target window
                target_rv_col = f'realized_vol_{target_window}d'
                if target_rv_col in target_window_data.columns:
                    target_rv = target_window_data[target_rv_col].mean()
                else:
                    # Try alternative column names
                    rv_columns = [col for col in target_window_data.columns if 'vol' in col.lower()]
                    if rv_columns:
                        target_rv = target_window_data[rv_columns[0]].mean()
                    else:
                        continue  # Skip if no realized volatility data
                
                if not np.isnan(target_rv) and not any(np.isnan(feature_vector)):
                    features.append(feature_vector)
                    targets.append(target_rv)
                    dates.append(ticker_data.iloc[i]['date'])
                    tickers.append(ticker)
    
    if len(features) == 0:
        print("No valid regression data points found")
        return None
    
    # Convert to DataFrame
    regression_df = pd.DataFrame(features, columns=feature_names)
    regression_df['target_rv'] = targets
    regression_df['date'] = dates
    regression_df['ticker'] = tickers
    
    print(f"Built regression dataset with {len(regression_df)} observations")
    print(f"Feature correlation with target:")
    for feature in feature_names:
        corr = regression_df[feature].corr(regression_df['target_rv'])
        print(f"  {feature}: {corr:.3f}")
    
    return regression_df

def run_single_name_case_study(pipeline, ticker, start_date='2023-01-01', end_date='2024-12-31'):
    """
    Comprehensive single-name case study as discussed in the meeting
    
    Args:
        pipeline: EarningsIVDataPipeline instance
        ticker: Stock ticker to analyze
        start_date: Analysis start date
        end_date: Analysis end date
    """
    print(f"\n{'='*60}")
    print(f"SINGLE-NAME CASE STUDY: {ticker}")
    print(f"{'='*60}")
    
    # Get data for this ticker
    securities = pipeline.get_securities_info([ticker])
    if len(securities) == 0:
        print(f"No security data found for {ticker}")
        return None
    
    secid = securities.iloc[0]['secid']
    
    # Get all data
    earnings = pipeline.get_earnings_dates([ticker], start_date, end_date)
    stock_prices = pipeline.get_stock_prices([secid], start_date, end_date)
    options = pipeline.get_option_data([secid], start_date, end_date)
    
    if options is None or len(options) == 0:
        print(f"No options data found for {ticker}")
        return None
    
    # Calculate realized volatility
    realized_vol = calculate_realized_volatility_estimators(stock_prices)
    
    # Calculate option metrics and apply filters
    enhanced_options = pipeline.calculate_option_metrics()
    filtered_options = pipeline.apply_data_filters()
    
    # Merge with earnings
    merged_earnings = pipeline.merge_securities_earnings()
    earnings_options = pipeline.merge_earnings_options()
    
    # Run analyses
    pipeline.analyze_volatility_smile()
    pipeline.earnings_event_study()
    pipeline.analyze_term_structure()
    
    # Volume analysis
    volume_analysis = analyze_option_volume_vs_stock_adv(filtered_options, stock_prices, ticker)
    
    # Build regression dataset
    regression_data = build_regression_dataset(earnings_options, realized_vol)
    
    if regression_data is not None and len(regression_data) > 10:
        # Prepare features and target - use dynamic feature columns
        feature_cols = [col for col in regression_data.columns if col.startswith('avg_')]
        X = regression_data[feature_cols]
        y = regression_data['target_rv']
        
        print(f"Using features: {feature_cols}")
        print(f"Feature matrix shape: {X.shape}")
        
        # Run kernel regression
        regression_results = enhanced_kernel_regression_analysis(X, y)
        
        print(f"\n✅ Single-name case study complete for {ticker}")
        return {
            'ticker': ticker,
            'regression_data': regression_data,
            'regression_results': regression_results,
            'volume_analysis': volume_analysis
        }
    else:
        print(f"Insufficient data for regression analysis on {ticker}")
        return None

def get_large_cap_universe(pipeline, min_market_cap=1e9, min_option_volume=1000):
    """
    Get universe of large cap stocks with sufficient option volume
    
    Args:
        pipeline: EarningsIVDataPipeline instance
        min_market_cap: Minimum market cap threshold
        min_option_volume: Minimum average daily option volume
    """
    print("Building large cap universe with option volume filters...")
    
    query = f"""
    WITH market_caps AS (
        SELECT DISTINCT s.secid, s.ticker, s.issuer,
               p.close * s.share_vol as market_cap,
               p.date
        FROM optionm.securd1 s
        JOIN optionm.secprd p ON s.secid = p.secid
        WHERE s.exchange_d != 0
          AND p.date >= '2023-01-01'
          AND p.close * s.share_vol >= {min_market_cap}
    ),
    option_volumes AS (
        SELECT secid, 
               AVG(COALESCE(volume, 0)) as avg_daily_volume,
               COUNT(*) as trading_days
        FROM optionm.opprcd2023
        WHERE volume IS NOT NULL AND volume > 0
        GROUP BY secid
        HAVING AVG(COALESCE(volume, 0)) >= {min_option_volume}
    )
    SELECT DISTINCT m.secid, m.ticker, m.issuer, 
           AVG(m.market_cap) as avg_market_cap,
           v.avg_daily_volume as avg_option_volume
    FROM market_caps m
    JOIN option_volumes v ON m.secid = v.secid
    GROUP BY m.secid, m.ticker, m.issuer, v.avg_daily_volume
    ORDER BY avg_market_cap DESC
    LIMIT 200
    """
    
    try:
        universe = pipeline.db.raw_sql(query)
        print(f"Universe contains {len(universe)} stocks")
        return universe
    except Exception as e:
        print(f"Error building universe: {e}")
        # Fallback to manual list
        fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'UNH']
        return pipeline.get_securities_info(fallback_tickers)

def debug_data_structure(pipeline):
    """
    Debug function to understand the data structure and available columns
    
    Args:
        pipeline: EarningsIVDataPipeline instance
    """
    print("🔍 Debugging data structure...")
    
    if 'earnings_options' in pipeline.data:
        print(f"\n📊 Earnings Options Data:")
        print(f"Shape: {pipeline.data['earnings_options'].shape}")
        print(f"Columns: {list(pipeline.data['earnings_options'].columns)}")
        print(f"Sample data:")
        print(pipeline.data['earnings_options'].head())
    else:
        print("❌ No earnings_options data found")
    
    if 'stock_prices' in pipeline.data:
        print(f"\n📈 Stock Prices Data:")
        print(f"Shape: {pipeline.data['stock_prices'].shape}")
        print(f"Columns: {list(pipeline.data['stock_prices'].columns)}")
    else:
        print("❌ No stock_prices data found")
    
    if 'options' in pipeline.data:
        print(f"\n📋 Options Data:")
        print(f"Shape: {pipeline.data['options'].shape}")
        print(f"Columns: {list(pipeline.data['options'].columns)}")
    else:
        print("❌ No options data found")
    
    if 'filtered_options' in pipeline.data:
        print(f"\n🔍 Filtered Options Data:")
        print(f"Shape: {pipeline.data['filtered_options'].shape}")
        print(f"Columns: {list(pipeline.data['filtered_options'].columns)}")
    else:
        print("❌ No filtered_options data found")

# Example usage
def run_enhanced_analysis_example():
    """
    Example of how to run the enhanced analysis pipeline
    """
    import wrds
    
    # Connect to WRDS
    db = wrds.Connection(wrds_username='your_wrds_username')
    
    # Initialize pipeline (assuming you have the EarningsIVDataPipeline class)
    # pipeline = EarningsIVDataPipeline(db)
    
    # Run single-name case study
    # case_study = run_single_name_case_study(pipeline, 'AAPL')
    
    # if case_study:
    #     print(f"Case study results for {case_study['ticker']}:")
    #     print(f"Regression data points: {len(case_study['regression_data'])}")
    #     print(f"Test R²: {case_study['regression_results']['test_r2']:.3f}")
    
    db.close()
    print("Enhanced analysis functions loaded successfully!")
    print("Import this module and use the functions in your notebook.")

if __name__ == "__main__":
    print("Enhanced Analysis Functions for Earnings IV Project")
    print("Available functions:")
    print("- calculate_realized_volatility_estimators()")
    print("- analyze_option_volume_vs_stock_adv()")
    print("- enhanced_kernel_regression_analysis()")
    print("- build_regression_dataset()")
    print("- run_single_name_case_study()")
    print("- get_large_cap_universe()")
    print("- debug_data_structure()")
    print("\nImport this module in your notebook to use these functions!") 