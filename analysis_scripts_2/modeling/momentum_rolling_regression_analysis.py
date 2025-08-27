#!/usr/bin/env python3
"""
Momentum-Enhanced Model 3 Rolling Regression Analysis
Rolling walk-forward validation with 5-year training, 6-month validation, 6-month testing
Testing Model 3 + momentum_6m and Model 3 + z_score_momentum with Linear Regression
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load data and prepare for momentum-enhanced Model 3 analysis.
    """
    print("MOMENTUM-ENHANCED MODEL 3 ROLLING REGRESSION ANALYSIS")
    print("="*70)
    
    # Load the final merged dataset with momentum features
    file_path = 'data_files/final_merged_dataset_with_momentum_final.csv'
    df = pd.read_csv(file_path)
    print("Loaded dataset: {} observations".format(len(df)))
    
    # Convert dates and add time components
    df['earnings_date'] = pd.to_datetime(df['earnings_date'])
    df['year'] = df['earnings_date'].dt.year
    df['quarter'] = df['earnings_date'].dt.quarter
    df['month'] = df['earnings_date'].dt.month
    
    # Create season identifier
    df['season'] = df['year'].astype(str) + ' Q' + df['quarter'].astype(str)
    
    # Calculate normative_iv_rv_ratio feature
    print("Creating normative_iv_rv_ratio feature...")
    df['normative_iv_rv_ratio'] = df['avg_pre'] / df['normative_realized_vol']
    
    # Handle infinite values and NaN
    df['normative_iv_rv_ratio'] = df['normative_iv_rv_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Show feature statistics
    valid_ratio = df['normative_iv_rv_ratio'].notna().sum()
    print("  Created normative_iv_rv_ratio feature: {} valid values".format(valid_ratio))
    
    # Check momentum features availability
    momentum_features = ['momentum_6m', 'z_score_momentum']
    print("\nMomentum features for analysis:")
    for feature in momentum_features:
        if feature in df.columns:
            valid_count = df[feature].notna().sum()
            total_count = len(df)
            coverage = 100.0 * valid_count / total_count
            print("  {}: {} ({:.1f}% coverage)".format(feature, valid_count, coverage))
        else:
            print("  {}: Not found in dataset".format(feature))
    
    # Sort by date
    df = df.sort_values('earnings_date').reset_index(drop=True)
    
    # Remove extreme outliers (z-score > 3)
    revr_zscore = np.abs((df['revr'] - df['revr'].mean()) / df['revr'].std())
    ievr_zscore = np.abs((df['ievr'] - df['ievr'].mean()) / df['ievr'].std())
    
    df_clean = df[(revr_zscore <= 3) & (ievr_zscore <= 3)].copy()
    print("After outlier removal: {} observations".format(len(df_clean)))
    
    # Show date range
    print("Date range: {} to {}".format(
        df_clean['earnings_date'].min().strftime('%Y-%m'),
        df_clean['earnings_date'].max().strftime('%Y-%m')
    ))
    
    return df_clean

def create_time_windows(df, train_years=5, val_months=6, test_months=6):
    """
    Create rolling time windows for walk-forward validation.
    """
    print("\nCREATING TIME WINDOWS")
    print("="*60)
    print("Training window: {} years".format(train_years))
    print("Validation window: {} months".format(val_months))
    print("Testing window: {} months".format(test_months))
    
    # Get unique dates and sort
    unique_dates = df['earnings_date'].dt.to_period('M').unique()
    unique_dates = np.sort(unique_dates)
    
    windows = []
    current_idx = 0
    
    while current_idx < len(unique_dates):
        # Training period
        train_end = current_idx + (train_years * 12) - 1
        if train_end >= len(unique_dates):
            break
            
        # Validation period
        val_start = train_end + 1
        val_end = val_start + val_months - 1
        if val_end >= len(unique_dates):
            break
            
        # Testing period
        test_start = val_end + 1
        test_end = test_start + test_months - 1
        if test_end >= len(unique_dates):
            break
            
        # Create window
        window = {
            'train_start': unique_dates[current_idx],
            'train_end': unique_dates[train_end],
            'val_start': unique_dates[val_start],
            'val_end': unique_dates[val_end],
            'test_start': unique_dates[test_start],
            'test_end': unique_dates[test_end],
            'window_id': len(windows) + 1
        }
        
        windows.append(window)
        current_idx += test_months  # Move forward by test window size
    
    print("Created {} rolling windows".format(len(windows)))
    
    # Show first few windows
    for i, window in enumerate(windows[:3]):
        print("  Window {}: Train {}-{}, Val {}-{}, Test {}-{}".format(
            i+1, window['train_start'], window['train_end'],
            window['val_start'], window['val_end'],
            window['test_start'], window['test_end']
        ))
    
    return windows

def get_data_for_window(df, window):
    """
    Extract data for a specific time window.
    """
    # Convert periods to datetime for filtering
    train_start = window['train_start'].to_timestamp()
    train_end = window['train_end'].to_timestamp()
    val_start = window['val_start'].to_timestamp()
    val_end = window['val_end'].to_timestamp()
    test_start = window['test_start'].to_timestamp()
    test_end = window['test_end'].to_timestamp()
    
    # Filter data
    train_data = df[(df['earnings_date'] >= train_start) & (df['earnings_date'] <= train_end)]
    val_data = df[(df['earnings_date'] >= val_start) & (df['earnings_date'] <= val_end)]
    test_data = df[(df['earnings_date'] >= test_start) & (df['earnings_date'] <= test_end)]
    
    return train_data, val_data, test_data

def run_single_model(X_train, y_train, X_val, y_val, X_test, y_test, model_type='LinearRegression'):
    """
    Train and evaluate a linear regression model.
    """
    # Initialize Linear Regression model
    model = LinearRegression()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    val_r2 = r2_score(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    
    # Get feature importance (coefficients for Linear Regression)
    feature_importance = np.abs(model.coef_)
    
    return {
        'model_type': model_type,
        'val_r2': val_r2,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'feature_importance': feature_importance,
        'val_pred': val_pred,
        'test_pred': test_pred
    }

def run_momentum_rolling_regression(df, windows):
    """
    Run rolling regression for momentum-enhanced Model 3 combinations.
    """
    print("\nRUNNING MOMENTUM-ENHANCED MODEL 3 ROLLING REGRESSION")
    print("="*70)
    
    # Define feature combinations
    base_features = ['ievr', 'normative_iv_rv_ratio', 'SKEW', 'KURT', 'IV_RATIO', 'SMIRK', 'vol_hl7', 'vol_hl10', 'vol_hl21']
    
    feature_combinations = {
        'Model3_plus_momentum_6m': base_features + ['momentum_6m'],
        'Model3_plus_z_score_momentum': base_features + ['z_score_momentum']
    }
    
    # Define algorithms - only Linear Regression for detailed predictions
    algorithms = ['LinearRegression']
    
    # Store all results
    all_results = {}
    
    for combo_name, features in feature_combinations.items():
        print("\nTesting feature combination: {}".format(combo_name))
        print("Features: {}".format(features))
        
        # Check which features are available
        available_features = [f for f in features if f in df.columns]
        missing_features = set(features) - set(available_features)
        
        if missing_features:
            print("  WARNING: Missing features: {} - skipping".format(missing_features))
            continue
        
        all_results[combo_name] = {}
        
        for algorithm in algorithms:
            print("\n  Running with {}...".format(algorithm))
            
            algorithm_results = []
            
            for i, window in enumerate(windows):
                # Get data for this window
                train_data, val_data, test_data = get_data_for_window(df, window)
                
                # Check if we have enough data
                if len(train_data) < 50 or len(val_data) < 10 or len(test_data) < 10:
                    continue
                
                # Prepare data
                X_train = train_data[available_features].values
                y_train = train_data['revr'].values
                X_val = val_data[available_features].values
                y_val = val_data['revr'].values
                X_test = test_data[available_features].values
                y_test = test_data['revr'].values
                
                # Remove rows with NaN values
                train_valid_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
                val_valid_mask = ~np.isnan(X_val).any(axis=1) & ~np.isnan(y_val)
                test_valid_mask = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test)
                
                X_train_clean = X_train[train_valid_mask]
                y_train_clean = y_train[train_valid_mask]
                X_val_clean = X_val[val_valid_mask]
                y_val_clean = y_val[val_valid_mask]
                X_test_clean = X_test[test_valid_mask]
                y_test_clean = y_test[test_valid_mask]
                
                if len(X_train_clean) < 30 or len(X_val_clean) < 5 or len(X_test_clean) < 5:
                    continue
                
                # Run model
                try:
                    result = run_single_model(
                        X_train_clean, y_train_clean,
                        X_val_clean, y_val_clean,
                        X_test_clean, y_test_clean,
                        model_type=algorithm
                    )
                    
                    # Add window information
                    result['window_id'] = window['window_id']
                    result['window_test_start'] = window['test_start']
                    result['window_test_end'] = window['test_end']
                    result['train_size'] = len(X_train_clean)
                    result['val_size'] = len(X_val_clean)
                    result['test_size'] = len(X_test_clean)
                    result['features'] = available_features.copy()
                    
                    algorithm_results.append(result)
                    
                except Exception as e:
                    print("    WARNING: Error in window {}: {}".format(i+1, str(e)))
                    continue
            
            all_results[combo_name][algorithm] = algorithm_results
            print("    Completed {} windows".format(len(algorithm_results)))
    
    return all_results

def analyze_results_over_time(all_results):
    """
    Analyze how test R2 changes over time for each combination and algorithm.
    """
    print("\nANALYZING RESULTS OVER TIME")
    print("="*50)
    
    # Prepare data for time series analysis
    time_series_data = []
    
    for combo_name, combo_results in all_results.items():
        for algorithm, results in combo_results.items():
            for result in results:
                time_series_data.append({
                    'combination': combo_name,
                    'algorithm': result['model_type'],  # Use actual model type
                    'window_id': result['window_id'],
                    'test_start': result['window_test_start'],
                    'test_end': result['window_test_end'],
                    'test_r2': result['test_r2'],
                    'test_rmse': result['test_rmse'],
                    'val_r2': result['val_r2'],
                    'train_size': result['train_size'],
                    'test_size': result['test_size']
                })
    
    # Convert to DataFrame
    ts_df = pd.DataFrame(time_series_data)
    
    if len(ts_df) == 0:
        print("No results to analyze")
        return None
    
    # Add test period year for aggregation
    # Convert Period objects to timestamp first, then to datetime
    ts_df['test_year'] = ts_df['test_start'].apply(lambda x: x.to_timestamp() if hasattr(x, 'to_timestamp') else pd.to_datetime(x)).dt.year
    
    # Print summary statistics
    print("PERFORMANCE SUMMARY BY COMBINATION AND ALGORITHM:")
    print("-" * 60)
    
    summary_stats = ts_df.groupby(['combination', 'algorithm']).agg({
        'test_r2': ['count', 'mean', 'std', 'min', 'max'],
        'test_rmse': 'mean',
        'window_id': 'count'
    }).round(4)
    
    print(summary_stats)
    
    return ts_df

def create_feature_importance_analysis(all_results):
    """
    Analyze feature importance across all windows and algorithms.
    """
    print("\nANALYZING FEATURE IMPORTANCE")
    print("="*50)
    
    importance_data = []
    
    for combo_name, combo_results in all_results.items():
        for algorithm, results in combo_results.items():
            if not results:
                continue
            
            # Get feature names from first result
            feature_names = results[0]['features']
            
            # Aggregate importance across all windows
            importance_matrix = np.array([r['feature_importance'] for r in results])
            avg_importance = np.mean(importance_matrix, axis=0)
            std_importance = np.std(importance_matrix, axis=0)
            
            for i, feature in enumerate(feature_names):
                importance_data.append({
                    'combination': combo_name,
                    'algorithm': results[0]['model_type'],  # Use actual model type
                    'feature': feature,
                    'avg_importance': avg_importance[i],
                    'std_importance': std_importance[i],
                    'n_windows': len(results)
                })
    
    importance_df = pd.DataFrame(importance_data)
    
    if len(importance_df) == 0:
        print("No feature importance data to analyze")
        return None
    
    # Print top features by combination and algorithm
    print("TOP FEATURES BY IMPORTANCE:")
    print("-" * 40)
    
    for combo in importance_df['combination'].unique():
        print("\n{}:".format(combo))
        for algo in importance_df[importance_df['combination'] == combo]['algorithm'].unique():
            combo_algo_data = importance_df[
                (importance_df['combination'] == combo) & 
                (importance_df['algorithm'] == algo)
            ].sort_values('avg_importance', ascending=False)
            
            print("  {} - Top 3 features:".format(algo))
            for _, row in combo_algo_data.head(3).iterrows():
                print("    {}: {:.4f} (+/- {:.4f})".format(
                    row['feature'], row['avg_importance'], row['std_importance']
                ))
    
    return importance_df

def create_visualizations(ts_df, importance_df):
    """
    Create visualizations for test R2 over time and feature importance.
    """
    print("\nCREATING VISUALIZATIONS")
    print("="*50)
    
    # Set up plotting style
    plt.style.use('default')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Test R2 over time
    ax1 = plt.subplot(2, 3, (1, 2))
    
    # Plot time series for each combination and algorithm
    for combo in ts_df['combination'].unique():
        for algo in ts_df[ts_df['combination'] == combo]['algorithm'].unique():
            combo_data = ts_df[
                (ts_df['combination'] == combo) & 
                (ts_df['algorithm'] == algo)
            ].sort_values('test_start')
            
            if len(combo_data) > 0:
                label = "{} - {}".format(combo.replace('Model3_plus_', ''), algo)
                # Convert Period to datetime for plotting
                test_dates = combo_data['test_start'].apply(lambda x: x.to_timestamp() if hasattr(x, 'to_timestamp') else pd.to_datetime(x))
                ax1.plot(test_dates, combo_data['test_r2'], 
                        marker='o', label=label, alpha=0.7)
    
    ax1.set_xlabel('Test Period Start Date')
    ax1.set_ylabel('Test R2')
    ax1.set_title('Test R2 Performance Over Time')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Average performance comparison
    ax2 = plt.subplot(2, 3, 3)
    
    performance_summary = ts_df.groupby(['combination', 'algorithm'])['test_r2'].mean().reset_index()
    performance_pivot = performance_summary.pivot(index='combination', columns='algorithm', values='test_r2')
    
    performance_pivot.plot(kind='bar', ax=ax2)
    ax2.set_title('Average Test R2 by Combination and Algorithm')
    ax2.set_ylabel('Average Test R2')
    ax2.set_xlabel('Feature Combination')
    ax2.legend(title='Algorithm')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Feature importance heatmaps
    if importance_df is not None:
        # Create heatmap for each algorithm
        algorithms = importance_df['algorithm'].unique()
        
        for i, algorithm in enumerate(algorithms):
            ax = plt.subplot(2, 3, 4 + i)
            
            algo_data = importance_df[importance_df['algorithm'] == algorithm]
            heatmap_data = algo_data.pivot(index='feature', columns='combination', values='avg_importance')
            
            # Fill NaN values with 0
            heatmap_data = heatmap_data.fillna(0)
            
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
            ax.set_title('Feature Importance - {}'.format(algorithm))
            ax.set_xlabel('Feature Combination')
            ax.set_ylabel('Features')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'output_files/momentum_rolling_regression_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("Visualization saved to: {}".format(output_file))
    plt.show()

def collect_detailed_predictions(all_results, df, windows):
    """
    Collect detailed predicted vs actual values for all combinations and algorithms.
    """
    print("\nCOLLECTING DETAILED PREDICTIONS")
    print("="*50)
    
    detailed_predictions = []
    
    for combo_name, combo_results in all_results.items():
        print("Processing predictions for {}...".format(combo_name))
        
        for algorithm, results in combo_results.items():
            if not results:
                continue
                
            for result in results:
                window_id = result['window_id']
                window = windows[window_id - 1]  # Convert to 0-based index
                
                # Get the original data for this window
                train_data, val_data, test_data = get_data_for_window(df, window)
                
                # Get feature names and prepare data (same as in main analysis)
                features = result['features']
                
                # Process test data
                X_test = test_data[features].values
                y_test = test_data['revr'].values
                test_valid_mask = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test)
                
                if test_valid_mask.sum() > 0:
                    # Get clean test data
                    test_data_clean = test_data[test_valid_mask].copy()
                    
                    # Add predictions to the dataframe
                    test_data_clean['predicted_revr'] = result['test_pred']
                    test_data_clean['actual_revr'] = result['test_pred']  # This will be overwritten below
                    test_data_clean['actual_revr'] = y_test[test_valid_mask]
                    
                    # Add metadata
                    test_data_clean['combination'] = combo_name
                    test_data_clean['algorithm'] = result['model_type']
                    test_data_clean['window_id'] = window_id
                    test_data_clean['test_start'] = window['test_start']
                    test_data_clean['test_end'] = window['test_end']
                    test_data_clean['train_start'] = window['train_start']
                    test_data_clean['train_end'] = window['train_end']
                    test_data_clean['period'] = 'test'
                    
                    # Calculate prediction error
                    test_data_clean['prediction_error'] = test_data_clean['predicted_revr'] - test_data_clean['actual_revr']
                    test_data_clean['abs_prediction_error'] = np.abs(test_data_clean['prediction_error'])
                    
                    # Select relevant columns for output
                    output_columns = [
                        'ticker', 'earnings_date', 'combination', 'algorithm', 'window_id',
                        'test_start', 'test_end', 'train_start', 'train_end', 'period',
                        'actual_revr', 'predicted_revr', 'prediction_error', 'abs_prediction_error'
                    ] + features
                    
                    # Only include columns that exist
                    available_columns = [col for col in output_columns if col in test_data_clean.columns]
                    
                    detailed_predictions.append(test_data_clean[available_columns])
    
    if detailed_predictions:
        # Combine all predictions
        all_predictions_df = pd.concat(detailed_predictions, ignore_index=True)
        print("Collected {} detailed predictions".format(len(all_predictions_df)))
        return all_predictions_df
    else:
        print("No detailed predictions to collect")
        return None

def save_detailed_results(ts_df, importance_df, predictions_df=None):
    """
    Save detailed results to CSV files.
    """
    print("\nSAVING DETAILED RESULTS")
    print("="*50)
    
    # Save time series results
    if ts_df is not None:
        ts_output_file = 'output_files/momentum_rolling_regression_time_series.csv'
        ts_df.to_csv(ts_output_file, index=False)
        print("Time series results saved to: {}".format(ts_output_file))
    
    # Save feature importance results
    if importance_df is not None:
        importance_output_file = 'output_files/momentum_rolling_regression_feature_importance.csv'
        importance_df.to_csv(importance_output_file, index=False)
        print("Feature importance results saved to: {}".format(importance_output_file))
    
    # Save detailed predictions - both combined and separate files
    if predictions_df is not None:
        # Save combined predictions file
        combined_output_file = 'output_files/momentum_rolling_regression_predictions_combined.csv'
        predictions_df.to_csv(combined_output_file, index=False)
        print("Combined predictions saved to: {}".format(combined_output_file))
        
        # Save separate files for each combination
        print("\nSaving separate prediction files for each combination:")
        for combination in predictions_df['combination'].unique():
            combo_data = predictions_df[predictions_df['combination'] == combination]
            
            # Create filename based on combination name
            if 'momentum_6m' in combination:
                filename = 'output_files/model3_plus_momentum_6m_predictions.csv'
                combo_name = 'Model 3 + momentum_6m'
            elif 'z_score_momentum' in combination:
                filename = 'output_files/model3_plus_z_score_momentum_predictions.csv'
                combo_name = 'Model 3 + z_score_momentum'
            else:
                # Fallback filename
                safe_name = combination.replace(' ', '_').lower()
                filename = 'output_files/{}_predictions.csv'.format(safe_name)
                combo_name = combination
            
            combo_data.to_csv(filename, index=False)
            print("  {}: {} predictions saved to {}".format(
                combo_name, len(combo_data), filename
            ))
        
        # Print overall summary of predictions
        print("\nOverall Predictions Summary:")
        print("  Total predictions: {}".format(len(predictions_df)))
        print("  Combinations: {}".format(list(predictions_df['combination'].unique())))
        print("  Algorithms: {}".format(list(predictions_df['algorithm'].unique())))
        
        # Summary by combination
        print("\nBy combination:")
        for combination in predictions_df['combination'].unique():
            combo_data = predictions_df[predictions_df['combination'] == combination]
            print("  {}: {} predictions".format(combination, len(combo_data)))
            
            # Show algorithm breakdown
            for algorithm in combo_data['algorithm'].unique():
                algo_data = combo_data[combo_data['algorithm'] == algorithm]
                print("    {}: {} predictions".format(algorithm, len(algo_data)))
        
        if len(predictions_df) > 0:
            print("  Date range: {} to {}".format(
                predictions_df['earnings_date'].min(), 
                predictions_df['earnings_date'].max()
            ))

def main():
    """
    Main function to run momentum-enhanced Model 3 rolling regression analysis.
    """
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Create time windows
    windows = create_time_windows(df)
    
    # Run rolling regression analysis
    all_results = run_momentum_rolling_regression(df, windows)
    
    if any(combo_results for combo_results in all_results.values()):
        # Analyze results over time
        ts_df = analyze_results_over_time(all_results)
        
        # Analyze feature importance
        importance_df = create_feature_importance_analysis(all_results)
        
        # Collect detailed predictions (predicted vs actual REVR)
        predictions_df = collect_detailed_predictions(all_results, df, windows)
        
        # Create visualizations
        if ts_df is not None:
            create_visualizations(ts_df, importance_df)
        
        # Save detailed results including predictions
        save_detailed_results(ts_df, importance_df, predictions_df)
        
        print("\nMomentum-enhanced Model 3 rolling regression analysis completed successfully!")
        
    else:
        print("No results generated")

if __name__ == "__main__":
    main()
