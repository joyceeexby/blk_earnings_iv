#!/usr/bin/env python3
"""
Comprehensive Model Comparison with Walk-Forward Validation
Using final_merged_dataset.csv with 3 different feature sets:

Model 1: IEVR + normative_iv_rv_ratio + optimal volatility features (best of 7)
Model 2: IEVR + normative_iv_rv_ratio + 4 options features (SKEW, KURT, IV_RATIO, SMIRK)
Model 3: IEVR + normative_iv_rv_ratio + 4 options features + optimal volatility features

All models use the same walk-forward validation: 5-year training, 6-month validation, 6-month testing
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load the final merged dataset and prepare it for analysis.
    """
    print("📊 LOADING AND PREPARING FINAL MERGED DATASET")
    print("="*60)
    
    # Load the final merged dataset
    file_path = 'data_files/final_merged_dataset.csv'
    df = pd.read_csv(file_path)
    print(f"✅ Loaded dataset: {len(df):,} observations")
    
    # Convert dates and add time components
    df['earnings_date'] = pd.to_datetime(df['earnings_date'])
    df['year'] = df['earnings_date'].dt.year
    df['quarter'] = df['earnings_date'].dt.quarter
    df['month'] = df['earnings_date'].dt.month
    
    # Create season identifier
    df['season'] = df['year'].astype(str) + ' Q' + df['quarter'].astype(str)
    
    # Calculate normative_iv_rv_ratio feature
    print("🔬 Creating normative_iv_rv_ratio feature...")
    df['normative_iv_rv_ratio'] = df['avg_pre'] / df['normative_realized_vol']
    
    # Handle infinite values and NaN
    df['normative_iv_rv_ratio'] = df['normative_iv_rv_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Show feature statistics
    valid_ratio = df['normative_iv_rv_ratio'].notna().sum()
    print(f"  ✅ Created normative_iv_rv_ratio feature: {valid_ratio:,} valid values")
    if valid_ratio > 0:
        print(f"  📊 Mean: {df['normative_iv_rv_ratio'].mean():.4f}")
        print(f"  📊 Std: {df['normative_iv_rv_ratio'].std():.4f}")
        print(f"  📊 Range: {df['normative_iv_rv_ratio'].min():.4f} to {df['normative_iv_rv_ratio'].max():.4f}")
    
    # Sort by date
    df = df.sort_values('earnings_date').reset_index(drop=True)
    
    # Remove extreme outliers (z-score > 3)
    revr_zscore = np.abs((df['revr'] - df['revr'].mean()) / df['revr'].std())
    ievr_zscore = np.abs((df['ievr'] - df['ievr'].mean()) / df['ievr'].std())
    
    df_clean = df[(revr_zscore <= 3) & (ievr_zscore <= 3)].copy()
    print(f"🧹 After outlier removal: {len(df_clean):,} observations")
    
    # Show date range
    print(f"📅 Date range: {df_clean['earnings_date'].min().strftime('%Y-%m')} to {df_clean['earnings_date'].max().strftime('%Y-%m')}")
    
    return df_clean

def select_optimal_volatility_features(df, target_col='revr'):
    """
    Select the best volatility features using feature selection.
    """
    print("🔍 SELECTING OPTIMAL VOLATILITY FEATURES")
    print("="*50)
    
    # Define volatility features
    vol_features = ['ret', 'vol_hl5', 'vol_hl7', 'vol_hl10', 'vol_hl21', 'vol_hl63', 'vol_hl126']
    
    # Check which features exist and have data
    available_vol_features = [col for col in vol_features if col in df.columns and df[col].notna().sum() > 0]
    
    if len(available_vol_features) == 0:
        print("❌ No volatility features available")
        return []
    
    print(f"Available volatility features: {available_vol_features}")
    
    # Prepare data for feature selection
    X_vol = df[available_vol_features].fillna(df[available_vol_features].mean())
    y = df[target_col]
    
    # Remove rows with NaN in target
    valid_mask = y.notna()
    X_vol_clean = X_vol[valid_mask]
    y_clean = y[valid_mask]
    
    if len(X_vol_clean) == 0:
        print("❌ No valid data for feature selection")
        return []
    
    # Use SelectKBest to select top features
    k = min(3, len(available_vol_features))  # Select top 3 or all if less than 3
    selector = SelectKBest(score_func=f_regression, k=k)
    
    try:
        selector.fit(X_vol_clean, y_clean)
        selected_features = [available_vol_features[i] for i in selector.get_support(indices=True)]
        
        # Get feature scores
        scores = selector.scores_
        feature_scores = list(zip(available_vol_features, scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Feature selection scores:")
        for feature, score in feature_scores:
            status = "✅ SELECTED" if feature in selected_features else "❌ NOT SELECTED"
            print(f"  {feature}: {score:.2f} {status}")
        
        print(f"✅ Selected {len(selected_features)} optimal volatility features: {selected_features}")
        return selected_features
        
    except Exception as e:
        print(f"❌ Error in feature selection: {e}")
        # Fallback: return first 3 available features
        fallback_features = available_vol_features[:3]
        print(f"🔄 Fallback: using first 3 features: {fallback_features}")
        return fallback_features

def create_time_windows(df, train_years=5, val_months=6, test_months=6):
    """
    Create rolling time windows for walk-forward validation.
    """
    print(f"\n🕒 CREATING TIME WINDOWS")
    print("="*60)
    print(f"Training window: {train_years} years")
    print(f"Validation window: {val_months} months")
    print(f"Testing window: {test_months} months")
    
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
    
    print(f"📊 Created {len(windows)} rolling windows")
    
    # Show first few windows
    for i, window in enumerate(windows[:3]):
        print(f"  Window {i+1}: Train {window['train_start']}-{window['train_end']}, "
              f"Val {window['val_start']}-{window['val_end']}, "
              f"Test {window['train_start']}-{window['test_end']}")
    
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

def run_model_comparison(df, windows, optimal_vol_features):
    """
    Run all three models for comparison.
    """
    print(f"\n🔬 RUNNING MODEL COMPARISON")
    print("="*60)
    
    # Define the three models
    models = {
        'Model 1': {
            'name': 'IEVR + normative_iv_rv_ratio + optimal volatility',
            'features': ['ievr', 'normative_iv_rv_ratio'] + optimal_vol_features
        },
        'Model 2': {
            'name': 'IEVR + normative_iv_rv_ratio + options features',
            'features': ['ievr', 'normative_iv_rv_ratio', 'SKEW', 'KURT', 'IV_RATIO', 'SMIRK']
        },
        'Model 3': {
            'name': 'IEVR + normative_iv_rv_ratio + options + optimal volatility',
            'features': ['ievr', 'normative_iv_rv_ratio', 'SKEW', 'KURT', 'IV_RATIO', 'SMIRK'] + optimal_vol_features
        }
    }
    
    # Store results for each model
    all_results = {model_name: [] for model_name in models.keys()}
    all_predictions = {model_name: [] for model_name in models.keys()}
    
    for i, window in enumerate(windows):
        print(f"\n📊 Processing Window {i+1}/{len(windows)}")
        print(f"  Train: {window['train_start']} to {window['train_end']}")
        print(f"  Val:   {window['val_start']} to {window['val_end']}")
        print(f"  Test:  {window['test_start']} to {window['test_end']}")
        
        # Get data for this window
        train_data, val_data, test_data = get_data_for_window(df, window)
        
        # Check if we have enough data
        if len(train_data) < 50 or len(val_data) < 10 or len(test_data) < 10:
            print(f"  ⚠️  Insufficient data - skipping window")
            continue
        
        # Run each model
        for model_name, model_config in models.items():
            print(f"  🔍 Running {model_name}...")
            
            # Check if all features are available
            available_features = [f for f in model_config['features'] if f in df.columns]
            if len(available_features) != len(model_config['features']):
                missing = set(model_config['features']) - set(available_features)
                print(f"    ⚠️  Missing features: {missing} - skipping model")
                continue
            
            # Prepare training data
            X_train = train_data[available_features].values
            y_train = train_data['revr'].values
            
            # Remove rows with NaN values in features
            valid_mask = ~np.isnan(X_train).any(axis=1)
            X_train_clean = X_train[valid_mask]
            y_train_clean = y_train[valid_mask]
            
            if len(X_train_clean) < 30:
                print(f"    ⚠️  Insufficient clean training data - skipping model")
                continue
            
            # Train model
            model = LinearRegression()
            model.fit(X_train_clean, y_train_clean)
            
            # Get coefficients
            intercept = model.intercept_
            coefficients = dict(zip(available_features, model.coef_))
            
            # Validation performance
            X_val = val_data[available_features].values
            y_val = val_data['revr'].values
            
            val_valid_mask = ~np.isnan(X_val).any(axis=1)
            X_val_clean = X_val[val_valid_mask]
            y_val_clean = y_val[val_valid_mask]
            
            if len(X_val_clean) < 5:
                print(f"    ⚠️  Insufficient clean validation data - skipping model")
                continue
            
            y_val_pred = model.predict(X_val_clean)
            
            val_r2 = r2_score(y_val_clean, y_val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val_clean, y_val_pred))
            val_mae = mean_absolute_error(y_val_clean, y_val_pred)
            
            # Test performance
            X_test = test_data[available_features].values
            y_test = test_data['revr'].values
            
            test_valid_mask = ~np.isnan(X_test).any(axis=1)
            X_test_clean = X_test[test_valid_mask]
            y_test_clean = y_test[test_valid_mask]
            
            if len(X_test_clean) < 5:
                print(f"    ⚠️  Insufficient clean test data - skipping model")
                continue
            
            y_test_pred = model.predict(X_test_clean)
            
            test_r2 = r2_score(y_test_clean, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test_clean, y_test_pred))
            test_mae = mean_absolute_error(y_test_clean, y_test_pred)
            
            # Store results
            window_result = {
                'model_name': model_name,
                'window_id': window['window_id'],
                'train_start': window['train_start'].strftime('%Y-%m'),
                'train_end': window['train_end'].strftime('%Y-%m'),
                'val_start': window['val_start'].strftime('%Y-%m'),
                'val_end': window['val_end'].strftime('%Y-%m'),
                'test_start': window['test_start'].strftime('%Y-%m'),
                'test_end': window['test_end'].strftime('%Y-%m'),
                'train_obs': len(train_data),
                'val_obs': len(val_data),
                'test_obs': len(test_data),
                'features_used': available_features,
                'intercept': intercept,
                'val_r2': val_r2,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae
            }
            
            # Add coefficients
            for feature, coef in coefficients.items():
                window_result[f'coef_{feature}'] = coef
            
            all_results[model_name].append(window_result)
            
            # Store predictions
            val_data_clean = val_data[val_valid_mask].copy()
            val_data_clean['predicted_revr'] = y_val_pred
            val_data_clean['actual_revr'] = y_val_clean
            val_data_clean['model_name'] = model_name
            val_data_clean['window_id'] = window['window_id']
            val_data_clean['period'] = 'validation'
            
            test_data_clean = test_data[test_valid_mask].copy()
            test_data_clean['predicted_revr'] = y_test_pred
            test_data_clean['actual_revr'] = y_test_clean
            test_data_clean['model_name'] = model_name
            test_data_clean['window_id'] = window['window_id']
            test_data_clean['period'] = 'test'
            
            # Add window dates
            for data_clean in [val_data_clean, test_data_clean]:
                data_clean['train_start'] = window['train_start'].strftime('%Y-%m')
                data_clean['train_end'] = window['train_end'].strftime('%Y-%m')
                data_clean['val_start'] = window['val_start'].strftime('%Y-%m')
                data_clean['val_end'] = window['val_end'].strftime('%Y-%m')
                data_clean['test_start'] = window['test_start'].strftime('%Y-%m')
                data_clean['test_end'] = window['test_end'].strftime('%Y-%m')
            
            window_predictions = pd.concat([val_data_clean, test_data_clean], ignore_index=True)
            all_predictions[model_name].append(window_predictions)
            
            print(f"    ✅ {model_name}: Val R²={val_r2:.4f}, Test R²={test_r2:.4f}")
    
    return all_results, all_predictions

def analyze_and_save_results(all_results, all_predictions, optimal_vol_features):
    """
    Analyze and save all results.
    """
    print(f"\n📊 ANALYZING AND SAVING RESULTS")
    print("="*60)
    
    # Save results for each model
    for model_name, results in all_results.items():
        if results:
            results_df = pd.DataFrame(results)
            output_file = f'data_files/model_comparison_{model_name.replace(" ", "_").lower()}_results.csv'
            results_df.to_csv(output_file, index=False)
            print(f"✅ {model_name} results saved to: {output_file}")
            
            # Show summary
            print(f"📊 {model_name} Summary:")
            print(f"  - Windows: {len(results_df)}")
            print(f"  - Average validation R²: {results_df['val_r2'].mean():.4f}")
            print(f"  - Average test R²: {results_df['test_r2'].mean():.4f}")
            print(f"  - Best test R²: {results_df['test_r2'].max():.4f}")
            print(f"  - Worst test R²: {results_df['test_r2'].min():.4f}")
    
    # Save predictions for each model
    for model_name, predictions in all_predictions.items():
        if predictions:
            all_preds_df = pd.concat(predictions, ignore_index=True)
            preds_file = f'data_files/model_comparison_{model_name.replace(" ", "_").lower()}_predictions.csv'
            all_preds_df.to_csv(preds_file, index=False)
            print(f"✅ {model_name} predictions saved to: {preds_file}")
            print(f"  - Total predictions: {len(all_preds_df):,}")
    
    # Create comprehensive comparison
    create_model_comparison_visualization(all_results, all_predictions, optimal_vol_features)
    
    return all_results, all_predictions

def create_model_comparison_visualization(all_results, all_predictions, optimal_vol_features):
    """
    Create visualizations comparing all models.
    """
    print(f"\n📊 CREATING MODEL COMPARISON VISUALIZATIONS")
    print("="*60)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison: Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Test R² comparison across windows
    ax1 = axes[0, 0]
    for model_name, results in all_results.items():
        if results:
            results_df = pd.DataFrame(results)
            ax1.plot(results_df['window_id'], results_df['test_r2'], 
                    marker='o', label=model_name, alpha=0.7)
    
    ax1.set_xlabel('Window ID')
    ax1.set_ylabel('Test R²')
    ax1.set_title('Model Performance Across Windows')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average performance comparison
    ax2 = axes[0, 1]
    model_performance = []
    model_names = []
    
    for model_name, results in all_results.items():
        if results:
            results_df = pd.DataFrame(results)
            avg_test_r2 = results_df['test_r2'].mean()
            avg_val_r2 = results_df['val_r2'].mean()
            model_performance.append([avg_val_r2, avg_test_r2])
            model_names.append(model_name)
    
    if model_performance:
        model_performance = np.array(model_performance)
        x = np.arange(len(model_names))
        width = 0.35
        
        ax2.bar(x - width/2, model_performance[:, 0], width, label='Validation R²', alpha=0.7)
        ax2.bar(x + width/2, model_performance[:, 1], width, label='Test R²', alpha=0.7)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Average Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels([name.replace('Model ', '') for name in model_names])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Feature importance (coefficient stability)
    ax3 = axes[1, 0]
    feature_importance = {}
    
    for model_name, results in all_results.items():
        if results:
            results_df = pd.DataFrame(results)
            coef_cols = [col for col in results_df.columns if col.startswith('coef_')]
            
            for col in coef_cols:
                feature = col.replace('coef_', '')
                if feature not in feature_importance:
                    feature_importance[feature] = []
                feature_importance[feature].extend(results_df[col].tolist())
    
    if feature_importance:
        features = list(feature_importance.keys())
        means = [np.mean(feature_importance[feat]) for feat in features]
        stds = [np.std(feature_importance[feat]) for feat in features]
        
        y_pos = np.arange(len(features))
        ax3.barh(y_pos, means, xerr=stds, alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(features)
        ax3.set_xlabel('Coefficient Value')
        ax3.set_title('Feature Importance (Coefficient Stability)')
        ax3.grid(True, alpha=0.3)
    
    # 4. Prediction accuracy comparison
    ax4 = axes[1, 1]
    for model_name, predictions in all_predictions.items():
        if predictions:
            all_preds_df = pd.concat(predictions, ignore_index=True)
            ax4.scatter(all_preds_df['actual_revr'], all_preds_df['predicted_revr'], 
                       alpha=0.6, s=20, label=model_name)
    
    # Add perfect prediction line
    all_actual = []
    all_predicted = []
    for predictions in all_predictions.values():
        if predictions:
            all_preds_df = pd.concat(predictions, ignore_index=True)
            all_actual.extend(all_preds_df['actual_revr'].tolist())
            all_predicted.extend(all_preds_df['predicted_revr'].tolist())
    
    if all_actual and all_predicted:
        min_val = min(min(all_actual), min(all_predicted))
        max_val = max(max(all_actual), max(all_predicted))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax4.set_xlabel('Actual REVR')
    ax4.set_ylabel('Predicted REVR')
    ax4.set_title('Prediction Accuracy Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'data_files/model_comparison_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_file}")
    
    plt.show()
    
    return fig

def main():
    """
    Main function to run the comprehensive model comparison.
    """
    print("🔬 COMPREHENSIVE MODEL COMPARISON WITH WALK-FORWARD VALIDATION")
    print("="*80)
    print("Models:")
    print("  • Model 1: IEVR + normative_iv_rv_ratio + optimal volatility features")
    print("  • Model 2: IEVR + normative_iv_rv_ratio + options features")
    print("  • Model 3: IEVR + normative_iv_rv_ratio + options + optimal volatility")
    print("="*80)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Select optimal volatility features
    optimal_vol_features = select_optimal_volatility_features(df)
    
    # Create time windows
    windows = create_time_windows(df)
    
    # Run model comparison
    all_results, all_predictions = run_model_comparison(df, windows, optimal_vol_features)
    
    # Analyze and save results
    if any(all_results.values()):
        analyze_and_save_results(all_results, all_predictions, optimal_vol_features)
        print(f"\n🎉 Model comparison completed successfully!")
        print(f"📊 Generated results for {sum(len(results) > 0 for results in all_results.values())} models")
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()

