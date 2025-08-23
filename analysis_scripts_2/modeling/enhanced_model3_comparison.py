#!/usr/bin/env python3
"""
Enhanced Model 3 Comparison: Linear Regression vs Random Forest vs XGBoost
Compare performance and feature importance across different algorithms for Model 3
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
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
              f"Test {window['test_start']}-{window['test_end']}")
    
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

def run_enhanced_model3_comparison(df, windows):
    """
    Run Model 3 with Linear Regression, Random Forest, and XGBoost.
    """
    print(f"\n🔬 RUNNING ENHANCED MODEL 3 COMPARISON")
    print("="*60)
    
    # Define Model 3 features
    model3_features = ['ievr', 'normative_iv_rv_ratio', 'SKEW', 'KURT', 'IV_RATIO', 'SMIRK', 'vol_hl7', 'vol_hl10', 'vol_hl21']
    
    # Check which features are available
    available_features = [f for f in model3_features if f in df.columns]
    print(f"📋 Available features: {available_features}")
    print(f"📊 Target variable: revr")
    
    # Store results for each algorithm
    algorithms = {
        'Linear_Regression': LinearRegression(),
        'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    all_results = {algo_name: [] for algo_name in algorithms.keys()}
    all_predictions = {algo_name: [] for algo_name in algorithms.keys()}
    all_feature_importance = {algo_name: [] for algo_name in algorithms.keys()}
    
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
        
        # Run each algorithm
        for algo_name, model in algorithms.items():
            print(f"  🔍 Running {algo_name}...")
            
            # Prepare training data
            X_train = train_data[available_features].values
            y_train = train_data['revr'].values
            
            # Remove rows with NaN values in features
            valid_mask = ~np.isnan(X_train).any(axis=1)
            X_train_clean = X_train[valid_mask]
            y_train_clean = y_train[valid_mask]
            
            if len(X_train_clean) < 30:
                print(f"    ⚠️  Insufficient clean training data - skipping algorithm")
                continue
            
            # Train model
            model.fit(X_train_clean, y_train_clean)
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_)
            else:
                feature_importance = np.ones(len(available_features)) / len(available_features)
            
            # Validation performance
            X_val = val_data[available_features].values
            y_val = val_data['revr'].values
            
            val_valid_mask = ~np.isnan(X_val).any(axis=1)
            X_val_clean = X_val[val_valid_mask]
            y_val_clean = y_val[val_valid_mask]
            
            if len(X_val_clean) < 5:
                print(f"    ⚠️  Insufficient clean validation data - skipping algorithm")
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
                print(f"    ⚠️  Insufficient clean test data - skipping algorithm")
                continue
            
            y_test_pred = model.predict(X_test_clean)
            
            test_r2 = r2_score(y_test_clean, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test_clean, y_test_pred))
            test_mae = mean_absolute_error(y_test_clean, y_test_pred)
            
            # Store results
            window_result = {
                'algorithm': algo_name,
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
                'val_r2': val_r2,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae
            }
            
            all_results[algo_name].append(window_result)
            
            # Store predictions
            val_data_clean = val_data[val_valid_mask].copy()
            val_data_clean['predicted_revr'] = y_val_pred
            val_data_clean['actual_revr'] = y_val_clean
            val_data_clean['algorithm'] = algo_name
            val_data_clean['window_id'] = window['window_id']
            val_data_clean['period'] = 'validation'
            
            test_data_clean = test_data[test_valid_mask].copy()
            test_data_clean['predicted_revr'] = y_test_pred
            test_data_clean['actual_revr'] = y_test_clean
            test_data_clean['algorithm'] = algo_name
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
            all_predictions[algo_name].append(window_predictions)
            
            # Store feature importance
            feature_importance_dict = {
                'algorithm': algo_name,
                'window_id': window['window_id'],
                'test_start': window['test_start'].strftime('%Y-%m'),
                'test_end': window['test_end'].strftime('%Y-%m')
            }
            
            for j, feature in enumerate(available_features):
                feature_importance_dict[f'importance_{feature}'] = feature_importance[j]
            
            all_feature_importance[algo_name].append(feature_importance_dict)
            
            print(f"    ✅ {algo_name}: Val R²={val_r2:.4f}, Test R²={test_r2:.4f}")
    
    return all_results, all_predictions, all_feature_importance, available_features

def analyze_and_save_results(all_results, all_predictions, all_feature_importance, available_features):
    """
    Analyze and save all results.
    """
    print(f"\n📊 ANALYZING AND SAVING RESULTS")
    print("="*60)
    
    # Save results for each algorithm
    for algo_name, results in all_results.items():
        if results:
            results_df = pd.DataFrame(results)
            output_file = f'data_files/enhanced_model3_{algo_name.lower()}_results.csv'
            results_df.to_csv(output_file, index=False)
            print(f"✅ {algo_name} results saved to: {output_file}")
            
            # Show summary
            print(f"📊 {algo_name} Summary:")
            print(f"  - Windows: {len(results_df)}")
            print(f"  - Average validation R²: {results_df['val_r2'].mean():.4f}")
            print(f"  - Average test R²: {results_df['test_r2'].mean():.4f}")
            print(f"  - Best test R²: {results_df['test_r2'].max():.4f}")
            print(f"  - Worst test R²: {results_df['test_r2'].min():.4f}")
    
    # Save predictions for each algorithm
    for algo_name, predictions in all_predictions.items():
        if predictions:
            all_preds_df = pd.concat(predictions, ignore_index=True)
            preds_file = f'data_files/enhanced_model3_{algo_name.lower()}_predictions.csv'
            all_preds_df.to_csv(preds_file, index=False)
            print(f"✅ {algo_name} predictions saved to: {preds_file}")
            print(f"  - Total predictions: {len(all_preds_df):,}")
    
    # Save feature importance for each algorithm
    for algo_name, feature_imp in all_feature_importance.items():
        if feature_imp:
            feature_imp_df = pd.DataFrame(feature_imp)
            imp_file = f'data_files/enhanced_model3_{algo_name.lower()}_feature_importance.csv'
            feature_imp_df.to_csv(imp_file, index=False)
            print(f"✅ {algo_name} feature importance saved to: {imp_file}")
    
    # Create comprehensive comparison
    create_algorithm_comparison_visualization(all_results, all_predictions, all_feature_importance, available_features)
    
    return all_results, all_predictions, all_feature_importance

def create_algorithm_comparison_visualization(all_results, all_predictions, all_feature_importance, available_features):
    """
    Create visualizations comparing all algorithms.
    """
    print(f"\n📊 CREATING ALGORITHM COMPARISON VISUALIZATIONS")
    print("="*60)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Enhanced Model 3: Algorithm Comparison', fontsize=16, fontweight='bold')
    
    # 1. Test R² comparison across windows
    ax1 = axes[0, 0]
    for algo_name, results in all_results.items():
        if results:
            results_df = pd.DataFrame(results)
            ax1.plot(results_df['window_id'], results_df['test_r2'], 
                    marker='o', label=algo_name, alpha=0.7)
    
    ax1.set_xlabel('Window ID')
    ax1.set_ylabel('Test R²')
    ax1.set_title('Algorithm Performance Across Windows')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average performance comparison
    ax2 = axes[0, 1]
    algorithm_performance = []
    algorithm_names = []
    
    for algo_name, results in all_results.items():
        if results:
            results_df = pd.DataFrame(results)
            avg_test_r2 = results_df['test_r2'].mean()
            avg_val_r2 = results_df['val_r2'].mean()
            algorithm_performance.append([avg_val_r2, avg_test_r2])
            algorithm_names.append(algo_name)
    
    if algorithm_performance:
        algorithm_performance = np.array(algorithm_performance)
        x = np.arange(len(algorithm_names))
        width = 0.35
        
        ax2.bar(x - width/2, algorithm_performance[:, 0], width, label='Validation R²', alpha=0.7)
        ax2.bar(x + width/2, algorithm_performance[:, 1], width, label='Test R²', alpha=0.7)
        
        ax2.set_xlabel('Algorithms')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Average Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(algorithm_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Feature importance comparison
    ax3 = axes[1, 0]
    feature_importance_summary = {}
    
    for algo_name, feature_imp in all_feature_importance.items():
        if feature_imp:
            feature_imp_df = pd.DataFrame(feature_imp)
            imp_cols = [col for col in feature_imp_df.columns if col.startswith('importance_')]
            
            for col in imp_cols:
                feature = col.replace('importance_', '')
                if feature not in feature_importance_summary:
                    feature_importance_summary[feature] = {}
                feature_importance_summary[feature][algo_name] = feature_imp_df[col].mean()
    
    if feature_importance_summary:
        features = list(feature_importance_summary.keys())
        x = np.arange(len(features))
        width = 0.25
        
        for i, algo_name in enumerate(algorithm_names):
            if algo_name in all_feature_importance:
                values = [feature_importance_summary[feat].get(algo_name, 0) for feat in features]
                ax3.bar(x + i*width, values, width, label=algo_name, alpha=0.7)
        
        ax3.set_xlabel('Features')
        ax3.set_ylabel('Feature Importance')
        ax3.set_title('Feature Importance Comparison Across Algorithms')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(features, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Prediction accuracy comparison
    ax4 = axes[1, 1]
    for algo_name, predictions in all_predictions.items():
        if predictions:
            all_preds_df = pd.concat(predictions, ignore_index=True)
            ax4.scatter(all_preds_df['actual_revr'], all_preds_df['predicted_revr'], 
                       alpha=0.6, s=20, label=algo_name)
    
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
    output_file = 'data_files/enhanced_model3_algorithm_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_file}")
    
    plt.show()
    
    return fig

def create_feature_importance_heatmap(all_feature_importance, available_features):
    """
    Create a heatmap showing feature importance across algorithms and time.
    """
    print(f"\n📊 CREATING FEATURE IMPORTANCE HEATMAP")
    print("="*60)
    
    # Aggregate feature importance by algorithm
    algo_feature_importance = {}
    
    for algo_name, feature_imp in all_feature_importance.items():
        if feature_imp:
            feature_imp_df = pd.DataFrame(feature_imp)
            imp_cols = [col for col in feature_imp_df.columns if col.startswith('importance_')]
            
            for col in imp_cols:
                feature = col.replace('importance_', '')
                if feature not in algo_feature_importance:
                    algo_feature_importance[feature] = {}
                algo_feature_importance[feature][algo_name] = feature_imp_df[col].mean()
    
    if algo_feature_importance:
        # Create heatmap DataFrame
        heatmap_df = pd.DataFrame(algo_feature_importance).T
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_df, annot=True, cmap='YlOrRd', fmt='.3f', cbar_kws={'label': 'Feature Importance'})
        plt.title('Feature Importance Heatmap: Algorithm Comparison')
        plt.xlabel('Algorithms')
        plt.ylabel('Features')
        plt.tight_layout()
        
        # Save the plot
        output_file = 'data_files/enhanced_model3_feature_importance_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Feature importance heatmap saved to: {output_file}")
        
        plt.show()
        
        return heatmap_df
    
    return None

def main():
    """
    Main function to run the enhanced Model 3 comparison.
    """
    print("🔬 ENHANCED MODEL 3 COMPARISON: LINEAR REGRESSION vs RANDOM FOREST vs XGBOOST")
    print("="*80)
    print("Features: IEVR + normative_iv_rv_ratio + SKEW + KURT + IV_RATIO + SMIRK + vol_hl7 + vol_hl10 + vol_hl21")
    print("="*80)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Create time windows
    windows = create_time_windows(df)
    
    # Run enhanced Model 3 comparison
    all_results, all_predictions, all_feature_importance, available_features = run_enhanced_model3_comparison(df, windows)
    
    # Analyze and save results
    if any(all_results.values()):
        analyze_and_save_results(all_results, all_predictions, all_feature_importance, available_features)
        
        # Create feature importance heatmap
        create_feature_importance_heatmap(all_feature_importance, available_features)
        
        print(f"\n🎉 Enhanced Model 3 comparison completed successfully!")
        print(f"📊 Generated results for {sum(len(results) > 0 for results in all_results.values())} algorithms")
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()

