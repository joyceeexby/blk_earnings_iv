#!/usr/bin/env python3
"""
Configuration file for Comprehensive REVR Analysis
"""

# WRDS Connection Settings
WRDS_CONFIG = {
    'username': 'joycexu020113',  # Add your WRDS username
    'password': 'JoyceXu020205'   # Add your WRDS password
}

# Analysis Parameters
ANALYSIS_CONFIG = {
    'start_quarter': 'Q1',
    'start_year': 2005,
    'end_quarter': 'Q3', 
    'end_year': 2023,
    'analysis_days_before': 30,  # Days before earnings to analyze
    'num_top_stocks': 500        # Number of top stocks by dollar volume
}

# Output Settings
OUTPUT_CONFIG = {
    'intermediate_save_frequency': 10,  # Save intermediate results every N seasons
    'final_filename': 'data_files/bulk_revr_comprehensive.csv',
    'intermediate_filename_pattern': 'data_files/bulk_revr_intermediate_{}_seasons.csv'
}

# Progress Tracking
PROGRESS_CONFIG = {
    'progress_update_frequency': 50,  # Show progress every N stocks
    'verbose_logging': True,          # Detailed logging
    'save_intermediate_results': True # Save intermediate results
}

# Error Handling
ERROR_CONFIG = {
    'continue_on_error': True,        # Continue processing if individual stocks fail
    'max_retries': 3,                 # Maximum retries for failed operations
    'log_errors': True                # Log all errors
}

# Data Quality Checks
QUALITY_CONFIG = {
    'min_revr_value': 0.01,          # Minimum valid REVR value
    'max_revr_value': 100.0,         # Maximum valid REVR value
    'require_positive_volume': True,  # Require positive trading volume
    'min_price_threshold': 1.0        # Minimum stock price threshold
}
