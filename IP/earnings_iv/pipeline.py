# pipeline.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .queries import build_optionm_query, build_rdq_query_from_tickers, build_secprd_query
from .analysis import calculate_option_metrics, apply_data_filters, merge_earnings_options, calculate_eiv, print_options_summary
from .plotting import plot_volatility_smile, plot_term_structure, plot_event_study_iv
import matplotlib.pyplot as plt

class EarningsIVDataPipeline:
    """
    Enhanced data pipeline for earnings implied volatility analysis using WRDS data
    """
    def __init__(self, db_connection):
        self.db = db_connection
        self.data = {}
        self.available_tables = None

    def setup_optionm_tables(self):
        if self.available_tables is None:
            tables_df = self.db.raw_sql("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'optionm'
                ORDER BY table_name
            """)
            self.available_tables = set(tables_df['table_name'].str.lower())
        return self.available_tables

    def get_securities_info(self, ticker_list):
        print("Fetching security information from OptionMetrics...")
        formatted_tickers = ', '.join([f"'{ticker}'" for ticker in ticker_list])
        query = f"""
        SELECT DISTINCT *
        FROM optionm.securd1
        WHERE ticker IN ({formatted_tickers})
          AND exchange_d != 0
        ORDER BY ticker
        """
        self.data['securities'] = self.db.raw_sql(query)
        print(f"Retrieved {len(self.data['securities'])} securities")
        return self.data['securities']

    def get_earnings_dates(self, ticker_list, start_date='2023-01-01', end_date='2024-12-31'):
        print("Fetching earnings announcement dates from Compustat...")
        query = build_rdq_query_from_tickers(ticker_list, start_date, end_date)
        try:
            self.data['earnings'] = self.db.raw_sql(query)
            print(f"Retrieved {len(self.data['earnings'])} earnings announcements")
            return self.data['earnings']
        except Exception as e:
            print(f"Error fetching earnings data: {e}")
            return None

    def get_option_data(self, secid_list, start_date='2023-01-01', end_date='2024-12-31'):
        print("Fetching option data from OptionMetrics...")
        self.setup_optionm_tables()
        fields = [
            'date', 'secid', 'exdate', 'strike_price', 'cp_flag',
            'best_bid', 'best_offer', 'open_interest',
            'impl_volatility', 'delta', 'gamma', 'theta', 'vega', 'volume'
        ]
        query = build_optionm_query(self.available_tables, 'opprcd', start_date, end_date, fields, secid_list)
        if "not found" in query or "No available" in query:
            print(f"Query build failed: {query}")
            return None
        try:
            print("Executing options query...")
            self.data['options'] = self.db.raw_sql(query)
            print(f"Available columns in options data: {list(self.data['options'].columns)}")
            volume_field_candidates = ['volume', 'vol', 'contract_volume', 'opt_volume']
            volume_col = None
            for col_candidate in volume_field_candidates:
                if col_candidate in self.data['options'].columns:
                    volume_col = col_candidate
                    break
            if volume_col and volume_col != 'volume':
                print(f"Found volume column: {volume_col}, renaming to 'volume'")
                self.data['options']['volume'] = self.data['options'][volume_col]
            print(f"Retrieved {len(self.data['options'])} option records")
            return self.data['options']
        except Exception as e:
            print(f"Error fetching options data: {e}")
            return None

    def get_stock_prices(self, secid_list, start_date='2023-01-01', end_date='2024-12-31'):
        print("Fetching stock prices from OptionMetrics...")
        query = build_secprd_query(secid_list, start_date, end_date)
        try:
            self.data['stock_prices'] = self.db.raw_sql(query)
            print(f"Retrieved {len(self.data['stock_prices'])} stock price records")
            return self.data['stock_prices']
        except Exception as e:
            print(f"Error fetching stock prices: {e}")
            return None

    def merge_securities_earnings(self):
        if 'securities' not in self.data or 'earnings' not in self.data:
            print("Need both securities and earnings data")
            return None
        merged = self.data['earnings'].merge(
            self.data['securities'][['secid', 'ticker', 'cusip', 'issuer']],
            on='ticker', how='inner')
        self.data['earnings_securities'] = merged
        print(f"Merged {len(merged)} earnings-securities records")
        return merged

    def calculate_option_metrics(self):
        if 'options' not in self.data:
            raise ValueError("Options data not loaded. Run get_option_data() first.")
        stock_prices = self.data.get('stock_prices', None)
        self.data['options_enhanced'] = calculate_option_metrics(self.data['options'], stock_prices)
        print(f"Enhanced {len(self.data['options_enhanced'])} option records with calculated metrics")
        return self.data['options_enhanced']

    def apply_data_filters(self, min_volume=10, max_bid_ask_spread=0.5, tte_range=(7, 60), moneyness_range=(0.8, 1.2)):
        if 'options_enhanced' not in self.data:
            self.calculate_option_metrics()
        filtered = apply_data_filters(self.data['options_enhanced'], min_volume, max_bid_ask_spread, tte_range, moneyness_range)
        self.data['options_filtered'] = filtered
        print(f"Filtered options data to {len(filtered)} records")
        return filtered

    def merge_earnings_options(self, event_window_days=30):
        if 'earnings_securities' not in self.data or 'options_filtered' not in self.data:
            print("Need both earnings_securities and options_filtered data")
            return None
        merged = merge_earnings_options(self.data['earnings_securities'], self.data['options_filtered'], event_window_days)
        self.data['earnings_options'] = merged
        if merged is not None:
            print(f"Merged dataset contains {len(merged)} records")
        else:
            print("No matching earnings-options data found")
        return merged

    def compute_eiv_for_events(self):
        """
        Compute EIV for each earnings event and store in self.data['eiv_results'] as a DataFrame.
        Requires 'earnings_securities', 'options_filtered' in self.data.
        """
        if 'earnings_securities' not in self.data or 'options_filtered' not in self.data:
            print("Need both earnings_securities and options_filtered data")
            return None
        earnings = self.data['earnings_securities'].copy()
        options = self.data['options_filtered'].copy()
        eiv_records = []
        for _, earning in earnings.iterrows():
            secid = earning['secid']
            earnings_date = pd.to_datetime(earning['earnings_date'])
            ticker = earning['ticker']
            # Filter options for this secid
            secid_options = options[options['secid'] == secid].copy()
            if secid_options.empty:
                continue
            # Use the last available date before earnings as current_date
            current_date = secid_options[secid_options['date'] < earnings_date]['date'].max()
            if pd.isnull(current_date):
                continue
            option_slice = secid_options[secid_options['date'] == current_date]
            if option_slice.empty:
                continue
            eiv = calculate_eiv(option_slice, earnings_date, current_date)
            eiv_records.append({
                'secid': secid,
                'ticker': ticker,
                'earnings_date': earnings_date,
                'current_date': current_date,
                'EIV': eiv
            })
        eiv_df = pd.DataFrame(eiv_records)
        self.data['eiv_results'] = eiv_df
        print(f"Computed EIV for {len(eiv_df)} earnings events.")
        return eiv_df

    def plot_eiv_distribution(self):
        """
        Plot the distribution and summary statistics of EIV across events.
        """
        if 'eiv_results' not in self.data or self.data['eiv_results'].empty:
            print("No EIV results to plot.")
            return
        eiv_df = self.data['eiv_results']
        plt.figure(figsize=(10, 6))
        plt.hist(eiv_df['EIV'].dropna(), bins=30, color='skyblue', edgecolor='k', alpha=0.7)
        plt.title('Distribution of Earnings-Induced Volatility (EIV)')
        plt.xlabel('EIV')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()
        print("EIV Summary Statistics:")
        print(eiv_df['EIV'].describe())

    def plot_volatility_smile(self):
        if 'earnings_options' not in self.data:
            print("No earnings-options data available for smile analysis")
            return
        plot_volatility_smile(self.data['earnings_options'])

    def plot_term_structure(self):
        if 'earnings_options' not in self.data:
            print("No earnings-options data available for term structure analysis")
            return
        df = self.data['earnings_options'].copy()
        if 'moneyness' in df.columns:
            atm_options = df[(df['moneyness'] >= 0.8) & (df['moneyness'] <= 1.2)].copy()
        else:
            atm_options = df.copy()
        if len(atm_options) == 0:
            print("No ATM options available for term structure analysis")
            return
        term_structure = atm_options.groupby('tte')['impl_volatility'].agg(['mean', 'std', 'count']).reset_index()
        term_structure = term_structure[term_structure['count'] >= 5]
        if len(term_structure) == 0:
            print("Insufficient data for term structure analysis")
            return
        plot_term_structure(term_structure, atm_options)

    def generate_summary_report(self):
        print("\n" + "="*70)
        print("EARNINGS IMPLIED VOLATILITY ANALYSIS - SUMMARY REPORT")
        print("="*70)

        # Data availability summary
        print("\n📊 DATA AVAILABILITY:")
        print("-" * 30)
        for key, value in self.data.items():
            if isinstance(value, pd.DataFrame):
                print(f"{key:20}: {len(value):,} records")
            else:
                print(f"{key:20}: {type(value)}")

        # Options data quality
        if 'options_filtered' in self.data:
            print("\n⚙️ OPTIONS DATA QUALITY:")
            print("-" * 30)
            print_options_summary(self.data['options_filtered'])

        # EIV summary
        if 'eiv_results' in self.data:
            print("\n📊 EIV DISTRIBUTION:")
            print("-" * 30)
            self.plot_eiv_distribution()

        # Volatility smile plot
        self.plot_volatility_smile()

        # Term structure plot
        self.plot_term_structure()

        # Event study plot (if available)
        if 'event_study' in self.data:
            plot_event_study_iv(self.data['event_study'])

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)

    # Methods to be imported from queries.py, analysis.py, plotting.py
    # e.g. get_securities_info, get_earnings_dates, get_option_data, etc. 