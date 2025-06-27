# main.py
import wrds
from .pipeline import EarningsIVDataPipeline
import warnings
warnings.filterwarnings("ignore")
from earnings_iv.analysis import run_iv_predicts_rv_workflow


if __name__ == "__main__":
    # Connect to WRDS
    db = wrds.Connection(wrds_username="sami_sellami",
                     password="")

    # Define analysis parameters
    TICKERS = ['AAPL']
    START_DATE = '2023-01-01'
    END_DATE = '2024-12-31'

    # Initialize pipeline
    pipeline = EarningsIVDataPipeline(db)

    # Run pipeline steps
    pipeline.get_securities_info(TICKERS)
    pipeline.get_earnings_dates(TICKERS, START_DATE, END_DATE)
    pipeline.merge_securities_earnings()
    secids = pipeline.data['earnings_securities']['secid'].unique().tolist()
    pipeline.get_option_data(secids, START_DATE, END_DATE)
    pipeline.get_stock_prices(secids, START_DATE, END_DATE)
    pipeline.calculate_option_metrics()

    # Custom filter values
    MIN_VOLUME = 1
    MAX_BID_ASK_SPREAD = 1.0
    TTE_RANGE = (1, 90)
    MONEYNESS_RANGE = (0.5, 1.5)

    pipeline.apply_data_filters(
        min_volume=MIN_VOLUME,
        max_bid_ask_spread=MAX_BID_ASK_SPREAD,
        tte_range=TTE_RANGE,
        moneyness_range=MONEYNESS_RANGE
    )

    pipeline.merge_earnings_options()

    # Compute EIV for events
    pipeline.compute_eiv_for_events()  

    # Generate summary report
    pipeline.generate_summary_report()

    # Close connection
    db.close()
    print("Earnings IV Analysis Pipeline completed.")

    run_iv_predicts_rv_workflow(
        pipeline.data['options_filtered'],
        pipeline.data['stock_prices'],
        pipeline.data['earnings_securities'],
        window=5,         # or 1, 3, etc.
        days_before=20    # or 0 for the day of earnings
    ) 