#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import wrds


import pandas as pd

def get_linktable_by_permnos(db, permno_list):
    """
    Fetch CRSP-Compustat link history records for a list of PERMNOs.

    Parameters
    ----------
    db : wrds.Connection
        Active WRDS connection object.
    permno_list : list of int or str
        List of CRSP PERMNO identifiers.

    Returns
    -------
    DataFrame
        Subset of crsp.ccmxpf_lnkhist with link history for the given PERMNOs.
    """
    if not permno_list:
        return pd.DataFrame()

    # Convert to comma-separated string
    formatted_permnos = ', '.join(str(p) for p in permno_list)

    sql = f"""
        SELECT *
        FROM crsp.ccmxpf_lnkhist
        WHERE lpermno IN ({formatted_permnos})
        ORDER BY lpermno, linkdt
    """

    df = db.raw_sql(sql, date_cols=['linkdt','linkenddt'])
    return df

# --- your function as-is ---
def get_earnings_data(gvkey_list, start_date, end_date, db):
    """
    Fetches quarterly earnings announcement dates and related data
    for a list of gvkeys within a specific date range.

    Returns DataFrame with columns:
    [cusip8, cusip, ticker, datadate, earnings_date, fyearq, fqtr, gvkey]
    """
    if not gvkey_list:
        return pd.DataFrame(columns=["cusip8", "cusip", "ticker", "datadate", "earnings_date", "fyearq", "fqtr", "gvkey"])

    formatted_gvkeys = "', '".join(gvkey_list)
    query = f"""
        SELECT SUBSTRING(cusip, 1, 8) AS cusip8,
               cusip,
               tic AS ticker,
               datadate,
               rdq AS earnings_date,
               fyearq,
               fqtr,
               gvkey
        FROM comp.fundq
        WHERE gvkey IN ('{formatted_gvkeys}')
          AND rdq BETWEEN '{start_date}' AND '{end_date}'
          AND rdq IS NOT NULL
    """
    return db.raw_sql(query)


#--------------------------main--------------------------

db = wrds.Connection()

#--------------------------Connect holdings to gvkeys--------------------------
df = pd.read_csv("analysis_scripts_2/data_files/top500_liquidity_2005_2023.csv")
permno_lst = df['permno'].unique()
link_df = get_linktable_by_permnos(db, list(permno_lst))
link_df.to_csv("analysis_scripts_2/data_files/link_table.csv", index=False) #saves link table for future use
gv_lst = link_df['gvkey'].unique()

#--------------------------Use gvkeys to get earnings dates--------------------------
start_date = '2005-01-01'
end_date = '2023-12-31'
df = get_earnings_data(list(gv_lst), start_date, end_date, db)
df.to_csv("analysis_scripts_2/data_files/earnings_dates.csv", index=False)