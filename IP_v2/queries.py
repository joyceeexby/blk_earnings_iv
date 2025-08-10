import pandas as pd

def build_optionm_query(available_tables, table_base, start_date, end_date, fields, secids=None):
    table_base = table_base.lower()
    matching_tables = [t for t in available_tables if t.startswith(table_base)]
    if not matching_tables:
        return f"Table '{table_base}' not found in OptionMetrics."
    secid_filter = ""
    if secids is not None:
        if isinstance(secids, (list, tuple, set)):
            secid_list = ", ".join(str(s) for s in secids)
            secid_filter = f"AND secid IN ({secid_list})"
        else:
            secid_filter = f"AND secid = {secids}"
    years = list(range(pd.to_datetime(start_date).year, pd.to_datetime(end_date).year + 1))
    if table_base in matching_tables:
        return f"""
SELECT {', '.join(fields)}
FROM optionm.{table_base}
WHERE date BETWEEN '{start_date}' AND '{end_date}'
{secid_filter}
        """.strip()
    union_queries = []
    for year in years:
        table_year = f"{table_base}{year}"
        if table_year in matching_tables:
            query = f"""
SELECT {', '.join(fields)}
FROM optionm.{table_year}
WHERE date BETWEEN '{start_date}' AND '{end_date}'
{secid_filter}
""".strip()
            union_queries.append(query)
    if not union_queries:
        return f"No available year-specific tables for '{table_base}' in range {years}."
    return "\nUNION ALL\n".join(union_queries)

def build_rdq_query_from_tickers(ticker_list, start_date, end_date):
    if not ticker_list:
        raise ValueError("You must provide at least one ticker.")
    formatted_tickers = ', '.join([f"'{ticker}'" for ticker in ticker_list])
    query = f"""
    SELECT cusip,
           tic as ticker,
           datadate,
           rdq as earnings_date,
           fyearq,
           fqtr
    FROM comp.fundq
    WHERE tic IN ({formatted_tickers})
      AND rdq BETWEEN '{start_date}' AND '{end_date}'
      AND rdq IS NOT NULL
    ORDER BY tic, rdq;
    """
    return query

def build_rdq_query_from_gvkeys(gvkey_list, start_date, end_date):
    """
    Build query to get earnings dates using GVKEYs (for index constituents).
    """
    if not gvkey_list:
        raise ValueError("You must provide at least one GVKEY.")
    formatted_gvkeys = ', '.join([f"'{gvkey}'" for gvkey in gvkey_list])
    query = f"""
    SELECT gvkey, cusip,
           tic as ticker,
           datadate,
           rdq as earnings_date,
           fyearq,
           fqtr
    FROM comp.fundq
    WHERE gvkey IN ({formatted_gvkeys})
      AND rdq BETWEEN '{start_date}' AND '{end_date}'
      AND rdq IS NOT NULL
    ORDER BY ticker, rdq;
    """
    return query

def build_secprd_query(secid_list, start_date, end_date):
    if not secid_list:
        raise ValueError("SECID list is empty.")
    formatted_secids = ', '.join([str(int(secid)) for secid in secid_list])
    query = f"""
    SELECT *
    FROM optionm.secprd
    WHERE secid IN ({formatted_secids})
      AND date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY secid, date;
    """
    return query

def build_index_holdings_query(index_name):
    """
    Build query to get index constituents.
    """
    query = f"""
    SELECT *
    FROM comp_na_daily_all.wrds_idx_cst_current t
    WHERE indexname = '{index_name}'
    """
    return query

def build_ibes_summary_query(cusip_list, start_date, end_date):
    """
    Build query to get IBES EPS summary estimates.
    """
    if not cusip_list:
        return ""
    
    cusip_list_str = ', '.join(f"'{cusip}'" for cusip in cusip_list)
    
    query = f"""
    SELECT ticker, cusip, statpers, fpedats, anndats_act,
           meanest, stdev, numest, fpi
    FROM tr_ibes.statsum_epsus
    WHERE cusip IN ({cusip_list_str})
      AND statpers BETWEEN '{start_date}' AND '{end_date}'
      AND measure = 'EPS'
      AND fiscalp = 'QTR'
    """
    return query

def build_security_info_query(cusip_list):
    """
    Build query to get latest ticker information for CUSIPs.
    """
    if not cusip_list:
        return ""
    
    formatted_cusips = ', '.join(f"'{str(cusip)}'" for cusip in cusip_list)
    
    query = f"""
    SELECT cusip, ticker, secid, effect_date
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY cusip ORDER BY effect_date DESC) AS rn
        FROM optionm_all.secnmd
        WHERE cusip IN ({formatted_cusips})
    ) sub
    WHERE rn = 1
    ORDER BY cusip;
    """
    return query 