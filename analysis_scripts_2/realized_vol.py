# =========================
# 0) Imports & WRDS connect
# =========================
import pandas as pd
import numpy as np

import wrds  # pip install wrds if needed
db = wrds.Connection()  # or wrds.Connection(wrds_username="your_user")

# ==============================================
# 1) Data pull: CRSP daily returns for PERMNO(s)
# ==============================================
def get_crsp_returns(db, permnos, start_date, end_date):
    """
    Fetch daily returns for a list of CRSP PERMNOs within a date range.

    Parameters
    ----------
    db         : wrds.Connection
        Active WRDS connection.
    permnos    : list[int] or int or str
        One or many PERMNOs.
    start_date : str 'YYYY-MM-DD'
    end_date   : str 'YYYY-MM-DD'

    Returns
    -------
    DataFrame with columns [permno, date, ret].
    """
    # Normalize inputs
    if isinstance(permnos, (int, str)):
        permnos = [permnos]
    permnos = [int(p) for p in permnos if pd.notna(p)]
    if not permnos:
        return pd.DataFrame(columns=["permno", "date", "ret"])

    permnos_sql = ",".join(str(p) for p in permnos)

    query = f"""
        SELECT permno, date, ret
        FROM crsp.dsf
        WHERE permno IN ({permnos_sql})
          AND date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        ORDER BY permno, date;
    """
    return db.raw_sql(query, date_cols=["date"])

# ============================================================
# 2) Feature: EWMA (half-life) realized volatility per PERMNO
# ============================================================
def compute_ewma_vol(df, halflives=[5, 21, 63]):
    """
    Compute EWMA realized volatility for each (permno, date) with given half-lives.

    Parameters
    ----------
    df : DataFrame with ['permno', 'date', 'ret'] (daily returns)
    halflives : list[int]
        Half-life parameters in TRADING DAYS.

    Returns
    -------
    DataFrame: original columns + one vol column per half-life (e.g., 'vol_hl21').
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["permno", "date"])

    # variance proxy
    out["ret2"] = out["ret"] ** 2

    for hl in halflives:
        # Convert half-life -> alpha (decay)
        # Convention: weight halves every hl days
        alpha = 1 - np.exp(np.log(0.5) / hl)  # 0<alpha<1
        col = f"vol_hl{hl}"

        # Group by permno and compute EWMA of squared returns, then sqrt
        out[col] = (
            out.groupby("permno")["ret2"]
               .transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean() ** 0.5)
        )

    return out.drop(columns=["ret2"])

# =========================
# 3) Example: pull & compute
# =========================
# Example input: list of PERMNOs you want, and the time window
permno_list = [10078, 93436]  # <-- replace with your list (or df['permno'].unique().tolist())
start_date = "2005-01-01"
end_date   = "2023-01-01"

# Pull daily returns
price_df = get_crsp_returns(db, permno_list, start_date, end_date)
print(f"Pulled {len(price_df):,} rows of daily returns.")

# Compute EWMA vols for multiple half-lives
halflives = [5, 7, 10, 21, 63, 126]
vol_df = compute_ewma_vol(price_df, halflives)

out_path = "data_files/vol_df.csv"  # change as needed
vol_df.to_csv(out_path, index=False)
