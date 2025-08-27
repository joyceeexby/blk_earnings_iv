#!/usr/bin/env python3
"""
Linear REVR path builder (business days)

Input CSV columns (required):
  - ticker
  - earnings_date
  - analysis_date_ievr
  - predicted_revr

Output:
  - revr_linear_paths.csv (long format)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -------- CONFIG --------
IN_PATH  = Path("analysis_scripts_2/data_files/filtered_test_predictions.csv")   # change if needed
OUT_PATH = Path("analysis_scripts_2/data_files/revr_linear_paths.csv")
START_VALUE = 1.0  # REVR at analysis_date_ievr
USE_BUSINESS_DAYS = True  # set False to include calendar days

# -------- CORE --------
def build_linear_paths(df: pd.DataFrame,
                       start_value: float = START_VALUE,
                       use_business_days: bool = True) -> pd.DataFrame:
    """Build straight-line REVR paths per row from analysis_date_ievr -> earnings_date."""
    # Parse and clean
    df = df.copy()
    for c in ["earnings_date", "analysis_date_ievr"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["predicted_revr"] = pd.to_numeric(df["predicted_revr"], errors="coerce")
    df = df.dropna(subset=["ticker", "earnings_date", "analysis_date_ievr", "predicted_revr"])

    freq = "B" if use_business_days else "D"

    def _one_row(row) -> pd.DataFrame | None:
        tkr    = row["ticker"]
        a_dt   = row["analysis_date_ievr"]
        e_dt   = row["earnings_date"]
        target = float(row["predicted_revr"])

        if pd.isna(a_dt) or pd.isna(e_dt):
            return None

        # Ensure start <= end for range construction (we still keep endpoints as start_value -> target)
        start, end = (a_dt, e_dt) if a_dt <= e_dt else (e_dt, a_dt)

        # Inclusive range
        rng = pd.date_range(start=start, end=end, freq=freq)
        if len(rng) == 0:
            return None

        # Determine order of endpoints relative to the generated range
        # We always want value(start_of_line)=start_value at analysis_date_ievr
        # and value(end_of_line)=target at earnings_date
        # If dates were swapped, we still output a straight line start_value -> target across rng
        n = len(rng)
        if n == 1:
            vals = np.array([target if a_dt == e_dt else start_value], dtype=float)
        else:
            vals = np.linspace(start_value, target, num=n)

        out = pd.DataFrame(
            {"ticker": tkr, "date": rng, "interp_revr": vals,
             "analysis_date_ievr": a_dt, "earnings_date": e_dt, "predicted_revr": target}
        )
        return out

    chunks = []
    for _, r in df.iterrows():
        path = _one_row(r)
        if path is not None:
            chunks.append(path)

    if not chunks:
        return pd.DataFrame(columns=[
            "ticker", "date", "interp_revr", "analysis_date_ievr", "earnings_date", "predicted_revr"
        ])

    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    data = pd.read_csv(IN_PATH)
    long_df = build_linear_paths(data, start_value=START_VALUE, use_business_days=USE_BUSINESS_DAYS)
    long_df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH.resolve()}")
