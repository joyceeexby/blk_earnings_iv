#!/usr/bin/env python3
"""
CRSP Liquidity Universe Module
Build Top-N-by-Dollar-Volume universes on the first month-end of each quarter.

- Filters to US common stocks (shrcd 10/11; exchcd 1/2/3) BEFORE ranking
- Saves to data_files/topN_liquidity_YYYY_YYYY.csv
"""

import os
import sys
import argparse
import pandas as pd
import wrds
import warnings
warnings.filterwarnings("ignore")


class CRSPLiquidityUniverse:
    """
    Utilities to build liquidity-driven universes from CRSP.
    """

    def __init__(self, db, security_table="stocknames"):
        """
        Parameters
        ----------
        db : wrds.Connection
            Active WRDS connection.
        security_table : {"stocknames", "msenames"}
            Source for security/name metadata and time-valid filters.
        """
        self.db = db
        if security_table not in {"stocknames", "msenames"}:
            raise ValueError("security_table must be 'stocknames' or 'msenames'")
        self.security_table = security_table

    def _build_sql_top_n_first_monthend_per_quarter(
        self, start_year, end_year, num_top_stocks, shrcd=(10, 11), exchcd=(1, 2, 3)
    ):
        names_table = f"crsp.{self.security_table}"
        sql = f"""WITH qdates AS (
    SELECT MIN(date) AS date
    FROM (
        SELECT DISTINCT date,
               DATE_TRUNC('quarter', date)::date AS qstart
        FROM crsp.msf
        WHERE date >= '{start_year}-01-01' AND date <= '{end_year}-12-31'
    ) d
    GROUP BY qstart
),
ms AS (
    SELECT m.permno, m.date, ABS(m.prc) AS prc_abs, m.vol, m.cusip
    FROM crsp.msf m
    JOIN qdates q ON q.date = m.date
    WHERE m.vol IS NOT NULL AND m.vol > 0
),
names AS (
    SELECT permno, ticker, comnam, namedt,
           COALESCE(nameenddt, DATE '9999-12-31') AS nameenddt,
           shrcd, exchcd
    FROM crsp.stocknames
),
joined AS (
    SELECT
        ms.date AS qdate,
        ms.permno,
        ms.cusip,                 -- ✅ include cusip here
        (ms.prc_abs * ms.vol)::double precision AS dollar_vol,
        n.ticker,
        n.comnam
    FROM ms
    JOIN names n
      ON n.permno = ms.permno
     AND n.namedt <= ms.date
     AND n.nameenddt >= ms.date
    WHERE n.shrcd IN (10, 11)
      AND n.exchcd IN (1, 2, 3)
),
ranked AS (
    SELECT
        qdate,
        permno,
        cusip,                    
        ticker,
        comnam,
        dollar_vol,
        ROW_NUMBER() OVER (PARTITION BY qdate ORDER BY dollar_vol DESC) AS rn
    FROM joined
)
SELECT
    qdate,
    permno,
    cusip,                        
    ticker,
    comnam,
    dollar_vol
FROM ranked
WHERE rn <= {num_top_stocks}
ORDER BY qdate, dollar_vol DESC;
    """
        return sql

    def top_dollar_volume_quarterly(
        self,
        start_year: int,
        end_year: int,
        num_top_stocks: int = 500,
        add_year_quarter: bool = True,
        shrcd=(10, 11),
        exchcd=(1, 2, 3),
    ) -> pd.DataFrame:
        """
        Top-N per quarter by dollar volume (|prc| * vol), using FIRST month-end each quarter.
        Filters to US common stocks BEFORE ranking.

        Returns columns:
        ['quarter_start_date','quarter_end_date','permno','ticker','comnam','dollar_vol']
        plus ['year','quarter'] if add_year_quarter=True.
        """
        print(f"\n{'='*80}")
        print(f"CRSP Liquidity Universe: {start_year}–{end_year} | Top N per quarter = {num_top_stocks}")
        print(f"Security table: {self.security_table} | Filters: shrcd={shrcd}, exchcd={exchcd}")
        print(f"{'='*80}")

        sql = self._build_sql_top_n_first_monthend_per_quarter(
            start_year, end_year, num_top_stocks, shrcd=shrcd, exchcd=exchcd
        )
        df = self.db.raw_sql(sql, date_cols=["qdate"])

        if df.empty:
            print("No rows returned from WRDS.")
            cols = ['quarter_start_date','quarter_end_date','permno','ticker','comnam','dollar_vol']
            if add_year_quarter:
                cols = ['year','quarter'] + cols
            return pd.DataFrame(columns=cols)

        # Derive quarter labels from qdate (first month-end used for ranking)
        qper = df['qdate'].dt.to_period('Q')
        df['quarter_start_date'] = qper.apply(lambda p: p.start_time.date())
        df['quarter_end_date']   = qper.apply(lambda p: p.end_time.date())

        if add_year_quarter:
            df['year'] = df['qdate'].dt.year
            df['quarter'] = df['qdate'].dt.quarter

        base_cols = ['quarter_start_date','quarter_end_date','permno','ticker','cusip','comnam','dollar_vol']
        cols = (['year','quarter'] + base_cols) if add_year_quarter else base_cols

        out = (
            df[cols]
            .drop_duplicates(subset=['quarter_start_date','permno'], keep='first')
            .sort_values(['quarter_start_date','dollar_vol'], ascending=[True, False])
            .reset_index(drop=True)
        )

        # Summary prints like your REVR module
        n_quarters = out['quarter_start_date'].nunique()
        print(f"✓ Retrieved {len(out):,} rows across {n_quarters} quarters.")
        if len(out) > 0:
            first_q = out['quarter_start_date'].min()
            last_q  = out['quarter_start_date'].max()
            print(f"  Coverage: {first_q} to {last_q}")
            print("  Example rows:")
            print(out.head(10).to_string(index=False))

        return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build Top-N liquidity universe from CRSP.")
    p.add_argument("--start-year", type=int, default=2005, help="Start year (inclusive)")
    p.add_argument("--end-year", type=int, default=2023, help="End year (inclusive)")
    p.add_argument("--top-n", type=int, default=500, help="Top N per quarter (default 500)")
    p.add_argument("--security-table", choices=["stocknames", "msenames"], default="stocknames",
                   help="Use CRSP stocknames or msenames (default stocknames)")
    p.add_argument("--no-year-quarter", action="store_true",
                   help="Do NOT add year/quarter columns")
    p.add_argument("--wrds-user", type=str, default=os.getenv("WRDS_USERNAME", ""),
                   help="WRDS username (or set WRDS_USERNAME env)")
    return p.parse_args(argv)



def main(argv=None):
    args = parse_args(argv)
    add_yq = not args.no_year_quarter

    print("CRSP LIQUIDITY UNIVERSE")
    print("=" * 80)

    # Connect to WRDS
    try:
        if args.wrds_user:
            db = wrds.Connection(wrds_username=args.wrds_user)
        else:
            db = wrds.Connection()
        print("✓ Connected to WRDS")
    except Exception as e:
        print(f"Error connecting to WRDS: {e}")
        sys.exit(1)

    try:
        builder = CRSPLiquidityUniverse(db, security_table=args.security_table)
        df = builder.top_dollar_volume_quarterly(
            start_year=args.start_year,
            end_year=args.end_year,
            num_top_stocks=args.top_n,
            add_year_quarter=add_yq,
        )

        # --- Automatic naming into data_files ---
        script_dir = os.path.dirname(__file__)
        out_dir = os.path.join(script_dir, "data_files")
        os.makedirs(out_dir, exist_ok=True)

        filename = f"top{args.top_n}_liquidity_{args.start_year}_{args.end_year}.csv"
        outpath = os.path.join(out_dir, filename)

        df.to_csv(outpath, index=False)
        print(f"✓ Saved CSV -> {outpath}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)
    finally:
        try:
            db.close()
            print("✓ Database connection closed")
        except Exception:
            pass


if __name__ == "__main__":
    main()
