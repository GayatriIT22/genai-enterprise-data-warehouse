"""ETL: train.csv  ->  cleaned star schema  ->  SQLite data warehouse.

Run from the project root:   python -m warehouse.build_warehouse
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from config import DATA_CSV, DB_PATH

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


# ----------------------------------------------------------------- Extract
def extract() -> pd.DataFrame:
    return pd.read_csv(DATA_CSV)


# --------------------------------------------------------------- Transform
def _add_key(frame: pd.DataFrame, key_name: str) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    frame.insert(0, key_name, range(1, len(frame) + 1))
    return frame


def transform(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = raw.copy()

    # 1. Dates are dd/mm/yyyy in this file -> parse with an explicit format.
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")

    # 2. 11 rows have no postal code; all are Burlington, Vermont (05401).
    missing = df["Postal Code"].isna() & df["City"].eq("Burlington") & df["State"].eq("Vermont")
    df.loc[missing, "Postal Code"] = 5401
    df["Postal Code"] = df["Postal Code"].fillna(0).astype(int).astype(str).str.zfill(5)

    # 3. Dimensions
    dim_customer = _add_key(
        df[["Customer ID", "Customer Name", "Segment"]]
        .drop_duplicates("Customer ID").sort_values("Customer ID"),
        "customer_key",
    ).rename(columns={"Customer ID": "customer_id", "Customer Name": "customer_name",
                      "Segment": "segment"})

    # 32 product ids carry more than one name in the raw data -> keep the first.
    dim_product = _add_key(
        df[["Product ID", "Product Name", "Category", "Sub-Category"]]
        .drop_duplicates("Product ID").sort_values("Product ID"),
        "product_key",
    ).rename(columns={"Product ID": "product_id", "Product Name": "product_name",
                      "Category": "category", "Sub-Category": "sub_category"})

    loc_cols = ["Country", "Region", "State", "City", "Postal Code"]
    dim_location = _add_key(
        df[loc_cols].drop_duplicates().sort_values(loc_cols), "location_key"
    ).rename(columns={"Country": "country", "Region": "region", "State": "state",
                      "City": "city", "Postal Code": "postal_code"})

    # Continuous calendar covering every order AND ship date.
    days = pd.date_range(df[["Order Date", "Ship Date"]].min().min(),
                         df[["Order Date", "Ship Date"]].max().max(), freq="D")
    dim_date = pd.DataFrame({
        "date_key": days.strftime("%Y%m%d").astype(int),
        "full_date": days.strftime("%Y-%m-%d"),
        "day": days.day, "month": days.month, "month_name": days.strftime("%B"),
        "quarter": days.quarter, "year": days.year, "day_name": days.strftime("%A"),
    })

    # 4. Fact table: swap business columns for surrogate keys.
    fact = (
        df.merge(dim_customer[["customer_key", "customer_id"]],
                 left_on="Customer ID", right_on="customer_id")
          .merge(dim_product[["product_key", "product_id"]],
                 left_on="Product ID", right_on="product_id")
          .merge(dim_location, left_on=loc_cols,
                 right_on=["country", "region", "state", "city", "postal_code"])
    )
    fact["order_date_key"] = fact["Order Date"].dt.strftime("%Y%m%d").astype(int)
    fact["ship_date_key"] = fact["Ship Date"].dt.strftime("%Y%m%d").astype(int)
    fact = (fact.rename(columns={"Row ID": "row_id", "Order ID": "order_id",
                                 "Ship Mode": "ship_mode", "Sales": "sales"})
                [["row_id", "order_id", "order_date_key", "ship_date_key", "customer_key",
                  "product_key", "location_key", "ship_mode", "sales"]]
                .sort_values("row_id"))

    return {"dim_date": dim_date, "dim_customer": dim_customer, "dim_product": dim_product,
            "dim_location": dim_location, "fact_sales": fact}


# -------------------------------------------------------------------- Load
def load(tables: dict[str, pd.DataFrame]) -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
        for name in ["dim_date", "dim_customer", "dim_product", "dim_location", "fact_sales"]:
            tables[name].to_sql(name, conn, if_exists="append", index=False)
        conn.commit()
        bad = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert not bad, f"Foreign-key violations: {bad[:5]}"
    finally:
        conn.close()


# ---------------------------------------------------------------- Validate
def validate(raw: pd.DataFrame) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        n_fact, total = conn.execute("SELECT COUNT(*), SUM(sales) FROM fact_sales").fetchone()
    finally:
        conn.close()
    assert n_fact == len(raw), f"Row count mismatch: {n_fact} vs {len(raw)}"
    assert abs(total - raw["Sales"].sum()) < 0.01, "Total sales mismatch"
    print(f"Validation OK: {n_fact:,} fact rows, total sales {total:,.2f} match the CSV.")


def build() -> None:
    raw = extract()
    tables = transform(raw)
    load(tables)
    validate(raw)
    print("\nTable sizes:")
    for name, frame in tables.items():
        print(f"  {name:<14} {len(frame):>6,} rows")
    print(f"\nDatabase written to {DB_PATH}")


if __name__ == "__main__":
    build()
