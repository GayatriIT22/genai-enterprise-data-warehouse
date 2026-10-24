"""OLAP operations on the star schema (ROLAP: plain SQL + pandas, no cube server).

  roll-up   -> aggregate UP a hierarchy      (city -> state -> region -> country)
  drill-down-> go DOWN a hierarchy           (year -> quarter -> month)
  slice     -> fix ONE dimension to one value (category = 'Technology')
  dice      -> fix SEVERAL dimensions         (Technology + Furniture, West + East, 2017-2018)
  pivot     -> rotate the cube                (regions as rows, years as columns)

Run the demo:   python -m olap_mining.olap_queries
"""
from __future__ import annotations

import pandas as pd

from db import query

LOCATION_LEVELS = ["country", "region", "state", "city"]
TIME_LEVELS = ["year", "quarter", "month"]

# dimension name -> SQL column (whitelist: user input never goes into the SQL text)
DIMENSIONS = {
    "year": "d.year", "quarter": "d.quarter", "month": "d.month", "month_name": "d.month_name",
    "country": "l.country", "region": "l.region", "state": "l.state", "city": "l.city",
    "category": "p.category", "sub_category": "p.sub_category",
    "segment": "c.segment", "ship_mode": "f.ship_mode",
}

BASE_FROM = """
FROM fact_sales f
JOIN dim_date     d ON f.order_date_key = d.date_key
JOIN dim_customer c ON f.customer_key   = c.customer_key
JOIN dim_product  p ON f.product_key    = p.product_key
JOIN dim_location l ON f.location_key   = l.location_key
"""


def cube(group_by: list[str], filters: dict | None = None) -> pd.DataFrame:
    """Total sales and order count grouped by `group_by`, optionally filtered."""
    filters = filters or {}
    for name in list(group_by) + list(filters):
        if name not in DIMENSIONS:
            raise ValueError(f"Unknown dimension '{name}'. Choose from {sorted(DIMENSIONS)}")

    select = "".join(f"{DIMENSIONS[g]} AS {g}, " for g in group_by)
    where, params = [], []
    for dim, values in filters.items():
        values = [values] if isinstance(values, (str, int)) else list(values)
        where.append(f"{DIMENSIONS[dim]} IN ({','.join('?' * len(values))})")
        params += values

    sql = f"SELECT {select}ROUND(SUM(f.sales), 2) AS total_sales, COUNT(DISTINCT f.order_id) AS orders {BASE_FROM}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if group_by:
        positions = ", ".join(str(i + 1) for i in range(len(group_by)))
        sql += f" GROUP BY {positions} ORDER BY {positions}"
    return query(sql, params)


# ---------------------------------------------------------------- Roll-up
def rollup_location(level: str = "region") -> pd.DataFrame:
    """Sales at a chosen level of the location hierarchy (with its parent levels)."""
    return cube(LOCATION_LEVELS[: LOCATION_LEVELS.index(level) + 1])


# ------------------------------------------------------------- Drill-down
def drilldown_time(level: str = "year", year: int | None = None) -> pd.DataFrame:
    """year -> quarter -> month. Optionally restrict to one year."""
    group = {"year": ["year"], "quarter": ["year", "quarter"],
             "month": ["year", "month", "month_name"]}[level]
    return cube(group, {"year": year} if year else None)


# ------------------------------------------------------------------ Slice
def slice_cube(dimension: str, value, group_by: list[str] | None = None) -> pd.DataFrame:
    """Fix ONE dimension to ONE value and look at the rest."""
    return cube(group_by or ["year", "region"], {dimension: value})


# ------------------------------------------------------------------- Dice
def dice_cube(filters: dict, group_by: list[str] | None = None) -> pd.DataFrame:
    """Fix SEVERAL dimensions to sets of values (a sub-cube)."""
    return cube(group_by or ["category", "region"], filters)


# ------------------------------------------------------------------ Pivot
def pivot(rows: str = "region", cols: str = "year") -> pd.DataFrame:
    """Rotate the cube: one dimension as rows, another as columns, with totals."""
    data = cube([rows, cols])
    table = data.pivot_table(index=rows, columns=cols, values="total_sales",
                             aggfunc="sum", margins=True, margins_name="Total").round(2)
    table.columns = table.columns.astype(str)      # years are ints, "Total" is text
    return table


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    print("ROLL-UP  (sales by region)\n", rollup_location("region"), "\n")
    print("DRILL-DOWN  (2017 by quarter)\n", drilldown_time("quarter", 2017), "\n")
    print("SLICE  (Technology only: year x region)\n", slice_cube("category", "Technology"), "\n")
    print("DICE  (Technology+Furniture, West+East, 2017-2018)\n",
          dice_cube({"category": ["Technology", "Furniture"], "region": ["West", "East"],
                     "year": [2017, 2018]}), "\n")
    print("PIVOT  (region x year)\n", pivot("region", "year"))
