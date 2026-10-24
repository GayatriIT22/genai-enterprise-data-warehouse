"""Database helpers.

Everything that only *reads* the warehouse (OLAP, dashboard, GenAI layer) uses a
read-only SQLite connection, so even a bad SQL query can never change data.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from config import DB_PATH


def connect(read_only: bool = True) -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Warehouse database not found at {DB_PATH}. Run: python run_pipeline.py"
        )
    if read_only:
        return sqlite3.connect(DB_PATH.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
    return sqlite3.connect(DB_PATH)


def query(sql: str, params: tuple | list = ()) -> pd.DataFrame:
    conn = connect(read_only=True)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
