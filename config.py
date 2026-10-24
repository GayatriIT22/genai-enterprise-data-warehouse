"""Project-wide settings. Every script imports paths from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "train.csv"
DB_PATH = ROOT / "warehouse" / "superstore_dw.db"

# The dataset's last order is on 2018-12-30, so RFM "recency" is measured
# from 2018-12-31 (not from today's date).
REFERENCE_DATE = "2018-12-31"
