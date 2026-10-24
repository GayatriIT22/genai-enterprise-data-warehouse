"""Customer segmentation with RFM + K-Means (the data-mining part of the project).

RFM per customer:
  Recency   = days since the last order (measured from REFERENCE_DATE)
  Frequency = number of distinct orders
  Monetary  = total sales amount

Steps: build RFM -> log-transform (they are right-skewed) -> standardise -> K-Means
-> name clusters by how good their RFM profile is.

Run:   python -m olap_mining.customer_segmentation
"""
from __future__ import annotations

from contextlib import closing

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import REFERENCE_DATE
from db import connect, query

RFM_SQL = """
SELECT c.customer_key, c.customer_id, c.customer_name,
       MAX(d.full_date)            AS last_order_date,
       COUNT(DISTINCT f.order_id)  AS frequency,
       ROUND(SUM(f.sales), 2)      AS monetary
FROM fact_sales f
JOIN dim_customer c ON f.customer_key   = c.customer_key
JOIN dim_date     d ON f.order_date_key = d.date_key
GROUP BY c.customer_key
"""

# Cluster names, best profile first.
LABELS = {
    2: ["High-value active", "Low-value inactive"],
    3: ["Champions", "Regular", "At risk"],
    4: ["Champions", "Loyal customers", "At risk", "Lost"],
    5: ["Champions", "Loyal customers", "Potential loyalists", "At risk", "Lost"],
    6: ["Champions", "Loyal customers", "Potential loyalists", "Needs attention", "At risk", "Lost"],
}
RFM_COLS = ["recency", "frequency", "monetary"]


def build_rfm() -> pd.DataFrame:
    rfm = query(RFM_SQL)
    ref = pd.Timestamp(REFERENCE_DATE)
    rfm["recency"] = (ref - pd.to_datetime(rfm["last_order_date"])).dt.days
    return rfm.drop(columns="last_order_date")


def scale(rfm: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(np.log1p(rfm[RFM_COLS]))


def evaluate_k(k_range=range(2, 9)) -> pd.DataFrame:
    """Inertia (elbow) and silhouette score for each k, to justify the choice of k."""
    X = scale(build_rfm())
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def run_segmentation(k: int = 4, save: bool = False) -> pd.DataFrame:
    rfm = build_rfm()
    X = scale(rfm)
    rfm["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)

    # Rank clusters: high frequency + high monetary + LOW recency = best.
    z = pd.DataFrame(X, columns=RFM_COLS).groupby(rfm["cluster"]).mean()
    best_first = (z["frequency"] + z["monetary"] - z["recency"]).sort_values(ascending=False).index
    names = LABELS.get(k, [f"Segment {i + 1}" for i in range(k)])
    rfm["rfm_segment"] = rfm["cluster"].map({c: names[i] for i, c in enumerate(best_first)})

    if save:
        with closing(connect(read_only=False)) as conn:
            rfm[["customer_key", "customer_id", "customer_name", *RFM_COLS, "cluster",
                 "rfm_segment"]].to_sql("customer_segments", conn, if_exists="replace", index=False)
    return rfm


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    out = (rfm.groupby("rfm_segment")
              .agg(customers=("customer_key", "count"),
                   avg_recency_days=("recency", "mean"),
                   avg_orders=("frequency", "mean"),
                   avg_sales=("monetary", "mean"),
                   total_sales=("monetary", "sum"))
              .round(1).sort_values("avg_sales", ascending=False))
    out["pct_of_sales"] = (100 * out["total_sales"] / out["total_sales"].sum()).round(1)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    print("Choosing k:\n", evaluate_k().round(3), "\n")
    result = run_segmentation(k=4, save=True)
    print("Segment summary (k=4):\n", segment_summary(result))
    print("\nSaved table 'customer_segments' into the warehouse.")
