"""One command to (re)build everything:   python run_pipeline.py

  1. ETL: data/train.csv -> star schema -> warehouse/superstore_dw.db
  2. Data mining: RFM + K-Means (k=4) -> table 'customer_segments' inside the warehouse
"""
from olap_mining.customer_segmentation import run_segmentation, segment_summary
from warehouse.build_warehouse import build

if __name__ == "__main__":
    build()
    print("\nRunning customer segmentation (RFM + K-Means, k=4)...")
    print(segment_summary(run_segmentation(k=4, save=True)).to_string())
    print("\nDone. Start the app with:  streamlit run app.py")
