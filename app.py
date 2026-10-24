"""Streamlit app.   Run:  streamlit run app.py"""
import pandas as pd
import plotly.express as px
import streamlit as st

import olap_mining.olap_queries as olap
from config import DB_PATH
from db import query
from genai.fallback_questions import FALLBACK
from genai.text_to_sql import ask, get_providers
from olap_mining.customer_segmentation import evaluate_k, run_segmentation, segment_summary

st.set_page_config(page_title="GenAI-Ready Enterprise Data Warehouse", page_icon="📊", layout="wide")
st.title("GenAI-Ready Enterprise Data Warehouse")
st.caption("Business analytics and decision support on Superstore sales, 2015-2018  |  CSN447, Group 8")

if not DB_PATH.exists():
    # First run on a fresh machine (Colab, Streamlit Cloud): build the warehouse automatically.
    with st.spinner("First run: building the data warehouse and customer segments (~15 seconds)..."):
        from warehouse.build_warehouse import build
        build()
        run_segmentation(k=4, save=True)

with st.sidebar:
    st.subheader("GenAI status")
    providers = get_providers()
    if providers:
        st.success("AI providers (tried in this order):\n\n" +
                   "\n\n".join(f"{p.name}: {p.model}" for p in providers))
    else:
        st.warning("No API key found. Ask AI will use pre-written backup queries.")
    st.caption("Data: 9,800 order lines. Only the Sales amount is available (no profit, quantity "
               "or discount).")


@st.cache_data
def cached_segmentation(k: int):
    return run_segmentation(k=k, save=False)


@st.cache_data
def cached_k_scores():
    return evaluate_k()


tab_overview, tab_olap, tab_seg, tab_ask = st.tabs(
    ["Overview", "OLAP explorer", "Customer segments", "Ask AI"])

# ---------------------------------------------------------------- Overview
with tab_overview:
    kpi = query("""SELECT ROUND(SUM(sales), 0) AS sales, COUNT(DISTINCT order_id) AS orders,
                          COUNT(DISTINCT customer_key) AS customers,
                          COUNT(DISTINCT product_key) AS products FROM fact_sales""").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total sales", f"${kpi.sales:,.0f}")
    c2.metric("Orders", f"{int(kpi.orders):,}")
    c3.metric("Customers", f"{int(kpi.customers):,}")
    c4.metric("Products", f"{int(kpi.products):,}")

    yearly = olap.cube(["year", "category"])
    st.plotly_chart(px.bar(yearly, x="year", y="total_sales", color="category", barmode="group",
                           title="Sales by year and category",
                           labels={"total_sales": "Sales ($)", "year": "Year"}))

    st.subheader("What is inside the warehouse")
    sizes = pd.DataFrame({
        "table": ["fact_sales", "dim_date", "dim_customer", "dim_product", "dim_location",
                  "customer_segments"],
        "role": ["Fact: one row per order line (measure: sales)", "Dimension: calendar",
                 "Dimension: who bought", "Dimension: what was bought",
                 "Dimension: where (country > region > state > city)",
                 "Mining result: RFM + K-Means segment per customer"]})
    sizes["rows"] = [int(query(f"SELECT COUNT(*) AS n FROM {t}").iat[0, 0]) for t in sizes["table"]]
    st.dataframe(sizes, hide_index=True)

# -------------------------------------------------------------------- OLAP
with tab_olap:
    op = st.radio("Operation", ["Roll-up", "Drill-down", "Slice", "Dice", "Pivot"], horizontal=True)

    if op == "Roll-up":
        st.caption("Aggregate UP the location hierarchy: city > state > region > country.")
        level = st.select_slider("Level", options=olap.LOCATION_LEVELS, value="region")
        data = olap.rollup_location(level)
        top = data.sort_values("total_sales", ascending=False).head(25)
        st.plotly_chart(px.bar(top, x=level, y="total_sales", labels={"total_sales": "Sales ($)"},
                               title=f"Sales by {level}" + (" (top 25)" if len(data) > 25 else "")))
        st.dataframe(data, hide_index=True)

    elif op == "Drill-down":
        st.caption("Go DOWN the time hierarchy: year > quarter > month.")
        level = st.select_slider("Level", options=olap.TIME_LEVELS, value="year")
        year = st.selectbox("Restrict to a year", ["All", 2015, 2016, 2017, 2018])
        data = olap.drilldown_time(level, None if year == "All" else year)
        label = data["year"].astype(str)
        if level == "quarter":
            label = label + "-Q" + data["quarter"].astype(str)
        elif level == "month":
            label = label + "-" + data["month"].astype(str).str.zfill(2)
        data = data.assign(period=label)
        st.plotly_chart(px.line(data, x="period", y="total_sales", markers=True,
                                labels={"total_sales": "Sales ($)", "period": "Period"},
                                title=f"Sales by {level}"))
        st.dataframe(data.drop(columns="period"), hide_index=True)

    elif op == "Slice":
        st.caption("Fix ONE dimension to ONE value and look at the rest of the cube.")
        col1, col2, col3 = st.columns(3)
        dim = col1.selectbox("Dimension to fix", ["category", "region", "segment", "ship_mode", "year"])
        values = olap.cube([dim])[dim].tolist()
        val = col2.selectbox("Value", values)
        other = col3.selectbox("Break down by", [d for d in ["region", "category", "segment", "year",
                                                              "ship_mode", "state"] if d != dim],
                               index=0 if dim != "region" else 1)
        data = olap.slice_cube(dim, val, [other])
        st.plotly_chart(px.bar(data, x=other, y="total_sales", labels={"total_sales": "Sales ($)"},
                               title=f"{dim} = {val}: sales by {other}"))
        st.dataframe(data, hide_index=True)

    elif op == "Dice":
        st.caption("Fix SEVERAL dimensions to sets of values (a sub-cube).")
        col1, col2, col3 = st.columns(3)
        cats = col1.multiselect("Category", olap.cube(["category"])["category"].tolist(),
                                default=["Technology", "Furniture"])
        regs = col2.multiselect("Region", olap.cube(["region"])["region"].tolist(),
                                default=["West", "East"])
        yrs = col3.multiselect("Year", [2015, 2016, 2017, 2018], default=[2017, 2018])
        filters = {k: v for k, v in {"category": cats, "region": regs, "year": yrs}.items() if v}
        data = olap.dice_cube(filters, ["category", "region"])
        if data.empty:
            st.info("No data for this combination.")
        else:
            st.plotly_chart(px.bar(data, x="region", y="total_sales", color="category", barmode="group",
                                   labels={"total_sales": "Sales ($)"}, title="Diced sub-cube"))
            st.dataframe(data, hide_index=True)

    else:
        st.caption("Rotate the cube: one dimension as rows, another as columns, with totals.")
        col1, col2 = st.columns(2)
        dims = ["region", "category", "segment", "year", "ship_mode", "sub_category"]
        rows = col1.selectbox("Rows", dims, index=0)
        cols = col2.selectbox("Columns", [d for d in dims if d != rows], index=2)
        table = olap.pivot(rows, cols)
        st.dataframe(table.style.format("{:,.0f}"))

# ---------------------------------------------------------------- Segments
with tab_seg:
    st.caption("Data mining: customers grouped by Recency, Frequency and Monetary value using K-Means.")
    k = st.slider("Number of clusters (k)", 2, 6, 4)
    rfm = cached_segmentation(k)
    summary = segment_summary(rfm)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(px.scatter(rfm, x="frequency", y="monetary", color="rfm_segment",
                                   hover_name="customer_name", log_y=True,
                                   hover_data=["recency"], opacity=0.75,
                                   labels={"frequency": "Orders (frequency)",
                                           "monetary": "Total sales, $ (log scale)"},
                                   title="Customers by segment"))
    with col2:
        scores = cached_k_scores()
        st.plotly_chart(px.line(scores, x="k", y="inertia", markers=True,
                                title="Elbow curve (inertia vs k)"))
    st.dataframe(summary)
    st.caption("The Ask AI tab uses the saved k=4 segmentation (run_pipeline.py).")

# ------------------------------------------------------------------ Ask AI
with tab_ask:
    st.caption("Ask a question in plain English. The assistant writes SQL, runs it on the "
               "warehouse (read-only) and explains the result.")

    def set_question(text: str):
        st.session_state["q"] = text

    st.write("Try an example:")
    cols = st.columns(3)
    for i, item in enumerate(FALLBACK[:6]):
        cols[i % 3].button(item["question"], key=f"ex{i}", on_click=set_question,
                           args=(item["question"],))

    question = st.text_input("Your question", key="q",
                             placeholder="e.g. Which region had the highest sales in 2018?")
    if st.button("Ask", type="primary", key="ask_btn") and question.strip():
        with st.spinner("Thinking..."):
            st.session_state["result"] = ask(question.strip())

    res = st.session_state.get("result")
    if res:
        st.caption(f"Answered by: {res['source']}" + (f" ({res['model']})" if res["model"] else "")
                   + (" - from cache" if res["cached"] else ""))
        if res["note"]:
            st.info(res["note"])
        if res["error"]:
            st.error(res["error"])
        if res["summary"]:
            st.success(res["summary"].replace("$", "\\$"))
        if res["sql"]:
            with st.expander("Generated SQL", expanded=True):
                st.code(res["sql"], language="sql")
        df = res["df"]
        if df is not None and not df.empty:
            st.dataframe(df, hide_index=True)
            numeric = df.select_dtypes("number").columns.tolist()
            if df.shape[1] >= 2 and numeric and 2 <= len(df) <= 25:
                x = [c for c in df.columns if c != numeric[-1]][0]
                st.plotly_chart(px.bar(df, x=x, y=numeric[-1]))
