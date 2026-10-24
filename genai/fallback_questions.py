"""Pre-written question -> SQL pairs.

Used as a BACKUP when the Claude API is unreachable (no internet / no key / no credits),
so the demo never dies. They also work as the list of example questions in the app.
"""

_JOIN_DATE = "JOIN dim_date d ON f.order_date_key = d.date_key"
_JOIN_LOC = "JOIN dim_location l ON f.location_key = l.location_key"
_JOIN_PROD = "JOIN dim_product p ON f.product_key = p.product_key"
_JOIN_CUST = "JOIN dim_customer c ON f.customer_key = c.customer_key"

FALLBACK = [
    {"question": "Which region has the highest total sales?",
     "sql": f"SELECT l.region, ROUND(SUM(f.sales), 2) AS total_sales FROM fact_sales f {_JOIN_LOC} "
            "GROUP BY l.region ORDER BY total_sales DESC"},
    {"question": "Top 5 products by sales in 2017",
     "sql": f"SELECT p.product_name, ROUND(SUM(f.sales), 2) AS total_sales FROM fact_sales f {_JOIN_PROD} "
            f"{_JOIN_DATE} WHERE d.year = 2017 GROUP BY p.product_key ORDER BY total_sales DESC LIMIT 5"},
    {"question": "Monthly sales trend for 2018",
     "sql": f"SELECT d.month_name, ROUND(SUM(f.sales), 2) AS total_sales FROM fact_sales f {_JOIN_DATE} "
            "WHERE d.year = 2018 GROUP BY d.month, d.month_name ORDER BY d.month"},
    {"question": "Which customer segment buys the most?",
     "sql": f"SELECT c.segment, ROUND(SUM(f.sales), 2) AS total_sales FROM fact_sales f {_JOIN_CUST} "
            "GROUP BY c.segment ORDER BY total_sales DESC"},
    {"question": "Sales of the Technology category in each region",
     "sql": f"SELECT l.region, ROUND(SUM(f.sales), 2) AS total_sales FROM fact_sales f {_JOIN_LOC} "
            f"{_JOIN_PROD} WHERE p.category = 'Technology' GROUP BY l.region ORDER BY total_sales DESC"},
    {"question": "Who are the top 10 customers by total sales?",
     "sql": f"SELECT c.customer_name, ROUND(SUM(f.sales), 2) AS total_sales FROM fact_sales f {_JOIN_CUST} "
            "GROUP BY c.customer_key ORDER BY total_sales DESC LIMIT 10"},
    {"question": "Which state has the lowest sales?",
     "sql": f"SELECT l.state, ROUND(SUM(f.sales), 2) AS total_sales FROM fact_sales f {_JOIN_LOC} "
            "GROUP BY l.state ORDER BY total_sales ASC LIMIT 5"},
    {"question": "Total sales by category for each year",
     "sql": f"SELECT d.year, p.category, ROUND(SUM(f.sales), 2) AS total_sales FROM fact_sales f {_JOIN_DATE} "
            f"{_JOIN_PROD} GROUP BY d.year, p.category ORDER BY d.year, total_sales DESC"},
    {"question": "Which shipping mode is used most often?",
     "sql": "SELECT ship_mode, COUNT(DISTINCT order_id) AS orders FROM fact_sales "
            "GROUP BY ship_mode ORDER BY orders DESC"},
    {"question": "Which sub-category has the highest sales?",
     "sql": f"SELECT p.sub_category, ROUND(SUM(f.sales), 2) AS total_sales FROM fact_sales f {_JOIN_PROD} "
            "GROUP BY p.sub_category ORDER BY total_sales DESC LIMIT 5"},
    {"question": "Quarterly sales for 2017",
     "sql": f"SELECT d.quarter, ROUND(SUM(f.sales), 2) AS total_sales FROM fact_sales f {_JOIN_DATE} "
            "WHERE d.year = 2017 GROUP BY d.quarter ORDER BY d.quarter"},
    {"question": "How many customers are in each RFM segment?",
     "sql": "SELECT rfm_segment, COUNT(*) AS customers, ROUND(SUM(monetary), 2) AS total_sales "
            "FROM customer_segments GROUP BY rfm_segment ORDER BY total_sales DESC"},
    {"question": "Year-wise total sales and number of orders",
     "sql": f"SELECT d.year, ROUND(SUM(f.sales), 2) AS total_sales, COUNT(DISTINCT f.order_id) AS orders "
            f"FROM fact_sales f {_JOIN_DATE} GROUP BY d.year ORDER BY d.year"},
]
