-- Star schema for the Superstore sales warehouse (SQLite).
-- One fact table (fact_sales) surrounded by four dimension tables.
-- Comments below are kept in sqlite_master and are also shown to the LLM.

CREATE TABLE dim_date (
    date_key    INTEGER PRIMARY KEY,   -- yyyymmdd, e.g. 20170811
    full_date   TEXT NOT NULL,         -- YYYY-MM-DD
    day         INTEGER,
    month       INTEGER,               -- 1-12
    month_name  TEXT,                  -- January ... December
    quarter     INTEGER,               -- 1-4
    year        INTEGER,
    day_name    TEXT
);

CREATE TABLE dim_customer (
    customer_key   INTEGER PRIMARY KEY,
    customer_id    TEXT UNIQUE NOT NULL,   -- business id, e.g. CG-12520
    customer_name  TEXT,
    segment        TEXT                    -- business segment: Consumer / Corporate / Home Office
);

CREATE TABLE dim_product (
    product_key   INTEGER PRIMARY KEY,
    product_id    TEXT UNIQUE NOT NULL,    -- e.g. FUR-BO-10001798
    product_name  TEXT,
    category      TEXT,                    -- Furniture / Office Supplies / Technology
    sub_category  TEXT
);

CREATE TABLE dim_location (
    location_key  INTEGER PRIMARY KEY,
    country       TEXT,                    -- hierarchy: country > region > state > city
    region        TEXT,                    -- East / West / Central / South
    state         TEXT,
    city          TEXT,
    postal_code   TEXT
);

CREATE TABLE fact_sales (
    row_id          INTEGER PRIMARY KEY,   -- one row = one product line of an order
    order_id        TEXT NOT NULL,         -- an order can have several rows
    order_date_key  INTEGER NOT NULL REFERENCES dim_date(date_key),
    ship_date_key   INTEGER NOT NULL REFERENCES dim_date(date_key),
    customer_key    INTEGER NOT NULL REFERENCES dim_customer(customer_key),
    product_key     INTEGER NOT NULL REFERENCES dim_product(product_key),
    location_key    INTEGER NOT NULL REFERENCES dim_location(location_key),
    ship_mode       TEXT,                  -- Standard Class / Second Class / First Class / Same Day
    sales           REAL NOT NULL          -- sales amount in USD (the only measure)
);

CREATE INDEX idx_fact_order_date ON fact_sales(order_date_key);
CREATE INDEX idx_fact_customer   ON fact_sales(customer_key);
CREATE INDEX idx_fact_product    ON fact_sales(product_key);
CREATE INDEX idx_fact_location   ON fact_sales(location_key);
CREATE INDEX idx_fact_order_id   ON fact_sales(order_id);
