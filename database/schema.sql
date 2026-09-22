-- Sales Analytics Database Schema
-- A realistic normalized schema: categories -> products, customers -> orders -> order_items,
-- plus regions and payment info to support genuinely "complex" analytical queries
-- (joins, window functions, cohort/time-series analysis).

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS categories;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS regions;

CREATE TABLE regions (
    region_id     INTEGER PRIMARY KEY,
    region_name   TEXT NOT NULL,
    country       TEXT NOT NULL
);

CREATE TABLE customers (
    customer_id     INTEGER PRIMARY KEY,
    first_name      TEXT NOT NULL,
    last_name       TEXT NOT NULL,
    email           TEXT NOT NULL UNIQUE,
    signup_date     DATE NOT NULL,
    region_id       INTEGER NOT NULL REFERENCES regions(region_id),
    customer_segment TEXT NOT NULL CHECK (customer_segment IN ('Consumer','SMB','Enterprise'))
);

CREATE TABLE categories (
    category_id     INTEGER PRIMARY KEY,
    category_name   TEXT NOT NULL UNIQUE,
    parent_category TEXT
);

CREATE TABLE products (
    product_id      INTEGER PRIMARY KEY,
    product_name    TEXT NOT NULL,
    category_id     INTEGER NOT NULL REFERENCES categories(category_id),
    unit_cost       DECIMAL(10,2) NOT NULL,
    unit_price      DECIMAL(10,2) NOT NULL,
    launch_date     DATE NOT NULL,
    is_active       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE orders (
    order_id        INTEGER PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date      DATE NOT NULL,
    order_status    TEXT NOT NULL CHECK (order_status IN ('Completed','Refunded','Cancelled','Pending')),
    payment_method  TEXT NOT NULL,
    discount_pct    DECIMAL(4,3) NOT NULL DEFAULT 0
);

CREATE TABLE order_items (
    order_item_id   INTEGER PRIMARY KEY,
    order_id        INTEGER NOT NULL REFERENCES orders(order_id),
    product_id      INTEGER NOT NULL REFERENCES products(product_id),
    quantity        INTEGER NOT NULL,
    unit_price_at_sale DECIMAL(10,2) NOT NULL
);

-- Indexes that matter for the kinds of analytical queries this system will generate
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_customers_region ON customers(region_id);

-- A view that pre-joins the most commonly needed analytical grain (one row per line item)
CREATE VIEW sales_flat AS
SELECT
    oi.order_item_id,
    o.order_id,
    o.order_date,
    o.order_status,
    o.payment_method,
    o.discount_pct,
    c.customer_id,
    c.customer_segment,
    r.region_name,
    r.country,
    p.product_id,
    p.product_name,
    cat.category_name,
    oi.quantity,
    oi.unit_price_at_sale,
    p.unit_cost,
    ROUND(oi.quantity * oi.unit_price_at_sale * (1 - o.discount_pct), 2) AS net_revenue,
    ROUND(oi.quantity * p.unit_cost, 2) AS total_cost
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
JOIN customers c ON o.customer_id = c.customer_id
JOIN regions r ON c.region_id = r.region_id
JOIN products p ON oi.product_id = p.product_id
JOIN categories cat ON p.category_id = cat.category_id;
