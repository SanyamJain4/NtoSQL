"""
Builds sales.db: a realistic, moderately-sized SQLite sales database used to
demo and test the NL2SQL system. Deterministic (seeded) so results are
reproducible across runs.
"""
import sqlite3
import random
import datetime
import os

random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "sales.db")
SCHEMA_PATH = os.path.join(HERE, "schema.sql")

REGIONS = [
    ("North America", "USA"), ("North America", "Canada"),
    ("Europe", "Germany"), ("Europe", "France"), ("Europe", "UK"),
    ("Asia Pacific", "India"), ("Asia Pacific", "Japan"), ("Asia Pacific", "Australia"),
    ("Latin America", "Brazil"), ("Latin America", "Mexico"),
]

CATEGORIES = [
    ("Laptops", "Electronics"), ("Smartphones", "Electronics"),
    ("Audio", "Electronics"), ("Wearables", "Electronics"),
    ("Office Chairs", "Furniture"), ("Desks", "Furniture"),
    ("Notebooks", "Office Supplies"), ("Pens & Pencils", "Office Supplies"),
    ("Monitors", "Electronics"), ("Keyboards & Mice", "Electronics"),
]

PRODUCT_NAMES = {
    "Laptops": ["Aria 14 Laptop", "Aria 16 Pro Laptop", "Vantage Ultrabook", "Vantage Business 15"],
    "Smartphones": ["Nova X Phone", "Nova X Lite", "Pulse 5G Phone"],
    "Audio": ["Echo Buds", "Echo Buds Pro", "BassLine Headphones", "BassLine Speaker"],
    "Wearables": ["FitTrack Band", "FitTrack Watch"],
    "Office Chairs": ["ErgoFlex Chair", "ErgoFlex Mesh Chair", "Classic Task Chair"],
    "Desks": ["StandUp Desk 48in", "StandUp Desk 60in", "Compact Desk"],
    "Notebooks": ["Quad Ruled Notebook", "Dot Grid Notebook 3-Pack"],
    "Pens & Pencils": ["Precision Gel Pens 12-Pack", "Mechanical Pencil Set"],
    "Monitors": ["ClearView 27in 4K", "ClearView 24in FHD", "ClearView 32in Ultrawide"],
    "Keyboards & Mice": ["Silent Mechanical Keyboard", "Ergo Wireless Mouse", "Combo Keyboard+Mouse"],
}

SEGMENTS = ["Consumer", "SMB", "Enterprise"]
PAYMENT_METHODS = ["Credit Card", "PayPal", "Bank Transfer", "Gift Card"]
STATUS_WEIGHTS = [("Completed", 0.86), ("Refunded", 0.05), ("Cancelled", 0.06), ("Pending", 0.03)]

FIRST_NAMES = ["Amit","Priya","John","Emma","Liam","Sofia","Noah","Mia","Lucas","Ava",
               "Ethan","Zoe","Raj","Anna","Carlos","Yuki","Chen","Fatima","Omar","Elena"]
LAST_NAMES = ["Sharma","Smith","Garcia","Müller","Tanaka","Silva","Kumar","Johnson","Rossi","Dubois"]


def random_date(start, end):
    delta = (end - start).days
    return start + datetime.timedelta(days=random.randint(0, delta))


def weighted_choice(pairs):
    items, weights = zip(*pairs)
    return random.choices(items, weights=weights, k=1)[0]


def build():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    with open(SCHEMA_PATH) as f:
        cur.executescript(f.read())

    # Regions
    for i, (name, country) in enumerate(REGIONS, start=1):
        cur.execute("INSERT INTO regions VALUES (?,?,?)", (i, name, country))

    # Categories
    cat_ids = {}
    for i, (name, parent) in enumerate(CATEGORIES, start=1):
        cur.execute("INSERT INTO categories VALUES (?,?,?)", (i, name, parent))
        cat_ids[name] = i

    # Products
    product_id = 1
    products = []  # (id, category_id, cost, price)
    launch_start = datetime.date(2022, 1, 1)
    launch_end = datetime.date(2024, 6, 1)
    for cat_name, names in PRODUCT_NAMES.items():
        for pname in names:
            cost = round(random.uniform(8, 600), 2)
            price = round(cost * random.uniform(1.4, 2.8), 2)
            launch = random_date(launch_start, launch_end)
            active = 1 if random.random() > 0.05 else 0
            cur.execute(
                "INSERT INTO products VALUES (?,?,?,?,?,?,?)",
                (product_id, pname, cat_ids[cat_name], cost, price, launch.isoformat(), active),
            )
            products.append((product_id, cat_ids[cat_name], cost, price))
            product_id += 1

    # Customers
    n_customers = 400
    signup_start = datetime.date(2021, 1, 1)
    signup_end = datetime.date(2025, 6, 1)
    customers = []
    for cid in range(1, n_customers + 1):
        fn, ln = random.choice(FIRST_NAMES), random.choice(LAST_NAMES)
        email = f"{fn.lower()}.{ln.lower()}{cid}@example.com"
        signup = random_date(signup_start, signup_end)
        region_id = random.randint(1, len(REGIONS))
        segment = weighted_choice([("Consumer", 0.65), ("SMB", 0.25), ("Enterprise", 0.10)])
        cur.execute(
            "INSERT INTO customers VALUES (?,?,?,?,?,?,?)",
            (cid, fn, ln, email, signup.isoformat(), region_id, segment),
        )
        customers.append((cid, signup))

    # Orders + order_items
    order_id = 1
    item_id = 1
    order_start = datetime.date(2024, 1, 1)
    order_end = datetime.date(2026, 8, 1)  # up through "today" for recency queries

    for cid, signup in customers:
        # each customer places a random number of orders after their signup date
        n_orders = max(0, int(random.gauss(6, 4)))
        for _ in range(n_orders):
            lo = max(order_start, signup)
            if lo >= order_end:
                continue
            odate = random_date(lo, order_end)
            status = weighted_choice(STATUS_WEIGHTS)
            payment = random.choice(PAYMENT_METHODS)
            discount = round(random.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20]), 2)
            cur.execute(
                "INSERT INTO orders VALUES (?,?,?,?,?,?)",
                (order_id, cid, odate.isoformat(), status, payment, discount),
            )
            # 1-4 line items per order
            for _ in range(random.randint(1, 4)):
                pid, _, cost, price = random.choice(products)
                qty = random.randint(1, 5)
                cur.execute(
                    "INSERT INTO order_items VALUES (?,?,?,?,?)",
                    (item_id, order_id, pid, qty, price),
                )
                item_id += 1
            order_id += 1

    conn.commit()

    # Quick sanity counts
    counts = {}
    for t in ["regions", "categories", "products", "customers", "orders", "order_items"]:
        counts[t] = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    conn.close()
    return counts


if __name__ == "__main__":
    counts = build()
    print(f"Built database at {DB_PATH}")
    for t, c in counts.items():
        print(f"  {t}: {c} rows")
