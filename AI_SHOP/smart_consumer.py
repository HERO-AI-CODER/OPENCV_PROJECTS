import sqlite3
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re


# ============================================================
# DATABASE
# ============================================================

DB_NAME = "smart_consumer.db"


def create_database():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT,
            quantity REAL DEFAULT 1,
            unit TEXT DEFAULT 'piece'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL,
            shop TEXT NOT NULL,
            price REAL NOT NULL,
            url TEXT,
            checked_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS bills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bill_date TEXT,
            total REAL,
            savings REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS bill_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bill_id INTEGER,
            product_name TEXT,
            quantity REAL,
            price REAL,
            shop TEXT,
            FOREIGN KEY (bill_id) REFERENCES bills(id)
        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# CUSTOMER ITEM INPUT
# ============================================================

def collect_customer_items():

    print("\n================================")
    print("     SMART CONSUMER ASSISTANT")
    print("================================")

    items = []

    while True:

        name = input("\nEnter required item (or 'done'): ").strip()

        if name.lower() == "done":
            break

        if not name:
            continue

        quantity = float(input("Quantity: "))

        unit = input("Unit (piece/kg/litre/etc.): ").strip()

        items.append({
            "name": name,
            "quantity": quantity,
            "unit": unit
        })

    return items


# ============================================================
# STORE CUSTOMER ITEMS
# ============================================================

def save_products(items):

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    for item in items:

        cur.execute("""
            INSERT INTO products
            (name, quantity, unit)
            VALUES (?, ?, ?)
        """, (
            item["name"],
            item["quantity"],
            item["unit"]
        ))

    conn.commit()
    conn.close()


# ============================================================
# COMPUTER VISION
# ============================================================

def identify_product_camera():

    """
    Basic camera module.

    This version displays the camera.
    YOLO can be connected here for actual
    product recognition.
    """

    import cv2

    camera = cv2.VideoCapture(0)

    print("\nCamera started.")
    print("Press Q to quit.")

    detected_product = None

    while True:

        success, frame = camera.read()

        if not success:
            break

        cv2.imshow(
            "Smart Consumer Camera",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    return detected_product


# ============================================================
# WEB SCRAPING
# ============================================================

def scrape_product_price(url, product_name):

    headers = {
        "User-Agent":
        "Mozilla/5.0"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        text = soup.get_text(" ", strip=True)

        # Find prices such as ₹499, Rs.499, 499.00
        prices = re.findall(
            r'(?:₹|Rs\.?|INR)?\s?(\d+(?:\.\d{1,2})?)',
            text
        )

        if not prices:
            return None

        numbers = []

        for price in prices:

            try:
                value = float(price)

                if 1 <= value <= 100000:
                    numbers.append(value)

            except ValueError:
                pass

        if numbers:
            return min(numbers)

        return None

    except Exception as e:

        print("Scraping error:", e)
        return None


# ============================================================
# PRICE DATABASE
# ============================================================

def add_price(product, shop, price, url=""):

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO prices
        (
            product_name,
            shop,
            price,
            url,
            checked_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        product,
        shop,
        price,
        url,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


# ============================================================
# PRICE COMPARISON
# ============================================================

def compare_prices(product_name):

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        SELECT shop, price, url
        FROM prices
        WHERE product_name = ?
        ORDER BY price ASC
    """, (product_name,))

    results = cur.fetchall()

    conn.close()

    return results


def display_price_comparison(product_name):

    results = compare_prices(product_name)

    print("\n--------------------------------")
    print("PRICE COMPARISON")
    print("--------------------------------")

    if not results:
        print("No prices found.")
        return None

    for shop, price, url in results:

        print(
            f"{shop:20} ₹{price:.2f}"
        )

    cheapest = results[0]

    print(
        f"\nBEST PRICE: {cheapest[0]} "
        f"₹{cheapest[1]:.2f}"
    )

    return cheapest


# ============================================================
# SMART RECOMMENDATION ENGINE
# ============================================================

def smart_recommendation(product_name):

    results = compare_prices(product_name)

    if not results:
        return None

    cheapest_shop, cheapest_price, _ = results[0]

    average_price = sum(
        item[1] for item in results
    ) / len(results)

    savings = average_price - cheapest_price

    print("\n🧠 SMART RECOMMENDATION")
    print("--------------------------------")

    print(
        f"Product       : {product_name}"
    )

    print(
        f"Recommended   : {cheapest_shop}"
    )

    print(
        f"Best price    : ₹{cheapest_price:.2f}"
    )

    print(
        f"Average price : ₹{average_price:.2f}"
    )

    print(
        f"Potential save: ₹{savings:.2f}"
    )

    return {
        "shop": cheapest_shop,
        "price": cheapest_price,
        "savings": savings
    }


# ============================================================
# BILL GENERATION
# ============================================================

def create_bill(items):

    bill_items = []
    total = 0
    total_savings = 0

    print("\n================================")
    print("             BILL")
    print("================================")

    for item in items:

        recommendation = smart_recommendation(
            item["name"]
        )

        if recommendation is None:
            print(
                f"\nNo price found for "
                f"{item['name']}"
            )
            continue

        quantity = item["quantity"]

        item_total = (
            recommendation["price"] *
            quantity
        )

        item_saving = (
            recommendation["savings"] *
            quantity
        )

        total += item_total
        total_savings += item_saving

        bill_items.append({
            "name": item["name"],
            "quantity": quantity,
            "price": recommendation["price"],
            "shop": recommendation["shop"]
        })

        print(
            f"{item['name']:20} "
            f"{quantity:g} x "
            f"₹{recommendation['price']:.2f} "
            f"= ₹{item_total:.2f}"
        )

    print("--------------------------------")

    print(
        f"TOTAL: ₹{total:.2f}"
    )

    print(
        f"POTENTIAL SAVINGS: "
        f"₹{total_savings:.2f}"
    )

    save_bill(
        bill_items,
        total,
        total_savings
    )


# ============================================================
# SAVE BILL
# ============================================================

def save_bill(items, total, savings):

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    bill_date = datetime.now().isoformat()

    cur.execute("""
        INSERT INTO bills
        (bill_date, total, savings)
        VALUES (?, ?, ?)
    """, (
        bill_date,
        total,
        savings
    ))

    bill_id = cur.lastrowid

    for item in items:

        cur.execute("""
            INSERT INTO bill_items
            (
                bill_id,
                product_name,
                quantity,
                price,
                shop
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            bill_id,
            item["name"],
            item["quantity"],
            item["price"],
            item["shop"]
        ))

    conn.commit()
    conn.close()

    print(
        f"\nBill saved successfully."
    )

    print(
        f"Bill ID: {bill_id}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    create_database()

    print("\nSMART CONSUMER SYSTEM")
    print("=====================")

    items = collect_customer_items()

    if not items:
        print("No items entered.")
        return

    save_products(items)

    print("\nCustomer requirements:")

    for item in items:

        print(
            f"- {item['name']} "
            f"({item['quantity']} "
            f"{item['unit']})"
        )

    # --------------------------------------------------------
    # Camera can be activated here
    # --------------------------------------------------------

    # identify_product_camera()

    # --------------------------------------------------------
    # Example price data
    # In the full version this comes from web scraping.
    # --------------------------------------------------------

    for item in items:

        name = item["name"]

        # Demo prices
        add_price(
            name,
            "Shop A",
            100
        )

        add_price(
            name,
            "Shop B",
            92
        )

        add_price(
            name,
            "Shop C",
            97
        )

    # --------------------------------------------------------
    # Compare prices and create bill
    # --------------------------------------------------------

    create_bill(items)


if __name__ == "__main__":
    main()