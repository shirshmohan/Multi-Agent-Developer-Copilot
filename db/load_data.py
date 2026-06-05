"""
load_data.py
Reads the Telco Customer Churn CSV, normalizes it into the 6 Postgres tables,
and seeds MongoDB with synthetic per-customer network telemetry.

Run AFTER `docker compose up -d` and AFTER applying schema.sql.

Usage:  python load_data.py path/to/telco_churn.csv
"""
import sys
import random
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pymongo import MongoClient

# --- connection settings: match docker-compose.yml. We connect as ADMIN to LOAD. ---
PG = dict(host="localhost", port=5433, dbname="telecom", user="admin", password="admin_pw")
MONGO_URI = "mongodb://admin:admin_pw@localhost:27017/"

# Synthetic cities so the schema has geography to join on.
CITIES = [("Mumbai","MH",20_000_000), ("Delhi","DL",19_000_000),
          ("Bengaluru","KA",13_000_000), ("Chennai","TN",11_000_000),
          ("Kolkata","WB",15_000_000), ("Hyderabad","TG",10_000_000)]

def yn(v):  # the CSV uses 'Yes'/'No' strings; convert to real booleans
    return str(v).strip().lower() in ("yes", "true", "1")

def load_postgres(df):
    conn = psycopg2.connect(**PG); conn.autocommit = False
    cur = conn.cursor()

    # 1. cities — insert the lookup rows, capture their generated IDs
    cur.execute("DELETE FROM billing; DELETE FROM services; DELETE FROM contracts; "
                "DELETE FROM customers; DELETE FROM cities;")   # idempotent re-runs
    city_ids = []
    for name, state, pop in CITIES:
        cur.execute("INSERT INTO cities (city_name,state,population) VALUES (%s,%s,%s) "
                    "RETURNING city_id", (name, state, pop))
        city_ids.append(cur.fetchone()[0])

    # 2. customers — identity + target, each randomly assigned a city
    customers = [(r.customerID, r.gender, bool(r.SeniorCitizen), yn(r.Partner),
                  yn(r.Dependents), random.choice(city_ids), yn(r.Churn))
                 for r in df.itertuples()]
    execute_values(cur,
        "INSERT INTO customers (customer_id,gender,senior_citizen,partner,"
        "dependents,city_id,churn) VALUES %s", customers)

    # 3. contracts
    contracts = [(r.customerID, int(r.tenure), r.Contract,
                  yn(r.PaperlessBilling), r.PaymentMethod) for r in df.itertuples()]
    execute_values(cur,
        "INSERT INTO contracts (customer_id,tenure_months,contract_type,"
        "paperless_billing,payment_method) VALUES %s", contracts)

    # 4. services
    services = [(r.customerID, yn(r.PhoneService), r.MultipleLines, r.InternetService,
                 r.OnlineSecurity, r.OnlineBackup, r.DeviceProtection, r.TechSupport,
                 r.StreamingTV, r.StreamingMovies) for r in df.itertuples()]
    execute_values(cur,
        "INSERT INTO services (customer_id,phone_service,multiple_lines,internet_service,"
        "online_security,online_backup,device_protection,tech_support,streaming_tv,"
        "streaming_movies) VALUES %s", services)

    # 5. billing — TotalCharges has blanks in the raw data; coerce to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    billing = [(r.customerID, float(r.MonthlyCharges), float(r.TotalCharges))
               for r in df.itertuples()]
    execute_values(cur,
        "INSERT INTO billing (customer_id,monthly_charges,total_charges) VALUES %s", billing)

    conn.commit(); cur.close(); conn.close()
    print(f"Postgres: loaded {len(customers)} customers across 6 tables.")

def seed_mongo(df):
    # MongoDB's DISTINCT purpose: semi-structured telemetry that doesn't fit rows.
    client = MongoClient(MONGO_URI)
    coll = client["telecom"]["network_events"]
    coll.drop()  # idempotent
    docs = []
    for r in df.itertuples():
        n = random.randint(2, 6)  # each customer gets a few event logs of varying shape
        for _ in range(n):
            docs.append({
                "customer_id": r.customerID,            # KEYED to the real Postgres customer
                "event_type": random.choice(["dropped_call","slow_data","sms_fail","ok"]),
                "signal_dbm": random.randint(-120, -60),
                "tower": f"TWR-{random.randint(1,50):03d}",
                "metrics": {                            # nested doc — the "why Mongo" part
                    "latency_ms": random.randint(10, 400),
                    "packet_loss": round(random.random()*0.1, 4),
                },
            })
    coll.insert_many(docs)
    print(f"Mongo: seeded {len(docs)} network_events documents.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python load_data.py path/to/telco_churn.csv")
    df = pd.read_csv(sys.argv[1])
    load_postgres(df)
    seed_mongo(df)
    print("Done.")
