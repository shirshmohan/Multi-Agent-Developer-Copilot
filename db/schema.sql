-- schema.sql
-- 6-table normalized telecom schema, derived from the Telco Customer Churn dataset.
-- Run this ONCE against the fresh Postgres container before loading data.
--
-- DESIGN: the flat churn CSV mixes four entities into one row. We split them into
-- linked tables so the SQL agent has real joins to generate and optimize.

-- ---------- 1. cities : a geography lookup (synthesized) ----------
-- Exists so queries have something to GROUP BY / JOIN on geographically.
CREATE TABLE cities (
    city_id     SERIAL PRIMARY KEY,
    city_name   TEXT NOT NULL,
    state       TEXT NOT NULL,
    population  INTEGER
);

-- ---------- 2. customers : identity + demographics + the ML TARGET ----------
-- churn is the column the Data agent will learn to predict.
CREATE TABLE customers (
    customer_id     TEXT PRIMARY KEY,          -- the natural key from the dataset (e.g. '7590-VHVEG')
    gender          TEXT,
    senior_citizen  BOOLEAN,
    partner         BOOLEAN,
    dependents      BOOLEAN,
    city_id         INTEGER REFERENCES cities(city_id),   -- FK -> cities
    churn           BOOLEAN                     -- TARGET: did the customer leave?
);

-- ---------- 3. contracts : the customer's plan/tenure ----------
CREATE TABLE contracts (
    contract_id     SERIAL PRIMARY KEY,
    customer_id     TEXT REFERENCES customers(customer_id),  -- FK -> customers
    tenure_months   INTEGER,                    -- how long they've been a customer
    contract_type   TEXT,                       -- 'Month-to-month' / 'One year' / 'Two year'
    paperless_billing BOOLEAN,
    payment_method  TEXT
);

-- ---------- 4. services : which products the customer subscribes to ----------
CREATE TABLE services (
    service_id          SERIAL PRIMARY KEY,
    customer_id         TEXT REFERENCES customers(customer_id),  -- FK -> customers
    phone_service       BOOLEAN,
    multiple_lines      TEXT,
    internet_service    TEXT,                   -- 'DSL' / 'Fiber optic' / 'No'
    online_security     TEXT,
    online_backup       TEXT,
    device_protection   TEXT,
    tech_support        TEXT,
    streaming_tv        TEXT,
    streaming_movies    TEXT
);

-- ---------- 5. billing : the money ----------
CREATE TABLE billing (
    billing_id      SERIAL PRIMARY KEY,
    customer_id     TEXT REFERENCES customers(customer_id),  -- FK -> customers
    monthly_charges NUMERIC(10,2),
    total_charges   NUMERIC(10,2)
);

-- ---------- 6. agent_workspace schema : where agents are ALLOWED to write ----------
-- Agents NEVER write to the tables above. They create/insert here only.
CREATE SCHEMA agent_workspace;

-- =====================================================================
--  SECURITY: the load-bearing part. A separate, deliberately limited role.
--  Even if every layer above it is bypassed, the database itself refuses.
-- =====================================================================
CREATE ROLE agent_user WITH LOGIN PASSWORD 'agent_pw';

-- Read-only on the real data: SELECT, nothing else.
GRANT USAGE ON SCHEMA public TO agent_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO agent_user;

-- Full rights ONLY inside the sandbox schema.
GRANT USAGE, CREATE ON SCHEMA agent_workspace TO agent_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA agent_workspace TO agent_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA agent_workspace
    GRANT ALL PRIVILEGES ON TABLES TO agent_user;

-- Explicitly DENY the destructive verbs on the real data.
-- (REVOKE is belt-and-suspenders; we never granted them, but we make it loud.)
REVOKE INSERT, UPDATE, DELETE, TRUNCATE
    ON ALL TABLES IN SCHEMA public FROM agent_user;
