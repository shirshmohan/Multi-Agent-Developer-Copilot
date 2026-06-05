# Telecom Multi-Agent System

A hierarchical multi-agent system over a telecom database. A LangGraph supervisor
routes natural-language requests to specialist agents (currently a SQL agent),
verifies their output, and returns results. Built to be model-agnostic (OpenAI
now, local Qwen later) and security-first (layered guards on all data access).

## Architecture (layers, bottom-up)
- **Data**: PostgreSQL (6 connected telecom tables) + MongoDB (telemetry), in Docker.
- **Access**: read-only DB access as a limited role + a deterministic SQL guard.
- **Model**: provider-agnostic LLM layer (`get_llm()`), swappable OpenAI/Qwen.
- **Agents**: the SQL agent (English to validated SQL to results).
- **Orchestration**: a LangGraph supervisor graph with routing + QA verification.

## Prerequisites
- Docker Desktop
- Python 3.12+
- An OpenAI API key (for now)

## Setup
1. Copy the env template and fill in real values:
   ```
   cp .env.example .env
   # edit .env with your real OpenAI key and chosen passwords
   ```
2. Start the databases:
   ```
   docker compose up -d
   ```
3. Apply the schema:
   ```
   Get-Content db/schema.sql | docker exec -i telecom_postgres psql -U admin -d telecom
   ```
4. Load the dataset (download the Telco Customer Churn CSV into `db/` first):
   ```
   pip install -r requirements.txt
   python db/load_data.py db/telco_churn.csv
   ```

## Run
```
python test_supervisor.py "what is the churn rate for month-to-month contracts?"
```

## Note on data
The Telco Customer Churn dataset is NOT included in this repo (it's git-ignored).
Download it from Kaggle and place it in `db/` before running the loader.
