# LLM-Based Natural Language to SQL Analytics System

Ask a database questions in plain English and get back validated SQL,
results, optimization suggestions, and an auto-generated chart.

```
"Show top 5 products by revenue last quarter"
        │
        ▼
 Schema retriever (RAG over table/column metadata, TF-IDF)
        │
        ▼
 Prompt builder (schema + few-shot examples)
        │
        ▼
 LLM SQL generator (Google Gemini)
        │
        ▼
 Validator (safety guardrails + syntax/column checks)
        │
        ├─── fails ──► feed error back to LLM, retry (bounded)
        ▼
 Executor (SQLite)
        │
        ├──► Optimizer (EXPLAIN QUERY PLAN + rule-based suggestions)
        └──► Visualizer (auto-picks bar / line / scatter / KPI)
```

## 1. Setup

```bash
cd nl2sql_analytics
pip install -r requirements.txt

python database/build_db.py       # builds database/sales.db (seeded, reproducible)
```

## 2. Configure Gemini

```bash
export GEMINI_API_KEY=...
# optional, defaults to gemini-2.5-flash:
export NL2SQL_MODEL=gemini-2.5-flash
```

## 3. Run

**Web UI (recommended):**
```bash
pip install streamlit
streamlit run app.py
```
Opens at `http://localhost:8501`. Click a sample question in the sidebar or
type your own, then hit **Ask**. Results appear in three tabs: Results
(table), Chart (auto-picked bar/line/scatter/KPI), and Optimization
(EXPLAIN QUERY PLAN + suggestions). No API key? Check **Demo mode** in the
sidebar — it's on by default when no key is detected, and runs the full
pipeline (retrieval, validation, execution, optimization, charting) against
five canned example questions so you can see it work end-to-end for free.

**CLI:**
```bash
# Interactive:
python cli.py

# One-shot:
python cli.py "What is the month-over-month revenue growth this year?"
```

Each question prints the generated SQL, the result table, any optimizer
warnings, and saves a chart to `output/`.

## 4. Programmatic use

```python
from nl2sql.generator import NL2SQLEngine

engine = NL2SQLEngine("database/sales.db")
result = engine.ask("Which customer segment has the highest average order value?")

print(result.sql)          # the validated, executed SQL
print(result.dataframe)    # a pandas DataFrame of results
print(result.tables_used)  # which tables the retriever selected
print(result.warnings)     # e.g. "no LIMIT clause, capped to 1000 rows"
```

## Database

`database/schema.sql` defines a normalized sales schema (`regions`,
`customers`, `categories`, `products`, `orders`, `order_items`) plus a
`sales_flat` view — one row per order line item with revenue and cost
precomputed — which is what most analytical questions actually want.
`database/build_db.py` seeds it with ~400 customers and ~2,300 orders of
deterministic synthetic data.

Example complex query already supported by the schema (window function,
month-over-month growth):

```sql
WITH monthly AS (
  SELECT strftime('%Y-%m', order_date) AS month, SUM(net_revenue) AS revenue
  FROM sales_flat
  WHERE order_status = 'Completed' AND strftime('%Y', order_date) = strftime('%Y', 'now')
  GROUP BY month
)
SELECT month, revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 1) AS mom_growth_pct
FROM monthly
ORDER BY month;
```

## Module map

| File | Responsibility |
|---|---|
| `nl2sql/schema_retriever.py` | Introspects the DB; TF-IDF retrieval of the most relevant tables per question (RAG over metadata, not raw text) |
| `nl2sql/prompt_builder.py` | System prompt + few-shot examples |
| `nl2sql/llm_client.py` | Thin wrapper around the Google Gemini SDK |
| `nl2sql/generator.py` | Orchestrates retrieve → prompt → generate → validate → execute, with retry-on-error |
| `nl2sql/validator.py` | Blocks non-SELECT statements, checks hallucinated tables/columns, enforces row limits |
| `nl2sql/optimizer.py` | Runs `EXPLAIN QUERY PLAN`, flags full scans / `SELECT *` / missing `LIMIT`, optional LLM-generated rewrite suggestions |
| `nl2sql/visualizer.py` | Picks bar / line / grouped-bar / scatter / KPI based on result shape, renders PNG |
| `cli.py` | Interactive/one-shot entry point |

## Design notes

- **Schema retrieval, not full-schema dumping.** Only the top-K relevant
  tables go into the prompt. This scales to real warehouses (hundreds of
  tables) where the full schema wouldn't fit in context, and it measurably
  reduces wrong-column hallucinations.
- **Validation is layered, not optional.** Every write/DDL keyword is
  blocked regardless of framing (comments, multi-statement injection,
  CTEs). Table/column references are checked against the schema actually
  retrieved, catching hallucinations before execution — not after.
- **Retry uses execution feedback.** When a query fails (bad syntax, an
  unknown column, a SQLite runtime error), the actual error message is fed
  back to the LLM for a bounded number of retries, rather than failing
  immediately or retrying blind.
- **Optimizer is rule-based first, LLM second.** `EXPLAIN QUERY PLAN`
  gives deterministic, free, reliable signal (full scan vs. index search).
  The LLM is only invoked for a deeper rewrite suggestion once the
  rule-based pass has found something concrete to hand it.

## Extending this

- Swap TF-IDF for real embeddings + a vector DB (pgvector, FAISS) once the
  schema is large enough that keyword overlap stops being a reliable
  retrieval signal.
- Add a semantic layer / business glossary (e.g. "revenue" → `net_revenue`,
  not gross) to `prompt_builder.py` to resolve business-term ambiguity.
- Track execution accuracy (does the query run and return the right rows)
  separately from exact SQL match — the latter is too strict since many
  valid queries look syntactically different.
