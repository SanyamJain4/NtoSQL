"""
Builds the system + user prompt handed to the LLM. Keeping this separate
from generator.py makes it easy to iterate on prompt quality (the highest-
leverage lever in most NL2SQL systems) without touching call logic.

Two prompt "modes":
  - "sales":   used when the bundled demo database (with its `sales_flat`
               view) is detected. Keeps the original hand-tuned business
               rules and few-shot examples for that specific schema.
  - "generic": used for ANY other database (including uploaded ones).
               Domain-specific business rules are replaced with rules
               derived dynamically from the schema itself (enum hints,
               view detection, etc.) instead of hard-coded table/column
               names, so the same pipeline works unmodified on a new DB.

generator.py picks the mode automatically -- see NL2SQLEngine._prompt_mode.
"""

SALES_FEW_SHOT_EXAMPLES = [
    {
        "question": "Show top 5 products by revenue last quarter",
        "sql": """SELECT product_name, ROUND(SUM(net_revenue), 2) AS revenue
FROM sales_flat
WHERE order_status = 'Completed'
  AND order_date >= date('now', 'start of month', '-3 months')
  AND order_date < date('now', 'start of month')
GROUP BY product_name
ORDER BY revenue DESC
LIMIT 5;""",
    },
    {
        "question": "What is the month-over-month revenue growth this year?",
        "sql": """WITH monthly AS (
  SELECT strftime('%Y-%m', order_date) AS month, SUM(net_revenue) AS revenue
  FROM sales_flat
  WHERE order_status = 'Completed' AND strftime('%Y', order_date) = strftime('%Y', 'now')
  GROUP BY month
)
SELECT month, revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month)) / LAG(revenue) OVER (ORDER BY month), 1) AS mom_growth_pct
FROM monthly
ORDER BY month;""",
    },
    {
        "question": "Which customer segment has the highest average order value?",
        "sql": """SELECT customer_segment, ROUND(SUM(net_revenue) / COUNT(DISTINCT order_id), 2) AS avg_order_value
FROM sales_flat
WHERE order_status = 'Completed'
GROUP BY customer_segment
ORDER BY avg_order_value DESC;""",
    },
]

# Backwards-compatible alias (older code / notebooks may import this name).
FEW_SHOT_EXAMPLES = SALES_FEW_SHOT_EXAMPLES

SALES_DOMAIN_RULES = """\
- Prefer the `sales_flat` view for revenue/sales/product/category/region questions -- it is \
already joined and has `net_revenue` (after discount) precomputed. Only query raw tables \
directly when the question needs something sales_flat doesn't have (e.g. customer signup \
cohorts, product catalog fields not in the view).
- Always filter `order_status = 'Completed'` for revenue questions unless the user explicitly \
asks about refunds, cancellations, or all orders."""

GENERIC_DOMAIN_RULES = """\
- If a view is included in the schema below (marked "(view)"), it is usually already joined \
for a common analytical grain -- prefer it over manually re-joining the underlying tables \
when it covers what the question needs.
- Some columns below are annotated with "-- allowed values: ...". Only filter on those exact \
values; never invent a value that isn't listed.
- If the question implies filtering to a particular state (e.g. "completed", "active", \
"cancelled") and a matching status/state-like column exists, filter on it explicitly. \
Otherwise include all rows -- don't assume a default filter that isn't in the schema or the \
question."""

# Schema-agnostic query-pattern cheatsheet, used instead of literal few-shot
# Q/SQL pairs when we don't know the real table/column names in advance.
# Deliberately NOT phrased as copy-pasteable SQL (placeholders in angle
# brackets) so the model doesn't echo the placeholders back.
GENERIC_QUERY_PATTERNS = """\
Common query patterns (adapt to the ACTUAL table/column names in the schema \
below -- never copy the placeholder names shown here literally):
- Top-N by a metric: SELECT <dimension>, AGG(<metric>) AS m FROM <table> \
GROUP BY <dimension> ORDER BY m DESC LIMIT N;
- Change over time: bucket dates with strftime() in a CTE, then use \
LAG(<metric>) OVER (ORDER BY <period>) to compute period-over-period change.
- Rate or percentage: 100.0 * SUM(CASE WHEN <condition> THEN 1 ELSE 0 END) / COUNT(*)."""


SYSTEM_PROMPT_TEMPLATE = """You are an expert SQL analyst. You convert natural-language business \
questions into a single, correct, efficient SQLite query.

Rules:
- Output ONLY the SQL query. No explanation, no markdown code fences.
- Use ONLY the tables/views and columns listed in the schema below. Never invent columns.
{domain_rules}
- Use SQLite date functions (date('now'), strftime, julianday) for relative time ranges like \
"last quarter", "this year", "last 30 days".
- Never write INSERT, UPDATE, DELETE, DROP, ALTER, or any statement that modifies data. \
Read-only SELECT queries only.
- If the question is ambiguous (e.g. "top" without a metric), default to the most obviously \
relevant numeric column for the question (e.g. a revenue/amount/total-like column if one exists).
- End the query with a semicolon.

Schema (only the relevant tables/views for this question):
{schema}

{examples_label}:
{examples}
"""


def render_examples(examples: list) -> str:
    parts = []
    for ex in examples:
        parts.append(f"Q: {ex['question']}\nSQL:\n{ex['sql']}")
    return "\n\n".join(parts)


def build_prompt(question: str, table_infos: list, mode: str = "generic") -> tuple[str, str]:
    """Returns (system_prompt, user_prompt).

    mode: "sales" for the bundled demo DB (hand-tuned rules/examples for its
    known schema), "generic" for any other database -- including uploaded
    ones -- where rules are derived from the schema itself instead of
    hard-coded names.
    """
    schema_text = "\n\n".join(t.as_ddl_snippet() for t in table_infos)

    if mode == "sales":
        domain_rules = SALES_DOMAIN_RULES
        examples_label = "Examples"
        examples = render_examples(SALES_FEW_SHOT_EXAMPLES)
    else:
        domain_rules = GENERIC_DOMAIN_RULES
        examples_label = "Query pattern hints (not literal examples)"
        examples = GENERIC_QUERY_PATTERNS

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        domain_rules=domain_rules,
        schema=schema_text,
        examples_label=examples_label,
        examples=examples,
    )
    user_prompt = f"Q: {question}\nSQL:"
    return system_prompt, user_prompt
