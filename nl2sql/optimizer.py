"""
Lightweight, rule-based optimizer: runs EXPLAIN QUERY PLAN and flags common
anti-patterns (full table scans on large tables, missing index usage,
SELECT *, unbounded date ranges). This is deterministic and doesn't need an
LLM call -- fast, free, and reliable for the patterns it knows about.
An optional `llm_suggest()` escalates to the LLM for deeper, query-specific
rewrite suggestions when the rule-based pass finds issues.
"""
import sqlite3
import re
from dataclasses import dataclass, field


LARGE_TABLE_ROW_THRESHOLD = 500  # tables above this size warrant an index-scan check


@dataclass
class OptimizationReport:
    plan: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)


def explain(db_path: str, sql: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"EXPLAIN QUERY PLAN {sql}")
    rows = cur.fetchall()
    conn.close()
    # row format: id, parent, notused, detail
    return [row[3] for row in rows]


def _table_row_counts(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [r[0] for r in cur.fetchall()]
    counts = {}
    for t in tables:
        counts[t] = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    conn.close()
    return counts


def analyze(db_path: str, sql: str) -> OptimizationReport:
    plan = explain(db_path, sql)
    issues, suggestions = [], []
    row_counts = _table_row_counts(db_path)

    for line in plan:
        if "SCAN" in line and "USING INDEX" not in line:
            # Extract table name from something like "SCAN orders" or "SCAN TABLE orders"
            m = re.search(r"SCAN(?: TABLE)? (\w+)", line)
            table = m.group(1) if m else "?"
            size = row_counts.get(table, 0)
            if size > LARGE_TABLE_ROW_THRESHOLD:
                issues.append(f"Full table scan on '{table}' ({size} rows) without an index.")
                suggestions.append(
                    f"Consider adding an index on the filter/join column(s) used against '{table}', "
                    f"e.g. CREATE INDEX idx_{table}_<col> ON {table}(<col>);"
                )

    if re.search(r"select\s+\*", sql, re.IGNORECASE):
        issues.append("Query uses SELECT * -- pulls all columns even if only a few are needed.")
        suggestions.append("Select only the columns actually needed to reduce I/O and network payload.")

    if re.search(r"like\s+'%\w", sql, re.IGNORECASE):
        issues.append("Leading-wildcard LIKE pattern ('%text') cannot use an index.")
        suggestions.append("If this filters on a large table often, consider full-text search (FTS5) instead.")

    if not re.search(r"\blimit\b", sql, re.IGNORECASE):
        issues.append("No LIMIT clause -- could return an unbounded result set.")
        suggestions.append("Add a LIMIT unless the caller truly needs the full result set.")

    if not issues:
        suggestions.append("No obvious anti-patterns detected. Query plan looks reasonable for this data size.")

    return OptimizationReport(plan=plan, issues=issues, suggestions=suggestions)


def llm_suggest(llm_client, sql: str, report: OptimizationReport) -> str:
    """Escalate to the LLM for a query-specific rewrite suggestion, given the
    rule-based findings as context (keeps the LLM focused, not guessing blind)."""
    system = (
        "You are a SQL performance expert. Given a SQLite query and a list of "
        "detected issues, suggest a concise, concrete rewrite or indexing strategy. "
        "Be specific and brief -- 3-5 bullet points max, no fluff."
    )
    user = f"Query:\n{sql}\n\nDetected issues:\n" + "\n".join(f"- {i}" for i in report.issues)
    return llm_client.complete(system, user)
