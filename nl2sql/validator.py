"""
Validates LLM-generated SQL before it ever touches the database.

Two layers:
  1. Safety guardrails -- hard blocks on anything that isn't a read-only
     SELECT, regardless of how the query is phrased or wrapped (CTEs,
     subqueries, comments used to hide a second statement, etc).
  2. Syntax / semantic validation -- parses the SQL with sqlglot and checks
     that every referenced table/column actually exists in the schema we
     retrieved, catching hallucinated columns before execution.
"""
import re
import sqlglot
from sqlglot import exp
from dataclasses import dataclass


BLOCKED_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "truncate", "create",
    "replace", "attach", "detach", "pragma", "vacuum", "grant", "revoke",
}

MAX_ROWS_DEFAULT = 1000


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list
    warnings: list
    sanitized_sql: str = ""


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


def _extract_sql(raw: str) -> str:
    """LLMs sometimes wrap SQL in markdown fences despite instructions -- strip them."""
    raw = raw.strip()
    fence = re.match(r"^```(?:sql)?\s*(.*?)\s*```$", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    return raw


def validate(raw_sql: str, table_infos: list, max_rows: int = MAX_ROWS_DEFAULT) -> ValidationResult:
    errors, warnings = [], []
    sql = _extract_sql(raw_sql)
    clean = _strip_comments(sql).strip().rstrip(";")

    # 1. Must be a single statement.
    statements = [s for s in clean.split(";") if s.strip()]
    if len(statements) > 1:
        errors.append("Multiple SQL statements detected; only a single SELECT is allowed.")

    # 2. Block any write/DDL keyword anywhere in the query.
    lowered = clean.lower()
    for kw in BLOCKED_KEYWORDS:
        if re.search(rf"\b{kw}\b", lowered):
            errors.append(f"Blocked keyword detected: '{kw}'. Only read-only SELECT queries are permitted.")

    # 3. Must start with SELECT or WITH (CTE).
    if not re.match(r"^\s*(select|with)\b", lowered):
        errors.append("Query must start with SELECT or WITH (read-only queries only).")

    if errors:
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # 4. Parse with sqlglot to catch syntax errors and check table/column references.
    try:
        parsed = sqlglot.parse_one(clean, read="sqlite")
    except Exception as e:
        errors.append(f"SQL syntax error: {e}")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    known_tables = {t.name.lower() for t in table_infos}
    known_columns = set()
    for t in table_infos:
        for col, _ in t.columns:
            known_columns.add(col.lower())

    # CTEs (WITH x AS (...)) define their own names, which sqlglot also
    # surfaces via find_all(exp.Table) when they're referenced later in the
    # query. Those are not real schema tables and must not be flagged as
    # hallucinated/unknown.
    cte_names = {cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE)}

    referenced_tables = {tbl.name.lower() for tbl in parsed.find_all(exp.Table)}
    unknown_tables = referenced_tables - known_tables - cte_names
    if unknown_tables:
        errors.append(
            f"Query references table(s) not in the retrieved schema: {sorted(unknown_tables)}. "
            "Either the schema retriever missed a relevant table, or the model hallucinated one."
        )

    # Column check is a soft warning (aliases and computed columns make this
    # unreliable to hard-block on), but it catches obvious hallucinations.
    referenced_columns = {c.name.lower() for c in parsed.find_all(exp.Column)}
    unknown_columns = referenced_columns - known_columns
    unknown_columns = {c for c in unknown_columns if c not in ("*",)}
    if unknown_columns:
        warnings.append(f"Columns not found in schema (may be aliases): {sorted(unknown_columns)}")

    # 5. Enforce a row cap so a bad query can't blow up the response.
    has_limit = parsed.find(exp.Limit) is not None
    sanitized = clean
    if not has_limit:
        sanitized = f"{clean}\nLIMIT {max_rows}"
        warnings.append(f"No LIMIT clause found; capped result to {max_rows} rows.")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, sanitized_sql=sanitized)
