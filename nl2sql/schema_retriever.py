"""
Schema-aware retrieval ("RAG over metadata").

Instead of dumping the entire database schema into every prompt (wasteful,
and confusing for the LLM once you have dozens of tables), we:
  1. Introspect the DB once to build a "document" per table describing its
     columns, types, foreign keys, and a short natural-language purpose.
  2. Embed those documents with TF-IDF (works fully offline, no API key
     needed -- swap in real embeddings + a vector DB for production scale).
  3. At query time, retrieve only the top-K most relevant tables/views for
     the user's question and hand only those to the prompt builder.

This keeps prompts small, keeps accuracy high (less irrelevant schema =
less chance the model picks the wrong column), and scales to large
warehouses where the full schema wouldn't fit in context at all.
"""
import re
import sqlite3
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_CHECK_IN_RE = re.compile(
    r"CHECK\s*\(\s*[\"'`]?(\w+)[\"'`]?\s+IN\s*\(([^)]+)\)\s*\)", re.IGNORECASE
)


def _extract_enum_hints(create_sql: str) -> dict:
    """Parse `col TEXT CHECK (col IN ('A','B','C'))` style constraints out of
    a CREATE TABLE statement, generically, for ANY database -- this is what
    lets the LLM know valid values for status/category-like columns without
    us hand-coding them (the way the original prompt hard-coded
    order_status values for the bundled sales.db)."""
    hints = {}
    for col, values_raw in _CHECK_IN_RE.findall(create_sql or ""):
        values = [v.strip().strip("'\"") for v in values_raw.split(",")]
        if values:
            hints[col] = values
    return hints


# Hand-written business context per table/view. In production this would
# come from a data catalog (dbt descriptions, Alation, etc.) -- giving the
# LLM business meaning, not just column names, is what makes NL2SQL work
# on real warehouses.
TABLE_DESCRIPTIONS = {
    "sales_flat": (
        "Denormalized view, one row per order line item. Use this for almost "
        "all revenue, sales, product, category, region, and customer-segment "
        "analytics. Contains net_revenue (after discount) and total_cost. "
        "Prefer this view over joining raw tables manually."
    ),
    "orders": (
        "One row per order (order header). order_status: Completed, Refunded, "
        "Cancelled, Pending. discount_pct applies to the whole order."
    ),
    "order_items": (
        "One row per product line within an order. Links orders to products."
    ),
    "customers": (
        "One row per customer. customer_segment: Consumer, SMB, Enterprise. "
        "signup_date is when they registered."
    ),
    "products": (
        "One row per product (catalog). unit_cost vs unit_price gives margin. "
        "is_active flags discontinued products."
    ),
    "categories": "Product category lookup table (category_name, parent_category).",
    "regions": "Customer region/country lookup table.",
}


def _auto_description(name: str, kind: str, columns: list, foreign_keys: list) -> str:
    """Generate a reasonable description for a table/view we have no
    hand-written business context for (i.e. any table on a DB other than
    the bundled sales.db). Column names + FK relationships are the only
    signal we have, but they're usually enough for TF-IDF retrieval to
    work and for the LLM to understand the table's role."""
    col_names = ", ".join(c for c, _ in columns)
    parts = [f"{kind.capitalize()} '{name}' with columns: {col_names}."]
    if foreign_keys:
        refs = ", ".join(f"{c} references {rt}.{rc}" for c, rt, rc in foreign_keys)
        parts.append(f"Relationships: {refs}.")
    return " ".join(parts)


@dataclass
class TableInfo:
    name: str
    kind: str  # 'table' or 'view'
    columns: list = field(default_factory=list)  # list of (name, type)
    foreign_keys: list = field(default_factory=list)  # list of (col, ref_table, ref_col)
    description: str = ""
    enum_hints: dict = field(default_factory=dict)  # {column: [allowed values]}

    def as_document(self) -> str:
        col_str = ", ".join(f"{c}:{t}" for c, t in self.columns)
        fk_str = "; ".join(f"{c} -> {rt}.{rc}" for c, rt, rc in self.foreign_keys)
        return (
            f"{self.name} ({self.kind}). {self.description} "
            f"Columns: {col_str}. Foreign keys: {fk_str or 'none'}."
        )

    def as_ddl_snippet(self) -> str:
        lines = []
        for c, t in self.columns:
            if c in self.enum_hints:
                values = ", ".join(repr(v) for v in self.enum_hints[c])
                lines.append(f"  {c} {t}  -- allowed values: {values}")
            else:
                lines.append(f"  {c} {t}")
        cols = "\n".join(lines)
        return f"-- {self.description}\nTABLE {self.name} (\n{cols}\n)"


class SchemaRetriever:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.tables: dict[str, TableInfo] = {}
        self._introspect()
        self._build_index()

    def _introspect(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            "SELECT name, type, sql FROM sqlite_master "
            "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'"
        )
        objects = cur.fetchall()

        for name, kind, create_sql in objects:
            cur.execute(f"PRAGMA table_info({name})")
            columns = [(row[1], row[2]) for row in cur.fetchall()]

            fks = []
            cur.execute(f"PRAGMA foreign_key_list({name})")
            for row in cur.fetchall():
                # row: id, seq, table, from, to, ...
                fks.append((row[3], row[2], row[4]))

            description = TABLE_DESCRIPTIONS.get(name) or _auto_description(name, kind, columns, fks)
            enum_hints = _extract_enum_hints(create_sql or "")

            self.tables[name] = TableInfo(
                name=name,
                kind=kind,
                columns=columns,
                foreign_keys=fks,
                description=description,
                enum_hints=enum_hints,
            )
        conn.close()

    def _build_index(self):
        self._names = list(self.tables.keys())
        docs = [self.tables[n].as_document() for n in self._names]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(docs)

    def retrieve(self, question: str, top_k: int = 4) -> list[TableInfo]:
        """Return the top_k most relevant tables/views for a natural-language question."""
        q_vec = self._vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        ranked = sorted(zip(self._names, sims), key=lambda x: x[1], reverse=True)

        # Always include sales_flat if it exists -- it's the primary analytical
        # surface and cheap to include even when the keyword match is weak.
        selected = [n for n, s in ranked[:top_k]]
        if "sales_flat" in self.tables and "sales_flat" not in selected:
            selected = ["sales_flat"] + selected[: top_k - 1]

        return [self.tables[n] for n in selected]

    def full_schema_text(self) -> str:
        """Fallback: full schema, for small DBs or when retrieval isn't wanted."""
        return "\n\n".join(t.as_ddl_snippet() for t in self.tables.values())
