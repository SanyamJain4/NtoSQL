"""
Streamlit web UI for the NL2SQL Analytics System.

Run with:
    export GEMINI_API_KEY=...
    streamlit run app.py

Works against the bundled demo database (sales.db) out of the box, OR
against ANY database you upload from the sidebar:
    - an existing SQLite file (.db / .sqlite / .sqlite3), or
    - one or more CSV / Excel files, each of which becomes a table.

The whole pipeline (schema retrieval, prompt building, validation,
execution, optimization, charting) automatically adapts to whatever
schema is active -- see nl2sql/generator.py's prompt-mode auto-detection
and nl2sql/db_loader.py for the upload handling.

If no API key is set, the app falls back to "Demo Mode": a small set of
canned question -> SQL mappings so you can see the full pipeline work
end-to-end without needing a live LLM key. Demo mode is only meaningful
for the bundled sales database (the canned SQL references its schema),
so it's automatically disabled once you upload your own database.
"""
import os
import sys
import tempfile
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from nl2sql.generator import NL2SQLEngine
from nl2sql.optimizer import analyze as optimize_analyze
from nl2sql.visualizer import render, choose_chart_type
from nl2sql.llm_client import LLMClient
from nl2sql.db_loader import load_upload, DBLoadError, SQLITE_EXTENSIONS, TABULAR_EXTENSIONS

BUNDLED_DB_PATH = os.path.join(os.path.dirname(__file__), "database", "sales.db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_QUESTIONS = [
    "Show top 5 products by revenue last quarter",
    "What is the month-over-month revenue growth this year?",
    "Which customer segment has the highest average order value?",
    "What is revenue by region?",
    "Show refund rate by payment method",
]

# Canned responses for demo mode -- mirrors the few-shot examples in
# prompt_builder.py so the demo experience matches what the real LLM would
# produce. Only valid for the bundled sales.db schema.
DEMO_RESPONSES = {
    "Show top 5 products by revenue last quarter": """
        SELECT product_name, ROUND(SUM(net_revenue), 2) AS revenue
        FROM sales_flat
        WHERE order_status = 'Completed'
        GROUP BY product_name
        ORDER BY revenue DESC
        LIMIT 5;
    """,
    "What is the month-over-month revenue growth this year?": """
        WITH monthly AS (
          SELECT strftime('%Y-%m', order_date) AS month, SUM(net_revenue) AS revenue
          FROM sales_flat
          WHERE order_status = 'Completed' AND strftime('%Y', order_date) = strftime('%Y', 'now')
          GROUP BY month
        )
        SELECT month, revenue,
               ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month)) / LAG(revenue) OVER (ORDER BY month), 1) AS mom_growth_pct
        FROM monthly
        ORDER BY month;
    """,
    "Which customer segment has the highest average order value?": """
        SELECT customer_segment, ROUND(SUM(net_revenue) / COUNT(DISTINCT order_id), 2) AS avg_order_value
        FROM sales_flat
        WHERE order_status = 'Completed'
        GROUP BY customer_segment
        ORDER BY avg_order_value DESC;
    """,
    "What is revenue by region?": """
        SELECT region_name, ROUND(SUM(net_revenue), 2) AS revenue
        FROM sales_flat
        WHERE order_status = 'Completed'
        GROUP BY region_name
        ORDER BY revenue DESC;
    """,
    "Show refund rate by payment method": """
        SELECT payment_method,
               ROUND(100.0 * COUNT(DISTINCT CASE WHEN order_status = 'Refunded' THEN order_id END)
                     / COUNT(DISTINCT order_id), 2) AS refund_rate_pct
        FROM sales_flat
        GROUP BY payment_method
        ORDER BY refund_rate_pct DESC;
    """,
}


class DemoLLM:
    """Stand-in LLM for demo mode -- returns a canned query for known sample
    questions, otherwise a generic top-categories query so the pipeline never
    dead-ends during a demo. Only used against the bundled sales.db."""

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        for q, sql in DEMO_RESPONSES.items():
            if q.lower() in user_prompt.lower():
                return sql
        return """
            SELECT category_name, ROUND(SUM(net_revenue), 2) AS revenue
            FROM sales_flat
            WHERE order_status = 'Completed'
            GROUP BY category_name
            ORDER BY revenue DESC
            LIMIT 5;
        """


@st.cache_resource
def get_engine(db_path: str, demo_mode: bool):
    if demo_mode:
        return NL2SQLEngine(db_path, llm_client=DemoLLM())
    return NL2SQLEngine(db_path, llm_client=LLMClient())


def _upload_dir() -> str:
    if "upload_dir" not in st.session_state:
        st.session_state["upload_dir"] = tempfile.mkdtemp(prefix="nl2sql_upload_")
    return st.session_state["upload_dir"]


def _use_database(db_path: str, label: str, is_custom: bool):
    """Switch the active database and reset session state that's tied to
    the previous schema (query history, sample questions, etc.)."""
    if st.session_state.get("db_path") == db_path:
        return
    st.session_state["db_path"] = db_path
    st.session_state["db_label"] = label
    st.session_state["is_custom_db"] = is_custom
    st.session_state["history"] = []
    if is_custom:
        st.session_state["demo_mode_override"] = False


def _dynamic_sample_questions(db_path: str) -> list:
    """For a custom uploaded DB we don't know the domain, so generate a
    couple of always-valid starter questions from the actual table names
    instead of the sales-specific SAMPLE_QUESTIONS."""
    try:
        from nl2sql.schema_retriever import SchemaRetriever
        tables = list(SchemaRetriever(db_path).tables.keys())
    except Exception:
        return []
    if not tables:
        return []
    first = tables[0]
    qs = [f"Show the first 10 rows of {first}", f"How many rows are in {first}?"]
    if len(tables) > 1:
        qs.append(f"How many rows are in each of {', '.join(tables[:4])}?")
    return qs


def sidebar_database_section():
    st.subheader("Database")

    if "db_path" not in st.session_state:
        _use_database(BUNDLED_DB_PATH, "Built-in demo (sales.db)", is_custom=False)

    st.caption(f"Active: **{st.session_state['db_label']}**")

    with st.expander("Upload your own database", expanded=st.session_state.get("is_custom_db", False)):
        st.caption(
            "Upload a SQLite file, or one/more CSV or Excel files "
            "(each sheet/file becomes a table). The whole pipeline -- schema "
            "retrieval, prompting, validation, optimization, charting -- "
            "adapts automatically to whatever you upload."
        )
        exts = sorted(e.lstrip(".") for e in SQLITE_EXTENSIONS + TABULAR_EXTENSIONS)
        uploaded = st.file_uploader(
            "Database file(s)", type=exts, accept_multiple_files=True, key="db_uploader"
        )
        col_load, col_reset = st.columns(2)
        load_clicked = col_load.button("Load database", width="stretch", disabled=not uploaded)
        reset_clicked = col_reset.button("Use demo database", width="stretch")

        if load_clicked and uploaded:
            files = [(f.name, f.getvalue()) for f in uploaded]
            try:
                with st.spinner("Reading file(s) and building database..."):
                    db_path, table_names, warnings = load_upload(files, _upload_dir())
            except DBLoadError as e:
                st.error(str(e))
            else:
                for w in warnings:
                    st.warning(w)
                label = f"Uploaded: {', '.join(f for f, _ in files)}"[:80]
                _use_database(db_path, label, is_custom=True)
                if table_names:
                    st.success(f"Loaded {len(table_names)} table(s): {', '.join(table_names)}")
                else:
                    st.success("Loaded uploaded SQLite database.")
                st.rerun()

        if reset_clicked:
            _use_database(BUNDLED_DB_PATH, "Built-in demo (sales.db)", is_custom=False)
            st.rerun()

    if not st.session_state["is_custom_db"]:
        st.caption(
            "SQLite `sales.db` — regions, customers, products, orders, "
            "order_items, plus a pre-joined `sales_flat` view."
        )


def main():
    st.set_page_config(page_title="NL2SQL Analytics", page_icon="📊", layout="wide")
    st.title("📊 NL2SQL Analytics")
    st.caption("Ask any database a question in plain English.")

    with st.sidebar:
        st.header("Settings")

        # Database section first: it can force demo mode off, so settings
        # below need to know about it.
        sidebar_database_section()
        db_path = st.session_state["db_path"]
        is_custom_db = st.session_state["is_custom_db"]

        st.divider()
        has_key = bool(os.getenv("GEMINI_API_KEY"))
        if is_custom_db:
            st.checkbox(
                "Demo mode (no API key needed)",
                value=False,
                disabled=True,
                help="Demo mode only works against the built-in sample database, "
                     "since its canned answers are written for that schema. "
                     "Set GEMINI_API_KEY below to query your uploaded data.",
            )
            demo_mode = False
        else:
            demo_mode = st.checkbox("Demo mode (no API key needed)", value=not has_key)

        if not demo_mode and not has_key:
            st.warning("GEMINI_API_KEY is not set in the environment. Set it before running.")

        st.divider()
        st.subheader("Sample questions")
        sample_qs = SAMPLE_QUESTIONS if not is_custom_db else _dynamic_sample_questions(db_path)
        if not sample_qs:
            st.caption("No sample questions available for this database yet -- just type your own below.")
        for q in sample_qs:
            if st.button(q, width="stretch"):
                st.session_state["question_input"] = q

    st.session_state.setdefault("history", [])
    st.session_state.setdefault("question_input", "")

    question = st.text_input(
        "Ask a question about your data",
        key="question_input",
        placeholder="e.g. Show top 5 products by revenue last quarter",
    )
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and question.strip():
        engine = get_engine(db_path, demo_mode)
        with st.spinner("Generating SQL, validating, and executing..."):
            result = engine.ask(question)
        st.session_state["history"].insert(0, (question, result))

    if not st.session_state["history"]:
        st.info("Pick a sample question from the sidebar, or type your own, to get started.")
        return

    for question, result in st.session_state["history"]:
        st.markdown(f"### {question}")

        if not result.success:
            st.error(f"Failed after {result.attempts} attempt(s).")
            for e in result.errors:
                st.code(e, language=None)
            continue

        col_sql, col_meta = st.columns([2, 1])
        with col_sql:
            st.markdown("**Generated SQL**")
            st.code(result.sql, language="sql")
        with col_meta:
            st.markdown("**Tables used**")
            st.write(", ".join(result.tables_used))
            if result.warnings:
                st.markdown("**Warnings**")
                for w in result.warnings:
                    st.caption(f"⚠️ {w}")

        tab_results, tab_chart, tab_optimize = st.tabs(["Results", "Chart", "Optimization"])

        with tab_results:
            st.dataframe(result.dataframe, width="stretch")
            st.caption(f"{len(result.dataframe)} row(s) returned in {result.attempts} attempt(s).")

        with tab_chart:
            chart_type = choose_chart_type(result.dataframe)
            if chart_type == "table":
                st.info("Result shape doesn't map to a chart type -- showing table only.")
            else:
                safe_name = "".join(c if c.isalnum() else "_" for c in question)[:40]
                chart_path = os.path.join(OUTPUT_DIR, f"chart_{safe_name}.png")
                render(result.dataframe, question, chart_path)
                st.image(chart_path, caption=f"Auto-selected chart type: {chart_type}")

        with tab_optimize:
            opt = optimize_analyze(db_path, result.sql)
            with st.expander("EXPLAIN QUERY PLAN"):
                for line in opt.plan:
                    st.code(line, language=None)
            if opt.issues:
                for issue, suggestion in zip(opt.issues, opt.suggestions):
                    st.warning(issue)
                    st.caption(f"→ {suggestion}")
            else:
                st.success("No issues detected -- query plan looks reasonable for this data size.")

        st.divider()


if __name__ == "__main__":
    main()
