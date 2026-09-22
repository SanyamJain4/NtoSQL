#!/usr/bin/env python3
"""
Interactive CLI for the NL2SQL Analytics System.

Usage:
    export GEMINI_API_KEY=...
    python cli.py

    # or one-shot, against the bundled demo database:
    python cli.py "Show top 5 products by revenue last quarter"

    # or against ANY database -- pass an existing SQLite file:
    python cli.py --db /path/to/other.db "How many rows are in orders?"

    # or build a fresh SQLite database from CSV/Excel file(s) first:
    python cli.py --from-files customers.csv orders.xlsx -- "Top customers by orders"

Each question goes through: schema retrieval -> LLM SQL generation ->
validation/safety checks -> execution (with retry-on-error) -> optimizer
analysis -> auto-generated chart saved to ./output/. The prompt/rules
automatically adapt to whatever schema is active -- see
nl2sql/generator.py's prompt-mode auto-detection.
"""
import argparse
import sys
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()
from nl2sql.generator import NL2SQLEngine
from nl2sql.optimizer import analyze as optimize_analyze
from nl2sql.visualizer import render
from nl2sql.db_loader import load_upload, DBLoadError

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "database", "sales.db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def run_question(engine: NL2SQLEngine, question: str, chart_index: int):
    print(f"\nQ: {question}")
    result = engine.ask(question)

    if not result.success:
        print(f"  Failed after {result.attempts} attempt(s).")
        for e in result.errors:
            print(f"    error: {e}")
        return

    print(f"  Tables used: {result.tables_used}")
    print(f"  SQL:\n{result.sql}\n")
    if result.warnings:
        print(f"  Warnings: {result.warnings}")

    print(f"  Rows returned: {len(result.dataframe)}")
    print(result.dataframe.head(10).to_string(index=False))

    # Optimization pass
    opt = optimize_analyze(engine.db_path, result.sql)
    if opt.issues:
        print("\n  Optimization suggestions:")
        for s in opt.suggestions:
            print(f"    - {s}")
    else:
        print("\n  Optimizer: no issues detected.")

    # Visualization
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chart_path = os.path.join(OUTPUT_DIR, f"chart_{chart_index}.png")
    chart_type = render(result.dataframe, question, chart_path)
    print(f"  Chart ({chart_type}) saved to {chart_path}")


def resolve_db_path(args) -> str:
    """Figure out which database to use: an explicit --db path, a fresh
    database built from --from-files (CSV/Excel), or the bundled demo db."""
    if args.db:
        if not os.path.exists(args.db):
            print(f"Database not found: {args.db}")
            sys.exit(1)
        return args.db

    if args.from_files:
        files = []
        for path in args.from_files:
            if not os.path.exists(path):
                print(f"File not found: {path}")
                sys.exit(1)
            with open(path, "rb") as f:
                files.append((os.path.basename(path), f.read()))
        dest_dir = tempfile.mkdtemp(prefix="nl2sql_cli_")
        try:
            db_path, table_names, warnings = load_upload(files, dest_dir)
        except DBLoadError as e:
            print(f"Could not build database from files: {e}")
            sys.exit(1)
        for w in warnings:
            print(f"  warning: {w}")
        if table_names:
            print(f"Built database with table(s): {', '.join(table_names)}")
        return db_path

    if not os.path.exists(DEFAULT_DB_PATH):
        print("Database not found. Run: python database/build_db.py")
        sys.exit(1)
    return DEFAULT_DB_PATH


def main():
    parser = argparse.ArgumentParser(description="NL2SQL Analytics CLI")
    parser.add_argument("--db", help="Path to an existing SQLite database to query.")
    parser.add_argument(
        "--from-files", nargs="+", metavar="FILE",
        help="One or more CSV/Excel files to build a fresh SQLite database from "
             "(each file/sheet becomes a table). Ignored if --db is given. "
             "Put '--' before your question so it isn't swallowed as another "
             "filename, e.g.: --from-files a.csv b.csv -- \"my question\"",
    )
    parser.add_argument("question", nargs="*", help="One-shot question (omit for interactive mode).")
    args = parser.parse_args()

    db_path = resolve_db_path(args)
    engine = NL2SQLEngine(db_path)

    if args.question:
        run_question(engine, " ".join(args.question), chart_index=1)
        return

    print(f"NL2SQL Analytics -- database: {db_path}")
    print("Type a question, or 'quit' to exit.")
    i = 1
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("quit", "exit"):
            break
        run_question(engine, q, chart_index=i)
        i += 1


if __name__ == "__main__":
    main()
