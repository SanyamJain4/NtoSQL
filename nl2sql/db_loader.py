"""
Turns user-uploaded files into a queryable SQLite database, so the rest of
the pipeline (schema_retriever -> prompt_builder -> generator -> optimizer
-> visualizer) can run against ANY database, not just the bundled sales.db.

Supported inputs:
  - An existing SQLite database (.db / .sqlite / .sqlite3) -- used as-is.
  - One or more CSV files -- each file becomes a table.
  - One or more Excel files (.xlsx / .xls) -- each sheet becomes a table.
  - Any mix of CSV/Excel files in a single upload -- all resulting tables
    land in one new SQLite database so they can be joined/queried together.

Uploading a SQLite file together with CSV/Excel files is treated as
ambiguous (which one is "the" database?), so the SQLite file wins and the
rest are reported back as ignored via `warnings`.

This is intentionally best-effort (permissive type/date inference) so an
analyst can drag in whatever files they have and start asking questions
immediately, rather than hand-writing a schema first.
"""
import os
import re
import sqlite3
import pandas as pd


SQLITE_MAGIC = b"SQLite format 3\x00"
SQLITE_EXTENSIONS = (".db", ".sqlite", ".sqlite3")
TABULAR_EXTENSIONS = (".csv", ".xlsx", ".xls")


class DBLoadError(Exception):
    """Raised when the uploaded file(s) can't be turned into a usable DB."""


def _sanitize_name(name: str) -> str:
    """Turn a filename/sheet name into a safe, lowercase SQL identifier."""
    name = os.path.splitext(name)[0]
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()
    if not name:
        name = "table"
    if name[0].isdigit():
        name = f"t_{name}"
    return name


def _dedupe(name: str, used: set) -> str:
    base, i = name, 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: columns that look like dates get normalized to ISO
    'YYYY-MM-DD' strings so SQLite date functions (date(), strftime(),
    julianday()) work on them later, the same way they do for the bundled
    sales.db's DATE columns."""
    for col in df.columns:
        if df[col].dtype != object:
            continue
        sample = df[col].dropna().astype(str).head(20)
        if sample.empty:
            continue
        looks_date = sample.str.match(
            r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$|^\d{1,2}/\d{1,2}/\d{2,4}$"
        ).mean() > 0.8
        if looks_date:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                df[col] = parsed.dt.strftime("%Y-%m-%d")
    return df


def is_sqlite_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(SQLITE_MAGIC)) == SQLITE_MAGIC
    except OSError:
        return False


def _load_sqlite(tmp_path: str, dest_dir: str) -> str:
    if not is_sqlite_file(tmp_path):
        raise DBLoadError(
            "That file doesn't look like a valid SQLite database "
            "(bad header). If it's an export from another DB engine, "
            "convert it to SQLite first, or upload CSV/Excel instead."
        )
    dest = os.path.join(dest_dir, "uploaded.db")
    with open(tmp_path, "rb") as f_in, open(dest, "wb") as f_out:
        f_out.write(f_in.read())

    conn = sqlite3.connect(dest)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
        if not cur.fetchall():
            raise DBLoadError("That SQLite file has no tables or views in it.")
    finally:
        conn.close()
    return dest


def _build_db_from_tabular(files: list, dest_dir: str) -> tuple[str, list, list]:
    """files: list of (filename, bytes). Returns (db_path, table_names, warnings)."""
    dest = os.path.join(dest_dir, "uploaded.db")
    conn = sqlite3.connect(dest)
    used_names, created, warnings = set(), [], []

    for i, (filename, content) in enumerate(files):
        ext = os.path.splitext(filename)[1].lower()
        tmp_path = os.path.join(dest_dir, f"_src_{i}{ext}")
        with open(tmp_path, "wb") as f:
            f.write(content)

        try:
            if ext == ".csv":
                df = pd.read_csv(tmp_path)
                if df.empty:
                    warnings.append(f"Skipped '{filename}': no rows found.")
                    continue
                table_name = _dedupe(_sanitize_name(filename), used_names)
                _coerce_dates(df).to_sql(table_name, conn, index=False, if_exists="replace")
                created.append(table_name)

            elif ext in (".xlsx", ".xls"):
                sheets = pd.read_excel(tmp_path, sheet_name=None)
                for sheet_name, df in sheets.items():
                    if df.empty:
                        warnings.append(f"Skipped empty sheet '{sheet_name}' in '{filename}'.")
                        continue
                    base = filename if len(sheets) == 1 else f"{filename}_{sheet_name}"
                    table_name = _dedupe(_sanitize_name(base), used_names)
                    _coerce_dates(df).to_sql(table_name, conn, index=False, if_exists="replace")
                    created.append(table_name)
            else:
                warnings.append(f"Skipped '{filename}': unsupported file type '{ext}'.")
        except Exception as e:
            warnings.append(f"Couldn't read '{filename}': {e}")
        finally:
            os.remove(tmp_path)

    conn.commit()
    conn.close()

    if not created:
        raise DBLoadError(
            "None of the uploaded file(s) could be turned into a table. "
            "Check that CSV/Excel files have a header row and at least one data row."
        )
    return dest, created, warnings


def load_upload(files: list, dest_dir: str) -> tuple[str, list | None, list]:
    """
    Top-level entry point for the UI/CLI.

    files: list of (filename, file_bytes) tuples.
    dest_dir: a writable scratch directory (created if missing) -- the
        resulting SQLite file is written here as 'uploaded.db'.

    Returns (db_path, table_names, warnings):
        table_names is None when an existing SQLite DB was used as-is
        (its tables are discovered later by SchemaRetriever); it's a list
        of created table names when built from CSV/Excel.
    Raises DBLoadError on anything that leaves no usable database.
    """
    os.makedirs(dest_dir, exist_ok=True)
    warnings = []

    sqlite_files, tabular_files = [], []
    for filename, content in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in SQLITE_EXTENSIONS:
            sqlite_files.append((filename, content))
        elif ext in TABULAR_EXTENSIONS:
            tabular_files.append((filename, content))
        else:
            warnings.append(f"Skipped '{filename}': unsupported file type '{ext}'.")

    if sqlite_files:
        if len(sqlite_files) > 1 or tabular_files:
            ignored = [f for f, _ in sqlite_files[1:]] + [f for f, _ in tabular_files]
            warnings.append(
                f"Using SQLite database '{sqlite_files[0][0]}'; ignoring other "
                f"uploaded file(s): {', '.join(ignored)} (mixing a full database "
                "with loose CSV/Excel files is ambiguous)."
            )
        filename, content = sqlite_files[0]
        tmp_path = os.path.join(dest_dir, f"_src{os.path.splitext(filename)[1]}")
        with open(tmp_path, "wb") as f:
            f.write(content)
        try:
            db_path = _load_sqlite(tmp_path, dest_dir)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return db_path, None, warnings

    if tabular_files:
        db_path, created, more_warnings = _build_db_from_tabular(tabular_files, dest_dir)
        return db_path, created, warnings + more_warnings

    raise DBLoadError(
        "No supported files were uploaded. Supported: "
        + ", ".join(SQLITE_EXTENSIONS + TABULAR_EXTENSIONS)
    )
