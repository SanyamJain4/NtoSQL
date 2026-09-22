"""
Orchestrates the full pipeline: retrieve schema -> build prompt -> call LLM
-> validate -> execute -> on failure, feed the error back to the LLM and
retry (bounded). This "execution feedback" loop is one of the highest-
leverage accuracy improvements in NL2SQL systems.
"""
import sqlite3
import pandas as pd
from dataclasses import dataclass, field

from .schema_retriever import SchemaRetriever
from .prompt_builder import build_prompt
from .validator import validate
from .llm_client import LLMClient


@dataclass
class NL2SQLResult:
    question: str
    sql: str = ""
    dataframe: pd.DataFrame = None
    success: bool = False
    attempts: int = 0
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    tables_used: list = field(default_factory=list)


class NL2SQLEngine:
    def __init__(self, db_path: str, llm_client: LLMClient | None = None, max_retries: int = 2):
        self.db_path = db_path
        self.llm = llm_client or LLMClient()
        self.retriever = SchemaRetriever(db_path)
        self.max_retries = max_retries
        # Auto-detect which prompt mode to use: the bundled sales.db is
        # recognized by its `sales_flat` view, so it keeps the original
        # hand-tuned rules/examples. ANY other database -- including ones
        # uploaded through the UI -- automatically falls back to the
        # generic, schema-derived prompt instead. No manual config needed
        # to point this system at a new database.
        self.prompt_mode = "sales" if "sales_flat" in self.retriever.tables else "generic"

    def _execute(self, sql: str) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(sql, conn)
        finally:
            conn.close()
        return df

    def ask(self, question: str, top_k_tables: int = 4) -> NL2SQLResult:
        table_infos = self.retriever.retrieve(question, top_k=top_k_tables)
        system_prompt, user_prompt = build_prompt(question, table_infos, mode=self.prompt_mode)

        result = NL2SQLResult(question=question, tables_used=[t.name for t in table_infos])
        last_error_context = ""

        for attempt in range(1, self.max_retries + 2):  # first try + retries
            result.attempts = attempt
            prompt = user_prompt if not last_error_context else (
                f"{user_prompt}\n\n"
                f"Your previous attempt failed with this error:\n{last_error_context}\n"
                f"Fix the query and output ONLY the corrected SQL."
            )
            raw_sql = self.llm.complete(system_prompt, prompt)
            validation = validate(raw_sql, table_infos)
            result.warnings.extend(validation.warnings)

            if not validation.is_valid:
                result.errors.extend(validation.errors)
                last_error_context = "; ".join(validation.errors)

                # Self-healing: if the failure looks like a retrieval miss (the
                # model correctly reached for a real table our top-K search
                # didn't surface), widen to the full schema for the retry
                # instead of just repeating the same limited context.
                if any("not in the retrieved schema" in e for e in validation.errors):
                    all_tables = list(self.retriever.tables.values())
                    if len(all_tables) > len(table_infos):
                        table_infos = all_tables
                        result.tables_used = [t.name for t in table_infos]
                        system_prompt, user_prompt = build_prompt(question, table_infos, mode=self.prompt_mode)
                continue

            try:
                df = self._execute(validation.sanitized_sql)
            except Exception as e:
                result.errors.append(f"Execution error: {e}")
                last_error_context = str(e)
                continue

            result.sql = validation.sanitized_sql
            result.dataframe = df
            result.success = True
            return result

        return result  # exhausted retries without success
