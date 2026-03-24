"""
NL2SQL environment for AgentEvolver Self-Questioning.

Wraps a DuckDB / SQLite database as an interactive exploration environment.
The agent can inspect schemas, sample data, validate SQL, and execute queries.
Task Manager uses this to explore databases and synthesize NL2SQL training tasks.

Configuration via environment variables:
    NL2SQL_DATA_PATH  – path to JSON manifest (list of dicts). Each row needs db_id;
                        question_id optional (defaults to row_N). SQL from sql, SQL, or query.
    NL2SQL_DB_DIR     – root directory containing database files (see NL2SQL_SPIDER_PATH_STYLE)
    NL2SQL_DB_TYPE    – "duckdb" (default) or "sqlite"
    NL2SQL_SPIDER_PATH_STYLE – for SQLite + spider rows: "auto" (default) picks
                        database/{db_id}/{db_id}.sqlite if that file exists, else legacy layout;
                        "official" always uses the standard Spider release layout;
                        "legacy" always uses Spider_sqlite/spider_sqlite/spider_{db_id}.sqlite.
    NL2SQL_DEFAULT_DATASET_TYPE – if a JSON row has no dataset_type, treat as this value
                        (e.g. "spider" when using raw dev.json + official DB layout).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from env_service.base import BaseEnv
from env_service.registry import Registry

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Task-level data loading (class-level cache shared across instances)
# ---------------------------------------------------------------------------

_task_map: Dict[str, dict] | None = None


def _extract_gt_sql(item: dict) -> str:
    """Ground-truth SQL string from common NL2SQL JSON shapes."""
    s = item.get("sql")
    if isinstance(s, str) and s.strip():
        return s
    s = item.get("SQL")
    if isinstance(s, str) and s.strip():
        return s
    q = item.get("query")
    if isinstance(q, str) and q.strip():
        return q
    return ""


def _load_task_map() -> Dict[str, dict]:
    """Load once from NL2SQL_DATA_PATH and cache."""
    global _task_map
    if _task_map is not None:
        return _task_map

    data_path = os.environ.get("NL2SQL_DATA_PATH", "")
    db_dir = os.environ.get("NL2SQL_DB_DIR", "")
    db_type = os.environ.get("NL2SQL_DB_TYPE", "duckdb")

    if not data_path or not Path(data_path).is_file():
        _task_map = {}
        return _task_map

    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)

    default_ds = os.environ.get("NL2SQL_DEFAULT_DATASET_TYPE", "").strip().lower()

    _task_map = {}
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        qid_raw = item.get("question_id")
        if qid_raw is None or qid_raw == "":
            qid = f"row_{idx}"
        else:
            qid = str(qid_raw)
        db_id = item.get("db_id", "")
        if not db_id:
            continue
        ds = (item.get("dataset_type") or default_ds or "").strip().lower()
        _task_map[qid] = {
            "question": item.get("question", ""),
            "evidence": item.get("evidence", ""),
            "gt_sql": _extract_gt_sql(item),
            "db_id": db_id,
            "db_path": _resolve_db_path(db_dir, db_id, ds, db_type),
            "db_type": db_type,
        }

    return _task_map


def _resolve_db_path(db_dir: str, db_id: str, dataset_type: str, db_type: str) -> str:
    root = Path(db_dir)
    spider_style = os.environ.get("NL2SQL_SPIDER_PATH_STYLE", "auto").strip().lower()
    if dataset_type == "spider" and db_type == "sqlite":
        official_p = root / "database" / db_id / f"{db_id}.sqlite"
        legacy_p = root / "Spider_sqlite" / "spider_sqlite" / f"spider_{db_id}.sqlite"
        if spider_style in ("official", "canonical", "spider_data"):
            return str(official_p)
        if spider_style == "legacy":
            return str(legacy_p)
        # auto: prefer standard Spider release tree when present
        if official_p.is_file():
            return str(official_p)
        return str(legacy_p)
    suffix = "db" if db_type == "duckdb" else "sqlite"
    return str(root / f"bird_{db_id}.{suffix}")


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------

_FUNC_CALL_RE = re.compile(
    r"""(get_table_metadata|get_sample_rows|is_sql_executable|test_sql)\s*\(""",
    re.DOTALL,
)


def _parse_tool_calls(content: str) -> list[tuple[str, dict]]:
    """
    Extract tool calls from assistant content.

    Supports two formats:
      1. Python code blocks: ```python\nget_table_metadata("t")\n```
      2. Inline function calls: get_table_metadata("t")

    Returns list of (tool_name, kwargs) tuples.
    """
    code_blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", content, re.DOTALL)
    text = "\n".join(code_blocks) if code_blocks else content

    calls: list[tuple[str, dict]] = []
    for m in _FUNC_CALL_RE.finditer(text):
        func_name = m.group(1)
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        args_str = text[start : i - 1].strip()
        kwargs = _parse_args(func_name, args_str)
        calls.append((func_name, kwargs))

    return calls


def _parse_args(func_name: str, args_str: str) -> dict:
    """Best-effort argument parsing for tool calls."""
    if not args_str:
        return {}
    try:
        val = eval(f"dict(__args=[{args_str}])")  # noqa: S307
        positional = val.get("__args", [])
    except Exception:
        positional = [args_str.strip("'\"")]

    if func_name == "get_table_metadata":
        return {"table_name": str(positional[0]) if positional else ""}
    if func_name == "get_sample_rows":
        return {
            "table_name": str(positional[0]) if positional else "",
            "n": int(positional[1]) if len(positional) > 1 else 5,
        }
    if func_name == "is_sql_executable":
        return {"sql": str(positional[0]) if positional else ""}
    if func_name == "test_sql":
        return {
            "sql": str(positional[0]) if positional else "",
            "n": int(positional[1]) if len(positional) > 1 else 5,
        }
    return {}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _connect(db_path: str, db_type: str):
    if db_type == "sqlite":
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False, timeout=30)
        conn.execute("PRAGMA query_only = ON;")
        return conn
    if duckdb is None:
        raise ImportError("duckdb is required for db_type='duckdb'. pip install duckdb")
    return duckdb.connect(db_path, read_only=True)


def _list_tables(conn, db_type: str) -> list[str]:
    if db_type == "sqlite":
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
    else:
        rows = conn.execute("SHOW TABLES").fetchall()
    return [r[0] for r in rows]


def _get_table_columns(conn, table_name: str, db_type: str) -> list[dict]:
    rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    return [{"name": r[1], "type": r[2] or "TEXT", "notnull": bool(r[3]), "pk": bool(r[5])} for r in rows]


def _safe_execute(conn, sql: str, db_type: str, limit: int = 20) -> tuple[list[str], list[tuple]]:
    """Execute a read-only SELECT/WITH and return (columns, rows)."""
    s = sql.strip().rstrip(";")
    head = s.lstrip().upper()
    if not (head.startswith("SELECT") or head.startswith("WITH")):
        raise ValueError("Only SELECT / WITH queries are allowed.")
    q = f"SELECT * FROM ({s}) AS __sub LIMIT {limit}"
    cur = conn.execute(q)
    cols = [d[0] for d in cur.description] if cur.description else []
    return cols, cur.fetchall()


def _md_table(cols: list[str], rows: list[tuple]) -> str:
    if not cols:
        return "**No columns.**"
    if not rows:
        return "**No rows returned.**"

    def esc(v):
        return str(v).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(esc(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join("| " + " | ".join(esc(v) if v is not None else "NULL" for v in r) + " |" for r in rows)
    return f"{header}\n{sep}\n{body}"


# ---------------------------------------------------------------------------
# NL2SQL Environment
# ---------------------------------------------------------------------------

TOOL_DESCRIPTION = """\
You have the following tools to explore this database. Call them using Python-style syntax inside a code block.

**Available Tools:**

1. `get_table_metadata(table_name)` — Get column names, types, and primary keys for a table.
2. `get_sample_rows(table_name, n=5)` — Fetch N sample rows from a table.
3. `is_sql_executable(sql)` — Check if a SQL query is syntactically valid.
4. `test_sql(sql, n=5)` — Execute a SQL query and see up to N result rows.

**Example usage:**

```python
get_table_metadata("employees")
```

```python
test_sql("SELECT name, salary FROM employees WHERE department = 'Engineering' ORDER BY salary DESC", 10)
```

Explore the database schema, understand the data, and try various SQL queries.\
"""


@Registry.register("nl2sql")
class Nl2sqlEnv(BaseEnv):

    def __init__(self, task_id: str = None, instance_id: str = None, params: Dict = None):
        self.task_id = task_id
        self.instance_id = instance_id
        self.params = params or {}
        self._conn = None
        self._db_path: str = ""
        self._db_type: str = "duckdb"
        self._gt_sql: str = ""
        self._tables: list[str] = []

    def get_init_state(self, params: Dict = None) -> Dict[str, Any]:
        params = params or {}
        task_info = _load_task_map().get(str(self.task_id), {})

        self._db_path = task_info.get("db_path", params.get("db_path", ""))
        self._db_type = task_info.get("db_type", params.get("db_type", "duckdb"))
        self._gt_sql = task_info.get("gt_sql", params.get("gt_sql", ""))

        if not self._db_path or not Path(self._db_path).exists():
            raise FileNotFoundError(f"Database not found: {self._db_path} (task_id={self.task_id})")

        self._conn = _connect(self._db_path, self._db_type)
        self._tables = _list_tables(self._conn, self._db_type)

        db_desc = self._build_db_description()
        system_content = f"{db_desc}\n\n{TOOL_DESCRIPTION}"

        return {
            "state": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Explore this database."},
            ],
            "info": {"instance_id": self.instance_id, "task_id": self.task_id},
        }

    def step(self, action: Dict, params: Dict = None) -> Dict[str, Any]:
        content = action.get("content", "")
        calls = _parse_tool_calls(content)

        if not calls:
            return {
                "state": [{"role": "user", "content": "No tool call detected. Please use one of the available tools."}],
                "reward": 0.0,
                "is_terminated": False,
                "info": {},
            }

        results: list[str] = []
        for func_name, kwargs in calls:
            try:
                result = self._dispatch_tool(func_name, kwargs)
            except Exception as e:
                result = f"**Error in {func_name}:** {e}"
            results.append(f"### {func_name}\n{result}")

        observation = "\n\n".join(results)
        return {
            "state": [{"role": "user", "content": observation}],
            "reward": 0.0,
            "is_terminated": False,
            "info": {},
        }

    def evaluate(self, messages: Dict = None, params: Dict = None) -> float:
        return 0.0

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def get_info(self, messages: Dict = None, params: Dict = None) -> Dict:
        return {
            "db_path": self._db_path,
            "db_type": self._db_type,
            "tables": self._tables,
        }

    @staticmethod
    def get_query_list(split: str = "train", params: Dict = None) -> List[str]:
        task_map = _load_task_map()
        keys = sorted(task_map.keys(), key=lambda x: (len(str(x)), str(x)))
        cap = os.environ.get("NL2SQL_MAX_SEED_TASKS", "").strip()
        if params and params.get("max_seed_tasks") not in (None, ""):
            cap = str(params["max_seed_tasks"]).strip()
        if cap:
            try:
                n = int(cap)
                if n > 0:
                    keys = keys[:n]
            except ValueError:
                pass
        return keys

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_db_description(self) -> str:
        lines = ["### Database Schema\n", f"Database: `{Path(self._db_path).stem}`\n"]
        for tbl in self._tables:
            cols = _get_table_columns(self._conn, tbl, self._db_type)
            pk_cols = [c["name"] for c in cols if c["pk"]]
            col_strs = [f"  - `{c['name']}` ({c['type']})" + (" [PK]" if c["pk"] else "") for c in cols]
            lines.append(f"**{tbl}**" + (f"  (PK: {', '.join(pk_cols)})" if pk_cols else ""))
            lines.extend(col_strs)
            lines.append("")
        return "\n".join(lines)

    def _dispatch_tool(self, func_name: str, kwargs: dict) -> str:
        if func_name == "get_table_metadata":
            return self._tool_get_table_metadata(**kwargs)
        if func_name == "get_sample_rows":
            return self._tool_get_sample_rows(**kwargs)
        if func_name == "is_sql_executable":
            return self._tool_is_sql_executable(**kwargs)
        if func_name == "test_sql":
            return self._tool_test_sql(**kwargs)
        return f"Unknown tool: {func_name}"

    def _tool_get_table_metadata(self, table_name: str) -> str:
        if table_name not in self._tables:
            return f"**Error:** Table `{table_name}` not found. Available: {', '.join(self._tables)}"
        cols = _get_table_columns(self._conn, table_name, self._db_type)
        lines = [f"**Table: {table_name}** ({len(cols)} columns)\n"]
        lines.append("| Column | Type | PK | Not Null |")
        lines.append("| --- | --- | --- | --- |")
        for c in cols:
            lines.append(f"| {c['name']} | {c['type']} | {'Yes' if c['pk'] else ''} | {'Yes' if c['notnull'] else ''} |")
        return "\n".join(lines)

    def _tool_get_sample_rows(self, table_name: str, n: int = 5) -> str:
        if table_name not in self._tables:
            return f"**Error:** Table `{table_name}` not found. Available: {', '.join(self._tables)}"
        n = max(1, min(n, 50))
        safe_name = f'"{table_name}"'
        cur = self._conn.execute(f"SELECT * FROM {safe_name} LIMIT {n}")
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return _md_table(cols, rows)

    def _tool_is_sql_executable(self, sql: str) -> str:
        s = sql.strip().rstrip(";")
        if not s:
            return "**Error:** Empty SQL"
        try:
            explain = "EXPLAIN QUERY PLAN" if self._db_type == "sqlite" else "EXPLAIN"
            self._conn.execute(f"{explain} {s}")
            return "OK — query is executable."
        except Exception as e:
            return f"**Error:** {e}"

    def _tool_test_sql(self, sql: str, n: int = 5) -> str:
        n = max(1, min(n, 50))
        try:
            cols, rows = _safe_execute(self._conn, sql, self._db_type, limit=n)
        except Exception as e:
            return f"**Error:** {e}"
        return _md_table(cols, rows)
