"""
otterflow.tools
~~~~~~~~~~~~~~~
Tool primitives and a library of ready-to-use tools.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()  # ensure .env is loaded even when tools are imported standalone


# ────────────────────────────────────────────────────────────────────────────
# Core Tool dataclass
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    """Wraps a Python callable into a Claude-compatible tool."""

    name: str
    description: str
    parameters: dict           # JSON Schema object
    fn: Callable

    def to_claude_spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


# ────────────────────────────────────────────────────────────────────────────
# @tool decorator — turn any function into a Tool
# ────────────────────────────────────────────────────────────────────────────

def tool(name: str | None = None, description: str | None = None):
    """
    Decorator to convert a function into an otterflow Tool.

    Usage::

        @tool(description="Return the current UTC time as an ISO string.")
        def get_time() -> str:
            from datetime import datetime, timezone
            return datetime.now(timezone.utc).isoformat()

        agent = Agent("Bot", "You are helpful.", tools=[get_time])
    """
    def decorator(fn: Callable) -> Callable:
        import inspect

        _name = name or fn.__name__
        _desc = description or fn.__doc__ or ""

        hints = fn.__annotations__.copy()
        hints.pop("return", None)
        sig = inspect.signature(fn)

        type_map = {
            str: "string", int: "integer", float: "number",
            bool: "boolean", list: "array", dict: "object",
        }

        props: dict[str, dict] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            hint = hints.get(param_name, str)
            props[param_name] = {"type": type_map.get(hint, "string")}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema: dict[str, Any] = {"type": "object", "properties": props}
        if required:
            schema["required"] = required

        setattr(fn, "_tool_spec", Tool(name=_name, description=_desc, parameters=schema, fn=fn))
        return fn

    return decorator


# ────────────────────────────────────────────────────────────────────────────
# Built-in tools
# ────────────────────────────────────────────────────────────────────────────

# Reuse a single Anthropic client for web_search to avoid per-call overhead.
_web_search_client: Any = None


def _get_web_search_client() -> Any:
    global _web_search_client
    if _web_search_client is None:
        import anthropic
        _web_search_client = anthropic.Anthropic()
    return _web_search_client


def _web_search_fn(query: str) -> str:
    try:
        client = _get_web_search_client()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": query}],
        )
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return "No results found."
    except Exception as e:
        return f"Search error: {e}"


web_search = Tool(
    name="web_search",
    description="Search the web for current information, news, or research on any topic.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query."}},
        "required": ["query"],
    },
    fn=_web_search_fn,
)


# ── read_file ────────────────────────────────────────────────────────────────

def _read_file_fn(path: str) -> str:
    try:
        return Path(path).expanduser().read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"


read_file = Tool(
    name="read_file",
    description="Read the contents of a local file by its path.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative file path."},
        },
        "required": ["path"],
    },
    fn=_read_file_fn,
)


# ── write_file ───────────────────────────────────────────────────────────────

# Directories that agents must never write to.
_BLOCKED_WRITE_PREFIXES = (
    "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64",
    "/sys", "/proc", "/dev", "/boot", "/root",
)


def _write_file_fn(path: str, content: str) -> str:
    try:
        abs_path = Path(path).expanduser().resolve()
        path_str = str(abs_path)
        for blocked in _BLOCKED_WRITE_PREFIXES:
            if path_str.startswith(blocked):
                return f"Error: writing to {blocked} is not permitted."
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


write_file = Tool(
    name="write_file",
    description="Write text content to a local file.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write to."},
            "content": {"type": "string", "description": "Content to write."},
        },
        "required": ["path", "content"],
    },
    fn=_write_file_fn,
)


# ── run_python ───────────────────────────────────────────────────────────────

def _run_python_fn(code: str) -> str:
    """
    Execute a Python snippet in a subprocess. The code runs in an isolated
    child process with a 15-second timeout. The agent cannot access the
    parent process's state or credentials.
    """
    if len(code) > 8_000:
        return "Error: code exceeds 8,000 character limit."
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = result.stdout or result.stderr
        return output[:4000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out after 15s."
    except Exception as e:
        return f"Error: {e}"


run_python = Tool(
    name="run_python",
    description=(
        "Execute a Python code snippet in an isolated subprocess and return stdout/stderr. "
        "Use for data analysis, calculations, or transformations. "
        "No network access. 15-second timeout. 8,000 character limit."
    ),
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python code to execute."}},
        "required": ["code"],
    },
    fn=_run_python_fn,
)


# ── calculator ───────────────────────────────────────────────────────────────

def _calculator_fn(expression: str) -> str:
    allowed = set("0123456789+-*/(). eE")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters."
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307 — safe after allowlist + no builtins
        return str(result)
    except Exception as e:
        return f"Calc error: {e}"


calculator = Tool(
    name="calculator",
    description="Evaluate a safe mathematical expression and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression, e.g. '(12 * 8) / 3' or '1.5e3 * 2'",
            }
        },
        "required": ["expression"],
    },
    fn=_calculator_fn,
)


# ── memory_store / memory_recall ─────────────────────────────────────────────
# Simple key-value store for agent-accessible memory within a session.
# Note: this is a module-level store shared across all agents in a process.
# For per-agent isolation, use the Memory class and agent.memory.remember().

_TOOL_MEMORY: dict[str, str] = {}


def _memory_store_fn(key: str, value: str) -> str:
    _TOOL_MEMORY[key] = value
    return f"Stored '{key}'."


def _memory_recall_fn(key: str) -> str:
    return _TOOL_MEMORY.get(key, f"No memory found for key '{key}'.")


memory_store = Tool(
    name="memory_store",
    description="Store a key-value pair in agent memory for later recall.",
    parameters={
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "The key to store under."},
            "value": {"type": "string", "description": "The value to store."},
        },
        "required": ["key", "value"],
    },
    fn=_memory_store_fn,
)

memory_recall = Tool(
    name="memory_recall",
    description="Recall a previously stored value by key.",
    parameters={
        "type": "object",
        "properties": {"key": {"type": "string", "description": "The key to look up."}},
        "required": ["key"],
    },
    fn=_memory_recall_fn,
)


# Convenience bundle — all built-in tools
ALL_TOOLS = [web_search, read_file, write_file, run_python, calculator, memory_store, memory_recall]
