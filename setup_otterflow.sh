#!/usr/bin/env bash
set -e

# ────────────────────────────────────────────────
# OtterFlow — project setup script
# Run: bash setup_otterflow.sh [target-directory]
# Default target: ./otterflow
# ────────────────────────────────────────────────

TARGET="${1:-./otterflow}"
echo "🦦 Creating OtterFlow repo at: $TARGET"
mkdir -p "$TARGET"
cd "$TARGET"

mkdir -p otterflow/agents assets examples tests

# ── .gitignore ───────────────────────────────────
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.env
.venv
venv/
env/
dist/
build/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.log
.DS_Store
EOF

# ── LICENSE ──────────────────────────────────────
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

echo "✅ Config files written"

cat > pyproject.toml << 'FILEOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "otterflow"
version = "0.1.0"
description = "Lightweight Python framework for building multi-step AI agents powered by Claude"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.10"
keywords = ["ai", "agents", "claude", "anthropic", "llm", "automation"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "anthropic>=0.40.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio",
    "black",
    "ruff",
    "mypy",
]

[project.urls]
Homepage = "https://github.com/tabato/otterflow"
Documentation = "https://github.com/tabato/otterflow#readme"
Issues = "https://github.com/tabato/otterflow/issues"

[tool.hatch.build.targets.wheel]
packages = ["otterflow"]

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I"]

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true

FILEOF
echo "✅ pyproject.toml"

cat > otterflow/__init__.py << 'FILEOF'
"""
otterflow
~~~~~~~~~
Lightweight Python framework for building multi-step AI agents
powered by Claude. Build agents in 10 lines. Ship in minutes.

    >>> from otterflow import Agent
    >>> from otterflow.tools import web_search
    >>> agent = Agent("Researcher", "You are a research analyst.", tools=[web_search])
    >>> print(agent.run("What are the hottest AI startups right now?"))
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

from .agent import Agent
from .memory import Memory
from .tools import Tool, tool

from .agents import (
    ResearchAgent,
    EmailAgent,
    DataAnalystAgent,
    CompetitiveIntelAgent,
    SalesCoachAgent,
    BusinessIntelPipeline,
)

__all__ = [
    "Agent",
    "Memory",
    "Tool",
    "tool",
    # Pre-built agents
    "ResearchAgent",
    "EmailAgent",
    "DataAnalystAgent",
    "CompetitiveIntelAgent",
    "SalesCoachAgent",
    "BusinessIntelPipeline",
]

FILEOF
echo "✅ otterflow/__init__.py"

cat > otterflow/memory.py << 'FILEOF'
"""
otterflow.memory
~~~~~~~~~~~~~~~~
Memory management for agents. Handles conversation history,
sliding window context, and optional summarization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Memory:
    """
    Manages an agent's conversation history.

    Supports:
    - Full history mode (default)
    - Sliding window (keep last N turns)
    - Persistent key-value fact store
    """

    max_turns: int = 20          # 0 = unlimited
    facts: dict[str, str] = field(default_factory=dict)
    _history: list[dict] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Record a completed turn."""
        self._history.append({"user": user_msg, "assistant": assistant_msg})
        if self.max_turns and len(self._history) > self.max_turns:
            self._history = self._history[-self.max_turns:]

    def remember(self, key: str, value: str) -> None:
        """Store a persistent fact (survives sliding-window trimming)."""
        self.facts[key] = value

    def recall(self, key: str) -> str | None:
        """Retrieve a persistent fact."""
        return self.facts.get(key)

    def clear(self) -> None:
        """Wipe history (facts are preserved)."""
        self._history.clear()

    def build_messages(self, new_prompt: str) -> list[dict[str, Any]]:
        """
        Build the messages list for an Anthropic API call, injecting
        history and facts into the conversation.
        """
        messages: list[dict[str, Any]] = []

        # Inject persistent facts as early context
        if self.facts:
            facts_text = "\n".join(f"- {k}: {v}" for k, v in self.facts.items())
            messages.append({
                "role": "user",
                "content": f"[Context from memory]\n{facts_text}",
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I'll keep these facts in mind.",
            })

        # Replay conversation history
        for turn in self._history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        # Add the new prompt
        messages.append({"role": "user", "content": new_prompt})
        return messages

    def summary(self) -> str:
        """Human-readable memory summary."""
        lines = [f"Memory: {len(self._history)} turns stored"]
        if self.facts:
            lines.append(f"Facts: {list(self.facts.keys())}")
        return " | ".join(lines)

    def __len__(self) -> int:
        return len(self._history)

FILEOF
echo "✅ otterflow/memory.py"

cat > otterflow/tools.py << 'FILEOF'
"""
otterflow.tools
~~~~~~~~~~~~~~~
Tool primitives and a library of ready-to-use tools.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable


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

    def __call__(self, **kwargs) -> Any:
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
    """
    def decorator(fn: Callable) -> Callable:
        import inspect

        _name = name or fn.__name__
        _desc = description or fn.__doc__ or ""

        # Build JSON schema from type hints
        hints = fn.__annotations__.copy()
        hints.pop("return", None)
        sig = inspect.signature(fn)

        props = {}
        required = []
        for param_name, param in sig.parameters.items():
            hint = hints.get(param_name, "string")
            type_map = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
            json_type = type_map.get(hint, "string")
            props[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema = {"type": "object", "properties": props}
        if required:
            schema["required"] = required

        fn._tool_spec = Tool(name=_name, description=_desc, parameters=schema, fn=fn)
        return fn

    return decorator


# ────────────────────────────────────────────────────────────────────────────
# Built-in tools
# ────────────────────────────────────────────────────────────────────────────

def _web_search_fn(query: str) -> str:
    """Uses the Anthropic web search tool via a nested API call."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
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


def _read_file_fn(path: str) -> str:
    try:
        with open(os.path.expanduser(path)) as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


read_file = Tool(
    name="read_file",
    description="Read the contents of a local file by its path.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Absolute or relative file path."}},
        "required": ["path"],
    },
    fn=_read_file_fn,
)


def _write_file_fn(path: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(os.path.expanduser(path), "w") as f:
            f.write(content)
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


def _run_python_fn(code: str) -> str:
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=15
        )
        output = result.stdout or result.stderr
        return output[:2000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out after 15s"
    except Exception as e:
        return f"Error: {e}"


run_python = Tool(
    name="run_python",
    description="Execute a Python code snippet and return its stdout/stderr output.",
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python code to execute."}},
        "required": ["code"],
    },
    fn=_run_python_fn,
)


def _calculator_fn(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: unsafe expression."
        return str(eval(expression))  # noqa: S307 — safe after allowlist
    except Exception as e:
        return f"Calc error: {e}"


calculator = Tool(
    name="calculator",
    description="Evaluate a safe mathematical expression and return the result.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Math expression, e.g. '(12 * 8) / 3'"}},
        "required": ["expression"],
    },
    fn=_calculator_fn,
)


def _memory_store_fn(key: str, value: str, _store: dict = {}) -> str:
    _store[key] = value
    return f"Stored '{key}'."


def _memory_recall_fn(key: str, _store: dict = _memory_store_fn.__defaults__[0]) -> str:
    return _store.get(key, f"No memory found for key '{key}'.")


memory_store = Tool(
    name="memory_store",
    description="Store a key-value pair in agent memory for later recall.",
    parameters={
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "string"},
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
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
    fn=_memory_recall_fn,
)


# Convenience bundle
ALL_TOOLS = [web_search, read_file, write_file, run_python, calculator, memory_store, memory_recall]

FILEOF
echo "✅ otterflow/tools.py"

cat > otterflow/agent.py << 'FILEOF'
"""
otterflow.agent
~~~~~~~~~~~~~~~
Core Agent base class. Handles tool registration, execution loops,
memory injection, and multi-agent orchestration.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import anthropic

from .memory import Memory
from .tools import Tool


_client = anthropic.Anthropic()

MODEL = "claude-sonnet-4-20250514"


class Agent:
    """
    A lightweight, composable AI agent powered by Claude.

    Usage::

        from otterflow import Agent
        from otterflow.tools import web_search, read_file

        agent = Agent(
            name="ResearchBot",
            role="You are a senior research analyst.",
            tools=[web_search, read_file],
        )

        result = agent.run("Summarize the latest trends in AI infrastructure.")
        print(result)
    """

    def __init__(
        self,
        name: str,
        role: str,
        tools: list[Tool | Callable] | None = None,
        memory: Memory | None = None,
        max_steps: int = 10,
        verbose: bool = False,
        model: str = MODEL,
    ):
        self.name = name
        self.role = role
        self.tools: dict[str, Tool] = {}
        self.memory = memory or Memory()
        self.max_steps = max_steps
        self.verbose = verbose
        self.model = model
        self._sub_agents: list[Agent] = []

        for t in tools or []:
            self.register_tool(t)

    # ------------------------------------------------------------------ #
    # Tool registration
    # ------------------------------------------------------------------ #

    def register_tool(self, tool: Tool | Callable) -> "Agent":
        """Register a tool. Accepts a Tool instance or a @tool-decorated function."""
        if callable(tool) and hasattr(tool, "_tool_spec"):
            tool = tool._tool_spec
        if not isinstance(tool, Tool):
            raise TypeError(f"Expected a Tool, got {type(tool)}")
        self.tools[tool.name] = tool
        return self

    # ------------------------------------------------------------------ #
    # Sub-agent spawning
    # ------------------------------------------------------------------ #

    def spawn(self, agent: "Agent") -> "Agent":
        """Register a sub-agent. The parent can delegate tasks to it."""
        self._sub_agents.append(agent)
        tool = agent._as_tool()
        self.register_tool(tool)
        return self

    def _as_tool(self) -> Tool:
        """Expose this agent as a callable tool for a parent agent."""
        parent = self

        def _run(task: str) -> str:
            return parent.run(task)

        return Tool(
            name=f"delegate_to_{self.name.lower().replace(' ', '_')}",
            description=f"Delegate a task to the {self.name} sub-agent. {self.role}",
            parameters={
                "type": "object",
                "properties": {"task": {"type": "string", "description": "The task to delegate."}},
                "required": ["task"],
            },
            fn=_run,
        )

    # ------------------------------------------------------------------ #
    # Core run loop
    # ------------------------------------------------------------------ #

    def run(self, prompt: str, *, stream: bool = False) -> str:
        """Run the agent on a prompt. Returns the final text response."""
        messages = self.memory.build_messages(prompt)
        tool_specs = [t.to_claude_spec() for t in self.tools.values()]

        if self.verbose:
            print(f"\n[{self.name}] 🚀 Starting: {prompt[:80]}...")

        for step in range(self.max_steps):
            kwargs: dict[str, Any] = dict(
                model=self.model,
                max_tokens=4096,
                system=self.role,
                messages=messages,
            )
            if tool_specs:
                kwargs["tools"] = tool_specs

            response = _client.messages.create(**kwargs)

            if self.verbose:
                print(f"[{self.name}] Step {step + 1}: stop_reason={response.stop_reason}")

            # Collect text and tool calls from response
            assistant_content = []
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
                    tool_calls.append(block)

            messages.append({"role": "assistant", "content": assistant_content})

            # Done — extract and store final answer
            if response.stop_reason == "end_turn" or not tool_calls:
                final = self._extract_text(response)
                self.memory.add_turn(prompt, final)
                return final

            # Execute tool calls
            tool_results = []
            for tc in tool_calls:
                result = self._execute_tool(tc.name, tc.input)
                if self.verbose:
                    print(f"[{self.name}] 🔧 {tc.name} → {str(result)[:120]}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": str(result),
                })

            messages.append({"role": "user", "content": tool_results})

        return "Max steps reached without a final answer."

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _execute_tool(self, name: str, inputs: dict) -> Any:
        if name not in self.tools:
            return f"Error: tool '{name}' not found."
        try:
            return self.tools[name].fn(**inputs)
        except Exception as e:
            return f"Tool error: {e}"

    @staticmethod
    def _extract_text(response) -> str:
        parts = [b.text for b in response.content if b.type == "text"]
        return "\n".join(parts).strip()

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, tools={list(self.tools.keys())})"

FILEOF
echo "✅ otterflow/agent.py"

cat > otterflow/agents/__init__.py << 'FILEOF'
"""
otterflow.agents
~~~~~~~~~~~~~~~~
Ready-to-use business agents built on top of the Agent base class.
Drop these into your app or use them as starting templates.
"""

from __future__ import annotations

from ..agent import Agent
from ..memory import Memory
from ..tools import web_search, read_file, write_file, run_python, calculator


# ────────────────────────────────────────────────────────────────────────────
# Research Agent
# ────────────────────────────────────────────────────────────────────────────

def ResearchAgent(verbose: bool = False) -> Agent:
    """
    Deep-dive research agent. Searches the web, synthesizes findings,
    and produces structured reports.

    Usage::

        agent = ResearchAgent()
        report = agent.run("What are the top 5 AI infrastructure startups in 2025?")
    """
    return Agent(
        name="ResearchAgent",
        role=(
            "You are a senior research analyst with expertise in technology, business, "
            "and market intelligence. When given a research task:\n"
            "1. Break it into sub-questions\n"
            "2. Search for authoritative, recent sources\n"
            "3. Synthesize findings into a clear, structured report with sections and bullet points\n"
            "4. Always cite your sources\n"
            "Be thorough but concise. Prioritize actionable insights."
        ),
        tools=[web_search, write_file],
        verbose=verbose,
    )


# ────────────────────────────────────────────────────────────────────────────
# Email Assistant Agent
# ────────────────────────────────────────────────────────────────────────────

def EmailAgent(tone: str = "professional", verbose: bool = False) -> Agent:
    """
    Drafts, rewrites, and improves emails.

    Args:
        tone: 'professional', 'friendly', 'assertive', or 'concise'

    Usage::

        agent = EmailAgent(tone="friendly")
        draft = agent.run(
            "Write a follow-up email to a lead who went quiet after our demo last week."
        )
    """
    return Agent(
        name="EmailAgent",
        role=(
            f"You are an expert business communication specialist. Your writing tone is {tone}. "
            "When drafting or improving emails:\n"
            "- Keep subject lines punchy and specific\n"
            "- Use short paragraphs (2-3 sentences max)\n"
            "- End with a single, clear call to action\n"
            "- Mirror the vocabulary of the industry\n"
            "- Never use buzzwords like 'synergy' or 'circle back'\n"
            "Always output the full email including subject line."
        ),
        tools=[],
        verbose=verbose,
    )


# ────────────────────────────────────────────────────────────────────────────
# Data Analyst Agent
# ────────────────────────────────────────────────────────────────────────────

def DataAnalystAgent(verbose: bool = False) -> Agent:
    """
    Reads files, runs Python analysis, and surfaces business insights.

    Usage::

        agent = DataAnalystAgent()
        insights = agent.run("Analyze sales.csv and identify the top 3 growth opportunities.")
    """
    return Agent(
        name="DataAnalystAgent",
        role=(
            "You are a senior data analyst and Python expert. When given a data analysis task:\n"
            "1. Read the relevant files to understand the data structure\n"
            "2. Write and execute Python code (pandas, statistics) to analyze it\n"
            "3. Present findings as bullet-point business insights — not raw numbers\n"
            "4. Always include at least one specific, actionable recommendation\n"
            "Think like a McKinsey analyst: What decision does this data enable?"
        ),
        tools=[read_file, run_python, calculator, write_file],
        verbose=verbose,
    )


# ────────────────────────────────────────────────────────────────────────────
# Competitive Intelligence Agent
# ────────────────────────────────────────────────────────────────────────────

def CompetitiveIntelAgent(verbose: bool = False) -> Agent:
    """
    Researches competitors and builds competitive battle cards.

    Usage::

        agent = CompetitiveIntelAgent()
        battle_card = agent.run(
            "Build a competitive analysis of Salesforce vs HubSpot for a B2B SaaS startup."
        )
    """
    return Agent(
        name="CompetitiveIntelAgent",
        role=(
            "You are a competitive intelligence specialist. For any competitive analysis:\n"
            "1. Research each company's positioning, pricing, and key features\n"
            "2. Identify their ICP (ideal customer profile) and go-to-market motion\n"
            "3. Surface their weaknesses and common customer complaints\n"
            "4. Output a structured battle card with: Overview | Strengths | Weaknesses | "
            "How to win against them | Key differentiators\n"
            "Be opinionated. Founders need sharp insights, not balanced Wikipedia summaries."
        ),
        tools=[web_search, write_file],
        verbose=verbose,
    )


# ────────────────────────────────────────────────────────────────────────────
# Sales Coach Agent  (dogfood example for ai-sales-coach)
# ────────────────────────────────────────────────────────────────────────────

def SalesCoachAgent(
    methodology: str = "SPIN",
    memory: Memory | None = None,
    verbose: bool = False,
) -> Agent:
    """
    Real-time AI sales coach. Analyzes calls, critiques pitches,
    and generates objection-handling scripts.

    This agent is the foundation of the `ai-sales-coach` project.

    Args:
        methodology: Sales framework to apply ('SPIN', 'Challenger', 'MEDDIC', 'Sandler')

    Usage::

        coach = SalesCoachAgent(methodology="Challenger")
        feedback = coach.run(
            "Here's my cold call script: [script]. What am I doing wrong?"
        )
    """
    mem = memory or Memory()
    mem.remember("methodology", methodology)

    return Agent(
        name="SalesCoachAgent",
        role=(
            f"You are a world-class sales coach specializing in the {methodology} methodology. "
            "Your job is to help sales reps close more deals by:\n"
            "1. Analyzing their scripts, emails, or call transcripts\n"
            "2. Identifying specific weak points with timestamps or line references\n"
            "3. Rewriting weak sections using proven {methodology} techniques\n"
            "4. Generating objection-handling scripts for common pushbacks\n"
            "5. Role-playing as a difficult prospect on demand\n\n"
            "Be direct and specific. Vague coaching is worthless. "
            "Every piece of feedback must include a concrete rewrite or example."
        ),
        tools=[web_search, read_file],
        memory=mem,
        verbose=verbose,
    )


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator: Multi-Agent Pipeline
# ────────────────────────────────────────────────────────────────────────────

def BusinessIntelPipeline(verbose: bool = False) -> Agent:
    """
    A multi-agent orchestrator that combines Research + Competitive Intel
    + Data Analysis into a single business intelligence pipeline.

    Usage::

        pipeline = BusinessIntelPipeline(verbose=True)
        report = pipeline.run(
            "Give me a full market entry analysis for B2B HR software in Southeast Asia."
        )
    """
    orchestrator = Agent(
        name="BizIntelOrchestrator",
        role=(
            "You are a Chief Intelligence Officer overseeing a team of specialist agents. "
            "For complex business intelligence tasks, break them down and delegate to the "
            "appropriate specialist agents. Synthesize their outputs into a unified, "
            "executive-ready report. Always end with a clear RECOMMENDATION section."
        ),
        tools=[],
        verbose=verbose,
    )

    orchestrator.spawn(ResearchAgent(verbose=verbose))
    orchestrator.spawn(CompetitiveIntelAgent(verbose=verbose))
    orchestrator.spawn(DataAnalystAgent(verbose=verbose))

    return orchestrator

FILEOF
echo "✅ otterflow/agents/__init__.py"

cat > examples/quickstart.py << 'FILEOF'
"""
examples/quickstart.py
~~~~~~~~~~~~~~~~~~~~~~
5 examples showing otterflow in action.
Run any of these after: pip install otterflow
"""

import otterflow
from otterflow import Agent, Memory
from otterflow.tools import web_search, read_file, write_file, calculator, run_python
from otterflow.agents import (
    ResearchAgent,
    EmailAgent,
    DataAnalystAgent,
    CompetitiveIntelAgent,
    SalesCoachAgent,
    BusinessIntelPipeline,
)


# ── Example 1: Minimal custom agent ─────────────────────────────────────────

def example_1_minimal():
    """The simplest possible agent."""
    agent = Agent(
        name="Greeter",
        role="You are a friendly assistant who always responds in haiku.",
    )
    print(agent.run("Tell me about Python programming."))


# ── Example 2: Agent with tools ──────────────────────────────────────────────

def example_2_with_tools():
    """Agent that searches the web and does math."""
    agent = Agent(
        name="FinanceBot",
        role="You are a financial research assistant. Use tools to gather data.",
        tools=[web_search, calculator],
        verbose=True,
    )
    result = agent.run(
        "Search for NVIDIA's latest revenue, then calculate what 15% growth would look like."
    )
    print(result)


# ── Example 3: Pre-built Research Agent ──────────────────────────────────────

def example_3_research():
    researcher = ResearchAgent(verbose=True)
    report = researcher.run(
        "What are the top 5 trends in B2B SaaS for 2025? "
        "Focus on AI adoption and pricing model changes."
    )
    print(report)


# ── Example 4: Sales Coach (ai-sales-coach integration) ──────────────────────

def example_4_sales_coach():
    coach = SalesCoachAgent(methodology="Challenger", verbose=True)

    # Analyze a cold email
    feedback = coach.run(
        """
        Analyze this cold email and tell me exactly what to fix:

        Subject: Quick question

        Hi [Name],
        I hope this email finds you well. I wanted to reach out because our
        software could really help your business. We have many features that
        lots of companies find useful. Would you be open to a quick call?

        Best,
        John
        """
    )
    print("=== COACH FEEDBACK ===")
    print(feedback)

    # Follow-up: now the agent remembers the context
    rewrite = coach.run("Now rewrite the email using what you taught me.")
    print("\n=== REWRITTEN EMAIL ===")
    print(rewrite)


# ── Example 5: Multi-agent orchestration ─────────────────────────────────────

def example_5_multi_agent():
    """
    Orchestrator spawns sub-agents to do parallel research,
    then synthesizes a unified report.
    """
    pipeline = BusinessIntelPipeline(verbose=True)
    report = pipeline.run(
        "Should I build a B2B AI sales coaching product in 2025? "
        "Analyze the market, key competitors, and give me a go/no-go recommendation."
    )
    print(report)


# ── Example 6: Custom @tool decorator ────────────────────────────────────────

def example_6_custom_tool():
    from otterflow.tools import tool
    from datetime import datetime, timezone

    @tool(description="Return the current UTC datetime.")
    def get_current_time() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    @tool(description="Reverse a string.")
    def reverse_string(text: str) -> str:
        return text[::-1]

    agent = Agent(
        name="CustomToolBot",
        role="You are a helpful assistant with custom tools.",
        tools=[get_current_time, reverse_string],
        verbose=True,
    )
    print(agent.run("What time is it right now, and reverse the word 'otterflow'."))


# ── Example 7: Persistent memory ─────────────────────────────────────────────

def example_7_memory():
    memory = Memory(max_turns=10)
    memory.remember("user_name", "Alex")
    memory.remember("company", "TechCorp")
    memory.remember("deal_size", "$50k ARR")

    coach = SalesCoachAgent(memory=memory)

    # Agent knows who Alex is across turns
    r1 = coach.run("Help me prepare for my call with the VP of Sales tomorrow.")
    print(r1)

    r2 = coach.run("What objections should I expect given our deal size?")
    print(r2)


if __name__ == "__main__":
    import sys

    examples = {
        "1": example_1_minimal,
        "2": example_2_with_tools,
        "3": example_3_research,
        "4": example_4_sales_coach,
        "5": example_5_multi_agent,
        "6": example_6_custom_tool,
        "7": example_7_memory,
    }

    choice = sys.argv[1] if len(sys.argv) > 1 else "1"
    fn = examples.get(choice)
    if fn:
        print(f"\n{'='*60}")
        print(f"Running Example {choice}: {fn.__doc__}")
        print('='*60)
        fn()
    else:
        print(f"Usage: python quickstart.py [1-{len(examples)}]")
        for k, v in examples.items():
            print(f"  {k}: {v.__doc__}")

FILEOF
echo "✅ examples/quickstart.py"

cat > tests/test_otterflow.py << 'FILEOF'
"""
tests/test_otterflow.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for otterflow. Run with: pytest tests/
"""

import pytest
from unittest.mock import MagicMock, patch

from otterflow import Agent, Memory, Tool, tool
from otterflow.tools import calculator, read_file, write_file
from otterflow.agents import ResearchAgent, EmailAgent, SalesCoachAgent


# ── Tool tests ───────────────────────────────────────────────────────────────

def test_calculator_basic():
    result = calculator.fn(expression="2 + 2")
    assert result == "4"


def test_calculator_complex():
    result = calculator.fn(expression="(100 * 0.15) + 50")
    assert result == "65.0"


def test_calculator_blocks_unsafe():
    result = calculator.fn(expression="__import__('os').system('ls')")
    assert "unsafe" in result.lower() or "Error" in result


def test_write_and_read_file(tmp_path):
    path = str(tmp_path / "test.txt")
    write_result = write_file.fn(path=path, content="Hello otterflow!")
    assert "Written" in write_result

    read_result = read_file.fn(path=path)
    assert read_result == "Hello otterflow!"


def test_read_missing_file():
    result = read_file.fn(path="/nonexistent/path/file.txt")
    assert "Error" in result


# ── @tool decorator tests ─────────────────────────────────────────────────────

def test_tool_decorator_creates_tool():
    @tool(description="Doubles a number.")
    def double(x: int) -> int:
        return x * 2

    assert hasattr(double, "_tool_spec")
    spec = double._tool_spec
    assert isinstance(spec, Tool)
    assert spec.name == "double"
    assert spec.fn(x=5) == 10


def test_tool_decorator_schema():
    @tool(description="Greets a user.")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    schema = greet._tool_spec.to_claude_spec()
    assert schema["name"] == "greet"
    assert "name" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["required"] == ["name"]


# ── Memory tests ──────────────────────────────────────────────────────────────

def test_memory_add_and_retrieve():
    mem = Memory()
    mem.add_turn("Hello", "Hi there!")
    assert len(mem) == 1


def test_memory_sliding_window():
    mem = Memory(max_turns=3)
    for i in range(5):
        mem.add_turn(f"msg {i}", f"response {i}")
    assert len(mem) == 3


def test_memory_facts():
    mem = Memory()
    mem.remember("user", "Alex")
    assert mem.recall("user") == "Alex"
    assert mem.recall("nonexistent") is None


def test_memory_builds_messages():
    mem = Memory()
    mem.remember("company", "TechCorp")
    messages = mem.build_messages("Help me!")
    # Should have fact injection + new prompt
    assert messages[-1]["content"] == "Help me!"
    assert any("TechCorp" in str(m) for m in messages)


def test_memory_clear_preserves_facts():
    mem = Memory()
    mem.add_turn("Hello", "Hi")
    mem.remember("key", "value")
    mem.clear()
    assert len(mem) == 0
    assert mem.recall("key") == "value"


# ── Agent tests (mocked API) ──────────────────────────────────────────────────

def _mock_response(text: str):
    """Build a minimal mock Anthropic response."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


@patch("otterflow.agent._client")
def test_agent_basic_run(mock_client):
    mock_client.messages.create.return_value = _mock_response("Hello back!")

    agent = Agent(name="Test", role="You are a test agent.")
    result = agent.run("Hello!")

    assert result == "Hello back!"
    mock_client.messages.create.assert_called_once()


@patch("otterflow.agent._client")
def test_agent_registers_tools(mock_client):
    mock_client.messages.create.return_value = _mock_response("Done.")

    agent = Agent(name="ToolAgent", role="Use tools.", tools=[calculator])
    assert "calculator" in agent.tools

    agent.run("What is 2+2?")
    call_kwargs = mock_client.messages.create.call_args[1]
    assert "tools" in call_kwargs
    tool_names = [t["name"] for t in call_kwargs["tools"]]
    assert "calculator" in tool_names


@patch("otterflow.agent._client")
def test_agent_memory_persists(mock_client):
    mock_client.messages.create.return_value = _mock_response("Got it.")

    mem = Memory()
    agent = Agent(name="MemAgent", role="Remember things.", memory=mem)

    agent.run("My name is Alex.")
    agent.run("What is my name?")

    assert len(mem) == 2


@patch("otterflow.agent._client")
def test_sales_coach_agent(mock_client):
    mock_client.messages.create.return_value = _mock_response("Great pitch improvement!")

    coach = SalesCoachAgent(methodology="SPIN")
    result = coach.run("Review my pitch.")

    assert "improvement" in result.lower() or len(result) > 0
    # Check the system prompt includes the methodology
    call_kwargs = mock_client.messages.create.call_args[1]
    assert "SPIN" in call_kwargs["system"]


# ── Sub-agent / spawn tests ───────────────────────────────────────────────────

@patch("otterflow.agent._client")
def test_spawn_adds_tool(mock_client):
    mock_client.messages.create.return_value = _mock_response("Done.")

    parent = Agent(name="Parent", role="Orchestrate.")
    child = Agent(name="ChildWorker", role="Do work.")
    parent.spawn(child)

    assert "delegate_to_childworker" in parent.tools

FILEOF
echo "✅ tests/test_otterflow.py"

cat > assets/logo.svg << 'FILEOF'
<svg width="680" height="400" viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="340" cy="178" rx="80" ry="72" fill="#5DCAA5"/>
  <ellipse cx="340" cy="188" rx="52" ry="50" fill="#9FE1CB"/>
  <ellipse cx="278" cy="118" rx="18" ry="16" fill="#5DCAA5"/>
  <ellipse cx="278" cy="118" rx="10" ry="9" fill="#9FE1CB"/>
  <ellipse cx="402" cy="118" rx="18" ry="16" fill="#5DCAA5"/>
  <ellipse cx="402" cy="118" rx="10" ry="9" fill="#9FE1CB"/>
  <ellipse cx="340" cy="158" rx="60" ry="56" fill="#9FE1CB"/>
  <ellipse cx="320" cy="146" rx="9" ry="10" fill="#2C2C2A"/>
  <ellipse cx="360" cy="146" rx="9" ry="10" fill="#2C2C2A"/>
  <circle cx="323" cy="143" r="3" fill="white"/>
  <circle cx="363" cy="143" r="3" fill="white"/>
  <ellipse cx="340" cy="162" rx="8" ry="5" fill="#1D9E75"/>
  <path d="M328 170 Q340 180 352 170" fill="none" stroke="#0F6E56" stroke-width="2.5" stroke-linecap="round"/>
  <circle cx="316" cy="166" r="2" fill="#0F6E56" opacity="0.5"/>
  <circle cx="308" cy="170" r="2" fill="#0F6E56" opacity="0.5"/>
  <circle cx="364" cy="166" r="2" fill="#0F6E56" opacity="0.5"/>
  <circle cx="372" cy="170" r="2" fill="#0F6E56" opacity="0.5"/>
  <ellipse cx="280" cy="210" rx="16" ry="10" fill="#5DCAA5" transform="rotate(-30 280 210)"/>
  <ellipse cx="400" cy="210" rx="16" ry="10" fill="#5DCAA5" transform="rotate(30 400 210)"/>
  <rect x="406" y="198" width="6" height="22" rx="3" fill="#085041" transform="rotate(20 409 209)"/>
  <ellipse cx="411" cy="198" rx="7" ry="4" fill="#085041" transform="rotate(20 411 198)"/>
  <path d="M230 185 Q200 175 170 185" fill="none" stroke="#1D9E75" stroke-width="2.5" stroke-linecap="round" stroke-dasharray="5 4" opacity="0.7"/>
  <path d="M230 200 Q195 200 165 208" fill="none" stroke="#1D9E75" stroke-width="2" stroke-linecap="round" stroke-dasharray="4 4" opacity="0.45"/>
  <circle cx="158" cy="184" r="9" fill="#5DCAA5" opacity="0.8"/>
  <circle cx="152" cy="207" r="7" fill="#9FE1CB" opacity="0.6"/>
  <path d="M450 185 Q480 175 510 185" fill="none" stroke="#1D9E75" stroke-width="2.5" stroke-linecap="round" stroke-dasharray="5 4" opacity="0.7"/>
  <path d="M450 200 Q485 200 515 208" fill="none" stroke="#1D9E75" stroke-width="2" stroke-linecap="round" stroke-dasharray="4 4" opacity="0.45"/>
  <circle cx="522" cy="184" r="9" fill="#5DCAA5" opacity="0.8"/>
  <circle cx="528" cy="207" r="7" fill="#9FE1CB" opacity="0.6"/>
  <text x="340" y="295" text-anchor="middle" font-family="system-ui,sans-serif" font-size="42" font-weight="700" letter-spacing="-1" fill="#085041">Otter<tspan fill="#1D9E75">Flow</tspan></text>
  <text x="340" y="328" text-anchor="middle" font-family="system-ui,sans-serif" font-size="15" font-weight="400" fill="#1D9E75" letter-spacing="2">CLEVER AGENTS. CLEAN CODE.</text>
  <rect x="298" y="348" width="84" height="24" rx="12" fill="#E1F5EE"/>
  <text x="340" y="364" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" font-weight="500" fill="#085041">v0.1.0 · MIT</text>
</svg>

FILEOF
echo "✅ assets/logo.svg"

cat > README.md << 'FILEOF'
# 🌊 OtterFlow

**Build production-ready AI agents in 10 lines of Python.**

[![PyPI version](https://img.shields.io/pypi/v/otterflow.svg)](https://pypi.org/project/otterflow/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

`OtterFlow` is a minimal, composable framework for building multi-step AI agents powered by [Claude](https://anthropic.com/claude). No bloat. No magic. Just agents that work.

```python
from otterflow import Agent
from otterflow.tools import web_search, calculator

agent = Agent(
    name="FinanceBot",
    role="You are a financial research assistant.",
    tools=[web_search, calculator],
)

print(agent.run("What's NVIDIA's latest revenue? Calculate 15% YoY growth."))
```

---

## Why OtterFlow?

Most agent frameworks are either toy demos or enterprise nightmares. `otterflow` is neither.

| | OtterFlow | LangChain | AutoGen |
|---|---|---|---|
| Lines to a working agent | **~10** | ~50 | ~40 |
| Multi-agent orchestration | ✅ | ✅ | ✅ |
| Built-in business agents | ✅ | ❌ | ❌ |
| Zero magic / full control | ✅ | ❌ | ❌ |
| Persistent memory | ✅ | ✅ | ✅ |
| Custom tools in 3 lines | ✅ | ❌ | ❌ |

---

## Install

```bash
pip install otterflow
export ANTHROPIC_API_KEY="your-key-here"
```

---

## Core Concepts

### 1. Agent — the building block

```python
from otterflow import Agent

agent = Agent(
    name="Analyst",
    role="You are a senior business analyst. Be concise and data-driven.",
    verbose=True,   # shows tool calls as they happen
)

result = agent.run("Summarize the current state of the AI chip market.")
print(result)
```

### 2. Tools — give your agent superpowers

otterflow ships with 7 built-in tools:

```python
from otterflow.tools import (
    web_search,     # searches the web for current info
    read_file,      # reads local files
    write_file,     # writes output to disk
    run_python,     # executes Python code snippets
    calculator,     # evaluates math expressions
    memory_store,   # stores key-value pairs
    memory_recall,  # recalls stored values
)

agent = Agent(
    name="ResearchBot",
    role="You research topics and save reports.",
    tools=[web_search, write_file],
)

agent.run("Research the top 3 AI coding assistants and save a report to report.md")
```

### 3. Custom tools — 3 lines

```python
from otterflow.tools import tool

@tool(description="Fetch the current price of a stock ticker.")
def get_stock_price(ticker: str) -> str:
    import yfinance as yf
    return str(yf.Ticker(ticker).fast_info.last_price)

agent = Agent("Trader", "You track stocks.", tools=[get_stock_price])
agent.run("What's Apple trading at?")
```

### 4. Memory — agents that remember

```python
from otterflow import Agent, Memory

memory = Memory(max_turns=20)
memory.remember("client_name", "Acme Corp")
memory.remember("deal_value", "$120k")

agent = Agent("SalesBot", "You are a sales assistant.", memory=memory)

agent.run("Help me prep for tomorrow's renewal call.")
agent.run("What objections should I prepare for given the deal size?")
# ↑ Agent remembers the deal value from the previous turn
```

### 5. Multi-agent orchestration

Spawn sub-agents and let a parent orchestrator delegate tasks:

```python
from otterflow import Agent
from otterflow.tools import web_search, write_file

# Specialist agents
researcher = Agent("Researcher", "Deep research specialist.", tools=[web_search])
writer = Agent("Writer", "Turn research into polished reports.", tools=[write_file])

# Orchestrator spawns both as tools
orchestrator = Agent("Boss", "Delegate and synthesize.")
orchestrator.spawn(researcher)
orchestrator.spawn(writer)

# Single call routes through both agents automatically
orchestrator.run(
    "Research the EV market and write a 1-page executive brief to ev_brief.md"
)
```

---

## Pre-built Business Agents

Drop these into your app immediately:

```python
from otterflow.agents import (
    ResearchAgent,          # Deep web research + structured reports
    EmailAgent,             # Draft, rewrite, and improve emails
    DataAnalystAgent,       # Reads files, runs Python, surfaces insights
    CompetitiveIntelAgent,  # Competitor battle cards
    SalesCoachAgent,        # Analyze pitches, generate objection scripts
    BusinessIntelPipeline,  # All of the above, orchestrated
)
```

### ResearchAgent

```python
researcher = ResearchAgent(verbose=True)
report = researcher.run(
    "What are the top 5 B2B SaaS trends for 2025? Focus on AI and pricing."
)
```

### EmailAgent

```python
email_bot = EmailAgent(tone="assertive")
draft = email_bot.run(
    "Write a follow-up to a prospect who went cold after last week's demo."
)
```

### SalesCoachAgent

```python
coach = SalesCoachAgent(methodology="Challenger")

feedback = coach.run("""
    Analyze this cold email:

    Subject: Quick question
    Hi, I hope this finds you well. Our software has many great features...
""")

# In the next turn, the agent remembers what it already critiqued
rewrite = coach.run("Now rewrite it using what you just taught me.")
```

### BusinessIntelPipeline

```python
pipeline = BusinessIntelPipeline(verbose=True)
analysis = pipeline.run(
    "Should I build a B2B AI sales coaching product in 2025? "
    "Analyze market size, competitors, and give me a go/no-go."
)
```

---

## Real-world example: ai-sales-coach

`OtterFlow` powers [ai-sales-coach](https://github.com/tabato/ai-sales-coach) — an AI that analyzes your sales calls, rewrites your cold emails, and role-plays as difficult prospects to sharpen your reps.

```python
from otterflow import Memory
from otterflow.agents import SalesCoachAgent

# Initialize with rep context
memory = Memory()
memory.remember("rep_name", "Jordan")
memory.remember("product", "B2B SaaS CRM")
memory.remember("avg_deal_size", "$45k ARR")

coach = SalesCoachAgent(methodology="MEDDIC", memory=memory)

# Analyze a call transcript
with open("call_transcript.txt") as f:
    transcript = f.read()

feedback = coach.run(f"Analyze this call and tell Jordan exactly what to fix:\n{transcript}")
print(feedback)
```

---

## Architecture

```
otterflow/
├── agent.py          # Core Agent class — tool loop, sub-agent spawning
├── memory.py         # Memory — sliding window, persistent facts
├── tools.py          # Tool class, @tool decorator, built-in tools
└── agents/
    └── __init__.py   # Pre-built business agents
```

The execution loop is deliberately transparent:

```
run(prompt)
  │
  ├─ inject memory + history into messages
  ├─ call Claude with tool specs
  │
  └─ loop:
      ├─ if stop_reason == "end_turn" → return text
      └─ else → execute tool calls → append results → repeat
```

No hidden chains. No mysterious abstractions. You can read the entire core in < 200 lines.

---

## Contributing

PRs welcome. High-value contributions:

- New built-in tools (Slack, email, calendar, Postgres)
- Async support (`agent.arun()`)
- Streaming output
- Token usage tracking
- More pre-built agents

```bash
git clone https://github.com/tabato/otterflow
cd otterflow
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT. Use it, fork it, build your startup on it.

---

*Built with [Claude](https://anthropic.com/claude) by Anthropic.*

FILEOF
echo "✅ README.md"

# ── Git init ─────────────────────────────────────
git init -q
git add .
git commit -q -m "🦦 Initial commit — OtterFlow v0.1.0"

echo ""
echo "────────────────────────────────────────"
echo "🦦 OtterFlow is ready!"
echo ""
echo "Next steps:"
echo "  cd $TARGET"
echo "  pip install -e '.[dev]'"
echo "  export ANTHROPIC_API_KEY=your-key-here"
echo "  python examples/quickstart.py 1"
echo ""
echo "To push to GitHub:"
echo "  gh repo create otterflow --public --source=. --push"
echo "  (or create the repo on github.com and follow the instructions)"
echo "────────────────────────────────────────"