"""
otterflow.agent
~~~~~~~~~~~~~~~
Core Agent base class. Handles tool registration, execution loops,
memory injection, streaming, async, multi-agent orchestration,
token tracking, and retry logic.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, AsyncIterator, Callable, Iterator

import anthropic
from dotenv import load_dotenv

from .memory import Memory
from .tools import Tool

load_dotenv()  # no-op if no .env file present


# ────────────────────────────────────────────────────────────────────────────
# Lazy client singletons — fail at call time, not import time
# ────────────────────────────────────────────────────────────────────────────

_sync_client: anthropic.Anthropic | None = None
_async_client: anthropic.AsyncAnthropic | None = None

_KEY_HINT = (
    "ANTHROPIC_API_KEY is not set.\n"
    "  • Add it to a .env file:   ANTHROPIC_API_KEY=sk-ant-...\n"
    "  • Or export in your shell: export ANTHROPIC_API_KEY=sk-ant-...\n"
    "  • Get a key at https://console.anthropic.com/"
)


def _get_client() -> anthropic.Anthropic:
    global _sync_client
    if _sync_client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise EnvironmentError(_KEY_HINT)
        _sync_client = anthropic.Anthropic()
    return _sync_client


def _get_async_client() -> anthropic.AsyncAnthropic:
    global _async_client
    if _async_client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise EnvironmentError(_KEY_HINT)
        _async_client = anthropic.AsyncAnthropic()
    return _async_client


MODEL = "claude-sonnet-4-6"

# Pricing per million tokens — claude-sonnet-4-6
_COST_INPUT_PER_M = 3.00
_COST_OUTPUT_PER_M = 15.00

_MAX_RETRIES = 3


# ────────────────────────────────────────────────────────────────────────────
# Usage tracking
# ────────────────────────────────────────────────────────────────────────────

class Usage:
    """
    Cumulative token usage and estimated cost across all runs on an agent.

    Usage::

        agent = Agent("Bot", "You are helpful.")
        agent.run("Hello")
        print(agent.usage)
        # Usage(input=42, output=18, ~$0.0004)
    """

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * _COST_INPUT_PER_M
            + self.output_tokens / 1_000_000 * _COST_OUTPUT_PER_M
        )

    def __repr__(self) -> str:
        return (
            f"Usage(input={self.input_tokens:,}, output={self.output_tokens:,}, "
            f"~${self.estimated_cost_usd:.4f})"
        )


# ────────────────────────────────────────────────────────────────────────────
# Agent
# ────────────────────────────────────────────────────────────────────────────

class Agent:
    """
    A lightweight, composable AI agent powered by Claude.

    Supports synchronous, streaming, and async execution. Chain agents
    with the | operator to build multi-step pipelines.

    Usage::

        from otterflow import Agent
        from otterflow.tools import web_search, calculator

        agent = Agent(
            name="ResearchBot",
            role="You are a senior research analyst.",
            tools=[web_search, calculator],
        )

        # Synchronous
        result = agent.run("Summarize the latest trends in AI infrastructure.")

        # Streaming — yields chunks as they arrive
        for chunk in agent.stream("Write a report on the AI chip market."):
            print(chunk, end="", flush=True)

        # Async — non-blocking, works in FastAPI / async scripts
        result = await agent.arun("Summarize the latest trends.")

        # Pipe operator — chain agents
        pipeline = researcher | writer | editor
        result = pipeline.run("Research AI trends and write a 1-page brief.")

        # Token usage and estimated cost
        print(agent.usage)  # Usage(input=1,234, output=567, ~$0.0122)
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
        self.memory = memory if memory is not None else Memory()
        self.max_steps = max_steps
        self.verbose = verbose
        self.model = model
        self.usage = Usage()
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
        self.register_tool(agent._as_tool())
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
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_kwargs(self, messages: list[dict]) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=4096,
            system=self.role,
            messages=messages,
        )
        tool_specs = [t.to_claude_spec() for t in self.tools.values()]
        if tool_specs:
            kwargs["tools"] = tool_specs
        return kwargs

    def _call_api(self, **kwargs: Any) -> Any:
        """Sync API call with exponential-backoff retry on rate limits / server errors."""
        client = _get_client()
        for attempt in range(_MAX_RETRIES):
            try:
                return client.messages.create(**kwargs)
            except anthropic.RateLimitError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = 2 ** attempt
                if self.verbose:
                    print(f"[{self.name}] Rate limited — retrying in {wait}s...")
                time.sleep(wait)
            except anthropic.InternalServerError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = 2 ** attempt
                if self.verbose:
                    print(f"[{self.name}] Server error — retrying in {wait}s...")
                time.sleep(wait)

    def _track_usage(self, response: Any) -> None:
        if hasattr(response, "usage"):
            self.usage.input_tokens += response.usage.input_tokens
            self.usage.output_tokens += response.usage.output_tokens

    def _execute_tool(self, name: str, inputs: dict) -> Any:
        if name not in self.tools:
            return f"Error: tool '{name}' not found."
        try:
            return self.tools[name].fn(**inputs)
        except Exception as e:
            return f"Tool error: {e}"

    @staticmethod
    def _extract_text(response: Any) -> str:
        return "\n".join(b.text for b in response.content if b.type == "text").strip()

    @staticmethod
    def _parse_content(response: Any) -> tuple[list[dict], list[Any]]:
        """Return (assistant_content_for_messages, tool_use_blocks)."""
        content: list[dict] = []
        tool_calls: list[Any] = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                tool_calls.append(block)
        return content, tool_calls

    async def _async_call_api(self, **kwargs: Any) -> Any:
        """Async API call with exponential-backoff retry on rate limits / server errors."""
        client = _get_async_client()
        for attempt in range(_MAX_RETRIES):
            try:
                return await client.messages.create(**kwargs)
            except (anthropic.RateLimitError, anthropic.InternalServerError):
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = 2 ** attempt
                if self.verbose:
                    print(f"[{self.name}] Retrying in {wait}s...")
                await asyncio.sleep(wait)

    def _execute_tool_calls(self, tool_calls: list[Any]) -> list[dict]:
        results = []
        for tc in tool_calls:
            result = self._execute_tool(tc.name, tc.input)
            if self.verbose:
                print(f"[{self.name}] 🔧 {tc.name} → {str(result)[:120]}")
            results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": str(result),
            })
        return results

    # ------------------------------------------------------------------ #
    # Synchronous run
    # ------------------------------------------------------------------ #

    def run(self, prompt: str) -> str:
        """Run the agent on a prompt. Returns the final text response."""
        messages = self.memory.build_messages(prompt)

        if self.verbose:
            print(f"\n[{self.name}] 🦦 Starting: {prompt[:80]}...")

        for step in range(self.max_steps):
            response = self._call_api(**self._build_kwargs(messages))
            self._track_usage(response)

            if self.verbose:
                print(f"[{self.name}] Step {step + 1}: stop_reason={response.stop_reason}")

            assistant_content, tool_calls = self._parse_content(response)
            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "end_turn" or not tool_calls:
                final = self._extract_text(response)
                self.memory.add_turn(prompt, final)
                return final

            messages.append({
                "role": "user",
                "content": self._execute_tool_calls(tool_calls),
            })

        return "Max steps reached without a final answer."

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Stream text chunks as they arrive from Claude.

        Tool-call steps are executed silently between turns; only text
        content is yielded. The caller gets a smooth stream of the final
        response even if the agent used tools internally.

        Usage::

            for chunk in agent.stream("Write a report on AI chip stocks"):
                print(chunk, end="", flush=True)
            print()  # newline after stream ends
        """
        messages = self.memory.build_messages(prompt)
        client = _get_client()
        final_text = ""

        if self.verbose:
            print(f"\n[{self.name}] 🦦 Streaming: {prompt[:80]}...")

        for step in range(self.max_steps):
            kwargs = self._build_kwargs(messages)
            step_text = ""

            with client.messages.stream(**kwargs) as s:
                for chunk in s.text_stream:
                    step_text += chunk
                    yield chunk

                final = s.get_final_message()

            self._track_usage(final)
            assistant_content, tool_calls = self._parse_content(final)
            messages.append({"role": "assistant", "content": assistant_content})

            if not tool_calls:
                final_text = step_text
                break

            if self.verbose:
                print()  # newline before tool call logs
            messages.append({
                "role": "user",
                "content": self._execute_tool_calls(tool_calls),
            })

        self.memory.add_turn(prompt, final_text)

    # ------------------------------------------------------------------ #
    # Async streaming
    # ------------------------------------------------------------------ #

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """
        Async streaming — yields text chunks as they arrive from Claude.

        Tool-call steps are executed concurrently between turns; only text
        content is yielded. Use in FastAPI, async scripts, or any context
        where you need non-blocking streaming.

        Usage::

            async for chunk in agent.astream("Write a report on AI chip stocks"):
                print(chunk, end="", flush=True)
            print()
        """
        messages = self.memory.build_messages(prompt)
        client = _get_async_client()
        final_text = ""

        if self.verbose:
            print(f"\n[{self.name}] 🦦 Streaming (async): {prompt[:80]}...")

        for step in range(self.max_steps):
            kwargs = self._build_kwargs(messages)
            step_text = ""

            async with client.messages.stream(**kwargs) as s:
                async for chunk in s.text_stream:
                    step_text += chunk
                    yield chunk

                final = await s.get_final_message()

            self._track_usage(final)
            assistant_content, tool_calls = self._parse_content(final)
            messages.append({"role": "assistant", "content": assistant_content})

            if not tool_calls:
                final_text = step_text
                break

            if self.verbose:
                print()

            tool_results_raw = await asyncio.gather(
                *[asyncio.to_thread(self._execute_tool, tc.name, tc.input) for tc in tool_calls]
            )

            if self.verbose:
                for tc, result in zip(tool_calls, tool_results_raw):
                    print(f"[{self.name}] 🔧 {tc.name} → {str(result)[:120]}")

            messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tc.id, "content": str(result)}
                    for tc, result in zip(tool_calls, tool_results_raw)
                ],
            })

        self.memory.add_turn(prompt, final_text)

    # ------------------------------------------------------------------ #
    # Async
    # ------------------------------------------------------------------ #

    async def arun(self, prompt: str) -> str:
        """
        Async version of run(). Non-blocking — use in FastAPI, async scripts,
        or to run multiple agents concurrently with asyncio.gather().

        Usage::

            # Single call
            result = await agent.arun("Summarize the AI chip market.")

            # Concurrent — runs both agents at the same time
            results = await asyncio.gather(
                agent1.arun("Research topic A"),
                agent2.arun("Research topic B"),
            )
        """
        messages = self.memory.build_messages(prompt)

        if self.verbose:
            print(f"\n[{self.name}] 🦦 Starting (async): {prompt[:80]}...")

        for step in range(self.max_steps):
            kwargs = self._build_kwargs(messages)
            response = await self._async_call_api(**kwargs)
            self._track_usage(response)

            if self.verbose:
                print(f"[{self.name}] Step {step + 1}: stop_reason={response.stop_reason}")

            assistant_content, tool_calls = self._parse_content(response)
            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "end_turn" or not tool_calls:
                final = self._extract_text(response)
                self.memory.add_turn(prompt, final)
                return final

            # Execute tools concurrently in a thread pool (they're sync I/O)
            tool_results_raw = await asyncio.gather(
                *[asyncio.to_thread(self._execute_tool, tc.name, tc.input) for tc in tool_calls]
            )

            if self.verbose:
                for tc, result in zip(tool_calls, tool_results_raw):
                    print(f"[{self.name}] 🔧 {tc.name} → {str(result)[:120]}")

            messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tc.id, "content": str(result)}
                    for tc, result in zip(tool_calls, tool_results_raw)
                ],
            })

        return "Max steps reached without a final answer."

    # ------------------------------------------------------------------ #
    # Pipeline (|) operator
    # ------------------------------------------------------------------ #

    def __or__(self, other: "Agent") -> "Pipeline":
        """
        Chain this agent with another using the | operator.

            pipeline = researcher | writer | editor
            result = pipeline.run("Research AI trends and produce a report.")
        """
        return Pipeline([self, other])

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, tools={list(self.tools.keys())})"


# ────────────────────────────────────────────────────────────────────────────
# Pipeline — linear agent chain via the | operator
# ────────────────────────────────────────────────────────────────────────────

class Pipeline:
    """
    A linear chain of agents where each agent's output becomes the
    next agent's input. Build pipelines with the | operator.

    Usage::

        researcher = ResearchAgent()
        writer = Agent("Writer", "Turn research into polished prose.")
        editor = Agent("Editor", "Tighten copy. Remove filler. Keep it sharp.")

        pipeline = researcher | writer | editor
        result = pipeline.run("What are the top AI infrastructure startups in 2025?")

        # Async sequential pipeline
        result = await pipeline.arun("...")

        # Extend an existing pipeline
        pipeline = pipeline | Agent("Translator", "Translate to Spanish.")
    """

    def __init__(self, agents: list[Agent]) -> None:
        self.agents = agents

    def __or__(self, other: Agent) -> "Pipeline":
        return Pipeline(self.agents + [other])

    def run(self, prompt: str) -> str:
        """Pass the prompt through each agent in sequence."""
        result = prompt
        for agent in self.agents:
            result = agent.run(result)
        return result

    async def arun(self, prompt: str) -> str:
        """Async sequential pipeline — each agent awaits the previous."""
        result = prompt
        for agent in self.agents:
            result = await agent.arun(result)
        return result

    def __repr__(self) -> str:
        return "Pipeline(" + " | ".join(a.name for a in self.agents) + ")"
