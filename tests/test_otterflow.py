"""
tests/test_otterflow.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for otterflow. Run with: pytest tests/
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from otterflow import Agent, Memory, Pipeline, Tool, Usage, tool
from otterflow.agents import ContentCreatorAgent
from otterflow.tools import calculator, read_file, write_file

# ── Tool tests ───────────────────────────────────────────────────────────────

def test_calculator_basic():
    assert calculator.fn(expression="2 + 2") == "4"


def test_calculator_complex():
    assert calculator.fn(expression="(100 * 0.15) + 50") == "65.0"


def test_calculator_blocks_unsafe():
    result = calculator.fn(expression="__import__('os').system('ls')")
    assert "disallowed" in result.lower() or "error" in result.lower()


def test_calculator_scientific_notation():
    result = calculator.fn(expression="1.5e3 * 2")
    assert result == "3000.0"


def test_write_and_read_file(tmp_path):
    path = str(tmp_path / "test.txt")
    write_result = write_file.fn(path=path, content="Hello otterflow!")
    assert "Written" in write_result
    assert read_file.fn(path=path) == "Hello otterflow!"


def test_read_missing_file():
    assert "Error" in read_file.fn(path="/nonexistent/path/file.txt")


def test_write_file_blocks_system_paths():
    result = write_file.fn(path="/etc/passwd", content="hacked")
    assert "not permitted" in result.lower()


def test_write_file_blocks_usr():
    result = write_file.fn(path="/usr/local/bin/evil", content="x")
    assert "not permitted" in result.lower()


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


def test_tool_decorator_optional_param():
    @tool(description="Adds two numbers.")
    def add(a: int, b: int = 0) -> int:
        return a + b

    schema = add._tool_spec.to_claude_spec()
    assert "a" in schema["input_schema"]["required"]
    assert "b" not in schema["input_schema"].get("required", [])


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
    assert messages[-1]["content"] == "Help me!"
    assert any("TechCorp" in str(m) for m in messages)


def test_memory_clear_preserves_facts():
    mem = Memory()
    mem.add_turn("Hello", "Hi")
    mem.remember("key", "value")
    mem.clear()
    assert len(mem) == 0
    assert mem.recall("key") == "value"


# ── Usage tests ───────────────────────────────────────────────────────────────

def test_usage_totals():
    u = Usage()
    u.input_tokens = 1_000_000
    u.output_tokens = 1_000_000
    assert u.total_tokens == 2_000_000
    # 1M input @ $3 + 1M output @ $15 = $18
    assert abs(u.estimated_cost_usd - 18.0) < 0.001


def test_usage_repr():
    u = Usage()
    u.input_tokens = 500
    u.output_tokens = 200
    assert "500" in repr(u)
    assert "200" in repr(u)
    assert "$" in repr(u)


# ── Agent tests (mocked API) ──────────────────────────────────────────────────

def _mock_response(text: str):
    """Build a minimal mock Anthropic response."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    response.usage = MagicMock(input_tokens=10, output_tokens=5)
    return response


@patch("otterflow.agent._get_client")
def test_agent_basic_run(mock_get_client):
    mock_get_client.return_value.messages.create.return_value = _mock_response("Hello back!")

    agent = Agent(name="Test", role="You are a test agent.")
    result = agent.run("Hello!")

    assert result == "Hello back!"
    mock_get_client.return_value.messages.create.assert_called_once()


@patch("otterflow.agent._get_client")
def test_agent_tracks_usage(mock_get_client):
    mock_get_client.return_value.messages.create.return_value = _mock_response("Done.")

    agent = Agent(name="Test", role="You are a test agent.")
    agent.run("Hello!")

    assert agent.usage.input_tokens == 10
    assert agent.usage.output_tokens == 5


@patch("otterflow.agent._get_client")
def test_agent_registers_tools(mock_get_client):
    mock_get_client.return_value.messages.create.return_value = _mock_response("Done.")

    agent = Agent(name="ToolAgent", role="Use tools.", tools=[calculator])
    assert "calculator" in agent.tools

    agent.run("What is 2+2?")
    call_kwargs = mock_get_client.return_value.messages.create.call_args[1]
    assert "tools" in call_kwargs
    assert any(t["name"] == "calculator" for t in call_kwargs["tools"])


@patch("otterflow.agent._get_client")
def test_agent_memory_persists(mock_get_client):
    mock_get_client.return_value.messages.create.return_value = _mock_response("Got it.")

    mem = Memory()
    agent = Agent(name="MemAgent", role="Remember things.", memory=mem)
    agent.run("My name is Alex.")
    agent.run("What is my name?")

    assert len(mem) == 2


@patch("otterflow.agent._get_client")
def test_content_creator_agent(mock_get_client):
    mock_get_client.return_value.messages.create.return_value = _mock_response(
        "Your hook is weak. Here's a rewrite: ..."
    )

    coach = ContentCreatorAgent(platform="LinkedIn")
    result = coach.run("Analyze this post.")

    assert len(result) > 0
    call_kwargs = mock_get_client.return_value.messages.create.call_args[1]
    assert "LinkedIn" in call_kwargs["system"]


def test_content_creator_invalid_platform():
    with pytest.raises(ValueError, match="Unsupported platform"):
        ContentCreatorAgent(platform="TikTok")


# ── Pipeline tests ────────────────────────────────────────────────────────────

@patch("otterflow.agent._get_client")
def test_pipe_operator_creates_pipeline(mock_get_client):
    mock_get_client.return_value.messages.create.return_value = _mock_response("output")

    a = Agent(name="A", role="First.")
    b = Agent(name="B", role="Second.")
    pipeline = a | b

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.agents) == 2


@patch("otterflow.agent._get_client")
def test_pipeline_chains_output(mock_get_client):
    responses = [_mock_response("step1_output"), _mock_response("final_output")]
    mock_get_client.return_value.messages.create.side_effect = responses

    a = Agent(name="A", role="First.")
    b = Agent(name="B", role="Second.")
    pipeline = a | b

    result = pipeline.run("initial prompt")
    assert result == "final_output"

    calls = mock_get_client.return_value.messages.create.call_args_list
    # Second agent should have received "step1_output" as the prompt
    second_call_messages = calls[1][1]["messages"]
    assert any("step1_output" in str(m) for m in second_call_messages)


@patch("otterflow.agent._get_client")
def test_pipeline_extend_with_pipe(mock_get_client):
    mock_get_client.return_value.messages.create.return_value = _mock_response("done")

    a = Agent(name="A", role="A.")
    b = Agent(name="B", role="B.")
    c = Agent(name="C", role="C.")

    pipeline = a | b | c
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.agents) == 3


@patch("otterflow.agent._get_client")
def test_pipeline_repr(mock_get_client):
    a = Agent(name="Researcher", role="Research.")
    b = Agent(name="Writer", role="Write.")
    pipeline = a | b
    assert "Researcher" in repr(pipeline)
    assert "Writer" in repr(pipeline)


# ── Async tests ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("otterflow.agent._get_async_client")
async def test_agent_arun(mock_get_async_client):
    mock_response = _mock_response("Async response!")
    mock_get_async_client.return_value.messages.create = AsyncMock(return_value=mock_response)

    agent = Agent(name="AsyncAgent", role="You are async.")
    result = await agent.arun("Hello async!")

    assert result == "Async response!"


@pytest.mark.asyncio
@patch("otterflow.agent._get_async_client")
async def test_pipeline_arun(mock_get_async_client):
    responses = [_mock_response("intermediate"), _mock_response("final async")]
    mock_get_async_client.return_value.messages.create = AsyncMock(side_effect=responses)

    a = Agent(name="A", role="First.")
    b = Agent(name="B", role="Second.")
    pipeline = a | b

    result = await pipeline.arun("start")
    assert result == "final async"


@pytest.mark.asyncio
@patch("otterflow.agent._get_async_client")
async def test_agent_astream(mock_get_async_client):
    chunks = ["Hello", " from", " async", " stream!"]

    async def mock_text_stream():
        for chunk in chunks:
            yield chunk

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__.return_value.text_stream = mock_text_stream()
    mock_stream_ctx.__aenter__.return_value.get_final_message = AsyncMock(
        return_value=_mock_response("".join(chunks))
    )
    mock_get_async_client.return_value.messages.stream.return_value = mock_stream_ctx

    agent = Agent(name="StreamAgent", role="You stream.")
    collected = []
    async for chunk in agent.astream("Stream this!"):
        collected.append(chunk)

    assert collected == chunks
    assert "".join(collected) == "Hello from async stream!"


# ── Sub-agent / spawn tests ───────────────────────────────────────────────────

@patch("otterflow.agent._get_client")
def test_spawn_adds_tool(mock_get_client):
    mock_get_client.return_value.messages.create.return_value = _mock_response("Done.")

    parent = Agent(name="Parent", role="Orchestrate.")
    child = Agent(name="ChildWorker", role="Do work.")
    parent.spawn(child)

    assert "delegate_to_childworker" in parent.tools
