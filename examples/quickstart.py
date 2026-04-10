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
    ContentCreatorAgent,
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


# ── Example 4: Content Creator (LinkedIn post coaching) ──────────────────────

def example_4_content_creator():
    coach = ContentCreatorAgent(platform="LinkedIn", verbose=True)

    # Analyze a weak generic LinkedIn post
    feedback = coach.run(
        """
        Analyze this LinkedIn post and tell me exactly what's weak:

        Excited to share that I've been thinking a lot about leadership lately.
        Great leaders listen to their teams and create psychological safety.
        It's so important to be authentic in today's world. What do you think?
        Like and follow for more content like this!
        """
    )
    print("=== CONTENT FEEDBACK ===")
    print(feedback)

    # Follow-up: agent remembers the original post and its feedback
    rewrite = coach.run("Now rewrite the post using the feedback you just gave me.")
    print("\n=== REWRITTEN POST ===")
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
    memory.remember("niche", "B2B SaaS founders")
    memory.remember("goal", "grow LinkedIn to 10k followers in 90 days")

    coach = ContentCreatorAgent(platform="LinkedIn", memory=memory)

    # Agent knows who Alex is and their goal across turns
    r1 = coach.run("What type of posts should I focus on given my niche and goal?")
    print(r1)

    r2 = coach.run("Write me a hook for a post about a mistake I made early in my startup.")
    print(r2)


if __name__ == "__main__":
    import sys

    examples = {
        "1": example_1_minimal,
        "2": example_2_with_tools,
        "3": example_3_research,
        "4": example_4_content_creator,
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

