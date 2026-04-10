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
__author__ = "Thomas Abato"
__license__ = "MIT"

from .agent import Agent, Pipeline, Usage
from .memory import Memory
from .tools import Tool, tool

from .agents import (
    ResearchAgent,
    EmailAgent,
    DataAnalystAgent,
    CompetitiveIntelAgent,
    ContentCreatorAgent,
    BusinessIntelPipeline,
)

__all__ = [
    "Agent",
    "Pipeline",
    "Usage",
    "Memory",
    "Tool",
    "tool",
    # Pre-built agents
    "ResearchAgent",
    "EmailAgent",
    "DataAnalystAgent",
    "CompetitiveIntelAgent",
    "ContentCreatorAgent",
    "BusinessIntelPipeline",
]

