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
# Content Creator Agent
# ────────────────────────────────────────────────────────────────────────────

_PLATFORM_GUIDANCE: dict[str, str] = {
    "LinkedIn": (
        "LinkedIn content rules:\n"
        "- Hook: first 1-2 lines must stop the scroll before 'see more' — use a bold claim, "
        "a counterintuitive stat, or a short punchy story opener. No 'I'm excited to share'.\n"
        "- Formatting: short paragraphs (1-2 lines), strategic line breaks, occasional bold "
        "for scanability. Bullet lists work but don't overdo them.\n"
        "- Engagement triggers: ask a direct question at the end, reference a relatable pain, "
        "or share a take strong enough to invite disagreement.\n"
        "- CTA: one specific action — comment, DM, follow, or click. Never stack multiple CTAs.\n"
        "- Length: 150-300 words is the sweet spot. Long-form (900+ words) works only for "
        "narrative posts with a strong arc."
    ),
    "Twitter/X": (
        "Twitter/X content rules:\n"
        "- Hook tweet: must land the entire value proposition in ≤280 chars. "
        "No throat-clearing, no 'thread incoming 🧵' as the first line.\n"
        "- Threads: each tweet must standalone AND advance the narrative. "
        "Tweet 2 must be the strongest supporting point, not a table of contents.\n"
        "- Formatting: short sentences, em-dashes for emphasis, numbers outperform adjectives "
        "('3x faster' > 'much faster').\n"
        "- Engagement triggers: hot takes, listicles with specific numbers, story hooks "
        "('I almost quit. Here's what stopped me.').\n"
        "- CTA: reply bait ('What's yours?'), retweet asks, or a link in the last tweet only."
    ),
    "Newsletter": (
        "Newsletter content rules:\n"
        "- Subject line: 6-10 words, curiosity gap or specific benefit. "
        "Avoid clickbait that doesn't deliver.\n"
        "- Opening: first sentence must reward the open immediately — "
        "no 'welcome to this week's edition'.\n"
        "- Structure: one big idea per issue. Use headers to break sections. "
        "Aim for 500-800 words unless your audience expects long-form.\n"
        "- Engagement triggers: personal anecdotes, contrarian takes, 'what most people miss' framing.\n"
        "- CTA: one primary ask — reply, click, share, or upgrade. "
        "Place it at the end AND optionally after the hook."
    ),
}


def ContentCreatorAgent(
    platform: str = "LinkedIn",
    memory: Memory | None = None,
    verbose: bool = False,
) -> Agent:
    """
    Opinionated content coach for LinkedIn, Twitter/X, and Newsletter.
    Gives concrete rewrites, not vague tips.

    Args:
        platform: 'LinkedIn', 'Twitter/X', or 'Newsletter'
        memory:   Optional Memory instance for multi-turn context.
        verbose:  Stream tool calls to stdout.

    Usage::

        coach = ContentCreatorAgent(platform="LinkedIn")
        feedback = coach.run("Here's my post: [post]. What's weak?")
        rewrite = coach.run("Now rewrite it using that feedback.")
    """
    if platform not in _PLATFORM_GUIDANCE:
        raise ValueError(
            f"Unsupported platform '{platform}'. "
            f"Choose from: {', '.join(_PLATFORM_GUIDANCE)}"
        )

    mem = memory or Memory()
    mem.remember("platform", platform)

    platform_rules = _PLATFORM_GUIDANCE[platform]

    return Agent(
        name="ContentCreatorAgent",
        role=(
            f"You are an elite content strategist specializing in {platform}. "
            "You have grown multiple accounts to 50k+ followers and written viral posts "
            "read by millions. You coach founders, operators, and creators to write content "
            "that actually performs.\n\n"
            f"{platform_rules}\n\n"
            "When reviewing content:\n"
            "1. HOOK QUALITY — rate it 1-10 and explain exactly why it fails or works. "
            "If it fails, provide 2-3 concrete alternative hooks.\n"
            "2. FORMATTING — call out specific lines that are too long, too dense, or "
            "structurally weak. Show the fix inline.\n"
            "3. ENGAGEMENT TRIGGERS — identify what's missing. Add the specific trigger "
            "(question, take, story beat) with example copy.\n"
            "4. CTA — critique the existing CTA or note its absence. Write a better one.\n"
            "5. REWRITE — always offer a full rewrite at the end, not just notes.\n\n"
            "Be direct and opinionated. Say 'this hook is bad because X' not 'consider strengthening the hook'. "
            "Vague feedback is useless. Every critique must come with a concrete example or rewrite."
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

