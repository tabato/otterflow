# Learn OtterFlow
### From zero to building AI agents — no experience required.

---

```
        __   __
       /  \ /  \
      ( o   o  )
       \  ~~~  /     "Hey. You got this."
        |     |
        |  |  |
       /|  |  |\
```

*Hi. I'm Ollie. I'll be your guide.*

This guide will take you from **"I have no idea what any of this is"** to
**"I just built an AI agent."**

Six lessons. Plain English. No fluff.

Let's go.

---

## Lesson 1: What Is an AI Agent?

Imagine you hire an assistant.

You say: *"Go research the top 5 AI startups and write me a one-page summary."*

You don't tell them every step. You don't tell them which websites to visit.
You don't explain what a browser is. You just give them the goal.

They figure out the rest. They come back with a result.

**That is an AI agent.**

It's a program that takes a goal, figures out the steps to reach it,
and keeps going until the job is done.

A normal program does exactly what you tell it, step by step.
An agent decides its own steps.

```
  Normal program:                  Agent:

  You: "Do step 1."                You: "Handle this for me."
  Program: *does step 1*
  You: "Now do step 2."            Agent: *thinks*
  Program: *does step 2*           Agent: *searches*
  You: "Now do step 3..."          Agent: *reads*
                                   Agent: *writes*
                                   Agent: "Done. Here's your result."
```

OtterFlow lets you build agents like this in just a few lines of Python.

> **Key Takeaway:** An agent is like a smart assistant. You give it a goal. It handles the work.

---

## Lesson 2: What Is a Tool?

Here's the thing about agents.

A brand new agent knows how to *think*. It knows language. It can reason.
But it can't actually *do* anything in the real world.

It's a brain with no hands.

```
     Agent (no tools)           Agent (with tools)

        ( o o )                     ( o o )
        (     )                     (     )
          | |                       /   \
         (no hands)              [web] [files]

     "I want to help but        "Let me search that
      I literally can't          for you right now."
      do anything."
```

Tools are what give agents hands.

A tool is just a Python function with a description attached.
The agent reads the description, decides when to use it, and calls it.

Here's the simplest tool you can make:

```python
from otterflow.tools import tool

@tool(description="Say hello to someone by name.")
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

That's it. Four lines. You just gave an agent a new ability.

The `@tool` part is called a decorator. It wraps your function and tells
the agent: *"This exists. Here's what it does. Use it when you need it."*

Now add it to an agent:

```python
from otterflow import Agent

agent = Agent(name="Greeter", role="You are a friendly assistant.", tools=[greet])
print(agent.run("Say hello to Maria."))
```

The agent sees the `greet` tool, decides it's useful, and calls it.

> **Key Takeaway:** Tools are what agents use to act. Without tools, an agent can think but can't do. Give it tools, and it can do almost anything.

---

## Lesson 3: What Is Memory?

Picture this.

You walk into a coffee shop. You see your friend. You say:

*"Hey! How did that job interview go?"*

Your friend stares at you blankly.

*"What job interview?"*

*"The one you told me about last week."*

*"I don't remember. Who are you?"*

This is what talking to an agent feels like by default.

Every time you send a message, the agent starts fresh. It has no idea
what you said before. It forgets everything the moment the conversation ends.

**Memory fixes that.**

```
  Without Memory:              With Memory:

  You: "My name is Alex."      You: "My name is Alex."
  Agent: "Nice to meet you!"   Agent: "Nice to meet you!"

  You: "What's my name?"       You: "What's my name?"
  Agent: "I don't know         Agent: "Your name is Alex!"
          your name."
```

Here's how you use it:

```python
from otterflow import Agent, Memory

memory = Memory()
agent = Agent(name="Bot", role="You are helpful.", memory=memory)

agent.run("My favorite color is blue.")
print(agent.run("What is my favorite color?"))
# → "Your favorite color is blue."
```

You create a `Memory` object. You pass it to the agent. That's it.
Now the agent remembers everything from the conversation.

You can also store specific facts:

```python
memory.remember("name", "Alex")
memory.remember("goal", "grow a newsletter")
```

The agent will use these facts automatically. Every time.

> **Key Takeaway:** Memory lets agents remember. Without it, every message is a first meeting. With it, agents can have real, ongoing conversations.

---

## Lesson 4: How Do Agents Think?

This is the mental model. No code. Just the idea.

When you give an agent a task, it doesn't just answer immediately.
It goes through a loop.

**Think. Act. Observe. Repeat.**

```
         ┌─────────────────────────────┐
         │                             │
         │      You give a task        │
         │                             │
         └──────────────┬──────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │                              │
         │   THINK: "What should I do?" │
         │   "Do I need a tool?"        │
         │   "Do I have enough info?"   │
         │                              │
         └──────────────┬───────────────┘
                        │
           ┌────────────▼────────────┐
           │                         │
           │   ACT: Use a tool.      │
           │   Search. Calculate.    │
           │   Read. Write.          │
           │                         │
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │                         │
           │   OBSERVE: "What did    │
           │   I learn? Is this      │
           │   enough to answer?"    │
           │                         │
           └────────────┬────────────┘
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
       Not done yet.           Done!
       Loop back to            Give the
       THINK.                  final answer.
```

Here's a real example. You ask: *"What is NVIDIA's stock price, and what is 15% of it?"*

**Step 1 — Think:** "I need the stock price. I have a `web_search` tool. I'll use that."

**Step 2 — Act:** Calls `web_search("NVIDIA stock price today")`.

**Step 3 — Observe:** Gets back `$980`. "Now I need to calculate 15% of $980. I have a `calculator` tool."

**Step 4 — Think again:** "I'll use the calculator."

**Step 5 — Act:** Calls `calculator("980 * 0.15")`.

**Step 6 — Observe:** Gets back `$147`. "I have everything I need."

**Step 7 — Answer:** "NVIDIA's stock is ~$980. 15% of that is $147."

The agent made its own plan. You didn't tell it any of those steps.
You just asked a question.

This loop is called the **ReAct loop** (short for Reason + Act). It's how
almost every serious AI agent works.

> **Key Takeaway:** Agents don't just answer. They think, act, check their work, and loop until they're done. That's what makes them powerful.

---

## Lesson 5: Your First Real Agent

Let's build something real.

We're going to build a research agent from scratch. One line at a time.
I'll explain every single thing.

```python
# Line 1: Bring in the tools we need.
# "from X import Y" means: "go get Y from the otterflow library."
from otterflow import Agent
from otterflow.tools import web_search
```

`Agent` is the thing we're building. `web_search` is a tool that lets
the agent look things up online.

```python
# Line 2: Create the agent.
agent = Agent(
    name="Researcher",   # A name. Just for you to recognize it.
    role="You are a research analyst. Search the web and write clear summaries.",
    tools=[web_search],  # The tools it's allowed to use. A list.
    verbose=True,        # Print what the agent is doing as it works.
)
```

`name` — What to call it. Doesn't affect behavior.

`role` — This is the instruction. It shapes how the agent thinks and talks.
The clearer your role, the better the results.

`tools` — A list of tools it can use. If it's not in this list, the agent
can't use it. You're in control.

`verbose=True` — This makes the agent show its work as it goes.
Great for learning. You'll see every tool call it makes.

```python
# Line 3: Give it a task and print the result.
result = agent.run("What are the top 3 AI chip companies in 2025?")
print(result)
```

`agent.run(...)` — This starts the agent. You hand it your question.
It goes off, thinks, searches, and comes back with an answer.

The full thing, together:

```python
from otterflow import Agent
from otterflow.tools import web_search

agent = Agent(
    name="Researcher",
    role="You are a research analyst. Search the web and write clear summaries.",
    tools=[web_search],
    verbose=True,
)

result = agent.run("What are the top 3 AI chip companies in 2025?")
print(result)
```

That's 10 lines.

You now have a working AI research agent.

```
        __   __
       /  \ /  \
      ( ^   ^ )
       \  ~~~  /     "You just built an agent.
        |     |       Let that sink in."
```

> **Key Takeaway:** An agent needs three things: a role (instructions), tools (abilities), and a task (a goal). Give it those three things and it gets to work.

---

## Lesson 6: Chaining Agents Together

One agent is powerful.

Multiple agents working together? That's something else entirely.

Think about a company. The CEO doesn't do everything herself.
She has a team. A researcher. A writer. An analyst. An editor.

She tells the researcher: *"Go find the data."*
She tells the analyst: *"Turn that data into insights."*
She tells the writer: *"Write the report."*

Each person is great at one thing. Together, they produce something
no single person could do alone.

**OtterFlow lets you build this.**

**Option 1: The Pipe Operator `|`**

This is the simplest way to chain agents.
The output of one agent becomes the input of the next.

```python
from otterflow import Agent

researcher = Agent(name="Researcher", role="Find and summarize information on any topic.")
writer     = Agent(name="Writer",     role="Turn raw research into a polished blog post.")
editor     = Agent(name="Editor",     role="Tighten the writing. Cut anything weak.")

pipeline = researcher | writer | editor

result = pipeline.run("The future of nuclear energy.")
print(result)
```

```
  You
   │
   ▼
 Researcher ──→ "Here's the raw research..."
                     │
                     ▼
                  Writer ──→ "Here's a draft post..."
                                   │
                                   ▼
                                Editor ──→ "Here's the final version."
                                               │
                                               ▼
                                             You
```

Each agent gets the previous agent's full output as its input.
Simple. Powerful. Five lines of code.

**Option 2: The `spawn()` Method**

Sometimes you want an orchestrator — a boss agent who decides
*which* specialist to call and *when*.

This is smarter. The orchestrator uses tools, memory, and judgment
to coordinate the team.

```python
from otterflow import Agent

boss    = Agent(name="Boss",    role="Coordinate your team to answer complex questions.")
digger  = Agent(name="Digger",  role="Search the web for facts and data.")
numbers = Agent(name="Numbers", role="Do calculations and data analysis.")

boss.spawn(digger)   # Digger is now a tool the boss can call.
boss.spawn(numbers)  # Numbers is now a tool the boss can call.

result = boss.run("What's Tesla's revenue growth rate over the last 3 years?")
print(result)
```

```
                  ┌──────────┐
                  │   Boss   │   "I need data first."
                  └────┬─────┘
                       │
           ┌───────────┴────────────┐
           │                        │
           ▼                        ▼
     ┌──────────┐             ┌──────────┐
     │  Digger  │             │ Numbers  │
     │ (search) │             │  (math)  │
     └────┬─────┘             └────┬─────┘
          │                        │
          └───────────┬────────────┘
                      │
                      ▼
                  ┌──────────┐
                  │   Boss   │   "Here's the answer."
                  └──────────┘
```

`spawn()` turns an agent into a tool that another agent can call.
The boss decides when to use each specialist. You just watch it work.

> **Key Takeaway:** Chain agents with `|` when the work is linear (A → B → C). Use `spawn()` when you want a boss agent to decide who does what.

---

## Lesson 7: The Things That Make a Great Agent

You've learned the fundamentals. Now here are the three things that
separate okay agents from ones that actually work.

### 1. A great role is everything.

The `role` is the most important part of your agent.
Think of it as the job description.

Vague role → vague results.
Sharp role → sharp results.

```
  BAD:   role="You are helpful."

  GOOD:  role="You are a senior financial analyst.
               When asked about a company, you:
               1. Look up their revenue and growth rate.
               2. Compare it to competitors.
               3. Give a clear buy / hold / sell opinion.
               Be direct. Use numbers. No fluff."
```

The more specific you are, the smarter the agent seems.

### 2. Only give an agent the tools it needs.

Don't give a writer agent a `run_python` tool.
Don't give a calculator agent a `web_search` tool.

Tools are choices. The more choices an agent has, the more chances it has
to make a wrong one. Keep the toolkit tight.

### 3. Use memory for multi-turn conversations.

If your agent needs to remember what the user said earlier,
use a `Memory` object. If it's a one-shot task, you don't need it.

```python
# One-shot task — no memory needed.
agent.run("Summarize this document: ...")

# Multi-turn conversation — use memory.
memory = Memory()
agent = Agent(name="Coach", role="...", memory=memory)
agent.run("My name is Jordan and my goal is to get 10k followers.")
agent.run("What should my first post be about?")  # Agent remembers Jordan.
```

> **Key Takeaway:** Great agents come from clear roles, tight toolkits, and memory when you need it. Nail the role first. Everything else follows.

---

## What to Build Next

```
        __   __
       /  \ /  \
      ( o   o )
       \  ~~~  /     "Okay. You know the game now.
        | ^^^ |       Go build something."
        |     |
```

You've learned the fundamentals. Here's where to go next.

**Start with the examples in `examples/quickstart.py`.** Each one is short
and focused. Run them. Change things. Break them. That's how you learn.

```
  Example 1 → Build a minimal agent. (Start here.)
  Example 2 → Add tools. Make it search the web.
  Example 3 → Use the pre-built ResearchAgent.
  Example 4 → Use the ContentCreatorAgent. Give it a LinkedIn post.
  Example 5 → Build a multi-agent pipeline. Watch them work together.
  Example 6 → Make your own @tool from scratch.
  Example 7 → Add memory. Have a real conversation.
```

**Then look at the pre-built agents in `otterflow/agents/`.** We already built:

- `ResearchAgent` — searches and writes research reports.
- `EmailAgent` — drafts and rewrites emails with a specific tone.
- `DataAnalystAgent` — reads files and surfaces business insights.
- `CompetitiveIntelAgent` — builds battle cards on any competitor.
- `ContentCreatorAgent` — coaches you on LinkedIn, Twitter, or newsletters.
- `BusinessIntelPipeline` — a full multi-agent orchestrator. Drop a question in. Get a report back.

Use them as templates. Take them apart. Rewrite the `role`. Add tools.
Make them yours.

---

**The best way to learn this is to build something you actually want.**

Pick a real problem. Something you do manually today that you wish was automatic.
Then build an agent to do it.

You have everything you need.

```
  ┌─────────────────────────────────────────┐
  │                                         │
  │   Goal → Agent → Tools → Memory → Done  │
  │                                         │
  └─────────────────────────────────────────┘
```

Go build. — *Ollie* 🦦

---

*OtterFlow is open source. Found a bug? Have an idea?*
*Open an issue at github.com/tabato/otterflow*
