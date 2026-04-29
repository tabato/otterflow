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

    def forget(self, key: str) -> None:
        """Remove a persistent fact by key. No-op if the key doesn't exist."""
        self.facts.pop(key, None)

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

