"""
forge.context — Interaction-context management layer.

In an LLM-based coding agent the "context" is the conversation history
passed to the LLM on every turn.  Context management is the second major
research variable in FORGE:

  Research Arm B: Raw/full interaction history  vs  managed context
                  ↓                                   ↓
          (all messages kept)            (summarisation / pruning)

Step 1 Status
-------------
This module defines the ``ConversationContext`` container and its interface.
The actual pruning/summarisation strategies are NOT implemented yet.

Design
------
- ``ConversationContext`` holds an ordered list of Messages.
- It exposes ``add_message()`` and ``get_messages()`` with an optional
  ``strategy`` parameter (future hook for managed context research arm).
- Token counting is deliberately left as a stub — accurate counting
  requires knowing the tokeniser for the active model, which is a
  Step 2/3 concern.
- The context module is intentionally separated from the agent loop so
  that different context strategies can be swapped independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forge.llm import Message
from forge.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationContext:
    """Holds the live conversation history for a single agent run.

    Attributes
    ----------
    messages:
        Ordered list of all messages exchanged so far.
    max_messages:
        Soft ceiling on history length.  When exceeded, the context
        management strategy will be applied (Step 2/3 concern).
    """

    messages: list[Message] = field(default_factory=list)
    max_messages: int = 200

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    def add_message(self, message: Message) -> None:
        """Append *message* to the conversation history.

        When the history exceeds ``max_messages``, a warning is emitted.
        Pruning will be implemented once context strategies are built.
        """
        self.messages.append(message)
        logger.debug(
            "Context: added message role=%s len=%d total_messages=%d",
            message.role,
            len(message.content),
            len(self.messages),
        )
        if len(self.messages) > self.max_messages:
            logger.warning(
                "Context history has exceeded max_messages=%d. "
                "Context pruning is not yet implemented.",
                self.max_messages,
            )

    def get_messages(self) -> list[Message]:
        """Return the current message list.

        Future: accept a ``strategy`` parameter to apply managed-context
        transformations before returning the list to the LLM.
        """
        return list(self.messages)

    def clear(self) -> None:
        """Reset the conversation history (e.g. between agent runs)."""
        self.messages.clear()
        logger.debug("Context cleared.")

    # ------------------------------------------------------------------
    # Introspection helpers (Step 1: very basic)
    # ------------------------------------------------------------------

    @property
    def message_count(self) -> int:
        """Number of messages in the current history."""
        return len(self.messages)

    @property
    def approximate_char_count(self) -> int:
        """Sum of character lengths across all messages.

        This is a rough proxy for context size.  Accurate token counting
        will be added once provider tokeniser libraries are available.
        """
        return sum(len(m.content) for m in self.messages)
