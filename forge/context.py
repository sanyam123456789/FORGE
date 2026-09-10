"""
forge.context — Interaction-context management layer.

In an LLM-based coding agent the "context" is the conversation history
passed to the LLM on every turn.  Context management is the second major
research variable in FORGE:

  Research Arm B: Raw/full interaction history  vs  managed context
                  ↓                                   ↓
          (all messages kept)            (summarisation / pruning)

Status
------
- ``ConversationContext`` holds the ordered message history for one run.
- ``ContextStrategy`` decides which messages are actually sent to the LLM on
  a given turn.  Step 2 implements ``RawContextStrategy`` only (send the full
  history unchanged).  A future ``ManagedContextStrategy`` (pruning /
  summarisation) plugs in at the same seam without touching the agent loop.
- Token counting is deliberately a character-count proxy for now; accurate
  tokeniser-based counting is a later concern.

Design
------
- ``ConversationContext`` holds an ordered list of Messages and exposes
  ``add_message()`` / ``get_messages()`` (the raw, unmodified history).
- The agent runtime holds a ``ContextStrategy`` and calls
  ``strategy.prepare(context)`` to obtain the message list for each LLM call.
  This is the single integration point for the "raw vs managed" research arm.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
        """Return the full, unmodified message history.

        Any managed-context transformation is applied by a ``ContextStrategy``
        (see below), not here — this method always returns the raw history.
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
        """Sum of character lengths across all message contents.

        Rough proxy for context size.  Tool-call argument payloads are not
        included.  Accurate token counting is deferred until it is actually
        needed by an experiment.
        """
        return sum(len(m.content) for m in self.messages)


# ---------------------------------------------------------------------------
# Context strategies (research arm B seam)
# ---------------------------------------------------------------------------


class ContextStrategy(ABC):
    """Decides which messages are sent to the LLM for a given turn.

    The agent runtime depends on this abstraction, never on a concrete
    strategy.  Step 2 ships ``RawContextStrategy`` only.  Managed strategies
    (last-k, summarisation, ...) are a later research intervention and must
    slot in here without changes to the agent loop.
    """

    #: Stable identifier recorded in run traces.
    name: str = "abstract"

    @abstractmethod
    def prepare(self, context: "ConversationContext") -> list[Message]:
        """Return the message list to send to the LLM this turn."""


class RawContextStrategy(ContextStrategy):
    """Baseline: send the full interaction history, unmodified."""

    name = "raw"

    def prepare(self, context: "ConversationContext") -> list[Message]:
        return context.get_messages()
