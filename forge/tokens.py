"""
forge.tokens — dependency-free token estimate.

FORGE budgets context in tokens, because that is what providers bill and
what model context windows are measured in. Real tokenizers differ per
model and would add a dependency, so this is an *estimate*: it splits text
into words, numbers, punctuation and whitespace runs, and charges each piece
roughly what a BPE tokenizer would (a short word ~1 token, a long identifier
~len/5, a run of punctuation ~len/2).

It is deterministic, which is what matters for controlled comparisons: the
same text always gets the same estimate. Provider-reported usage (from the
API response) remains the source of truth for every reported metric; this
estimate is only used to decide what fits in a context budget.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from forge.llm import Message

_PIECE_RE = re.compile(r"[A-Za-z]+|\d+|\s+|[^\sA-Za-z\d]+")

#: Fixed overhead per chat message (role markers, separators).
MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    """Estimated token count of *text* (0 for empty text)."""
    if not text:
        return 0
    total = 0
    for piece in _PIECE_RE.findall(text):
        first = piece[0]
        if first.isspace():
            # Whitespace mostly merges into the next token; long indentation
            # runs cost about one token per 4 spaces/newlines.
            total += len(piece) // 4
        elif first.isdigit():
            total += (len(piece) + 2) // 3
        elif first.isalpha():
            total += max(1, (len(piece) + 3) // 5) if len(piece) > 4 else 1
        else:
            # Punctuation runs such as '": "' or '()' usually merge.
            total += (len(piece) + 1) // 2
    return total


def estimate_message_tokens(message: Message) -> int:
    """Estimated tokens for one message, including tool-call arguments."""
    total = MESSAGE_OVERHEAD + estimate_tokens(message.content or "")
    for call in message.tool_calls:
        args = call.arguments if isinstance(call.arguments, str) else repr(call.arguments)
        total += estimate_tokens(call.name) + estimate_tokens(args)
    return total


def estimate_messages_tokens(messages: Iterable[Message]) -> int:
    """Estimated tokens for a list of messages."""
    return sum(estimate_message_tokens(m) for m in messages)
