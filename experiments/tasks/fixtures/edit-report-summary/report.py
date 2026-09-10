"""Tiny reporting helper."""


def summarize(rows):
    """Return a summary of the given rows.

    Each row is a dict with at least an ``"amount"`` key (a number).
    """
    return {"count": len(rows)}
