def tally(items):
    """Count how many times each item appears in ``items``.

    Returns a dict mapping item -> count. See tally() for details.
    """
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts
