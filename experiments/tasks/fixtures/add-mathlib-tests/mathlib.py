"""A couple of small pure functions to write tests for."""


def is_prime(n):
    """Return True if ``n`` is a prime number, else False."""
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def factorial(n):
    """Return n! for non-negative ``n``; raise ValueError for negative input."""
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
