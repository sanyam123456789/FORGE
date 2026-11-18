"""Categorisation rules: regular expressions on the memo decide the category.

A rules file has one rule per line::

    # priority | pattern | category
    20 | rent|landlord | housing
    10 | grocer|supermarket | groceries

Higher priority wins. Patterns are case-insensitive regular expressions
searched anywhere in the memo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ledger.models import UNCATEGORIZED, Transaction


class RuleError(ValueError):
    """Raised for a rules file line that cannot be parsed."""


@dataclass
class Rule:
    priority: int
    pattern: str
    category: str
    _regex: re.Pattern | None = field(default=None, repr=False, compare=False)

    def regex(self) -> re.Pattern:
        if self._regex is None:
            try:
                self._regex = re.compile(self.pattern, re.IGNORECASE)
            except re.error as exc:
                raise RuleError(f"bad pattern {self.pattern!r}: {exc}") from exc
        return self._regex

    def matches(self, txn: Transaction) -> bool:
        return bool(self.regex().search(txn.memo or ""))


DEFAULT_RULES: list[Rule] = [
    Rule(30, r"salary|payroll|wages", "income"),
    Rule(20, r"\brent\b|landlord", "housing"),
    Rule(10, r"grocer|supermarket|whole foods|aldi|lidl|trader joe", "groceries"),
    Rule(10, r"uber|lyft|taxi|metro|transit|train ticket", "transport"),
    Rule(10, r"netflix|spotify|hulu|disney\+|youtube premium", "subscriptions"),
    Rule(10, r"restaurant|cafe|coffee|pizza|burger|sushi", "dining"),
    Rule(10, r"electric|water bill|gas bill|internet|mobile plan", "utilities"),
    Rule(5, r"pharmacy|doctor|dentist|clinic", "health"),
]


def sorted_rules(rules: list[Rule]) -> list[Rule]:
    """Rules in the order they are tried: priority high to low, then file order."""
    return [r for _, r in sorted(enumerate(rules), key=lambda p: (-p[1].priority, p[0]))]


def categorize(txn: Transaction, rules: list[Rule]) -> str | None:
    """The category of the first matching rule, or None."""
    for rule in sorted_rules(rules):
        if rule.matches(txn):
            return rule.category
    return None


def apply_rules(
    transactions: list[Transaction],
    rules: list[Rule] | None = None,
    *,
    overwrite: bool = False,
) -> tuple[list[Transaction], int]:
    """Categorise transactions. Returns ``(new list, number changed)``.

    Only uncategorised transactions are changed unless *overwrite* is set.
    The input list and its transactions are not modified.
    """
    rules = DEFAULT_RULES if rules is None else rules
    out: list[Transaction] = []
    changed = 0
    for txn in transactions:
        if not overwrite and txn.category != UNCATEGORIZED:
            out.append(txn)
            continue
        category = categorize(txn, rules)
        if category and category != txn.category:
            out.append(txn.with_category(category))
            changed += 1
        else:
            out.append(txn)
    return out, changed


def parse_rules(text: str) -> list[Rule]:
    """Parse rules from the ``priority | pattern | category`` format."""
    rules: list[Rule] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        first, sep, rest = stripped.partition("|")
        pattern, sep2, category = rest.rpartition("|")
        if not sep or not sep2:
            raise RuleError(f"line {line_no}: expected 'priority | pattern | category'")
        try:
            priority = int(first.strip())
        except ValueError as exc:
            raise RuleError(f"line {line_no}: priority must be a whole number") from exc
        pattern, category = pattern.strip(), category.strip().lower()
        if not pattern or not category:
            raise RuleError(f"line {line_no}: pattern and category must not be empty")
        rule = Rule(priority, pattern, category)
        rule.regex()  # fail early on a bad pattern
        rules.append(rule)
    return rules


def format_rules(rules: list[Rule]) -> str:
    return "\n".join(f"{r.priority} | {r.pattern} | {r.category}" for r in rules) + "\n"
