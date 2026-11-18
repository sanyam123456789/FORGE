"""Monthly budgets per category, with warnings before a budget runs out."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from ledger.models import Transaction
from ledger.money import format_amount, parse_amount, sum_amounts
from ledger.reports import render_table

LEVEL_OK = "ok"
LEVEL_WARNING = "warning"
LEVEL_OVER = "over"


class BudgetError(ValueError):
    """Raised for an invalid budget definition."""


@dataclass
class Budget:
    category: str
    monthly_limit: Decimal  # positive
    warn_at: Decimal = Decimal("0.8")  # share of the limit that triggers a warning

    def validate(self) -> None:
        if not self.category:
            raise BudgetError("budget category is empty")
        if self.monthly_limit <= 0:
            raise BudgetError(f"budget for {self.category!r}: limit must be positive")
        if not Decimal("0") < self.warn_at <= Decimal("1"):
            raise BudgetError(f"budget for {self.category!r}: warn_at must be in (0, 1]")


@dataclass
class BudgetStatus:
    category: str
    limit: Decimal
    spent: Decimal  # positive: how much was spent
    level: str

    @property
    def remaining(self) -> Decimal:
        return self.limit - self.spent

    @property
    def used(self) -> Decimal:
        """Share of the limit used, e.g. 0.75."""
        return (self.spent / self.limit).quantize(Decimal("0.01"))


def spent_in_month(
    transactions: Sequence[Transaction], category: str, year: int, month: int
) -> Decimal:
    """Total spending (as a positive number) in *category* in one month.

    Refunds (positive amounts) in the category reduce the spending.
    """
    in_month = [
        t for t in transactions
        if t.category == category and t.date.year == year and t.date.month == month
    ]
    return -sum_amounts(t.amount for t in in_month)


def status_for(budget: Budget, spent: Decimal) -> BudgetStatus:
    if spent > budget.monthly_limit:
        level = LEVEL_OVER
    elif spent >= budget.monthly_limit * budget.warn_at:
        level = LEVEL_WARNING
    else:
        level = LEVEL_OK
    return BudgetStatus(budget.category, budget.monthly_limit, spent, level)


def check_budgets(
    transactions: Sequence[Transaction],
    budgets: Sequence[Budget],
    year: int,
    month: int,
) -> list[BudgetStatus]:
    """Status of every budget for one month, most used first."""
    statuses = [
        status_for(b, spent_in_month(transactions, b.category, year, month))
        for b in budgets
    ]
    return sorted(statuses, key=lambda s: (-(s.spent / s.limit), s.category))


def load_budgets(data: dict) -> list[Budget]:
    """Budgets from ``{"groceries": "400", "dining": {"limit": "150", "warn_at": "0.9"}}``."""
    budgets = []
    for category, spec in data.items():
        if isinstance(spec, dict):
            limit = parse_amount(str(spec.get("limit", "")))
            warn_at = Decimal(str(spec.get("warn_at", "0.8")))
        else:
            limit = parse_amount(str(spec))
            warn_at = Decimal("0.8")
        budget = Budget(category.strip().lower(), limit, warn_at)
        budget.validate()
        budgets.append(budget)
    return budgets


def render_budget_report(statuses: Sequence[BudgetStatus], currency: str = "USD") -> str:
    marks = {LEVEL_OK: "", LEVEL_WARNING: "!", LEVEL_OVER: "!!"}
    rows = [
        [
            s.category,
            format_amount(s.limit, currency),
            format_amount(s.spent, currency),
            format_amount(s.remaining, currency),
            f"{int(s.used * 100)}%",
            marks[s.level],
        ]
        for s in statuses
    ]
    return render_table(
        ["budget", "limit", "spent", "left", "used", ""], rows, align="lrrrrl"
    )
