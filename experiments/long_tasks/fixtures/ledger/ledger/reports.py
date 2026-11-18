"""Reports: balances, category totals, monthly summaries and text tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from ledger.models import Ledger, Transaction
from ledger.money import format_amount, percent, sum_amounts


@dataclass
class MonthSummary:
    month: str  # YYYY-MM
    income: Decimal
    expenses: Decimal  # negative or zero
    count: int

    @property
    def net(self) -> Decimal:
        return self.income + self.expenses

    @property
    def savings_rate(self) -> Decimal | None:
        """Share of income kept, in percent (None without income)."""
        return percent(self.net, self.income)


def _month_key(day: date) -> str:
    return day.strftime("%Y-%m")


def balance_by_account(ledger: Ledger) -> dict[str, Decimal]:
    """Opening balance plus all transactions, per account."""
    balances = {name: acc.opening_balance for name, acc in ledger.accounts.items()}
    for txn in ledger.transactions:
        balances[txn.account] = balances.get(txn.account, Decimal("0")) + txn.amount
    return balances


def totals_by_category(
    transactions: Sequence[Transaction], *, expenses_only: bool = False
) -> dict[str, Decimal]:
    """Sum per category, largest spending first (most negative first)."""
    totals: dict[str, Decimal] = {}
    for txn in transactions:
        if expenses_only and not txn.is_expense:
            continue
        totals[txn.category] = totals.get(txn.category, Decimal("0")) + txn.amount
    return dict(sorted(totals.items(), key=lambda item: (item[1], item[0])))


def monthly_summary(transactions: Sequence[Transaction]) -> list[MonthSummary]:
    """One summary per month that has transactions, oldest first."""
    months: dict[str, list[Transaction]] = {}
    for txn in transactions:
        months.setdefault(_month_key(txn.date), []).append(txn)
    out = []
    for month in sorted(months):
        txns = months[month]
        out.append(
            MonthSummary(
                month=month,
                income=sum_amounts(t.amount for t in txns if t.is_income),
                expenses=sum_amounts(t.amount for t in txns if t.is_expense),
                count=len(txns),
            )
        )
    return out


def top_expenses(transactions: Sequence[Transaction], n: int = 5) -> list[Transaction]:
    """The *n* largest expenses, largest first; ties by date."""
    expenses = [t for t in transactions if t.is_expense]
    return sorted(expenses, key=lambda t: (t.amount, t.date))[:n]


def average_monthly_spending(transactions: Sequence[Transaction]) -> Decimal | None:
    """Mean of monthly expenses over months that have transactions."""
    summaries = monthly_summary(transactions)
    if not summaries:
        return None
    total = sum_amounts(s.expenses for s in summaries)
    return (total / len(summaries)).quantize(Decimal("0.01"))


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    align: str | None = None,
) -> str:
    """A plain-text table. *align* has one letter per column: l or r."""
    align = align or "l" + "r" * (len(headers) - 1)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            parts.append(cell.rjust(widths[i]) if align[i] == "r" else cell.ljust(widths[i]))
        return "  ".join(parts).rstrip()

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def render_monthly(summaries: Sequence[MonthSummary], currency: str = "USD") -> str:
    rows = []
    for s in summaries:
        rate = s.savings_rate
        rows.append(
            [
                s.month,
                format_amount(s.income, currency),
                format_amount(s.expenses, currency),
                format_amount(s.net, currency),
                f"{rate}%" if rate is not None else "-",
                str(s.count),
            ]
        )
    return render_table(["month", "income", "expenses", "net", "saved", "txns"], rows)


def render_categories(totals: dict[str, Decimal], currency: str = "USD") -> str:
    spent = sum_amounts(v for v in totals.values() if v < 0)
    rows = []
    for category, amount in totals.items():
        share = percent(amount, spent) if amount < 0 else None
        rows.append(
            [
                category,
                format_amount(amount, currency),
                f"{share}%" if share is not None else "-",
            ]
        )
    return render_table(["category", "total", "share"], rows)


def render_balances(balances: dict[str, Decimal], ledger: Ledger) -> str:
    rows = []
    for name in sorted(balances):
        account = ledger.accounts.get(name)
        currency = account.currency if account else ledger.base_currency
        kind = account.kind if account else "?"
        rows.append([name, kind, format_amount(balances[name], currency)])
    return render_table(["account", "kind", "balance"], rows, align="llr")
