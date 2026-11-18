"""Ledger data model: accounts, transactions and the ledger that holds them."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal

ACCOUNT_KINDS = ("checking", "savings", "credit", "cash")
UNCATEGORIZED = "uncategorized"


class ValidationError(ValueError):
    """Raised when an account or transaction is not valid."""


@dataclass
class Account:
    """A bank account, card or cash wallet."""

    name: str
    kind: str = "checking"
    currency: str = "USD"
    opening_balance: Decimal = Decimal("0")

    def validate(self) -> None:
        if not self.name or not self.name.strip():
            raise ValidationError("account name is empty")
        if self.kind not in ACCOUNT_KINDS:
            raise ValidationError(
                f"account {self.name!r}: kind must be one of {', '.join(ACCOUNT_KINDS)}"
            )
        if len(self.currency) != 3 or not self.currency.isalpha() or not self.currency.isupper():
            raise ValidationError(
                f"account {self.name!r}: currency must be a 3-letter code like USD"
            )


@dataclass
class Transaction:
    """One booked transaction. Expenses are negative, income positive."""

    date: date
    amount: Decimal
    account: str
    category: str = UNCATEGORIZED
    memo: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def is_expense(self) -> bool:
        return self.amount < 0

    @property
    def is_income(self) -> bool:
        return self.amount > 0

    @property
    def month(self) -> str:
        """The transaction's month as ``YYYY-MM``."""
        return self.date.strftime("%Y-%m")

    def validate(self, accounts: dict[str, Account] | None = None) -> None:
        if not isinstance(self.date, date):
            raise ValidationError("transaction date must be a date")
        if not isinstance(self.amount, Decimal):
            raise ValidationError("transaction amount must be a Decimal")
        if self.amount == 0:
            raise ValidationError("transaction amount must not be zero")
        if not self.account:
            raise ValidationError("transaction has no account")
        if accounts is not None and self.account not in accounts:
            raise ValidationError(f"unknown account {self.account!r}")
        if not self.category or not self.category.strip():
            raise ValidationError("transaction category is empty")
        for tag in self.tags:
            if not tag or " " in tag:
                raise ValidationError(f"invalid tag {tag!r} (tags are single words)")

    def with_category(self, category: str) -> "Transaction":
        return replace(self, category=category)

    def has_tag(self, tag: str) -> bool:
        return tag.lower() in (t.lower() for t in self.tags)


@dataclass
class Ledger:
    """All accounts and transactions of one person or household."""

    accounts: dict[str, Account] = field(default_factory=dict)
    transactions: list[Transaction] = field(default_factory=list)
    base_currency: str = "USD"

    def add_account(self, account: Account) -> None:
        account.validate()
        if account.name in self.accounts:
            raise ValidationError(f"account {account.name!r} already exists")
        self.accounts[account.name] = account

    def add_transaction(self, txn: Transaction) -> None:
        txn.validate(self.accounts)
        self.transactions.append(txn)

    def add_transactions(self, txns: list[Transaction]) -> int:
        """Add several transactions; all are validated before any is added."""
        for txn in txns:
            txn.validate(self.accounts)
        self.transactions.extend(txns)
        return len(txns)

    def transactions_for(
        self,
        *,
        account: str | None = None,
        category: str | None = None,
        start: date | None = None,
        end: date | None = None,
        tag: str | None = None,
    ) -> list[Transaction]:
        """Transactions matching every given filter (``end`` is inclusive)."""
        out = []
        for txn in self.transactions:
            if account is not None and txn.account != account:
                continue
            if category is not None and txn.category != category:
                continue
            if start is not None and txn.date < start:
                continue
            if end is not None and txn.date > end:
                continue
            if tag is not None and not txn.has_tag(tag):
                continue
            out.append(txn)
        return sorted(out, key=lambda t: (t.date, t.account))

    def months(self) -> list[str]:
        """Every ``YYYY-MM`` that has at least one transaction, in order."""
        return sorted({txn.month for txn in self.transactions})

    def categories(self) -> list[str]:
        return sorted({txn.category for txn in self.transactions})
