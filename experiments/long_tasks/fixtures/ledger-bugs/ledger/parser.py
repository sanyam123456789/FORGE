"""Import transactions from bank CSV exports.

Banks name their columns differently, so headers are matched through
``HEADER_ALIASES``. An export has either one signed ``amount`` column or
separate ``debit`` / ``credit`` columns. Bad rows do not stop the import:
they are reported in ``ImportResult.errors`` with their line number.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from ledger.models import UNCATEGORIZED, Transaction
from ledger.money import MoneyError, parse_amount

#: Canonical field -> header names that mean it (compared case-insensitively).
HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date", "posted", "posting date", "transaction date", "booking date"),
    "amount": ("amount", "value", "sum"),
    "memo": ("memo", "description", "details", "narrative", "payee"),
    "category": ("category",),
    "account": ("account",),
    "debit": ("debit", "withdrawal", "money out"),
    "credit": ("credit", "deposit", "money in"),
    "tags": ("tags", "labels"),
}

#: Date formats tried in order. Day-first formats come first unless
#: ``day_first=False`` is passed.
DAY_FIRST_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%Y/%m/%d", "%d-%m-%Y")
MONTH_FIRST_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m.%d.%Y", "%Y/%m/%d", "%m-%d-%Y")


class ParseError(ValueError):
    """Raised for a value that cannot be parsed."""


@dataclass
class ImportResult:
    transactions: list[Transaction] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        text = f"{len(self.transactions)} imported"
        if self.skipped:
            text += f", {self.skipped} blank skipped"
        if self.errors:
            text += f", {len(self.errors)} with errors"
        return text


def parse_date(text: str, *, day_first: bool = True) -> date:
    """Parse a date in any supported format (ISO always works)."""
    raw = (text or "").strip()
    if not raw:
        raise ParseError("date is empty")
    formats = DAY_FIRST_FORMATS if day_first else MONTH_FIRST_FORMATS
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ParseError(f"unrecognised date {raw!r}")


def map_headers(fieldnames: list[str] | None) -> dict[str, str]:
    """Map canonical field names to the CSV's actual header names."""
    mapping: dict[str, str] = {}
    for header in fieldnames or []:
        key = (header or "").strip().lower()
        for canonical, aliases in HEADER_ALIASES.items():
            if key in aliases and canonical not in mapping:
                mapping[canonical] = header
    if "date" not in mapping:
        raise ParseError("no date column (expected one of: " + ", ".join(HEADER_ALIASES["date"]) + ")")
    if "amount" not in mapping and not ("debit" in mapping or "credit" in mapping):
        raise ParseError("no amount column and no debit/credit columns")
    return mapping


def _row_amount(row: dict[str, str], mapping: dict[str, str]) -> Decimal:
    if "amount" in mapping:
        return parse_amount(row.get(mapping["amount"]))
    debit = (row.get(mapping.get("debit", ""), "") or "").strip()
    credit = (row.get(mapping.get("credit", ""), "") or "").strip()
    total = Decimal("0")
    if debit:
        total -= abs(parse_amount(debit))
    if credit:
        total += abs(parse_amount(credit))
    return total


def _row_tags(value: str | None) -> list[str]:
    if not value:
        return []
    parts = value.replace(",", ";").split(";")
    return [p.strip().lower() for p in parts if p.strip()]


def parse_csv(
    text: str,
    *,
    account: str = "main",
    default_category: str = UNCATEGORIZED,
    day_first: bool = True,
    delimiter: str | None = None,
) -> ImportResult:
    """Parse CSV *text* into transactions for *account*.

    The delimiter is sniffed (comma, semicolon or tab) unless given. A row
    whose amount is zero, or that is completely blank, is skipped.
    """
    result = ImportResult()
    if not text.strip():
        return result
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;\t").delimiter
        except csv.Error:
            delimiter = ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    try:
        mapping = map_headers(reader.fieldnames)
    except ParseError as exc:
        result.errors.append(f"header: {exc}")
        return result

    for line_no, row in enumerate(reader, start=2):
        if not any((v or "").strip() for v in row.values() if isinstance(v, str)):
            result.skipped += 1
            continue
        try:
            when = parse_date(row.get(mapping["date"], ""), day_first=day_first)
            amount = _row_amount(row, mapping)
        except (ParseError, MoneyError) as exc:
            result.errors.append(f"line {line_no}: {exc}")
            continue
        if amount == 0:
            result.skipped += 1
            continue
        memo = (row.get(mapping["memo"], "") or "").strip() if "memo" in mapping else ""
        category = (
            (row.get(mapping["category"], "") or "").strip().lower()
            if "category" in mapping
            else ""
        )
        row_account = (
            (row.get(mapping["account"], "") or "").strip() if "account" in mapping else ""
        )
        result.transactions.append(
            Transaction(
                date=when,
                amount=amount,
                account=row_account or account,
                category=category or default_category,
                memo=" ".join(memo.split()),
                tags=_row_tags(row.get(mapping["tags"]) if "tags" in mapping else None),
            )
        )
    return result


def parse_file(path: str | Path, **kwargs) -> ImportResult:
    """Parse a CSV file (UTF-8, with or without a byte-order mark)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    return parse_csv(text, **kwargs)


def dedupe(existing: list[Transaction], incoming: list[Transaction]) -> list[Transaction]:
    """Drop incoming transactions already in *existing* (same date, amount,
    account and memo). Duplicates inside *incoming* are kept: two identical
    coffees on one day are two transactions."""
    seen: dict[tuple, int] = {}
    for txn in existing:
        key = (txn.date, txn.amount, txn.account, txn.memo)
        seen[key] = seen.get(key, 0) + 1
    fresh = []
    for txn in incoming:
        key = (txn.date, txn.amount, txn.account, txn.memo)
        if seen.get(key, 0) > 0:
            seen[key] -= 1
            continue
        fresh.append(txn)
    return fresh
