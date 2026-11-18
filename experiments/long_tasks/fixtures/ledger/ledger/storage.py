"""Save and load a ledger as JSON.

Format version 2 stores accounts as objects. Version 1 files (accounts as a
plain list of names, all USD checking accounts) are still loaded.
"""

from __future__ import annotations

import json
import os
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from ledger.models import UNCATEGORIZED, Account, Ledger, Transaction

FORMAT_VERSION = 2


class StorageError(ValueError):
    """Raised when a ledger file cannot be read."""


def transaction_to_dict(txn: Transaction) -> dict[str, Any]:
    data: dict[str, Any] = {
        "date": txn.date.isoformat(),
        "amount": str(txn.amount),
        "account": txn.account,
        "category": txn.category,
    }
    if txn.memo:
        data["memo"] = txn.memo
    if txn.tags:
        data["tags"] = list(txn.tags)
    return data


def transaction_from_dict(data: dict[str, Any]) -> Transaction:
    try:
        return Transaction(
            date=date.fromisoformat(data["date"]),
            amount=Decimal(str(data["amount"])),
            account=data["account"],
            category=data.get("category") or UNCATEGORIZED,
            memo=data.get("memo", ""),
            tags=list(data.get("tags", [])),
        )
    except (KeyError, ValueError, ArithmeticError) as exc:
        raise StorageError(f"bad transaction {data!r}: {exc}") from exc


def ledger_to_dict(ledger: Ledger) -> dict[str, Any]:
    return {
        "version": FORMAT_VERSION,
        "base_currency": ledger.base_currency,
        "accounts": [
            {
                "name": a.name,
                "kind": a.kind,
                "currency": a.currency,
                "opening_balance": str(a.opening_balance),
            }
            for a in ledger.accounts.values()
        ],
        "transactions": [transaction_to_dict(t) for t in ledger.transactions],
    }


def ledger_from_dict(data: dict[str, Any]) -> Ledger:
    version = data.get("version", 1)
    if version not in (1, FORMAT_VERSION):
        raise StorageError(f"unsupported ledger format version {version!r}")
    ledger = Ledger(base_currency=data.get("base_currency", "USD"))
    for entry in data.get("accounts", []):
        if version == 1:
            account = Account(name=str(entry))
        else:
            account = Account(
                name=entry["name"],
                kind=entry.get("kind", "checking"),
                currency=entry.get("currency", "USD"),
                opening_balance=Decimal(str(entry.get("opening_balance", "0"))),
            )
        ledger.add_account(account)
    for entry in data.get("transactions", []):
        ledger.add_transaction(transaction_from_dict(entry))
    return ledger


def save_ledger(ledger: Ledger, path: str | Path) -> None:
    """Write the ledger atomically (a crash never leaves half a file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger_to_dict(ledger), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_ledger(path: str | Path) -> Ledger:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StorageError(f"no ledger file at {path}") from exc
    except json.JSONDecodeError as exc:
        raise StorageError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise StorageError(f"{path} does not contain a ledger")
    return ledger_from_dict(data)
