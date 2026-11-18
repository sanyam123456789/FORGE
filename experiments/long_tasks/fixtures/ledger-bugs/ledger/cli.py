"""Command line for the ledger: ``python -m ledger.cli <command> ...``.

Commands:
    init LEDGER [--currency USD]                  create an empty ledger file
    account LEDGER NAME [--kind K] [--currency C] [--opening AMOUNT]
    import LEDGER CSV [--account NAME] [--month-first]
    categorize LEDGER [--rules FILE] [--overwrite]
    balance LEDGER
    report LEDGER [--month YYYY-MM]
    categories LEDGER [--month YYYY-MM]
    budget LEDGER BUDGETS_JSON --month YYYY-MM
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from ledger import budget as budget_mod
from ledger import parser as csv_parser
from ledger import reports, rules, storage
from ledger.models import Account, Ledger, ValidationError
from ledger.money import MoneyError, parse_amount


def _month_range(text: str) -> tuple[date, date]:
    try:
        year, month = (int(p) for p in text.split("-"))
        start = date(year, month, 1)
    except ValueError as exc:
        raise SystemExit(f"error: --month must look like 2024-03, got {text!r}") from exc
    end = date(year + (month == 12), month % 12 + 1, 1)
    return start, date.fromordinal(end.toordinal() - 1)


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.ledger)
    if path.exists():
        print(f"error: {path} already exists", file=sys.stderr)
        return 1
    storage.save_ledger(Ledger(base_currency=args.currency), path)
    print(f"created {path}")
    return 0


def cmd_account(args: argparse.Namespace) -> int:
    ledger = storage.load_ledger(args.ledger)
    opening = parse_amount(args.opening) if args.opening else parse_amount("0")
    ledger.add_account(Account(args.name, args.kind, args.currency, opening))
    storage.save_ledger(ledger, args.ledger)
    print(f"added account {args.name}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    ledger = storage.load_ledger(args.ledger)
    result = csv_parser.parse_file(args.csv, account=args.account, day_first=not args.month_first)
    for error in result.errors:
        print(f"warning: {error}", file=sys.stderr)
    fresh = csv_parser.dedupe(ledger.transactions, result.transactions)
    ledger.add_transactions(fresh)
    storage.save_ledger(ledger, args.ledger)
    print(f"{result.summary()}; {len(fresh)} new")
    return 0 if result.ok else 1


def cmd_categorize(args: argparse.Namespace) -> int:
    ledger = storage.load_ledger(args.ledger)
    rule_list = (
        rules.parse_rules(Path(args.rules).read_text(encoding="utf-8"))
        if args.rules
        else None
    )
    ledger.transactions, changed = rules.apply_rules(
        ledger.transactions, rule_list, overwrite=args.overwrite
    )
    storage.save_ledger(ledger, args.ledger)
    print(f"categorised {changed} transaction(s)")
    return 0


def cmd_balance(args: argparse.Namespace) -> int:
    ledger = storage.load_ledger(args.ledger)
    print(reports.render_balances(reports.balance_by_account(ledger), ledger))
    return 0


def _selected(ledger: Ledger, month: str | None):
    if not month:
        return ledger.transactions
    start, end = _month_range(month)
    return ledger.transactions_for(start=start, end=end)


def cmd_report(args: argparse.Namespace) -> int:
    ledger = storage.load_ledger(args.ledger)
    txns = _selected(ledger, args.month)
    print(reports.render_monthly(reports.monthly_summary(txns), ledger.base_currency))
    top = reports.top_expenses(txns, 5)
    if top:
        print("\nlargest expenses:")
        for t in top:
            print(f"  {t.date}  {t.amount:>10}  {t.category:<14} {t.memo}")
    return 0


def cmd_categories(args: argparse.Namespace) -> int:
    ledger = storage.load_ledger(args.ledger)
    totals = reports.totals_by_category(_selected(ledger, args.month))
    print(reports.render_categories(totals, ledger.base_currency))
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    ledger = storage.load_ledger(args.ledger)
    budgets = budget_mod.load_budgets(json.loads(Path(args.budgets).read_text(encoding="utf-8")))
    start, _ = _month_range(args.month)
    statuses = budget_mod.check_budgets(ledger.transactions, budgets, start.year, start.month)
    print(budget_mod.render_budget_report(statuses, ledger.base_currency))
    return 1 if any(s.level == budget_mod.LEVEL_OVER for s in statuses) else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ledger", description="Personal finance ledger")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init")
    s.add_argument("ledger")
    s.add_argument("--currency", default="USD")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("account")
    s.add_argument("ledger")
    s.add_argument("name")
    s.add_argument("--kind", default="checking")
    s.add_argument("--currency", default="USD")
    s.add_argument("--opening", default=None)
    s.set_defaults(func=cmd_account)

    s = sub.add_parser("import")
    s.add_argument("ledger")
    s.add_argument("csv")
    s.add_argument("--account", default="main")
    s.add_argument("--month-first", action="store_true")
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("categorize")
    s.add_argument("ledger")
    s.add_argument("--rules", default=None)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_categorize)

    s = sub.add_parser("balance")
    s.add_argument("ledger")
    s.set_defaults(func=cmd_balance)

    s = sub.add_parser("report")
    s.add_argument("ledger")
    s.add_argument("--month", default=None)
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("categories")
    s.add_argument("ledger")
    s.add_argument("--month", default=None)
    s.set_defaults(func=cmd_categories)

    s = sub.add_parser("budget")
    s.add_argument("ledger")
    s.add_argument("budgets")
    s.add_argument("--month", required=True)
    s.set_defaults(func=cmd_budget)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (storage.StorageError, ValidationError, MoneyError, budget_mod.BudgetError,
            rules.RuleError, csv_parser.ParseError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
