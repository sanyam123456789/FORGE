import json
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from ledger.models import Account, Ledger, Transaction, ValidationError
from ledger.rules import Rule, RuleError, apply_rules, parse_rules
from ledger.storage import StorageError, load_ledger, save_ledger


class StorageTests(unittest.TestCase):
    def test_round_trip(self):
        ledger = Ledger(base_currency="EUR")
        ledger.add_account(Account("main", currency="EUR", opening_balance=Decimal("10.5")))
        ledger.add_transaction(
            Transaction(date(2024, 5, 1), Decimal("-4.20"), "main", "dining", "Cafe", ["work"])
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            save_ledger(ledger, path)
            loaded = load_ledger(path)
        self.assertEqual(loaded.base_currency, "EUR")
        self.assertEqual(loaded.accounts["main"].opening_balance, Decimal("10.5"))
        self.assertEqual(loaded.transactions, ledger.transactions)

    def test_version_1_file(self):
        data = {
            "accounts": ["main"],
            "transactions": [{"date": "2023-01-02", "amount": "-5", "account": "main",
                              "memo": "Old entry"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            loaded = load_ledger(path)
        self.assertEqual(loaded.accounts["main"].kind, "checking")
        self.assertEqual(loaded.transactions[0].category, "uncategorized")
        self.assertEqual(loaded.transactions[0].memo, "Old entry")

    def test_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StorageError):
                load_ledger(Path(tmp) / "missing.json")
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            with self.assertRaises(StorageError):
                load_ledger(bad)

    def test_unknown_account_rejected(self):
        ledger = Ledger()
        with self.assertRaises(ValidationError):
            ledger.add_transaction(Transaction(date(2024, 1, 1), Decimal("1"), "nope"))


class RuleTests(unittest.TestCase):
    def test_apply_default_rules(self):
        txns = [
            Transaction(date(2024, 1, 1), Decimal("-50"), "main", memo="ALDI store 42"),
            Transaction(date(2024, 1, 2), Decimal("-9.99"), "main", memo="Netflix.com"),
            Transaction(date(2024, 1, 3), Decimal("-5"), "main", "dining", memo="Aldi"),
        ]
        out, changed = apply_rules(txns)
        self.assertEqual([t.category for t in out], ["groceries", "subscriptions", "dining"])
        self.assertEqual(changed, 2)
        self.assertEqual(txns[0].category, "uncategorized")

    def test_priority(self):
        rules = [Rule(1, "coffee", "dining"), Rule(9, "work coffee", "business")]
        txn = Transaction(date(2024, 1, 1), Decimal("-3"), "main", memo="Work coffee beans")
        out, _ = apply_rules([txn], rules)
        self.assertEqual(out[0].category, "business")

    def test_parse_rules(self):
        rules = parse_rules("# comment\n20 | rent|landlord | Housing\n\n5|gym|health\n")
        self.assertEqual([(r.priority, r.pattern, r.category) for r in rules],
                         [(20, "rent|landlord", "housing"), (5, "gym", "health")])
        with self.assertRaises(RuleError):
            parse_rules("x | a | b")
        with self.assertRaises(RuleError):
            parse_rules("1 | ( | b")


if __name__ == "__main__":
    unittest.main()
