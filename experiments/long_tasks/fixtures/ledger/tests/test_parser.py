import unittest
from datetime import date
from decimal import Decimal

from ledger.models import Transaction
from ledger.parser import ParseError, dedupe, parse_csv, parse_date

SIGNED = """Date,Description,Amount,Category
2024-03-01,Whole Foods Market,-54.20,
2024-03-02,  ACME   payroll ,"2,500.00",
2024-03-03,Coffee corner,(3.80),dining
"""

DEBIT_CREDIT = """Posted;Narrative;Money Out;Money In;Tags
01/04/2024;Rent April;1200.00;;home
02/04/2024;Refund;;15.00;
03/04/2024;;;;
"""


class ParseDateTests(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(parse_date("2024-03-01"), date(2024, 3, 1))
        self.assertEqual(parse_date("01/04/2024"), date(2024, 4, 1))
        self.assertEqual(parse_date("01/04/2024", day_first=False), date(2024, 1, 4))
        self.assertEqual(parse_date("31.12.2023"), date(2023, 12, 31))

    def test_bad_date(self):
        with self.assertRaises(ParseError):
            parse_date("yesterday")


class ParseCsvTests(unittest.TestCase):
    def test_signed_amounts(self):
        result = parse_csv(SIGNED, account="checking")
        self.assertTrue(result.ok, result.errors)
        amounts = [t.amount for t in result.transactions]
        self.assertEqual(amounts, [Decimal("-54.20"), Decimal("2500.00"), Decimal("-3.80")])
        self.assertEqual(result.transactions[1].memo, "ACME payroll")
        self.assertEqual(result.transactions[2].category, "dining")
        self.assertEqual(result.transactions[0].category, "uncategorized")
        self.assertTrue(all(t.account == "checking" for t in result.transactions))

    def test_debit_credit_columns(self):
        result = parse_csv(DEBIT_CREDIT)
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(len(result.transactions), 2)
        rent, refund = result.transactions
        self.assertEqual(rent.amount, Decimal("-1200.00"))
        self.assertEqual(rent.tags, ["home"])
        self.assertEqual(refund.amount, Decimal("15.00"))
        self.assertEqual(result.skipped, 1)

    def test_bad_rows_are_reported(self):
        text = "date,amount\n2024-01-01,abc\nnot a date,5\n2024-01-02,7\n"
        result = parse_csv(text)
        self.assertEqual(len(result.transactions), 1)
        self.assertEqual(len(result.errors), 2)
        self.assertTrue(result.errors[0].startswith("line 2:"))

    def test_missing_headers(self):
        result = parse_csv("when,how much\n2024-01-01,5\n")
        self.assertFalse(result.ok)
        self.assertIn("header", result.errors[0])


class DedupeTests(unittest.TestCase):
    def test_dedupe_keeps_same_day_repeats(self):
        coffee = Transaction(date(2024, 1, 1), Decimal("-3"), "main", memo="coffee")
        existing = [coffee]
        incoming = [
            Transaction(date(2024, 1, 1), Decimal("-3"), "main", memo="coffee"),
            Transaction(date(2024, 1, 1), Decimal("-3"), "main", memo="coffee"),
        ]
        self.assertEqual(len(dedupe(existing, incoming)), 1)


if __name__ == "__main__":
    unittest.main()
