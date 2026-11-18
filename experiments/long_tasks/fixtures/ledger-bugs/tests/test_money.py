import unittest
from decimal import Decimal

from ledger.money import (
    MoneyError,
    convert,
    format_amount,
    parse_amount,
    percent,
    round_money,
    split_amount,
)


class ParseAmountTests(unittest.TestCase):
    def test_plain_and_signed(self):
        self.assertEqual(parse_amount("12.5"), Decimal("12.5"))
        self.assertEqual(parse_amount("-12"), Decimal("-12"))
        self.assertEqual(parse_amount("+3.10"), Decimal("3.10"))

    def test_thousands_and_symbols(self):
        self.assertEqual(parse_amount("1,234.50"), Decimal("1234.50"))
        self.assertEqual(parse_amount("$5.00"), Decimal("5.00"))
        self.assertEqual(parse_amount("5.00 USD"), Decimal("5.00"))
        self.assertEqual(parse_amount("-€7"), Decimal("-7"))

    def test_accounting_negative(self):
        self.assertEqual(parse_amount("(12.50)"), Decimal("-12.50"))
        self.assertEqual(parse_amount("($1,000.00)"), Decimal("-1000.00"))

    def test_rejects_garbage(self):
        for bad in ("", "  ", "abc", "1.2.3", None, "NaN"):
            with self.assertRaises(MoneyError):
                parse_amount(bad)


class FormatAndRoundTests(unittest.TestCase):
    def test_round_half_up(self):
        self.assertEqual(round_money(Decimal("2.345")), Decimal("2.35"))
        self.assertEqual(round_money(Decimal("1234.5"), "JPY"), Decimal("1235"))

    def test_format(self):
        self.assertEqual(format_amount(Decimal("-1234.5")), "-$1,234.50")
        self.assertEqual(format_amount(Decimal("99"), "EUR"), "€99.00")
        self.assertEqual(format_amount(Decimal("1500"), "JPY"), "¥1,500")
        self.assertEqual(format_amount(Decimal("3"), "CHF"), "3.00 CHF")


class ConvertSplitTests(unittest.TestCase):
    RATES = {"EUR": Decimal("0.9"), "JPY": Decimal("150")}

    def test_convert(self):
        self.assertEqual(convert(Decimal("10"), "USD", "EUR", self.RATES), Decimal("9.00"))
        self.assertEqual(convert(Decimal("9"), "EUR", "USD", self.RATES), Decimal("10.00"))
        self.assertEqual(convert(Decimal("9"), "EUR", "JPY", self.RATES), Decimal("1500"))

    def test_convert_unknown_currency(self):
        with self.assertRaises(MoneyError):
            convert(Decimal("1"), "USD", "GBP", self.RATES)

    def test_split_adds_up(self):
        shares = split_amount(Decimal("10"), 3)
        self.assertEqual(shares, [Decimal("3.34"), Decimal("3.33"), Decimal("3.33")])
        self.assertEqual(sum(shares), Decimal("10.00"))
        self.assertEqual(sum(split_amount(Decimal("-7.01"), 4)), Decimal("-7.01"))

    def test_percent(self):
        self.assertEqual(percent(Decimal("1"), Decimal("3")), Decimal("33.3"))
        self.assertIsNone(percent(Decimal("1"), Decimal("0")))


if __name__ == "__main__":
    unittest.main()
