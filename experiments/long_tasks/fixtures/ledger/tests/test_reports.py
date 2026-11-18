import unittest
from datetime import date
from decimal import Decimal

from ledger.models import Account, Ledger, Transaction
from ledger.reports import (
    average_monthly_spending,
    balance_by_account,
    monthly_summary,
    render_table,
    top_expenses,
    totals_by_category,
)


def txn(day, amount, category="misc", account="main", memo=""):
    return Transaction(day, Decimal(amount), account, category, memo)


TXNS = [
    txn(date(2024, 1, 5), "3000", "income"),
    txn(date(2024, 1, 20), "-1200", "housing"),
    txn(date(2024, 1, 21), "-80.50", "groceries"),
    txn(date(2024, 2, 3), "-95.25", "groceries"),
    txn(date(2024, 2, 14), "-60", "dining"),
    txn(date(2024, 2, 28), "3000", "income"),
]


class SummaryTests(unittest.TestCase):
    def test_monthly_summary_groups_by_month(self):
        summaries = monthly_summary(TXNS)
        self.assertEqual([s.month for s in summaries], ["2024-01", "2024-02"])
        jan, feb = summaries
        self.assertEqual(jan.income, Decimal("3000"))
        self.assertEqual(jan.expenses, Decimal("-1280.50"))
        self.assertEqual(jan.count, 3)
        self.assertEqual(feb.net, Decimal("2844.75"))
        self.assertEqual(jan.savings_rate, Decimal("57.3"))

    def test_average_monthly_spending(self):
        self.assertEqual(average_monthly_spending(TXNS), Decimal("-717.88"))
        self.assertIsNone(average_monthly_spending([]))

    def test_totals_by_category(self):
        totals = totals_by_category(TXNS, expenses_only=True)
        self.assertEqual(list(totals), ["housing", "groceries", "dining"])
        self.assertEqual(totals["groceries"], Decimal("-175.75"))

    def test_top_expenses(self):
        top = top_expenses(TXNS, 2)
        self.assertEqual([t.amount for t in top], [Decimal("-1200"), Decimal("-95.25")])


class BalanceTests(unittest.TestCase):
    def test_balance_includes_opening(self):
        ledger = Ledger()
        ledger.add_account(Account("main", opening_balance=Decimal("100")))
        ledger.add_account(Account("cash", kind="cash"))
        for t in TXNS:
            ledger.add_transaction(t)
        ledger.add_transaction(txn(date(2024, 2, 1), "-20", account="cash"))
        balances = balance_by_account(ledger)
        self.assertEqual(balances["main"], Decimal("4664.25"))
        self.assertEqual(balances["cash"], Decimal("-20"))


class TableTests(unittest.TestCase):
    def test_render_table_aligns(self):
        text = render_table(["name", "n"], [["a", "1"], ["bbb", "22"]])
        self.assertEqual(text.splitlines()[2], "a      1")
        self.assertEqual(text.splitlines()[3], "bbb   22")


if __name__ == "__main__":
    unittest.main()
