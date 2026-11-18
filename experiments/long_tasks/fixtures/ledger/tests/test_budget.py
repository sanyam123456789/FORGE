import unittest
from datetime import date
from decimal import Decimal

from ledger.budget import (
    LEVEL_OK,
    LEVEL_OVER,
    LEVEL_WARNING,
    Budget,
    BudgetError,
    check_budgets,
    load_budgets,
    spent_in_month,
)
from ledger.models import Transaction


def txn(day, amount, category):
    return Transaction(day, Decimal(amount), "main", category)


TXNS = [
    txn(date(2023, 3, 10), "-500", "groceries"),  # same month, previous year
    txn(date(2024, 3, 2), "-150", "groceries"),
    txn(date(2024, 3, 9), "-200", "groceries"),
    txn(date(2024, 3, 12), "10", "groceries"),  # refund
    txn(date(2024, 3, 15), "-130", "dining"),
    txn(date(2024, 4, 1), "-999", "groceries"),
]


class SpentTests(unittest.TestCase):
    def test_spent_counts_only_that_month_and_year(self):
        self.assertEqual(spent_in_month(TXNS, "groceries", 2024, 3), Decimal("340"))
        self.assertEqual(spent_in_month(TXNS, "groceries", 2023, 3), Decimal("500"))
        self.assertEqual(spent_in_month(TXNS, "rent", 2024, 3), Decimal("0"))


class CheckBudgetTests(unittest.TestCase):
    def test_levels_and_order(self):
        budgets = [
            Budget("groceries", Decimal("400")),
            Budget("dining", Decimal("120")),
            Budget("transport", Decimal("100")),
        ]
        statuses = check_budgets(TXNS, budgets, 2024, 3)
        self.assertEqual([s.category for s in statuses], ["dining", "groceries", "transport"])
        levels = {s.category: s.level for s in statuses}
        self.assertEqual(levels, {"dining": LEVEL_OVER, "groceries": LEVEL_WARNING,
                                  "transport": LEVEL_OK})
        groceries = next(s for s in statuses if s.category == "groceries")
        self.assertEqual(groceries.remaining, Decimal("60"))
        self.assertEqual(groceries.used, Decimal("0.85"))


class LoadBudgetTests(unittest.TestCase):
    def test_load(self):
        budgets = load_budgets({"Groceries": "400", "dining": {"limit": "$150", "warn_at": "0.9"}})
        self.assertEqual(budgets[0].category, "groceries")
        self.assertEqual(budgets[1].monthly_limit, Decimal("150"))
        self.assertEqual(budgets[1].warn_at, Decimal("0.9"))

    def test_invalid(self):
        with self.assertRaises(BudgetError):
            load_budgets({"x": "-5"})
        with self.assertRaises(BudgetError):
            load_budgets({"x": {"limit": "5", "warn_at": "1.5"}})


if __name__ == "__main__":
    unittest.main()
