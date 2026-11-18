# ledger

A small personal-finance ledger: import bank CSV exports, categorise
transactions with rules, and report spending and monthly budgets.

```
python -m ledger.cli init my.json
python -m ledger.cli account my.json main --opening 1000
python -m ledger.cli import my.json bank.csv --account main
python -m ledger.cli categorize my.json
python -m ledger.cli report my.json --month 2024-03
python -m ledger.cli budget my.json budgets.json --month 2024-03
```

Amounts are `Decimal`; expenses are negative, income positive.

Run the tests (standard library only):

```
python -m unittest discover -s tests
```

Modules: `money` (parsing, formatting, conversion), `models` (Account,
Transaction, Ledger), `parser` (CSV import), `rules` (categorisation),
`reports`, `budget`, `storage` (JSON files) and `cli`.
