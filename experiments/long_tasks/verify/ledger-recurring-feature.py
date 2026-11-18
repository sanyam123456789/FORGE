"""Hidden check for ledger-recurring-feature. Usage: python -I this.py WORKSPACE"""

import json
import os
import subprocess
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

ws = Path(sys.argv[1]).resolve()
# UTF-8 output from child processes (amounts like €5 on Windows).
ENV = {**os.environ, "PYTHONUTF8": "1"}
problems = []

sys.path.insert(0, str(ws))
try:
    from ledger.models import Account, Ledger, Transaction
    from ledger.recurring import find_recurring
    from ledger.storage import save_ledger

    def t(day, amount, memo, account="main"):
        return Transaction(day, Decimal(amount), account, memo=memo)

    txns = [
        t(date(2024, 1, 3), "-9.99", "NETFLIX.COM 0123"),
        t(date(2024, 2, 3), "-9.99", "Netflix.com   0224"),
        t(date(2024, 3, 3), "-9.99", "netflix.com 0325"),
        t(date(2024, 3, 20), "-9.99", "netflix.com 0399"),   # same month again
        t(date(2024, 1, 1), "-1200", "Rent Jan"),
        t(date(2024, 2, 1), "-1200", "Rent Feb"),
        t(date(2024, 3, 1), "-1200", "rent mar"),            # memo differs: not one group
        t(date(2024, 1, 5), "-30", "Gym 1"),
        t(date(2024, 2, 5), "-30", "gym 2"),
        t(date(2024, 3, 5), "-35", "gym 3"),                 # amount differs
        t(date(2024, 1, 9), "3000", "Payroll 1"),
        t(date(2024, 2, 9), "3000", "Payroll 2"),
        t(date(2024, 3, 9), "3000", "Payroll 3"),            # income: never recurring expense
        t(date(2023, 12, 7), "-15", "Spotify"),
        t(date(2024, 1, 7), "-15", "spotify"),
        t(date(2024, 2, 7), "-15", "SPOTIFY"),
    ]
    got = find_recurring(txns)
    want = [
        {"memo": "spotify", "amount": Decimal("-15"), "months": 3},
        {"memo": "netflix.com", "amount": Decimal("-9.99"), "months": 3},
    ]
    if got != want:
        problems.append(f"find_recurring default: got {got!r}")
    got2 = find_recurring(txns, min_months=2)
    memos2 = [(g["memo"], g["amount"], g["months"]) for g in got2]
    want2 = [
        ("gym", Decimal("-30"), 2),
        ("spotify", Decimal("-15"), 3),
        ("netflix.com", Decimal("-9.99"), 3),
    ]
    if memos2 != want2:
        problems.append(f"find_recurring min_months=2: got {memos2!r}")
    if find_recurring([]) != []:
        problems.append("find_recurring([]) is not []")

    ledger = Ledger(base_currency="EUR")
    ledger.add_account(Account("main", currency="EUR"))
    for x in txns:
        ledger.add_transaction(x)
    path = ws / "_verify_ledger.json"
    save_ledger(ledger, path)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ledger.cli", "recurring", str(path)],
            cwd=ws, env=ENV, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        lines = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
        expected = ["-€15.00  spotify  3 months", "-€9.99  netflix.com  3 months"]
        if proc.returncode != 0 or lines != expected:
            problems.append(f"cli recurring: exit {proc.returncode}, output {lines!r}")
        proc = subprocess.run(
            [sys.executable, "-m", "ledger.cli", "recurring", str(path), "--min-months", "5"],
            cwd=ws, env=ENV, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        if proc.returncode != 0 or proc.stdout.strip() != "no recurring payments":
            problems.append(f"cli recurring --min-months 5: {proc.stdout.strip()!r}")
    finally:
        path.unlink(missing_ok=True)
except Exception as exc:  # noqa: BLE001
    problems.append(f"checks crashed: {type(exc).__name__}: {exc}")

if not (ws / "tests" / "test_recurring.py").is_file():
    problems.append("tests/test_recurring.py is missing")
proc = subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
    cwd=ws, env=ENV, capture_output=True, text=True, timeout=60,
)
if proc.returncode != 0:
    problems.append("test suite fails: " + proc.stderr.strip().splitlines()[-1])
elif "Ran 34 tests" in proc.stderr:
    problems.append("no new tests were added")

if problems:
    print("\n".join(problems))
    sys.exit(1)
print("ok")
