"""Hidden check for ledger-fix-bugs. Usage: python -I this.py WORKSPACE

Passes when (1) the visible tests are unchanged, (2) they all pass, and
(3) the three bugs are fixed when checked directly.
"""

import hashlib
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

ORIGINAL_TESTS = {
    "test_budget.py": "5c23c4542b10ca30d4529b453c62a83dadf2f4cf98f1d4a01e558a49be6e53ae",
    "test_money.py": "969f97bd0df2bc6579951485fe05a4583e16538cbf95f9bb2fc21d8437ddefd8",
    "test_parser.py": "58e7189b666c6582db9370d49d77c6a567b74f4e8a8adce12e82feb9b484b868",
    "test_reports.py": "645a62370040c9adc35281e7fd9d95441ed9d5a63eaa8000b94bf88c66a11c53",
    "test_storage_rules.py": "4f6e043e254c2c73a91458167fd66d67602ff9c3c3eac5cbc7955226a3431431",
}
for name, digest in ORIGINAL_TESTS.items():
    path = ws / "tests" / name
    if not path.is_file():
        problems.append(f"tests/{name} is missing")
        continue
    actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != digest:
        problems.append(f"tests/{name} was changed")

proc = subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
    cwd=ws, env=ENV, capture_output=True, text=True, timeout=60,
)
if proc.returncode != 0:
    problems.append("test suite fails: " + proc.stderr.strip().splitlines()[-1])

sys.path.insert(0, str(ws))
try:
    from ledger.budget import spent_in_month
    from ledger.models import Transaction
    from ledger.money import parse_amount
    from ledger.reports import monthly_summary

    if parse_amount("(7.25)") != Decimal("-7.25"):
        problems.append("parse_amount('(7.25)') is not -7.25")
    months = [s.month for s in monthly_summary([
        Transaction(date(2022, 11, 3), Decimal("-1"), "a"),
        Transaction(date(2022, 2, 3), Decimal("-1"), "a"),
    ])]
    if months != ["2022-02", "2022-11"]:
        problems.append(f"monthly_summary months are {months}")
    txns = [
        Transaction(date(2021, 6, 1), Decimal("-40"), "a", "fun"),
        Transaction(date(2022, 6, 1), Decimal("-15"), "a", "fun"),
    ]
    if spent_in_month(txns, "fun", 2022, 6) != Decimal("15"):
        problems.append("spent_in_month mixes years")
except Exception as exc:  # noqa: BLE001
    problems.append(f"direct checks crashed: {type(exc).__name__}: {exc}")

if problems:
    print("\n".join(problems))
    sys.exit(1)
print("ok")
