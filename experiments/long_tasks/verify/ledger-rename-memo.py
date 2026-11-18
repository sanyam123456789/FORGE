"""Hidden check for ledger-rename-memo. Usage: python -I this.py WORKSPACE"""

import dataclasses
import json
import os
import re
import subprocess
import sys
import tempfile
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
    from ledger.parser import dedupe, parse_csv
    from ledger.rules import apply_rules
    from ledger.storage import load_ledger, save_ledger

    names = [f.name for f in dataclasses.fields(Transaction)]
    if "description" not in names or "memo" in names:
        problems.append(f"Transaction fields are {names}")

    with tempfile.TemporaryDirectory() as tmp:
        old = Path(tmp) / "old.json"
        old.write_text(json.dumps({
            "version": 2, "base_currency": "USD",
            "accounts": [{"name": "main"}],
            "transactions": [{"date": "2024-01-02", "amount": "-5", "account": "main",
                              "memo": "Old style"}],
        }), encoding="utf-8")
        if load_ledger(old).transactions[0].description != "Old style":
            problems.append("old files with 'memo' do not load into description")

        v1 = Path(tmp) / "v1.json"
        v1.write_text(json.dumps({
            "accounts": ["main"],
            "transactions": [{"date": "2024-01-02", "amount": "-5", "account": "main",
                              "memo": "Version one"}],
        }), encoding="utf-8")
        if load_ledger(v1).transactions[0].description != "Version one":
            problems.append("version 1 files no longer load their memo")

        ledger = Ledger()
        ledger.add_account(Account("main"))
        ledger.add_transaction(Transaction(date(2024, 2, 1), Decimal("-42"), "main",
                                           description="Big ALDI shop"))
        new = Path(tmp) / "new.json"
        save_ledger(ledger, new)
        stored = json.loads(new.read_text(encoding="utf-8"))["transactions"][0]
        if stored.get("description") != "Big ALDI shop" or "memo" in stored:
            problems.append(f"saved transaction is {stored}")
        if load_ledger(new).transactions[0].description != "Big ALDI shop":
            problems.append("new files do not round-trip")

        proc = subprocess.run(
            [sys.executable, "-m", "ledger.cli", "report", str(new)],
            cwd=ws, env=ENV, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        if proc.returncode != 0 or "Big ALDI shop" not in proc.stdout:
            problems.append(f"cli report fails: {proc.stderr.strip()[-200:]}")

    result = parse_csv("Date,Memo,Amount\n2024-01-01,Coffee corner,-3\n")
    if not result.ok or result.transactions[0].description != "Coffee corner":
        problems.append("CSV 'Memo' header no longer imported")
    result = parse_csv("Date,Description,Amount\n2024-01-01,Tea house,-3\n")
    if not result.ok or result.transactions[0].description != "Tea house":
        problems.append("CSV 'Description' header no longer imported")

    out, changed = apply_rules([Transaction(date(2024, 1, 1), Decimal("-9"), "main",
                                            description="netflix")])
    if out[0].category != "subscriptions":
        problems.append("rules no longer match on description")
    same = Transaction(date(2024, 1, 1), Decimal("-3"), "main", description="coffee")
    other = Transaction(date(2024, 1, 1), Decimal("-3"), "main", description="tea")
    if len(dedupe([same], [same, other])) != 1:
        problems.append("dedupe no longer compares descriptions")
except Exception as exc:  # noqa: BLE001
    problems.append(f"checks crashed: {type(exc).__name__}: {exc}")

leftovers = []
for path in sorted(list((ws / "ledger").glob("*.py")) + list((ws / "tests").glob("*.py"))):
    for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if re.search(r"\.memo\b", line):
            leftovers.append(f"{path.relative_to(ws).as_posix()}:{no}")
if leftovers:
    problems.append("still uses .memo: " + ", ".join(leftovers[:8]))

proc = subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
    cwd=ws, env=ENV, capture_output=True, text=True, timeout=60,
)
if proc.returncode != 0:
    problems.append("test suite fails: " + proc.stderr.strip().splitlines()[-1])

if problems:
    print("\n".join(problems))
    sys.exit(1)
print("ok")
