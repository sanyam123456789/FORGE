"""Hidden check for ledger-api-docs. Usage: python -I this.py WORKSPACE

Passes when API.md has a "## ledger.<module>" section for every module
(except __init__), every public top-level function and class of a module is
named in that module's section, and no source file was changed.
"""

import ast
import hashlib
import re
import sys
from pathlib import Path

ws = Path(sys.argv[1]).resolve()
problems = []

ORIGINAL_SOURCES = {
    "__init__.py": "b69810689d127fa2dfa9bc36982c9c10610c0ed1e541ae82fb27e20ac4887584",
    "budget.py": "b7cbb815b4c0cee5379a4d1e16f47c0b8e3610de763374e6c05d7b0f2b848f6a",
    "cli.py": "f681663a37759d56705f519c1d69a1e940dc128928912a3c8b40384b38344ea9",
    "models.py": "9eaadb0ee6b8c4dcc415f086f3dc0b0fd043600a3dd0e9c137bc45c8c05888ab",
    "money.py": "1c2b212e374329faeb1cc256c01b4a279fafcf3a90b669037be7c587221d38c7",
    "parser.py": "a397b9a3292836ef7f149fd38048a399884a39b71e6b9880b42a1da1c54df867",
    "reports.py": "80b894b4fef69652362008706af0ec22d01a27bba325d21f42683a656834a4fb",
    "rules.py": "dc61e8245ba95bc6f79c2d276b7a86648d8b022853acfd3470e91a5588cfb945",
    "storage.py": "05fe1f0e1ebcee7aa43aacb2c39b71d0e88f7b1732503be9a9d1ba4f57742c21",
}
for name, digest in ORIGINAL_SOURCES.items():
    path = ws / "ledger" / name
    if not path.is_file():
        problems.append(f"ledger/{name} is missing")
        continue
    if hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != digest:
        problems.append(f"ledger/{name} was changed")

doc_path = ws / "API.md"
if not doc_path.is_file():
    problems.append("API.md is missing")
else:
    doc = doc_path.read_text(encoding="utf-8", errors="replace")
    # Split into sections at level-2 headings.
    sections = {}
    current = None
    for line in doc.splitlines():
        m = re.match(r"^##\s+`?(ledger\.\w+)`?\s*$", line.strip())
        if m:
            current = m.group(1)
            sections[current] = []
        elif current:
            sections[current].append(line)
    for path in sorted((ws / "ledger").glob("*.py")):
        if path.name == "__init__.py":
            continue
        module = f"ledger.{path.stem}"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_")
        ]
        if module not in sections:
            problems.append(f"no '## {module}' section")
            continue
        text = "\n".join(sections[module])
        missing = [n for n in names if not re.search(rf"\b{re.escape(n)}\b", text)]
        if missing:
            problems.append(f"{module}: not documented: {', '.join(missing)}")

if problems:
    print("\n".join(problems))
    sys.exit(1)
print("ok")
