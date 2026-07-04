"""Static repository checks for RingMoE.

Goals (works on Windows without MindSpore/aicc_tools installed):
- Parse Python files with AST to ensure syntax is valid.
- Collect import statements and try to resolve local imports (within repo).
- Heuristically flag suspicious imports (e.g., mindspore, aicc_tools) as external deps.
- Find common footguns: hardcoded AK/SK strings in source.

This is NOT a full runtime test.

Usage:
  python tools_repo_check.py
  python tools_repo_check.py --json report.json

Exit code:
  0 if no "ERROR" findings, else 1.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class Finding:
    severity: str
    code: str
    file: str
    line: int | None
    message: str

# A lightweight allowlist for common stdlib modules so we don't warn on them.
# (We keep it small and conservative; false positives are worse than missing a stdlib entry.)
STDLIB_PREFIXES: tuple[str, ...] = (
    "argparse",
    "ast",
    "collections",
    "contextlib",
    "copy",
    "dataclasses",
    "datetime",
    "functools",
    "glob",
    "hashlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "os",
    "pathlib",
    "pickle",
    "random",
    "re",
    "shutil",
    "subprocess",
    "sys",
    "time",
    "typing",
    "warnings",
)


def is_stdlib_import(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in STDLIB_PREFIXES)


def iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        # skip caches
        if "__pycache__" in p.parts:
            continue
        yield p


def parse_ast(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))


def module_name_from_path(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root)
    parts = list(rel.parts)
    parts[-1] = parts[-1][:-3]  # strip .py
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def build_local_module_index(root: Path) -> set[str]:
    mods: set[str] = set()
    for py in iter_py_files(root):
        mods.add(module_name_from_path(root, py))
    # also treat package dirs with __init__.py as modules
    for init in root.rglob("__init__.py"):
        if "__pycache__" in init.parts:
            continue
        mods.add(module_name_from_path(root, init))
    # remove empty module name (root __init__.py)
    mods.discard("")
    return mods


def dotted_prefixes(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]


def resolve_relative(module: str, level: int, current_mod: str) -> str | None:
    # current_mod like ringmoe_framework.tools.helper
    base_parts = current_mod.split(".")
    if level > len(base_parts):
        return None
    prefix_parts = base_parts[:-level]
    if module:
        prefix_parts += module.split(".")
    return ".".join([p for p in prefix_parts if p])


def collect_imports(tree: ast.AST, current_mod: str) -> list[tuple[str, int]]:
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, getattr(node, "lineno", 1)))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                mod = ""
            else:
                mod = node.module
            if node.level and node.level > 0:
                resolved = resolve_relative(mod, node.level, current_mod)
                if resolved is not None:
                    imports.append((resolved, getattr(node, "lineno", 1)))
                else:
                    imports.append((f"<unresolved-relative level={node.level} module={mod}>", getattr(node, "lineno", 1)))
            else:
                imports.append((mod, getattr(node, "lineno", 1)))
    # drop empty module in `from . import x` patterns
    imports = [(m, l) for (m, l) in imports if m]
    return imports


AK_PAT = re.compile(r"\b(ak|AK)\s*=\s*['\"]([A-Za-z0-9]{10,})['\"]")
SK_PAT = re.compile(r"\b(sk|SK)\s*=\s*['\"]([^'\"]{10,})['\"]")


def scan_secrets(text: str) -> list[str]:
    hits = []
    for m in AK_PAT.finditer(text):
        hits.append(f"Possible access key assignment: {m.group(0)[:80]}...")
    for m in SK_PAT.finditer(text):
        hits.append(f"Possible secret key assignment: {m.group(0)[:80]}...")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="json_path", default=None, help="write report as JSON")
    args = ap.parse_args()

    local_mods = build_local_module_index(REPO_ROOT)
    findings: list[Finding] = []

    # 1) AST parse
    for py in iter_py_files(REPO_ROOT):
        mod = module_name_from_path(REPO_ROOT, py)
        try:
            tree = parse_ast(py)
        except SyntaxError as e:
            findings.append(Finding(
                severity="ERROR",
                code="PY_SYNTAX",
                file=str(py.relative_to(REPO_ROOT)),
                line=e.lineno or 1,
                message=str(e),
            ))
            continue

        # 2) imports
        imports = collect_imports(tree, mod or "")
        for imp, lineno in imports:
            # external deps: skip warnings but list as info
            if imp.startswith("mindspore") or imp.startswith("aicc_tools"):
                findings.append(Finding(
                    severity="INFO",
                    code="EXT_DEP",
                    file=str(py.relative_to(REPO_ROOT)),
                    line=lineno,
                    message=f"External dependency import: {imp}",
                ))
                continue

            if is_stdlib_import(imp):
                continue

            # local resolution heuristic: if any prefix exists as module
            prefixes = dotted_prefixes(imp)
            if not any(p in local_mods for p in prefixes):
                # if it contains a dot, it's more likely a package
                sev = "WARN" if "." in imp else "INFO"
                findings.append(Finding(
                    severity=sev,
                    code="IMPORT_UNRESOLVED",
                    file=str(py.relative_to(REPO_ROOT)),
                    line=lineno,
                    message=f"Import may not resolve within repo: {imp}",
                ))

        # 3) hardcoded keys
        text = py.read_text(encoding="utf-8", errors="ignore")
        secret_hits = scan_secrets(text)
        for msg in secret_hits:
            findings.append(Finding(
                severity="WARN",
                code="HARDCODED_SECRET",
                file=str(py.relative_to(REPO_ROOT)),
                line=None,
                message=msg,
            ))

    # Summarize
    error_count = sum(1 for f in findings if f.severity == "ERROR")
    warn_count = sum(1 for f in findings if f.severity == "WARN")
    info_count = sum(1 for f in findings if f.severity == "INFO")

    report = {
        "root": str(REPO_ROOT),
        "summary": {"errors": error_count, "warnings": warn_count, "infos": info_count},
        "findings": [asdict(f) for f in findings],
    }

    if args.json_path:
        out_path = Path(args.json_path)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plain text output
    print("RingMoE static check report")
    print(json.dumps(report["summary"], ensure_ascii=False))

    # print top warnings/errors
    for f in findings:
        if f.severity in {"ERROR", "WARN"}:
            loc = f"{f.file}:{f.line}" if f.line else f.file
            print(f"[{f.severity}] {f.code} {loc} - {f.message}")

    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
