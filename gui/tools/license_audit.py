#!/usr/bin/env python3
"""License audit of the GUI dependency graph (design D1 §2.1, gate G0).

For every crate resolved for x86_64 Linux with all features:

1. the crate's SPDX license expression must be satisfiable with permissive
   licenses only (proper AND/OR/parentheses evaluation);
2. no file in the crate may mention a GPL-family license (full names, SPDX
   ids, or shorthand such as "GPLv3"), unless the (crate, path) pair is in
   REVIEWED with the reason it is acceptable. Crate metadata does not reveal
   bundled files (e.g. Wayland protocol XML compiled into bindings), and any
   new mention fails the audit until a person reviews it.

Scope (D1 §2.1): build inputs, i.e. files that are compiled, included or used
to generate code. Files that are never built for dependents (examples, tests,
benches, docs) are not distributed with faradice; they are still reported and
must be listed in REVIEWED.

Run from gui/:  python3 tools/license_audit.py
"""

import fnmatch
import json
import os
import re
import subprocess
import sys

PERMISSIVE = {
    "MIT", "Apache-2.0", "Apache-2.0 WITH LLVM-exception", "BSD-2-Clause",
    "BSD-3-Clause", "ISC", "Zlib", "Unicode-3.0", "Unlicense", "CC0-1.0", "BSL-1.0",
}

# (crate, path glob relative to the crate root) -> why it is acceptable.
REVIEWED = {
    ("self_cell", "LICENSE-GPLv2"): "dual Apache-2.0 OR GPL-2.0-only; Apache-2.0 is used",
    ("self_cell", "README.md"): "describes the dual license",
    ("self_cell", "Cargo.toml*"): "license expression Apache-2.0 OR GPL-2.0-only",
    ("rustix", "LICENSE-Apache-2.0_WITH_LLVM-exception"): "LLVM exception text mentions GPLv2 compatibility",
    ("linux-raw-sys", "LICENSE-Apache-2.0_WITH_LLVM-exception"): "LLVM exception text mentions GPLv2 compatibility",
    ("ttf-parser", "README.md"): "comparison table mentions FreeType's license",
    ("winit", "src/platform_impl/linux/wayland/window/state.rs"): "faradice patch comment explaining the LGPL removal",
    ("x11rb", "examples/*"): "example program (derived from GPLv3 code), never compiled for dependents",
}

MENTION = re.compile(
    rb"\b(L|A)?GPL(v?[0-9]|-[0-9])"
    rb"|GNU (Lesser |Library |Affero )?General Public"
    rb"|General Public License",
    re.I,
)


def tokens(expr):
    return re.findall(r"\(|\)|[^\s()]+", expr.replace("/", " OR "))


def satisfiable(expr):
    """Evaluates an SPDX expression: True if some choice uses only PERMISSIVE.

    Grammar: or := and ("OR" and)* ; and := atom ("AND" atom)* ;
    atom := "(" or ")" | id ["WITH" id]
    """
    toks = tokens(expr)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def atom():
        if peek() == "(":
            take()
            v = or_()
            if take() != ")":
                raise ValueError(expr)
            return v
        lic = take()
        if peek() == "WITH":
            take()
            lic = f"{lic} WITH {take()}"
        return lic in PERMISSIVE

    def and_():
        v = atom()
        while peek() == "AND":
            take()
            v = atom() and v
        return v

    def or_():
        v = and_()
        while peek() == "OR":
            take()
            v = and_() or v
        return v

    v = or_()
    if pos != len(toks):
        raise ValueError(expr)
    return v


def reviewed(name, rel):
    return any(n == name and fnmatch.fnmatch(rel, g) for (n, g) in REVIEWED)


def main():
    assert satisfiable("(MIT OR Apache-2.0) AND Unicode-3.0")
    assert not satisfiable("(MIT OR Apache-2.0) AND GPL-3.0-only")
    assert satisfiable("Apache-2.0 OR GPL-2.0-only")
    assert not satisfiable("LGPL-2.1-or-later")
    meta = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1", "--locked", "--all-features",
         "--filter-platform", "x86_64-unknown-linux-gnu"]))
    ids = {n["id"] for n in meta["resolve"]["nodes"]}
    problems, used = [], set()
    for p in (p for p in meta["packages"] if p["id"] in ids):
        if p["source"] is None and "/vendor/" not in p["manifest_path"]:
            continue  # faradice's own crates
        lic = p.get("license") or ""
        try:
            ok = satisfiable(lic)
        except (ValueError, IndexError):
            ok = False
        if not ok:
            problems.append(f"{p['name']} {p['version']}: license `{lic}`")
        root = os.path.dirname(p["manifest_path"])
        for d, _, files in os.walk(root):
            for f in files:
                path = os.path.join(d, f)
                rel = os.path.relpath(path, root)
                with open(path, "rb") as fh:
                    m = MENTION.search(fh.read())
                if not m:
                    continue
                if reviewed(p["name"], rel):
                    used.add(p["name"])
                    continue
                problems.append(f"{p['name']} {p['version']}: {rel}: {m.group(0)[:40]!r}")
    for x in problems:
        print("license audit:", x)
    stale = sorted({n for (n, _) in REVIEWED} - used)
    if stale:
        print("license audit: note: REVIEWED entries no longer needed:", ", ".join(stale))
    print(f"license audit: {len(ids)} packages, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
