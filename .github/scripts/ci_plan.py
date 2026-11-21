#!/usr/bin/env python3
"""Decides what a CI run checks, from the files a pull request changes.

Writes `key=value` lines to $GITHUB_OUTPUT (or stdout):

- `rust`: `true` unless only documentation changed.
- `full`: `true` to run every test (not a pull request, the
  `full-validation` label, or a change to the build setup).
- `filter`: a nextest filterset of the workspace packages to test on a pull
  request: each changed package and every package that depends on it.
- `gui`: `true` if the GUI workspace or a crate it depends on changed.

Usage: ci_plan.py BASE_SHA HEAD_SHA (pull requests) or ci_plan.py --full.
"""

import json
import os
import subprocess
import sys
from pathlib import PurePosixPath

# Changes to these alone never affect a build or a test.
DOC_ONLY = ("docs/", ".github/ISSUE_TEMPLATE/", "LICENSE-")
DOC_SUFFIXES = (".md",)
# Changes to these can affect every crate: run everything.
FULL = ("Cargo.toml", "Cargo.lock", "deny.toml", "rustfmt.toml", ".config/", ".github/")
GUI_ROOTS = ("faradice-core", "faradice-geom", "faradice-io", "faradice-engine")


def emit(**kv):
    out = os.environ.get("GITHUB_OUTPUT")
    lines = "".join(f"{k}={v}\n" for k, v in kv.items())
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(lines)
    sys.stdout.write(lines)


def is_doc(path):
    if path.endswith(DOC_SUFFIXES) and not path.startswith(".github/workflows/"):
        return True
    return path.startswith(DOC_ONLY)


def main():
    if sys.argv[1:] == ["--full"]:
        emit(rust="true", full="true", filter="all()", gui="true")
        return
    base, head = sys.argv[1:3]
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    code = [p for p in changed if not is_doc(p)]
    if not code:
        emit(rust="false", full="false", filter="none()", gui="false")
        return
    if any(p == f or (f.endswith("/") and p.startswith(f)) for p in code for f in FULL):
        emit(rust="true", full="true", filter="all()", gui="true")
        return

    meta = json.loads(subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        check=True, capture_output=True, text=True,
    ).stdout)
    root = PurePosixPath(meta["workspace_root"])
    dirs = {
        str(PurePosixPath(p["manifest_path"]).parent.relative_to(root)): p["name"]
        for p in meta["packages"]
    }
    deps = {
        p["name"]: {d["name"] for d in p["dependencies"] if d["name"] in dirs.values()}
        for p in meta["packages"]
    }

    touched, other = set(), False
    for p in code:
        owner = next((n for d, n in dirs.items() if p.startswith(d + "/")), None)
        if owner:
            touched.add(owner)
        elif not p.startswith("gui/"):
            # Examples, technology files, benchmark cases: inputs of tests
            # across the workspace.
            other = True
    if other:
        touched = set(dirs.values())

    # Workspace crates the GUI builds on (transitively).
    gui_deps, stack = set(), list(GUI_ROOTS)
    while stack:
        n = stack.pop()
        if n not in gui_deps:
            gui_deps.add(n)
            stack.extend(deps.get(n, ()))
    gui = any(p.startswith("gui/") for p in code) or bool(touched & gui_deps)

    filt = " | ".join(f"rdeps(={n})" for n in sorted(touched)) or "none()"
    emit(rust="true", full="false", filter=filt, gui=str(gui).lower())


if __name__ == "__main__":
    main()
