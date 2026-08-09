#!/usr/bin/env python3
"""Deny a bare python/pytest/flake8 call, or a new test file, and name the fdk
command instead.

AGENTS.md says to call the virtual environment's interpreter, never a bare
`python`. It says to use `fdk test`/`fdk testraw`/`fdk lint`, never pytest or
flake8 directly. It says to add a test to the file `fdk testfile` names,
never a new one. An agent that never reads any of that still meets this
hook, before the command runs. It is a Claude Code PreToolUse hook, wired
into `.claude/settings.json` so it applies by default, to every session in
this checkout, including a subagent's.

`python` is the one to deny hardest. It is the first command almost any
session reaches for, well before a test or a lint pass. A bare `python` in
this checkout is the system interpreter, which has no firedrake. Denying it
turns that first command into the moment a session finds fdk, instead of a
stale-looking `ModuleNotFoundError` it might paper over.

Usage
-----
Reads the PreToolUse payload on stdin, and always exits 0; a denial goes in
the JSON reply, not the exit code. Test a case from the command line,
without the JSON envelope::

    .agents/tools/require-fdk.py --check "python -c 'import firedrake'"
    .agents/tools/require-fdk.py --check-file tests/firedrake/regression/test_new.py < new.py
"""
import json
import os
import re
import subprocess
import sys

# A command word at the start of a command, or right after a shell separator
# or substitution. Not the word turning up inside a quoted argument, such as
# `grep -n pytest.mark.parallel`. Not part of a path either, such as
# `$VIRTUAL_ENV/bin/python`.
BOUNDARY = r"(?:^|[;&|`]|\$\()\s*"
PYTEST = re.compile(BOUNDARY + r"(?:python3?\s+-m\s+)?pytest\b", re.MULTILINE)
FLAKE8 = re.compile(BOUNDARY + r"(?:python3?\s+-m\s+)?flake8\b", re.MULTILINE)
PYTHON = re.compile(BOUNDARY + r"python3?\b", re.MULTILINE)
# A command naming fdk is never what this denies -- not only its own
# subcommands, but a person reading the script's source, e.g. `cat
# .agents/tools/fdk`. fdk's own python/pytest/flake8 calls never reach this
# hook at all: they are a subprocess fdk spawns, not a separate Bash tool
# call.
FDK = re.compile(r"\bfdk\b")


def verdict(command):
    """Return a denial reason for `command`, or None to let it through."""
    if not command or FDK.search(command):
        return None
    if PYTEST.search(command):
        return (
            "DENIED: pytest does not run directly in this repo. fdk is the "
            "only tool for tests here -- not a preference, a requirement. "
            "Use `.agents/tools/fdk test <nprocs> <paths>` or "
            "`.agents/tools/fdk testraw <nprocs> <paths>`. "
            "Run `.agents/tools/fdk help` for every subcommand."
        )
    if FLAKE8.search(command):
        return (
            "DENIED: flake8 does not run directly in this repo. fdk is the "
            "only tool for linting here -- not a preference, a requirement. "
            "Use `.agents/tools/fdk lint [paths]`. "
            "Run `.agents/tools/fdk help` for every subcommand."
        )
    if PYTHON.search(command):
        return (
            "DENIED: a bare python/python3 does not run in this repo -- it "
            "is the system interpreter, which has no firedrake. fdk is the "
            "only tool for this -- not a preference, a requirement. "
            "Use `.agents/tools/fdk py [args]`. "
            "Run `.agents/tools/fdk help` for every subcommand."
        )
    return None


# A test file, by Firedrake's own convention: a `test_*.py` under a `tests`
# directory component, wherever that component sits in the path.
TEST_FILE = re.compile(r"(^|/)tests/.*/test_[^/]+\.py$")
FDK_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fdk")


def candidate_test_files(content, limit=5):
    """Return existing test files `fdk testfile --from-content` names.

    Delegates to fdk rather than ranking files itself. fdk is the one place
    that heuristic should live. That keeps it a single implementation an
    agent can run directly, not a second copy only this hook uses.
    """
    proc = subprocess.run(
        [FDK_BIN, "testfile", "--from-content"],
        input=content, capture_output=True, text=True,
    )
    return proc.stdout.splitlines()[:limit] if proc.returncode == 0 else []


def new_test_file_verdict(path, content):
    """Return a (decision, reason) pair for writing a new test file, or None.

    A genuinely new feature area can have nothing to extend, so this is not
    always wrong the way a bare python/pytest/flake8 is. When an existing
    file plausibly already covers this one, deny and name the candidates.
    Otherwise ask the person in the session to confirm. An unattended
    subagent has nobody to answer, so it gets the safe default, while a
    present human can approve the one call that needs it.
    """
    if not path or not TEST_FILE.search(path) or os.path.exists(path):
        return None
    candidates = [c for c in candidate_test_files(content) if c]
    if candidates:
        listed = ", ".join(os.path.relpath(c) for c in candidates)
        return "deny", (
            f"DENIED: this may already be covered -- {listed} reference the "
            "same names, per `fdk testfile --from-content`. Run "
            "`.agents/tools/fdk testfile <source-path>` on the file you are "
            "actually changing, which reads far more than that content-only "
            "search can, and add the test to what it names."
        )
    return "ask", (
        "No existing test file references what this one tests, per `fdk "
        "testfile --from-content`. Run `.agents/tools/fdk testfile "
        "<source-path>` to check properly before confirming that a new "
        "file is really needed."
    )


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        print(verdict(sys.argv[2]) or "allowed")
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "--check-file":
        # Content comes from stdin, not from `path` on disk: the whole point
        # is to judge a file as though it does not exist there yet.
        content = sys.stdin.read()
        print((new_test_file_verdict(sys.argv[2], content) or (None, "allowed"))[1])
        return 0

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input") or {}
    if tool_name == "Bash":
        decision, reason = "deny", verdict(tool_input.get("command", ""))
    elif tool_name == "Write":
        decision, reason = new_test_file_verdict(
            tool_input.get("file_path", ""), tool_input.get("content", "")
        ) or (None, None)
    else:
        decision, reason = None, None
    if reason is None:
        return 0
    json.dump({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        },
    }, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
