#!/usr/bin/env python3
"""Check an edit against the prose and style rules in AGENTS.md.

Point any repository at this by adding a PostToolUse hook on ``Write|Edit`` to
that repository's ``.claude/settings.local.json``. The checks are generic.

Reads a PostToolUse payload on stdin. Reports only on lines the edit added,
found by diffing the file against git. Advisory: always exits 0, and reports
through systemMessage (to the user) and additionalContext (to Claude).

Checks
------
sphinx-field-list
    ``:arg x:``/``:param x:``/``:returns:``/``:rtype:``. AGENTS.md allows
    numpydoc only, and warns that copying the neighbouring style is the wrong
    instinct here.
long-sentence
    A sentence of more than MAX_WORDS words in a docstring or a comment.
    ASD-STE100 asks for short sentences, one idea each.
past-tense
    Wording that describes code which is not there any more.
hasattr-guard
    ``if not hasattr(self, ...)`` standing in for a setup flag.
"""
import ast
import io
import json
import os
import re
import subprocess
import sys
import tokenize

MAX_WORDS = 25
SOURCE_SUFFIXES = (".py", ".pyx", ".pxd", ".md", ".rst")
# AGENTS.md and CLAUDE.md state the rules, so they quote the words they ban.
EXEMPT_NAMES = {"AGENTS.md", "CLAUDE.md"}

SPHINX = re.compile(r":(?:arg|param|returns?|rtype|raises|type|vartype)\b")
TELLS = re.compile(
    r"(?<![A-Za-z])(?:used to|previously|no longer|we removed|this replaces"
    r"|formerly|instead of the (?:old|previous|former))(?![A-Za-z])",
    re.IGNORECASE,
)
HASATTR = re.compile(r"if\s+not\s+hasattr\(\s*self\b")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def git(*args):
    return subprocess.run(args, capture_output=True, text=True)


def added_line_numbers(path):
    """Return the 1-based line numbers this edit added, or None if unknown."""
    directory = os.path.dirname(path) or "."
    if git("git", "-C", directory, "rev-parse", "--git-dir").returncode != 0:
        return None
    tracked = git("git", "-C", directory, "ls-files", "--error-unmatch", path)
    if tracked.returncode != 0:
        # git does not track the file, so all of it is new.
        with open(path, encoding="utf-8", errors="replace") as handle:
            return set(range(1, len(handle.readlines()) + 1))

    diff = git("git", "-C", directory, "diff", "-U0", "--", path).stdout
    added = set()
    for line in diff.splitlines():
        match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if match:
            start = int(match.group(1))
            count = 1 if match.group(2) is None else int(match.group(2))
            added.update(range(start, start + count))
    return added


def sentences(text):
    """Split prose into sentences, ignoring code-like fragments."""
    flat = " ".join(text.split())
    return [s for s in SENTENCE_SPLIT.split(flat) if s]


def prose_blocks(path, source, added):
    """Yield (line number, prose) for docstrings and comment runs that were added."""
    if not path.endswith((".py", ".pyx", ".pxd")):
        # Treat a run of added prose lines as one block.
        run, start = [], None
        for number, line in enumerate(source.splitlines(), 1):
            if number in added and line.strip() and not line.lstrip().startswith(("```", "|", ">")):
                run.append(line.strip())
                start = start or number
            else:
                if run:
                    yield start, " ".join(run)
                run, start = [], None
        if run:
            yield start, " ".join(run)
        return

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            continue
        target = node.body[0]
        span = range(target.lineno, (target.end_lineno or target.lineno) + 1)
        if any(n in added for n in span):
            # Stop at the first numpydoc section: those are structured, not prose.
            body = re.split(r"\n\s*(?:Parameters|Returns|Raises|Notes|Examples)\s*\n", doc)[0]
            yield target.lineno, body

    run, start = [], None
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):
        tokens = []
    for token in tokens:
        if token.type == tokenize.COMMENT and token.start[0] in added:
            run.append(token.string.lstrip("#").strip())
            start = start or token.start[0]
        elif run and token.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            yield start, " ".join(run)
            run, start = [], None
    if run:
        yield start, " ".join(run)


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    tool_input = payload.get("tool_input") or {}
    path = tool_input.get("file_path") or (payload.get("tool_response") or {}).get("filePath")
    if not path or not os.path.isfile(path):
        return 0
    if os.path.basename(path) in EXEMPT_NAMES or not path.endswith(SOURCE_SUFFIXES):
        return 0

    added = added_line_numbers(path)
    if not added:
        return 0
    with open(path, encoding="utf-8", errors="replace") as handle:
        source = handle.read()
    lines = source.splitlines()

    findings = []
    for number in sorted(added):
        if number > len(lines):
            continue
        line = lines[number - 1]
        if SPHINX.search(line):
            findings.append((number, "sphinx-field-list",
                             "Use numpydoc sections, not Sphinx field lists", line.strip()))
        if TELLS.search(line):
            findings.append((number, "past-tense",
                             "Describes code that may not be there any more", line.strip()))
        if HASATTR.search(line):
            findings.append((number, "hasattr-guard",
                             "Declare a boolean or use functools.cached_property", line.strip()))

    for number, text in prose_blocks(path, source, added):
        for sentence in sentences(text):
            words = len(sentence.split())
            if words > MAX_WORDS:
                findings.append((number, "long-sentence",
                                 f"{words} words; ASD-STE100 asks for one idea per sentence",
                                 sentence[:110] + ("..." if len(sentence) > 110 else "")))

    if not findings:
        return 0

    findings.sort()
    report = "\n".join(f"  {path}:{n}  [{rule}] {why}\n      {excerpt}"
                       for n, rule, why, excerpt in findings[:12])
    rules = sorted({rule for _, rule, _, _ in findings})
    message = (
        "AGENTS.md check on lines this edit added. Fix these, or say why each one "
        "is a false positive:\n\n" + report
    )
    json.dump({
        "systemMessage": f"AGENTS.md: {len(findings)} finding(s) in "
                         f"{os.path.basename(path)} ({', '.join(rules)})",
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": message,
        },
    }, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
