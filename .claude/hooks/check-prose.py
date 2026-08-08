#!/usr/bin/env python3
"""Check prose against the rules in AGENTS.md.

Checks the part of those rules that a machine can judge, at the moment of
writing rather than at review. Only lines that an edit added are reported,
found by diffing against git. A file that git does not track counts as new in
full.

The checks are a floor, not a substitute for reading the prose. Clause-stacking
needs judgement, and so does an argument against a branch that is no longer
there: "would" is far too common a word to match on.

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

Usage
-----
Check files from the command line. This exits 1 when it finds something, so a
pre-commit script can use it::

    .claude/hooks/check-prose.py firedrake/mg/utils.py

Check every edit as you make it, by adding a Claude Code ``PostToolUse`` hook.
Put this in ``.claude/settings.local.json``, which git does not track, so that
each checkout opts in for itself::

    {
      "hooks": {
        "PostToolUse": [
          {
            "matcher": "Write|Edit",
            "hooks": [
              {
                "type": "command",
                "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/check-prose.py\"",
                "timeout": 15
              }
            ]
          }
        ]
      }
    }

As a hook it reads the payload on stdin, always exits 0, and reports through
``systemMessage`` and ``additionalContext``.
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


def check(path):
    """Return the findings for one file, as (line, rule, why, excerpt) tuples."""
    if not path or not os.path.isfile(path):
        return []
    if os.path.basename(path) in EXEMPT_NAMES or not path.endswith(SOURCE_SUFFIXES):
        return []
    # git resolves a path against -C, so give it one that does not move.
    path = os.path.abspath(path)

    added = added_line_numbers(path)
    if not added:
        return []
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

    findings.sort()
    return findings


def report(path, findings, limit=12):
    """Format findings for one file as indented lines."""
    return "\n".join(f"  {path}:{n}  [{rule}] {why}\n      {excerpt}"
                     for n, rule, why, excerpt in findings[:limit])


def run_as_hook():
    """Report on the file a PostToolUse payload names. Always succeeds."""
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    tool_input = payload.get("tool_input") or {}
    path = tool_input.get("file_path") or (payload.get("tool_response") or {}).get("filePath")
    findings = check(path)
    if not findings:
        return 0
    rules = sorted({rule for _, rule, _, _ in findings})
    json.dump({
        "systemMessage": f"AGENTS.md: {len(findings)} finding(s) in "
                         f"{os.path.basename(path)} ({', '.join(rules)})",
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": (
                "AGENTS.md check on lines this edit added. Fix these, or say why "
                "each one is a false positive:\n\n" + report(path, findings)
            ),
        },
    }, sys.stdout)
    return 0


def run_as_command(paths):
    """Report on the named files. Returns 1 if anything was found."""
    total = 0
    for path in paths:
        findings = check(path)
        if findings:
            print(report(path, findings, limit=len(findings)))
            total += len(findings)
    if total:
        print(f"\n{total} finding(s). See AGENTS.md, and this file's docstring.")
        return 1
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    sys.exit(run_as_command(args) if args else run_as_hook())
