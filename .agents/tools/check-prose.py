#!/usr/bin/env python3
"""Check prose against the rules in AGENTS.md.

Checks the part of those rules that a machine can judge, at the moment of
writing rather than at review. Only lines that an edit added are reported,
found by diffing against git. A file that git does not track counts as new in
full.

The checks are a floor, not a substitute for reading the prose. Clause-stacking
needs judgement, and so does an argument against a branch that is no longer
there: "would" is far too common a word to match on.

A Markdown or reStructuredText file holds things that are not prose. The checks
skip a fenced code block, and the YAML frontmatter that opens the file. A
document that shows badly written prose as an example therefore keeps its
examples, as long as it fences them.

Checks
------
sphinx-field-list
    ``:arg x:``/``:param x:``/``:returns:``/``:rtype:``, anywhere in a
    docstring an edit touches -- not only on the lines it added. AGENTS.md
    allows numpydoc only, and says that copying the neighbouring style is the
    wrong instinct here: touching a docstring migrates the whole docstring,
    not just the lines the edit added.
long-sentence
    A sentence of more than MAX_WORDS words in a docstring or a comment.
    ASD-STE100 asks for short sentences, one idea each.
past-tense
    Wording that describes code which is not there any more.
hasattr-guard
    ``if not hasattr(self, ...)`` standing in for a setup flag.

Usage
-----
This is a plain command-line program. Any agent, or any person, can run it. It
needs Python and ``git``, and nothing particular to one assistant.

Check files from the command line. This exits 1 when it finds something, so a
pre-commit script can use it::

    .agents/tools/check-prose.py firedrake/mg/utils.py

Check a whole branch with ``--range``, which reads the lines a commit range
added rather than the lines the working tree changed. Give it the files to
look at, or none to take every file the range touches::

    .agents/tools/check-prose.py --range main...HEAD

Check out the head of the range first. The line numbers come from the diff,
and the prose comes from the working tree, so the two must agree.

One assistant can also drive it automatically. Claude Code checks every edit
as you make it, through a ``PostToolUse`` hook. Put this in
``.claude/settings.local.json``, which git does not track, so that each
checkout opts in for itself::

    {
      "hooks": {
        "PostToolUse": [
          {
            "matcher": "Write|Edit",
            "hooks": [
              {
                "type": "command",
                "command": "python3 \"$CLAUDE_PROJECT_DIR/.agents/tools/check-prose.py\"",
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
# These state the rules, so they quote the words they ban. This file names
# itself: the hook only ever sees the file an edit names, but --range sweeps
# every file a branch touches, and reaches this one.
EXEMPT_NAMES = {"AGENTS.md", "CLAUDE.md", "check-prose.py"}

SPHINX = re.compile(r":(?:arg|param|returns?|rtype|raises|type|vartype)\b")
TELLS = re.compile(
    r"(?<![A-Za-z])(?:used to|previously|no longer|we removed|this replaces"
    r"|formerly|instead of the (?:old|previous|former))(?![A-Za-z])",
    re.IGNORECASE,
)
HASATTR = re.compile(r"if\s+not\s+hasattr\(\s*self\b")
# A sentence can end inside the markup that emphasises it, as `**Do this.**`
# does, so step over the closing markers to find the space after the stop.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[*_`\"')\]]*\s+")
FENCE = re.compile(r"^\s*(```|~~~)")


def git(*args):
    return subprocess.run(args, capture_output=True, text=True)


def added_line_numbers(path, commit_range=None):
    """Return the 1-based line numbers this edit added, or None if unknown."""
    directory = os.path.dirname(path) or "."
    if git("git", "-C", directory, "rev-parse", "--git-dir").returncode != 0:
        return None
    if commit_range is None:
        tracked = git("git", "-C", directory, "ls-files", "--error-unmatch", path)
        if tracked.returncode != 0:
            # git does not track the file, so all of it is new.
            with open(path, encoding="utf-8", errors="replace") as handle:
                return set(range(1, len(handle.readlines()) + 1))

    diff_range = [commit_range] if commit_range else []
    diff = git("git", "-C", directory, "diff", "-U0", *diff_range, "--", path).stdout
    added = set()
    for line in diff.splitlines():
        match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if match:
            start = int(match.group(1))
            count = 1 if match.group(2) is None else int(match.group(2))
            added.update(range(start, start + count))
    return added


def unprosed_lines(lines: list[str]) -> set[int]:
    """Find the lines of a Markdown or reStructuredText file that hold no prose.

    Parameters
    ----------
    lines : list of str
        The lines of the file, without their line endings.

    Returns
    -------
    set of int
        The 1-based numbers of the lines a fenced code block holds, and of the
        YAML frontmatter that opens the file. Both fences and both frontmatter
        markers are included.

    """
    skip = set()
    body = 0
    # Frontmatter opens the file with a --- line, and a second one ends it. A
    # file that opens with a rule and never closes one keeps all of its lines.
    if lines and lines[0].strip() == "---":
        for number in range(2, len(lines) + 1):
            if lines[number - 1].strip() == "---":
                skip.update(range(1, number + 1))
                body = number
                break
    fence = None
    for number in range(body + 1, len(lines) + 1):
        match = FENCE.match(lines[number - 1])
        if fence is None:
            if match:
                # The fence opens a block. Only the same marker closes it, so
                # that ``` inside a ~~~ block stays part of the block.
                fence = match.group(1)
                skip.add(number)
        else:
            skip.add(number)
            if match and match.group(1) == fence:
                fence = None
    return skip


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
            if number in added and line.strip() and not line.lstrip().startswith(("|", ">")):
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


def touched_docstring_span(path, source, added):
    """Return every line of a docstring that an edit touched, not just those it added.

    Touching one line of a docstring migrates the whole docstring: this widens
    the sphinx-field-list check from the added lines to the docstring's full
    span, so an old field-list line the edit left alone still gets caught.

    Parameters
    ----------
    path : str
        The file the docstring lives in.
    source : str
        The file's current contents.
    added : set of int
        The 1-based line numbers the edit added.

    Returns
    -------
    set of int
        The 1-based line numbers of every docstring with at least one line in
        ``added``.
    """
    span_lines = set()
    if not path.endswith((".py", ".pyx", ".pxd")):
        return span_lines
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return span_lines
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not ast.get_docstring(node):
            continue
        target = node.body[0]
        span = range(target.lineno, (target.end_lineno or target.lineno) + 1)
        if any(n in added for n in span):
            span_lines.update(span)
    return span_lines


def check(path, commit_range=None):
    """Return the findings for one file, as (line, rule, why, excerpt) tuples."""
    if not path or not os.path.isfile(path):
        return []
    if os.path.basename(path) in EXEMPT_NAMES or not path.endswith(SOURCE_SUFFIXES):
        return []
    # git resolves a path against -C, so give it one that does not move.
    path = os.path.abspath(path)

    added = added_line_numbers(path, commit_range)
    if not added:
        return []
    with open(path, encoding="utf-8", errors="replace") as handle:
        source = handle.read()
    lines = source.splitlines()
    if not path.endswith((".py", ".pyx", ".pxd")):
        added -= unprosed_lines(lines)

    docstring_span = touched_docstring_span(path, source, added)

    findings = []
    for number in sorted(added | docstring_span):
        if number > len(lines):
            continue
        line = lines[number - 1]
        if SPHINX.search(line):
            why = "Use numpydoc sections, not Sphinx field lists"
            if number not in added:
                why += " -- pre-existing, but this docstring was touched elsewhere"
            findings.append((number, "sphinx-field-list", why, line.strip()))
        if number not in added:
            continue
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


def annotated_code(path, source, added, span=10):
    """Yield (line, comment, code) for the comment runs an edit added.

    ``code`` is what follows the comment, with every comment taken out, which
    is what a reader who skips the prose actually sees.
    """
    if not path.endswith((".py", ".pyx", ".pxd")):
        return
    lines = source.splitlines()

    def strip(number):
        """Return a source line with any comment removed, or None if it is only one."""
        text = lines[number - 1]
        bare = text.split("#")[0] if "#" in text and not _in_string(text) else text
        return None if not bare.strip() else bare.rstrip()

    run, start = [], None
    for number, text in enumerate(lines, 1):
        stripped = text.strip()
        if stripped.startswith("#") and number in added:
            run.append(stripped.lstrip("#").strip())
            start = start or number
            continue
        if run:
            code = []
            for following in range(number, min(number + span, len(lines)) + 1):
                kept = strip(following)
                if kept is None and code:
                    break
                if kept is not None:
                    code.append(f"{following}\t{kept}")
            yield start, " ".join(run), "\n".join(code)
            run, start = [], None


def _in_string(text):
    """Whether the first ``#`` of a line sits inside a string literal."""
    return text.count('"', 0, text.index("#")) % 2 or text.count("'", 0, text.index("#")) % 2


def run_explain(paths, commit_range=None):
    """Print each added comment beside the code it annotates. Always succeeds."""
    if commit_range and not paths:
        paths = files_in_range(commit_range)
    total = 0
    for path in paths:
        path = os.path.abspath(path)
        if not os.path.isfile(path) or os.path.basename(path) in EXEMPT_NAMES:
            continue
        added = added_line_numbers(path, commit_range)
        if not added:
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            source = handle.read()
        for line, comment, code in annotated_code(path, source, added):
            print(f"\n=== {os.path.relpath(path)}:{line}")
            print(f"  COMMENT  {comment}")
            print("  CODE WITHOUT IT")
            print("\n".join(f"    {row}" for row in code.splitlines()) or "    (nothing)")
            total += 1
    print(f"\n{total} added comment(s). For each: does the code still say it?")
    print("Yes -> delete the comment. No -> rewrite the code, not the comment.")
    return 0


def files_in_range(commit_range):
    """Return the source files a commit range touches, relative to its root."""
    root = git("git", "rev-parse", "--show-toplevel").stdout.strip()
    listing = git("git", "diff", "--name-only", commit_range).stdout
    return [os.path.join(root, name) for name in listing.splitlines()
            if name.endswith(SOURCE_SUFFIXES)]


def run_as_command(paths, commit_range=None):
    """Report on the named files. Returns 1 if anything was found."""
    if commit_range and not paths:
        paths = files_in_range(commit_range)
    total = 0
    for path in paths:
        findings = check(path, commit_range)
        if findings:
            print(report(os.path.relpath(path), findings, limit=len(findings)))
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
    explain = False
    if args and args[0] == "--explain":
        explain, args = True, args[1:]
    commit_range = None
    if args and args[0] == "--range":
        if len(args) < 2:
            print("--range needs a commit range, such as main...HEAD", file=sys.stderr)
            sys.exit(2)
        commit_range = args[1]
        args = args[2:]
    if explain:
        sys.exit(run_explain(args, commit_range))
    if commit_range or args:
        sys.exit(run_as_command(args, commit_range))
    sys.exit(run_as_hook())
