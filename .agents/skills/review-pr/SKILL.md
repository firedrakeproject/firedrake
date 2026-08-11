---
name: review-pr
description: Review code changes in a Firedrake GitHub pull request and report findings to stdout. Use when the user asks to "review this PR", "review PR <number>", "review #N", or provides a pull request URL, and wants the review printed rather than posted as comments.
argument-hint: <PR_NUMBER | diff-file | empty for current branch>
---

Reviews the **remote PR state**, not local `HEAD`.

Read `AGENTS.md` at the repository root before reviewing. Its Core Working Rules, Coding Style And
Conventions, and Anti-Patterns sections are the rubric. The Anti-Patterns section is written for
review time, not for authoring time alone.

## Identify and fetch

Follow @identify.md (Sections 1–3) to resolve `<PR_NUMBER>`, fetch metadata, and check for
local-vs-remote drift.

## Review

Follow @review-procedure.md (Sections 4–7) to read the diff, classify findings, verify each one, and
compose the report.

## Report

Print the report. Do not write files and do not post anything to GitHub.
