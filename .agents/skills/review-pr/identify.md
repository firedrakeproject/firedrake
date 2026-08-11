Run each command as a separate Bash call with no shell metacharacters (no `$(...)`, pipes, `;`,
`&&`/`||`, redirections, here-docs).

### 1. Identify the PR — resolve `<PR_NUMBER>`

- Number given (e.g. `5215`, `#5215`, or a `github.com/.../pull/5215` URL) → use it.
- Diff file given → ask the user for the source branch, then `gh pr list --head <branch>`.
- Nothing given → `git branch --show-current` (if empty or detached, ask); then
  `gh pr list --head <branch>`.

If `gh pr list` returns 0 PRs, stop and report. If more than one, ask which number.

### 2. Get PR metadata

`gh pr view <PR_NUMBER> --json number,title,body,isDraft,headRefName,headRefOid,baseRefName,additions,deletions,changedFiles,labels`

Record `headRefOid` as `<PR_HEAD_SHA>`, and `baseRefName` as `<BASE>`.

### 3. Stacked-PR and drift checks

**Stacking.** Firedrake PRs are frequently stacked: `<BASE>` is often another feature branch, not
`main` or `release`. This is load-bearing for review scope.

- **Review only the diff against `<BASE>`.** `gh pr diff <PR_NUMBER>` already does this. Never diff
  a stacked PR against `main` or `release`. That reads the parent PR's changes as this one's, and it
  produces findings the author cannot act on here.
- Say so at the top of the report when `<BASE>` is neither `main` nor `release`. Name the base
  branch, and say that the changes the parent introduced are out of scope.
- A PR based on a non-default branch needs the `base:main` or `base:release` label for CI to resolve
  its base build. If `<BASE>` is a feature branch and neither label is present, report it as a HIGH
  finding — CI cannot run without it.

**Drift.** The PR may not match what is checked out locally.

- `git show-ref --verify --quiet refs/heads/<headRefName>` — if non-zero, skip this check.
- Else `git rev-parse <headRefName>`. Warn that the local branch and the PR head have diverged if it
  differs from `<PR_HEAD_SHA>`. Recommend `/review-branch` if the user meant to review local work.

Section 5 requires reopening cited code. Verify a finding against the fetched PR head when local and
remote have drifted (`gh pr diff`, or `gh api` file contents at `<PR_HEAD_SHA>`). The working tree
is the wrong source. It makes you confirm or drop a finding against code the PR does not hold.
