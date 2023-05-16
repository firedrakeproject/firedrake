---
name: "Bug fix or new feature"
description: ''
title: ''
labels: ''
assignees: ''

---

# Description
Please include a summary of the changes introduced by this PR.
Additionally include any relevant motivation and context. List any
dependencies that are required for this change, other PRs upon
which this may depend, and associated issues.

## Associated Pull Requests:
- Make a list (with links!)
- [Like this PR#691](https://github.com/OP2/PyOP2/pull/691)

## Fixes Issues:
- Make a list of issues (with links!)
- [Like this Issue#2824](https://github.com/firedrakeproject/firedrake/issues/2824)

If issues are fixed by this PR, prepend each of them with the word
"fixes", so they are automatically closed when this PR is merged. For
example "fixes #123, fixes #456".

# Checklist for author:

<!--
If you think an option is not relevant to your PR, do not delete it but use ~strikethrough formating on it~. This helps keeping track of the entire list.
-->

- [ ] I have linted the codebase (`make lint` in the `firedrake` source directory).
- [ ] My changes generate no new warnings.
- [ ] All of my functions and classes have appropriate docstrings.
- [ ] I have commented my code where its purpose may be unclear.
- [ ] I have included and updated any relevant documentation.
- [ ] Documentation builds locally (`make linkcheck; make html; make latexpdf` in `firedrake/docs` directory)
- [ ] I have added tests specific to the issues fixed in this PR.
- [ ] I have added tests that exercise the new functionality I have introduced
- [ ] Tests pass locally (`pytest tests` in the `firedrake` source directory) (useful, but not essential if you don't have suitable hardware).
- [ ] I have performed a self-review of my own code using the below guidelines.

# Checklist for reviewer:

- [ ] Docstrings present
- [ ] New tests present
- [ ] Code correctly commented
- [ ] No bad "code smells"
- [ ] No issues in parallel
- [ ] No CI issues (excessive parallelism/memory usage/time/warnings generated)
- [ ] Upstream/dependent branches and PRs are ready

Feel free to add reviewers if you know there is someone who is already aware of this work.

Please open this PR initially as a draft and mark as ready for review once CI tests are passing.

<!--
Thanks for contributing!
-->
