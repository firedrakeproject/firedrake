---
name: "\U00002699 Installation"
about: Create a report for an installation issue
title: 'INSTALL: '
labels: [installation]
assignees: ''

---

**Describe the error**
A short description of what the installation error is.

**Steps to Reproduce**
Steps to reproduce the behavior:

ie: The exact command used for installation/update  '...'

**Expected behavior**
Describe what you expected to happen.

**Error message**
Add error message with full backtrace.
Please add these as text using three backticks (`) for highlighting.
Please do not add screenshots.

Please also include either `firedrake-install.log` which is found in
the directory `firedrake-install` was run or `firedrake-update.log`
found in the virtualenv directory.

Additionally please include the PETSc configure log located in
`$VIRTUAL_ENV/src/petsc/configure.log`.

**Environment:**
 - OS: [eg: Linux, MacOS, WSL (Windows Subsystem for Linux)] add this as an issue label too! Please be specific: state the Linux distribution or MacOS version you are attempting to install on.
 - Python version: [eg: 3.9.7]
 - Any relevant environment variables or modifications [eg: PETSC_CONFIGURE_OPTIONS=...]

**Additional context**
Add any other context about the problem here.
