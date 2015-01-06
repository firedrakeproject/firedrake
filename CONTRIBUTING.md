# Contributing to PyOP2

We value third-party contributions. To keep things simple for you and us,
please adhere to the following contributing guidelines.

## Getting Started

* You will need a [GitHub account](https://github.com/signup/free).
* Submit a [ticket for your issue][0], assuming one does not already exist.
  * Clearly describe the issue including steps to reproduce when it is a bug.
  * Make sure you specify the version that you know has the issue.
  * Bonus points for submitting a failing test along with the ticket.
* If you don't have push access, fork the repository on GitHub.

## Making Changes

* Create a topic branch for your feature or bug fix.
* Make commits of logical units.
* Make sure your commits adhere to the coding guidelines below.
* Make sure your commit messages are in the [proper format][1]: The first line
  of the message should have 50 characters or less, separated by a blank line
  from the (optional) body. The body should be wrapped at 70 characters and
  paragraphs separated by blank lines. Bulleted lists are also fine.
* Make sure you have added the necessary tests for your changes.
* Run _all_ the tests to assure nothing else was accidentally broken.

## Coding guidelines

[PEP 0008][2] is enforced, with the exception of [E501][3] and [E226][3]:
* Indent by 4 spaces, tabs are *strictly forbidden*.
* Lines should not exceed 79 characters where possible without severely
  impacting legibility. If breaking a line would make the code much less
  readable it's fine to overrun by a little bit.
* No trailing whitespace at EOL or trailing blank lines at EOF.

## Checking your commit conforms to coding guidelines

Install a Git pre-commit hook automatically checking for tab and whitespace
errors before committing and also calls `flake8` on your changed files. In the
`.git/hooks` directory of your local Git repository, run the following:

```
git config --local core.whitespace "space-before-tab, tab-in-indent, trailing-space, tabwidth=4"
wget https://gist.github.com/kynan/d233073b66e860c41484/raw/pre-commit
chmod +x pre-commit
```

Make sure the `pre-commit.sample` hook is still in place, since it is required.

## Submitting Changes

* We can only accept your contribution if you have signed the Contributor
  License Agreement (CLA).
* Push your changes to a topic branch in your fork of the repository.
* Submit a pull request to the repository in the OP2 organization.

[0]: https://github.com/OP2/PyOP2/issues
[1]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
[2]: http://www.python.org/dev/peps/pep-0008/
[3]: http://pep8.readthedocs.org/en/latest/intro.html#error-codes
