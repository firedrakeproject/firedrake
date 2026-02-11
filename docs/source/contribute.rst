:orphan:

.. raw:: latex

   \clearpage

.. contents::

=========================
Contributing to Firedrake
=========================

As Firedrake developers, nothing makes us happier than receiving
external contributions - it means fewer bugs for us to fix ourselves!
Contributing to Firedrake is not difficult but involves a number of
steps, which we explain below.

Deciding what to fix
--------------------

If you are interested in making your first contribution to Firedrake but don't
know what to fix please take a look at our
`list of issues <https://github.com/firedrakeproject/firedrake/issues>`__. In
particular look out for issues labelled as a
`'good first issue' <https://github.com/firedrakeproject/firedrake/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22>`__.

Once you have chosen what to fix please :doc:`let us know<contact>`
so we can help you and stay informed.

.. _main_vs_release:

``main`` or ``release`` branch?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Firedrake development takes place on two different branches, called ``main``
and ``release``. ``main`` is where major Firedrake developments occur -
things like new features and API-breaking changes. Changes added to ``main``
will be made available at the next major Firedrake release (e.g.
``2025.10.x`` to ``2026.4.0``) every 6 months. ``release``
is the stable development branch of Firedrake and will only accept
bug fixes and documentation or website improvements. Changes to ``release``
are made available in Firedrake's patch releases (e.g. ``2025.10.2`` to
``2025.10.3``), which are made as needed.

When you submit changes you should consider which branch is the better
candidate for your changes, then base your development from that branch
and submit your PR targetting it.

Setting up a developer environment
----------------------------------

To edit Firedrake code you will need an editable install of either the
``main`` or ``release`` branch. Instructions on how to do this can be
found :ref:`here <dev_install>`.

If you are using a :ref:`developer Docker container <dev_containers>` to edit
Firedrake code then an editable install of the correct branch of
Firedrake can be found in ``/opt/firedrake``.

Working with git
----------------

Once you have a working install of Firedrake you will need to checkout
a new branch to develop on. This branch will have to come from a fork
of the main `Firedrake repository <https://github.com/firedrakeproject/firedrake>`__. For clarity we recommend you give
your branch a helpful name like ``mygithubusername/fix-bug-with-XXX``.

When you are ready for the Firedrake development team to take a look at
your code please open a pull request (PR) from your branch into the
`Firedrake repository <https://github.com/firedrakeproject/firedrake>`__.
You are encouraged to do this early on in the development process so
that we can give you feedback - just make sure to mark the PR
as a 'draft'.

Once you have opened a PR the Firedrake developers will review your
changes and suggest changes. Once everything has been addressed and
approved then we will merge the code.

.. _presubmission_checks:

Pre-submission checklist
------------------------

Before opening a PR/marking it as ready for review you should make sure
to have done the following:

#. One or more tests should be added to the Firedrake test suite to verify
   the new functionality works as intended.

#. Code should conform to Firedrake's `coding guide <https://github.com/firedrakeproject/firedrake/wiki/Firedrake-Coding-Guide>`__.

#. Code should pass linting checks. To do this check locally you should
   run the command::

      $ make srclint

#. Documentation changes should be checked to make sure that they are
   correctly rendered. To build the documentation locally you should
   run::

      $ cd docs
      $ make html

   and then inspect the generate HTML files in ``docs/build/html``.

Contributing demos
------------------

If you have implemented a complex method using Firedrake then we welcome
submissions of 'demos' (examples :ref:`here <intro_tutorials>` and
:ref:`here <advanced_tutorials>`). Demos are short Python scripts
(<100 lines of code) that are augmented with RST such that they can be
nicely rendered on the website.

A good demo should convey the critical steps necessary to implement your
method for others to learn from. It should **not** be a full application.

To contribute a new demo the same checks as
:ref:`above <presubmission_checks>` should be followed with some
demo-specific changes:

#. Demos should be merged into the ``release`` branch of Firedrake,
   not ``main``.

#. The demo should live in a subdirectory of the ``demos/`` directory.

#. To test the demo it must be added to
   ``tests/firedrake/demos/test_demos_run.py``.

#. The demo should be referenced in ``docs/source/intro_tut.rst``
   or ``docs/source/advanced_tut.rst`` as appropriate.

For inspiration, an examplar pull request adding several demos to Firedrake
may be found
`here <https://github.com/firedrakeproject/firedrake/pull/4317>`__.
