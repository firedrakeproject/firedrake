.. raw:: latex

   \clearpage

=========================
Contributing to Firedrake
=========================

...

All Firedrake development is managed on GitHub and Slack...

Deciding what to fix
--------------------

If you are interested in making your first contribution to Firedrake but don't
know what to fix please take a look at our (extensive!)
`list of issues <https://github.com/firedrakeproject/firedrake/issues>`__. In
particular look out for issues labelled as a
`'good first issue' <https://github.com/firedrakeproject/firedrake/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22>`__.
Once you've chosen one to attempt to fix please comment on the issue thread
to notify us and so we can give you advice.

If you already know what you want to fix then great! Before you begin please
do :doc:`get in touch<contact>` on Slack or GitHub discussions so we know
that it is happening.

.. _main_vs_release:


``main`` or ``release`` branch?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Firedrake development takes place on two different branches, called ``main``
and ``release``. ``main`` is where major Firedrake developments occur -
things like new features and API-breaking changes. Changes added to ``main``
will be made available at the next major Firedrake release (e.g.
``2025.10.x`` to ``2026.4.0``) every 6 months. In contrast, ``release``
is the stable development branch of Firedrake and will only accept
bug fixes and documentation or website improvements. Changes to ``release``
are made available in Firedrake's patch releases (e.g. ``2025.10.2`` to
``2025.10.3``), which are made as needed.

When you submit changes you should consider which branch is the better
candidate for your changes and base your development from that branch
and submit your PR targetting it.

Setting up a developer environment
----------------------------------

To edit Firedrake code you will need an editable install of either the
``main`` or ``release`` branch. Instructions on how to do this can be
found :ref:`here<dev_install>`.

If you are using a :ref:`Docker container<dev_containers>` to edit
Firedrake code then an editable install of the correct branch of
Firedrake can be found in ``/opt/firedrake``.

Working with git
----------------

Once you have a working install of Firedrake you will need to checkout
a new branch to develop on. This branch will have to come from a fork
of the main Firedrake repository. For clarity we recommend you give
your branch a helpful name like
``mygithubusername/fix-bug-where-XXX-happens``.

When you are ready for the Firedrake development team to take a look at
your code please open a pull request (PR) from your branch into the
`Firedrake repository <https://github.com/firedrakeproject/firedrake>`__.
You are encouraged to do this early on in the development process so
that we can give you early feedback - just make sure to mark the PR
as a 'draft'!

Once you have opened a PR the Firedrake developers will review your
changes and suggest changes. Once everything has been addressed and there
is an approving review then we will merge the code.

Developing Firedrake code
-------------------------

branch + fork

policies (style)
documentation
tests
linting

Contributing documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

cd firedrake_repo/docs
make html

Contributing demos
~~~~~~~~~~~~~~~~~~
