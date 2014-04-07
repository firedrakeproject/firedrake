Obtaining Firedrake
===================

Firedrake depends on PyOP2_, FFC_, FIAT_, and UFL_. It is easiest to obtain
all of these components on Ubuntu Linux and related distributions such as Mint
or Debian. Installation on other Unix-like operating systems is likely to be
possible, although harder. Installation on a Mac is straightforward using the
commands below.

PyOP2
-----

Instructions for obtaining PyOP2 and its dependencies are at
:doc:`obtaining_pyop2`.

FFC, FIAT and UFL
-----------------

Firedrake currently requires a fork of FFC_, UFL_ and FIAT_.  Note that FFC_
requires a version of Instant_.

Install FFC_ and all dependencies via pip::

  sudo pip install \
    git+https://bitbucket.org/mapdes/ffc.git#egg=ffc
    git+https://bitbucket.org/mapdes/ufl.git#egg=ufl
    git+https://bitbucket.org/mapdes/fiat.git#egg=fiat
    git+https://bitbucket.org/fenics-project/instant.git#egg=instant
    hg+https://bitbucket.org/khinsen/scientificpython


Firedrake
---------

In addition to PyOP2, you will need to install Firedrake. There are two
routes, depending on whether you intend to contribute to Firedrake
development.

Pip instructions for users
..........................

If you only wish to use Firedrake, and will not be contributing to
development at all, you can install Firedrake using pip::

  sudo pip install git+https://github.com/firedrakeproject/firedrake.git

or::

  pip install --user git+https://github.com/firedrakeproject/firedrake.git

You're now ready to go. You might like to start with the tutorial
examples on the :doc:`documentation page <documentation>`.

Git instructions for developers
...............................

Next, obtain the Firedrake source from GitHub_

 git clone https://github.com/firedrakeproject/firedrake.git

You will also need to point Python at the right directories. You might
want to consider setting this permanently in your
``.bashrc`` or similar::

  cd firedrake
  export PYTHONPATH=$PWD:$PYTHONPATH

From the Firedrake directory build the relevant modules::

 make

.. _PyOP2: http://op2.github.io/PyOP2
.. _FFC: https://bitbucket.org/mapdes/ffc
.. _FIAT: https://bitbucket.org/mapdes/fiat
.. _UFL: https://bitbucket.org/mapdes/ufl
.. _Instant: https://bitbucket.org/fenics-project/instant
.. _GitHub: https://github.com/firedrakeproject/firedrake
