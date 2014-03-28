Obtaining Firedrake
===================

Firedrake depends on PyOP2, Fluidity and FFC. It is easiest to obtain
all of these components on Ubuntu Linux and related distributions such
as Mint or Debian. Installation on other Unix-like operating systems
is likely to be possible, although harder. Installation on a Mac is
straightforward using the commands below.

PyOP2
-----

Instructions for obtaining PyOP2 are at :doc:`obtaining_pyop2`. These
also give the instructions for downloading and installing the correct
version of FFC and its dependencies.

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

Next, obtain the Firedrake source from `Github
<http://github.com/firedrakeproject/firedrake>`_: ::

 git clone https://github.com/firedrakeproject/firedrake.git

You will also need to point Python at the right directories. You might
want to consider setting this permanently in your
``.bashrc`` or similar::

  cd firedrake
  export PYTHONPATH=$PWD:$PYTHONPATH

From the Firedrake directory build the relevant modules::

 make
