Obtaining Firedrake
===================

Firedrake depends on PyOP2, Fluidity and FFC. It is easiset to obtain
all of these components on Ubuntu Linux and related distributions such
as Mint or Debian. Installation on other Unix-like operating systems
is likely to be possible, although harder.

PyOP2
-----

Instructions for obtaining PyOP2 are at :doc:`obtaining_pyop2`. These
also give the instructions fordownloading and installing the correct
version of FFC and its dependencies.

Firedrake
---------

The Firedrake layer is a branch of Fluidity. *Having completed* the
installation of PyOP2, it is necessary to install the dependencies for
Fluidity. On an Ubuntu or similar system, this is easiest to achieve
by installing the ``fluidity-dev`` package from the Fluidity PPA::

  sudo apt-add-repository ppa:fluidity-core/ppa
  sudo apt-get update
  sudo apt-get install fluidity-dev

Users of other systems should follow the instructions for installing
the Fluidity dependencies in appendix C of the Fluidity manual,
obtainable from `Launchpad
<https://launchpad.net/fluidity/+download>`_.

Next, obtain the Firedrake source from `Github
<http://github.com/firedrakeproject/firedrake>`_: ::

 git clone https://github.com/firedrakeproject/firedrake.git

You will also need to point Python and the dynamic linker at the right
directories. You might want to consider setting these directories
permanently in your ``.bashrc`` or similar::

  cd firedrake
  export PYTHONPATH=$PWD/python:$PYTHONPATH
  export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH

From the Firedrake directory, configure and build::

 ./configure
 make

You can build in parallel by using ``make -jn`` where ``n`` is the
number of parallel build jobs to use.
