.. raw:: latex

   \clearpage

.. contents::

====================
Installing Firedrake
====================

Firedrake is installed in 3 steps:

#. :ref:`Install system dependencies<install_system_dependencies>`
#. :ref:`Install PETSc<install_petsc>`
#. :ref`Install Firedrake<install_firedrake>`


Prerequisites
-------------

Supported systems
-----------------



*** homebrew
*** python (homebrew python! as a note box)


firedrake-configure
-------------------

To simplify the installation process, Firedrake provides a utility script called
``firedrake-configure``. This script can be downloaded by executing::

  curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-configure

*** works by spitting out a string, pipe to file... can be edited, show example of piping directly into apt install

***officially support Ubuntu (x86_64 and aarch64) and macOS (arm64)
``firedrake-configure`` is able to specify the right packages and flags needed to ...

``--no-package-manager``

*** WSL

Prepared configurations
~~~~~~~~~~~~~~~~~~~~~~~

***arches 

* ``default``: ???
* ``complex``: ???


.. _install_system_dependencies:

Installing system dependencies
------------------------------

installing needs a compiler toolchain, python and git

though an MPI distribution is strongly recommended



For apt (x86_64) these are:

.. literalinclude:: apt_deps.txt

Or for homebrew...:

.. literalinclude:: homebrew_deps.txt


.. _install_petsc:

Installing PETSc
----------------


git clone https://github.com/firedrakeproject/petsc.git

`
cd petsc
./configure $(python3 ../firedrake-configure --show-petsc-configure-options)


.. _install_firedrake:

Installing Firedrake
--------------------

**env

With these environment variables set, Firedrake can now simply be installed with
the command::

python3 -m pip install --no-binary h5py git+https://github.com/firedrakeproject/firedrake.git

*** note petsc4py cache, also mpi4py and h5py


Checking the installation
-------------------------

We recommend that you run some simple tests after installation to check
that Firedrake is fully functional. Activate the venv_ as above and
then run::

  cd $VIRTUAL_ENV/src/firedrake
  make check

This command will run a few of the unit tests, which exercise a good
chunk of the functionality of the library. These tests should take a
minute or less. If they fail to run for any reason, please see the
section below on how to diagnose and debug a failed installation.


Updating Firedrake
------------------

TODO

.. _system-requirements:


Having trouble?
---------------

:doc:`Slack channel </contact>`.

us on Slack or create a post on github discussions_.


.. _venv: https://docs.python.org/3/tutorial/venv.html
.. _homebrew: https://brew.sh/
.. _PETSc: https://www.mcs.anl.gov/petsc/
.. _discussions: https://github.com/firedrakeproject/firedrake/discussions
.. _issue: https://github.com/firedrakeproject/firedrake/issues
.. _WSL: https://github.com/firedrakeproject/firedrake/wiki/Installing-on-Windows-Subsystem-for-Linux
