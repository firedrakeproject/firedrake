.. raw:: latex

   \clearpage

.. contents::

====================
Installing Firedrake
====================


Installing Firedrake using pip
==============================

Firedrake is installed in 3 steps:

#. :ref:`Install system dependencies<install_system_dependencies>`
#. :ref:`Install PETSc<install_petsc>`
#. :ref:`Install Firedrake<install_firedrake>`

Supported systems
-----------------

Firedrake has official support for Ubuntu and macOS

but support on other linux systems should be possible

In

* A compiler toolchain (i.e. a C compiler
* Python 3 (parametrise)





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


Modifying the PETSc installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. _install_firedrake:

Installing Firedrake
--------------------

**env

With these environment variables set, Firedrake can now simply be installed with
the command::

pip install --no-binary h5py git+https://github.com/firedrakeproject/firedrake.git

*** note petsc4py cache, also mpi4py and h5py

.. note::
   During the installation Firedrake will compile and install petsc4py_. If
   you have previously installed petsc4py on your computer with a different
   PETSc then ``pip`` will erroneously reuse the existing petsc4py which is 
   linked against the wrong library. To avoid this you need to run the
   command::

       pip cache remove petsc4py

   Equivalent commands may also be necessary for mpi4py and h5py if you are
   changing the MPI and/or HDF5 libraries in use.


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


Alternative installation methods
================================

As well as being installable through ``pip``, Firedrake also provides
`Docker containers<https://hub.docker.com/u/firedrakeproject>`_ and
Jupyter notebooks running on :ref:`Google Colab<google_colab>`.


Having trouble?
===============

:doc:`Slack channel </contact>`.

us on Slack or create a post on github discussions_.


.. _venv: https://docs.python.org/3/tutorial/venv.html
.. _homebrew: https://brew.sh/
.. _PETSc: https://www.mcs.anl.gov/petsc/
.. _discussions: https://github.com/firedrakeproject/firedrake/discussions
.. _issue: https://github.com/firedrakeproject/firedrake/issues
.. _WSL: https://github.com/firedrakeproject/firedrake/wiki/Installing-on-Windows-Subsystem-for-Linux
.. _petsc4py: https://petsc.org/release/petsc4py/reference/petsc4py.html
