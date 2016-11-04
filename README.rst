.. image:: https://travis-ci.org/OP2/PyOP2.png?branch=master
  :target: https://travis-ci.org/OP2/PyOP2
  :alt: build status

.. contents::

Installing PyOP2
================

The main testing platform for PyOP2 is Ubuntu 14.04 64-bit with Python
2.7. Other UNIX-like systems may or may not work. Mac OS X 10.7-10.12
are also known to work.


Manual Installation
-------------------

Dependencies
~~~~~~~~~~~~

.. hint::

   You can skip over the dependencies list for now, since the
   instructions below tell you how to install each of these packages.

PyOP2 requires a number of tools and libraries to be available:

* A C compiler (for example gcc or clang), make
* A Fortran compiler (for PETSc)
* MPI
* Blas and Lapack
* Git, Mercurial
* Python version 2.7
* pip and the Python headers 

The following dependencies are part of the Python subsystem:

* Cython >= 0.22
* decorator 
* numpy >= 1.9.1 
* mpi4py >= 1.3.1

PETSc. We require very recent versions of PETSc so you will need to follow the specific instructions given below to install the right version.

* PETSc_
* PETSc4py_

COFFEE. We require the current master version of COFFEE for which you will need to follow the instructions given below.

Testing dependencies (optional, required to run the tests):

* pytest >= 2.3
* flake8 >= 2.1.0
* gmsh
* triangle

With the exception of the PETSc_ dependencies, these can be installed
using the package management system of your OS, or via ``pip``.

Installing packages with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install dependencies system-wide use ``sudo pip install ...``, to
install to a user site use ``pip install --user ...``. If you don't want
PyOP2 or its dependencies interfering with your existing Python environment,
consider creating a `virtualenv <http://virtualenv.org/>`__.

.. note::

   In the following we will use ``sudo pip install ...``. If
   you want either of the other options you should change the command
   appropriately.

.. note::

   Installing to the user site does not always give packages
   priority over system installed packages on your ``sys.path``.


Obtaining a build environment on Ubuntu and similar systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a Debian-based system (Ubuntu, Mint, etc.) install core packages::

  sudo apt-get install -y build-essential python-dev git-core \
    mercurial python-pip libopenmpi-dev openmpi-bin libblas-dev \
    liblapack-dev gfortran

.. note::

   This may not give you recent enough versions of those packages
   (in particular the Cython version shipped with 12.04 is too old). You
   can selectively upgrade packages via ``pip``, see below.

Install dependencies via ``pip``::

  sudo pip install "Cython>=0.22" decorator "numpy>=1.6" "mpi4py>=1.3.1"

.. hint::
   
   You can now skip down to installing :ref:`petsc-install`.

.. _mac-install:

Obtaining a build environment on Mac OS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend using `Homebrew <http://brew.sh>`__ as a package manager
for the required packages on Mac OS systems.  Obtaining a build
environment for PyOP2 consists of the following:

1. Install Xcode.  For OS X 10.9 (Mavericks) and later this is
   possible through the App Store.  For earlier versions, try
   https://developer.apple.com/downloads (note that on OS X 10.7
   (Lion) you will need to obtain Xcode 4.6 rather than Xcode 5)

2. If you did not install Xcode 5, you will need to additionally
   install the Xcode command line tools through the downloads section
   of Xcode's preferences

3. Install homebrew, following the instructions at http://brew.sh

4. Install an MPI library (PyOP2 is tested with openmpi)::

     brew install openmpi

5. Install an up-to-date Python via homebrew::

     brew install python

   .. note::

      Do not follow the instructions to update pip, since they
      currently result in a broken pip installation (see
      https://github.com/Homebrew/homebrew/issues/26900)

6. Install numpy via homebrew::

     brew tap homebrew/python
     brew install numpy

7. Install python dependencies via pip::

     pip install decorator
     pip install cython
     pip install mpi4py
     pip install pytest
     pip install flake8

.. hint::

   Your system is now ready to move on to installation of PETSc_ and
   petsc4py_ described below.  

.. note::

   On Mac OS we do not recommend using sudo when installing, as such
   when following instructions below to install with pip just remove
   the ``sudo`` portion of the command.

.. _petsc-install:

PETSc
~~~~~

PyOP2 uses petsc4py_, the Python bindings for the PETSc_ linear algebra
library and requires:

* an MPI implementation built with *shared libraries* 
* A suitable very recent PETSc_ master branch built with *shared libraries*

The version of PETSc_ you install *must* be configured with HDF5
support.  This either requires appropriate operating system packages,
or else asking PETSc_ to download and build a compatible HDF5
(instructions below).

If you have a suitable PETSc_ installed on your system, ``PETSC_DIR``
and ``PETSC_ARCH`` need to be set for the petsc4py_ installer to find
it. 

.. note::

   There are no current OS PETSc packages which are new
   enough. Therefore, unless you really know you should be doing
   otherwise, always install PETSc_ using pip. The following
   instructions will install the firedrake branch of PETSc_ and
   petsc4py_. This is a recent version of the upstream master branch
   which has been verified to at least build correctly. You may also
   use the upstream next or master branch, but be aware that these are
   rapidly developing and tend to break regularly.

Then install PETSc_ via ``pip`` ::

  sudo PETSC_CONFIGURE_OPTIONS="--download-ctetgen --download-triangle --download-chaco --download-hdf5" \
    pip install https://bitbucket.org/mapdes/petsc/get/firedrake.tar.bz2
  unset PETSC_DIR
  unset PETSC_ARCH


If you built PETSc_ using ``pip``, ``PETSC_DIR`` and ``PETSC_ARCH``
should be left unset when building petsc4py_.

Install petsc4py_ via ``pip``::

  sudo pip install git+https://bitbucket.org/mapdes/petsc4py.git@firedrake#egg=petsc4py

If you have previously installed and older version of PETSc_ or petsc4py_,
``pip`` might tell you that the requirements are already satisfied when running
above commands. In that case, use ``pip install -U --no-deps`` to upgrade
(``--no-deps`` prevents also recursively upgrading any dependencies).

.. _coffee-install:

COFFEE
~~~~~~

If you do not intend to develop COFFEE, you can simply install it using ``pip``::

  sudo pip install git+https://github.com/coneoproject/COFFEE.git

If you *do* intend to contribute to COFFEE, then clone the repository::

  git clone git@github.com:coneoproject/COFFEE.git

COFFEE can be installed from the repository via::

  sudo python setup.py install

HDF5
~~~~

PyOP2 allows initializing data structures using data stored in HDF5
files. To use this feature you need the optional dependency
`h5py <http://h5py.org>`__.  This installation should be linked
against the *same* version of the HDF5 library used to build PETSc_.

.. _pyop2-install:

Building PyOP2
--------------

Clone the PyOP2 repository::

  git clone git://github.com/OP2/PyOP2.git
 
PyOP2 uses `Cython <http://cython.org>`__ extension modules, which need to be built
in-place when using PyOP2 from the source tree::

  python setup.py build_ext --inplace

When running PyOP2 from the source tree, make sure it is on your
``$PYTHONPATH``::

  export PYTHONPATH=/path/to/PyOP2:$PYTHONPATH

When installing PyOP2 via ``python setup.py install`` the extension
modules will be built automatically and amending ``$PYTHONPATH`` is not
necessary.

Setting up the environment
--------------------------

To make sure PyOP2 finds all its dependencies, create a file ``.env``
e.g. in your PyOP2 root directory and source it via ``. .env`` when
using PyOP2. Use the template below, adjusting paths and removing
definitions as necessary::

  #PETSc installation, not necessary when PETSc was installed via pip
  export PETSC_DIR=/path/to/petsc 
  export PETSC_ARCH=linux-gnu-c-opt

  #Add PyOP2 to PYTHONPATH
  export PYTHONPATH=/path/to/PyOP2:$PYTHONPATH

Alternatively, package the configuration in an `environment
module <http://modules.sourceforge.net/>`__.

Testing your installation
-------------------------

PyOP2 unit tests use `pytest <http://pytest.org>`__ >= 2.3. Install via package
manager::

  sudo apt-get install python-pytest

or pip::

  sudo pip install "pytest>=2.3"

The code linting test uses `flake8 <http://flake8.readthedocs.org>`__.
Install via pip::

  sudo pip install "flake8>=2.1.0"

If you install *pytest* and *flake8* using ``pip --user``, you should
include the binary folder of your local site in your path by adding the
following to ``~/.bashrc`` or ``.env``::

  # Add pytest binaries to the path
  export PATH=${PATH}:${HOME}/.local/bin

If all tests in our test suite pass, you should be good to go::

  make test

This will run code linting and unit tests, attempting to run for all backends
and skipping those for not available backends.

Troubleshooting
---------------

Start by verifying that PyOP2 picks up the "correct" dependencies, in
particular if you have several versions of a Python package installed in
different places on the system.

Run ``pydoc <module>`` to find out where a module/package is loaded
from. To print the module search path, run::

 python -c 'from pprint import pprint; import sys; pprint(sys.path)'

Troubleshooting test failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the tests as follows, to abort after the first failed test:

Start with the unit tests with the sequential backend ::

  py.test test/unit -vsx --tb=short --backend=sequential


.. _PETSc: http://www.mcs.anl.gov/petsc/
.. _petsc4py: http://pythonhosted.org/petsc4py/
