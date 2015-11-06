.. image:: https://travis-ci.org/OP2/PyOP2.png?branch=master
  :target: https://travis-ci.org/OP2/PyOP2
  :alt: build status

.. contents::

Installing PyOP2
================

The main testing platform for PyOP2 is Ubuntu 12.04 64-bit with Python
2.7.3. Other UNIX-like systems may or may not work. Mac OS X 10.7,
10.9 and 10.10 are also known to work. Microsoft Windows may work, but
is not a supported platform.

Quick start installations
-------------------------

Installation script for Ubuntu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the impatient there is a script for the unattended installation of
PyOP2 and its dependencies on a Ubuntu 12.04 or compatible platform.
Only the sequential and OpenMP backends are covered at the moment.

.. note::
  This script will only work reliably on a clean Ubuntu installation and is
  not intended to be used by PyOP2 developers. If you intend to contribute to
  PyOP2 it is recommended to follow the instructions below for a manual
  installation.

Running with superuser privileges will install missing packages and
Python dependencies will be installed system wide::

  wget -O - https://github.com/OP2/PyOP2/raw/master/install.sh | sudo bash

.. warning::
  This will fail if you if you require a password for ``sudo``. Run e.g. the
  following beforehand to assure your password is cached ::

      sudo whoami

Running without superuser privileges will instruct you which packages
need to be installed. Python dependencies will be installed to the user
site ``~/.local``::

  wget -O - https://github.com/OP2/PyOP2/raw/master/install.sh | bash

In each case, PyOP2 will be cloned to subdirectories of the current directory.

After installation has completed and a rudimentary functionality check,
the test suite is run. The script indicates whether all these steps have
completed successfully and only in this case will exit with return code
0.

Only high-level progress updates are printed to screen. Most of the
output is redirected to a log file ``pyop2_install.log``. Please consult
this log file in the case of errors. If you can't figure out the cause
of discover a bug in the installation script, please `report
it <https://github.com/OP2/PyOP2/issues>`__.

This completes the quick start installation. More complete
instructions follow for virtual machine and native installations.

Provisioning a virtual machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``Vagrantfile`` is provided for automatic provisioning of a Ubuntu
12.04 64bit virtual machine with PyOP2 preinstalled. It requires
`VirtualBox 4.2 <https://www.virtualbox.org/wiki/Linux_Downloads>`__ and
`Vagrant <http://www.vagrantup.com>`__ to be installed, which are
available for Linux, Mac and Windows.

Creating and launching a virtual machine is a single command: run
``vagrant up`` to automatically download the base VM image, configure it
for use with VirtualBox, boot the VM and install PyOP2 and all
dependencies using the above install script.


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

1. Install Xcode.  For OS X 10.9 (Mavericks) this is possible through
   the App Store.  For earlier versions, try
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


.. note::

   If you intend to run PyOP2's OpenMP backend, you should
   additionally pass the following options to the PETSc configure
   stage ::

     --with-threadcomm --with-openmp --with-pthreadclasses

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

.. hint::

   If you only intend to run PyOP2 on CPUs (not GPUs) you can now skip
   straight to :ref:`pyop2-install`, otherwise read on for additional
   dependencies.

.. _cuda-installation:

CUDA backend:
~~~~~~~~~~~~~

Dependencies: 

* boost-python 
* Cusp 0.3.1 
* codepy >= 2013.1 
* Jinja2 
* mako 
* pycparser >= 2.10
* pycuda >= 2013.1

The `cusp library <http://cusplibrary.github.io>`__ version 0.3.1
headers need to be in your (CUDA) include path.

**Note:** Using the trunk version of Cusp will *not* work, since
revision f525d61 introduces a change that break backwards compatibility
with CUDA 4.x.

Install dependencies via the package manager (Debian based systems)::

  sudo apt-get install libboost-python-dev python-jinja2 python-mako python-pycuda

**Note:** The version of pycparser available in the package repositories
is too old, you will need to install it via ``pip``, see below.

Install dependencies via ``pip``::

  sudo pip install codepy Jinja2 mako pycparser>=2.10

If a pycuda package is not available, it will be necessary to install it
manually. Make sure ``nvcc`` is in your ``$PATH`` and ``libcuda.so`` in
your ``$LIBRARY_PATH`` if in a non-standard location::

  export CUDA_ROOT=/usr/local/cuda # change as appropriate 
  git clone https://github.com/inducer/pycuda.git 
  cd pycuda 
  git submodule init 
  git submodule update 
  # libcuda.so is in a non-standard location on Ubuntu systems 
  ./configure.py --no-use-shipped-boost \
  --cudadrv-lib-dir="/usr/lib/nvidia-current,${CUDA_ROOT}/lib,${CUDA_ROOT}/lib64" 
  python setup.py build 
  sudo python setup.py install 
  sudo cp siteconf.py /etc/aksetup-defaults.py

.. _opencl-installation:

OpenCL backend:
~~~~~~~~~~~~~~~

Dependencies: 

* Jinja2 
* mako 
* pycparser >= 2.10
* pyopencl >= 2012.1

pyopencl requires the OpenCL header ``CL/cl.h`` in a standard include
path. On a Debian system, install it via the package manager::

  sudo apt-get install opencl-headers

If you want to use OpenCL headers and/or libraries from a non-standard
location you need to configure pyopencl manually::

  export OPENCL_ROOT=/usr/local/opencl # change as appropriate 
  git clone https://github.com/inducer/pyopencl.git 
  cd pyopencl 
  git submodule init 
  git submodule update 
  ./configure.py --no-use-shipped-boost \
  --cl-inc-dir=${OPENCL_ROOT}/include --cl-lib-dir=${OPENCL_ROOT}/lib 
  python setup.py build 
  sudo python setup.py install

Otherwise, install dependencies via ``pip``::

  sudo pip install Jinja2 mako pyopencl>=2012.1 pycparser>=2.10

Installing the Intel OpenCL toolkit (64bit systems only)::

  cd /tmp 
  # install alien to convert the rpm to a deb package 
  sudo apt-get install alien 
  fakeroot wget http://registrationcenter.intel.com/irc_nas/2563/intel_sdk_for_ocl_applications_2012_x64.tgz
  tar xzf intel_sdk_for_ocl_applications_2012_x64.tgz 
  fakeroot alien *.rpm 
  sudo dpkg -i --force-overwrite *.deb

The ``--force-overwrite`` option is necessary in order to resolve
conflicts with the opencl-headers package (if installed).

Installing the `AMD OpenCL
toolkit <http://developer.amd.com/tools/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/>`__
(32bit and 64bit systems)::

  wget http://developer.amd.com/wordpress/media/2012/11/AMD-APP-SDK-v2.8-lnx64.tgz 
  # on a 32bit system, instead 
  wget http://developer.amd.com/wordpress/media/2012/11/AMD-APP-SDK-v2.8-lnx32.tgz 
  tar xzf AMD-APP-SDK-v2.8-lnx*.tgz 
  # Install to /usr/local instead of /opt 
  sed -ie 's:/opt:/usr/local:g' default-install_lnx*.pl
  sudo ./Install-AMD-APP.sh

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

With all the sequential tests passing, move on to the next backend in the same
manner as required.

.. _PETSc: http://www.mcs.anl.gov/petsc/
.. _petsc4py: http://pythonhosted.org/petsc4py/
