.. raw:: latex

   \clearpage

===================
Obtaining Firedrake
===================

Firedrake is installed using its install script::

  curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install

In the simplest cases, such as on a Mac with Homebrew installed or on
an Ubuntu workstation on which the user has sudo acccess, the user can simply run::

  python3 firedrake-install

Running ``firedrake-install`` with no arguments will install Firedrake in
a python venv_ created in a ``firedrake`` subdirectory of the
current directory. Run::

  python3 firedrake-install --help

for a full list of install options.  In particular, you may wish to
customise PETSc by adding packages (for instance ``--download-fftw``).
To do so, set the environment variable ``PETSC_CONFIGURE_OPTIONS``
before running ``firedrake-install``. However, some configuration
options (for instance ``--LDFLAGS`` and ``--CFLAGS``) should not be set
in ``PETSC_CONFIGURE_OPTIONS`` as they are set by the install script.
You can see the set of options passed to PETSc by providing the flag
``--show-petsc-configure-options``.

You will need to activate the venv in each shell from which you
use Firedrake::

  source firedrake/bin/activate

.. note::

   Should you use ``csh``, you will need::

     source firedrake/bin/activate.csh


Installation and MPI
--------------------

By default, ``firedrake-install`` will prompt the PETSc installer to
download and install its own MPICH library and executables in the
virtual environment.  This has implications for the performance of the
resulting library when run in parallel.  Instructions on how best to
configure MPI for the installation process are `found here
<https://www.firedrakeproject.org/parallelism.html>`_.


Testing the installation
------------------------

We recommend that you run some simple tests after installation to check
that Firedrake is fully functional. Activate the venv_ as above and
then run::

  cd $VIRTUAL_ENV/src/firedrake
  make check

This command will run a few of the unit tests, which exercise a good
chunk of the functionality of the library. These tests should take a
minute or less. If they fail to run for any reason, please see the
section below on how to diagnose and debug a failed installation.

.. note::

  There is a known issue which causes parallel tests to hang without
  failing. This is particularly a problem on MacOS and is due to the
  version of MPICH installed with Firedrake failing to resolve the
  local host at ip address ``127.0.0.1``. To resolve this issue modify
  the hosts database at ``/etc/hosts`` to include the entries::

    127.0.0.1       LOCALHOSTNAME.local
    127.0.0.1       LOCALHOSTNAME

  where ``LOCALHOSTNAME`` is the name returned by running the `hostname`
  command. Should the local host name change, this may require updating.

Upgrade
-------

The install script will install an upgrade script in
`firedrake/bin/firedrake-update`. Running this script will update
Firedrake and all its dependencies.

.. note::

   You should activate the venv_ before running
   `firedrake-update`.

Just like the ``firedrake-install`` script, running::

    firedrake-update --help

gives a full list of update options. For instance additional Firedrake
packages can be installed into an existing Firedrake installation using
``firedrake-update``.

.. _system-requirements:

System requirements
-------------------

Firedrake requires Python 3.10 to 3.13. The installation script is
tested by CI on Ubuntu 24.04 LTS. On Ubuntu 22.04 or later, the system
installed Python 3 is supported. On MacOS, the homebrew_ installed
Python 3 is supported::

  brew install python3

Installation is likely to work well on other Linux platforms, although
the script may fail if dependency packages are not already installed.
Installation on other Unix platforms may work but is untested. On Linux
systems that do not use the Debian package management system, it will be
necessary to pass the `--no-package-manager` option to the install
script. In this case, it is the user's responsibilty to ensure that
they have the system dependencies:

* A C and C++ compiler (for example gcc/g++ or clang), GNU make
* A Fortran compiler (for PETSc)
* Blas and Lapack
* Git, Mercurial
* Python version 3.10-3.13
* The Python headers
* autoconf, automake, libtool
* CMake
* zlib
* flex, bison
* Ninja
* pkg-config

Firedrake has been successfully installed on Windows 10 using the
Windows Subsystem for Linux. There are more detailed instructions for
WSL_ on the Firedrake wiki.
Installation on previous versions of Windows is unlikely to work.

System anti-requirements
~~~~~~~~~~~~~~~~~~~~~~~~

We strive to make Firedrake work on as many platforms as we can. Some
tools, however, make this challenging or impossible for end users.

**Anaconda.** The Anaconda Python distribution and package manager are
often recommended in introductory data science courses because it does
effectively handle many aggravating problems of dependency management.
Unfortunately, Anaconda does a poor job of isolating itself from the
rest of your system and assumes that it will be both the only Python
installation and the only supplier of any dependent packages. Anaconda
will install compilers and MPI compiler wrappers and put its compilers
right at the top of your ``PATH``. This is a problem because Firedrake
needs to build and use its own MPI. (We keep our MPI isolated from the
rest of your system through virtual environments.) When installed on a
platform with Anaconda, Firedrake can accidentally try to link to the
incompatible Anaconda installation of MPI.

There are three ways to work around this problem.

1. Remove Anaconda entirely.
2. Modify your ``PATH`` environment variable to remove any traces of
   Anaconda, then install Firedrake. If you need Anaconda later, you
   can re-enable it with a shell script that will add those directories
   back onto your path.
3. Use a `Docker image <https://hub.docker.com/r/firedrakeproject/firedrake>`_
   that we've built with Firedrake and its dependencies already installed.

**MacOS system Python.** The official MacOS installer on the Python
website does not have a working SSL by default. A working SSL is
necessary to securely fetch dependent packages from the internet. You
can enable SSL with the system Python, but we strongly recommend using
a Python version installed via Homebrew instead.

**MacPorts.**
Mac OS has multiple competing package managers which sometimes cause
issues for users attempting to install Firedrake. In particular, the
assembler provided by MacPorts is incompatible with the Mac system
compilers in a manner which causes Firedrake to fail to install. For
this reason, if you are installing Firedrake on a Mac which also has
MacPorts installed, you should ensure that ``/opt/local/bin`` and
``/opt/local/sbin`` are removed from your ``PATH`` when installing or
using Firedrake. This should ensure that no MacPorts installed tools
are found.

Debugging install problems
--------------------------

If ``firedrake-install`` fails, the following flowchart describes some
common build problems and how to solve them. If you understand the
prognosis and feel comfortable making these fixes yourself then great!
If not, feel free to ask for more help in our
:doc:`Slack channel </contact>`.

.. graphviz:: install-debug.dot

If you don't see the issue you're experiencing in this chart, please ask
us on Slack or create a post on github discussions_.
To help us diagnose what's going wrong, **please include the following log files**:

* ``firedrake-install.log`` from Firedrake, which you can find in the
  directory where you invoked ``firedrake-install`` from
* ``configure.log`` and ``make.log`` from PETSc, which you can find in
  ``src/petsc/`` inside the directory where Firedrake virtual
  environment was created

Likewise, if it's ``firedrake-update`` that fails, please include the
file ``firedrake-update.log``. You can find this in the Firedrake
virtual environment.

Recovering from a broken installation script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you find yourself in the unfortunate position that
``firedrake-update`` won't run because of a bug, and the bug has been
fixed in Firedrake master, then the following procedure will rebuild
``firedrake-update`` using the latest version.

From the top directory of your Firedrake install,
type::

  cd src/firedrake
  git pull
  ./scripts/firedrake-install --rebuild-script

You should now be able to run ``firedrake-update``.

Installing Firedrake with pip (experimental)
--------------------------------------------------------

Firedrake has experimental support for installing using ``pip``, avoiding the need for the ``firedrake-install`` script.

Requirements
~~~~~~~~~~~~

* An activated virtual environment.
* All the system requirements listed in :ref:`system-requirements`.
* A Firedrake-compatible PETSc installation (using our `fork of PETSc <https://github.com/firedrakeproject/petsc.git>`_). The set of flags passed to PETSc can be retrieved by passing the command ``--show-petsc-configure-options`` to ``firedrake-install``.
*  The following environment variables to be set:

  * ``PETSC_DIR`` and ``PETSC_ARCH`` to point to the correct location for the PETSc installation.
  * ``HDF5_DIR`` to ``$PETSC_DIR/$PETSC_ARCH``.
  * ``CC`` and ``MPICC`` to point to the ``mpicc`` compiler wrapper.
  * ``CXX`` to point to the ``mpicxx`` compiler wrapper.

Installation
~~~~~~~~~~~~

Having set up this environment, Firedrake can now be installed with the command::

  pip install --no-binary mpi4py,h5py git+https://github.com/firedrakeproject/firedrake.git

Removing Firedrake
------------------
Firedrake and its dependencies can be removed by deleting the Firedrake
install directory. This is usually the ``firedrake`` subdirectory
created after having run ``firedrake-install``. Note that this will not
undo the installation of any system packages which are Firedrake
dependencies: removing these might affect subsequently installed
packages for which these are also dependencies.

.. _Paraview: http://www.paraview.org
.. _venv: https://docs.python.org/3/tutorial/venv.html
.. _homebrew: https://brew.sh/
.. _PETSc: https://www.mcs.anl.gov/petsc/
.. _discussions: https://github.com/firedrakeproject/firedrake/discussions
.. _issue: https://github.com/firedrakeproject/firedrake/issues
.. _WSL: https://github.com/firedrakeproject/firedrake/wiki/Installing-on-Windows-Subsystem-for-Linux

Visualisation software
----------------------

Firedrake can output data in VTK format, suitable for viewing in
Paraview_.  On Ubuntu and similar systems, you can obtain Paraview by
installing the ``paraview`` package.  On Mac OS, the easiest approach
is to download a binary from the `paraview website <Paraview_>`_.

Building the documentation
--------------------------
If you want to be able to view and edit the documentation locally, run::

    python3 firedrake-install --documentation-dependencies

when installing Firedrake, or in an existing instalation (after running
``source firedrake/bin/activate`` to activate the virtual env) run::

    firedrake-update --documentation-dependencies

The documentation can be found in
``firedrake/firedrake/src/firedrake/docs``
and can be built by executing::

    make html

This will generate the HTML documentation (this website) on your local
machine.
