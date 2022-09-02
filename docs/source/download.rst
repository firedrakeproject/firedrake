:orphan: true

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

for a full list of install options.  In particular, you may
wish to customise the set of options used to build PETSc.  To do so,
set the environment variable ``PETSC_CONFIGURE_OPTIONS`` before
running ``firedrake-install``.  You can see the set of options passed
to PETSc by providing the flag ``--show-petsc-configure-options``.

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


Installing SciPy
----------------

If you want to install the ``scipy`` package alongside Firedrake you
must ensure that the package is built against the same libraries as the
rest of the Firedrake toolchain. To do this at install time simply add
``scipy`` to the ``firedrake-install`` command line arguments::

  python3 firedrake-install --pip-install scipy

If you want to add ``scipy`` to your environment after installing
Firedrake, first activate the virtual environment, then run::

  firedrake-update --pip-install scipy


Testing the installation
------------------------

We recommend that you run the test suite after installation to check
that Firedrake is fully functional. Activate the venv_ as above and
then run::

  cd $VIRTUAL_ENV/src/firedrake
  pytest tests/regression/ -k "poisson_strong or stokes_mini or dg_advection"

This command will run a few of the unit tests, which exercise a good
chunk of the functionality of the library. These tests should take a
minute or less. If they fail to run for any reason, please see the
section below on how to diagnose and debug a failed installation. If
you want to run the entire test suite you can do ``make alltest``
instead, but this takes several hours.

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

System requirements
-------------------

Firedrake requires Python 3.6.x to 3.10.x. On MacOS Arm (M1 or M2) Python 3.9.x
or 3.10.x are required since these are the only versions for which VTK binary
packages are currently available. The installation script is tested on Ubuntu
and MacOS X. On Ubuntu 18.04 or later, the system installed Python 3 is
supported and tested. On MacOS, the homebrew_ installed Python 3 is supported
and tested::

  brew install python3

Installation is likely to work well on other Linux platforms, although
the script may stop to ask you to install some dependency packages.
Installation on other Unix platforms may work but is untested. On Linux
systems that do not use the Debian package management system, it will be
necessary to pass the `--no-package-manager` option to the install
script. In this case, it is the user's responsibilty to ensure that
they have the system dependencies:

* A C and C++ compiler (for example gcc/g++ or clang), GNU make
* A Fortran compiler (for PETSc)
* Blas and Lapack
* Git, Mercurial
* Python version 3.6.x-3.10.x (3.9.x-3.10.x on MacOS Arm)
* The Python headers
* autoconf, automake, libtool
* CMake
* zlib
* flex, bison

Firedrake has been successfully installed on Windows 10 using the
Windows Subsystem for Linux. There are more detailed
`instructions here <https://github.com/firedrakeproject/firedrake/wiki/Installing-on-Windows-Subsystem-for-Linux>`_.
Installation on previous versions of Windows is unlikely to work.


Anaconda
~~~~~~~~

.. warning::
    The following is recent as of Sept 2022 and still highly experimental.

Firedrake was incompatible with the Anaconda Python distribution for a
long time. This limitation has been lifted, but for the time being,
installing with Anaconda will require either using this special
`shell script <https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install-conda.sh>`_
or executing the commands in that script manually. If installation under
Anaconda with that script fails, please
`raise an issue <https://github.com/firedrakeproject/firedrake/issues`_
and we will try to fix it as soon as possible, but please bear with us
as we iron out the wrinkles on supporting Anaconda.


System anti-requirements
~~~~~~~~~~~~~~~~~~~~~~~~

We strive to make Firedrake work on as many platforms as we can. Some
tools, however, make this challenging or impossible for end users.

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

If you don't see the issue you're experiencing in this chart, please
ask us on Slack or report a bug by creating a new `github discussion
<https://github.com/firedrakeproject/firedrake/discussions>`__. To help us
diagnose what's going wrong, **please include the following log files**:

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
