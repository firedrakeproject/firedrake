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

Reporting installation bugs
---------------------------

If ``firedrake-install`` fails to work, please report a bug so that we
can fix it for you by creating a new `github issue
<https://github.com/firedrakeproject/firedrake/issues>`__.  Please
include the log file ``firedrake-install.log`` in your bug report.
Similarly if ``firedrake-update`` fails, it produces a
``firedrake-update.log`` file which will help us to debug the problem.

Testing the installation
------------------------

It is recommended to run the test suite after installation to check
that the Firedrake installation is fully functional.  Activate the
venv_ as above and then run::

  cd firedrake/src/firedrake
  make alltest

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


System requirements
-------------------

The installation script is tested on Ubuntu and MacOS X. Installation
is likely to work well on other Linux platforms, although the script
may stop to ask you to install some dependency packages. Installation
on other Unix platforms may work but is untested. On Linux systems
that do not use the Debian package management system, it will be
necessary to pass the `--no-package-manager` option to the install
script. In this case, it is the user's responsibilty to ensure that
they have the system dependencies:

* A C and C++ compiler (for example gcc/g++ or clang), GNU make
* A Fortran compiler (for PETSc)
* Blas and Lapack
* Git, Mercurial
* Python version >=3.5
* The Python headers
* autoconf, automake, libtool
* CMake
* zlib

Firedrake has been successfully installed on Windows 10 using the
Windows Subsystem for Linux. There are more detailed
`instructions here <https://github.com/firedrakeproject/firedrake/wiki/Installing-on-Windows-Subsystem-for-Linux>`_.
Installation on previous versions of Windows is unlikely to work.


Supported Python distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Firedrake requires Python 3.5 or later.

On Ubuntu (16.04 or later), the system installed Python 3 is supported and tested.

On Mac OS, the homebrew_ installed Python 3 is supported and tested::

  brew install python3

If instead you choose to install Python 3 using the official Mac OS
installer on the Python website, you need to be aware that that
installation will not have a working SSL by default. You need to
follow the SSL certificate instructions given in the installation process (or in
``/Applications/Python 3.6/ReadMe.rtf`` after installation).


.. note::

   Your system still needs to have Python 2 available to build PETSc_.

Additional considerations for MacPorts users
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mac OS has multiple competing package managers which sometimes cause
issues for users attempting to install Firedrake. In particular, the
assembler provided by MacPorts is incompatible with the Mac system
compilers in a manner which causes Firedrake to fail to install. For
this reason, if you are installing Firedrake on a Mac which also has
MacPorts installed, you should ensure that ``/opt/local/bin`` and
``/opt/local/sbin`` are removed from your ``PATH`` when installing or
using Firedrake. This should ensure that no MacPorts installed tools
are found.

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
