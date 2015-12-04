Obtaining Firedrake
===================

The simplest way to install Firedrake is to use our install script::

  curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
  python firedrake-install

Running `firedrake-install` with no arguments will install firedrake in
a python virtualenv_ created in a `firedrake` subdirectory of the
current directory. Run::

  python firedrake-install --help

for a full list of install options including user- or system-wide
installs and installation in developer mode. If you install in
virtualenv_ mode, you will need to activate the virtualenv in each
shell from which you use Firedrake::

  source firedrake/bin/activate

It is recommended to run the test suite after installation to check
that the Firedrake installation is fully functional. Enter the
virtualenv_ as above, install pytest_ with::

  pip install pytest

and then run::

  cd firedrake/src/firedrake
  make alltest


System requirements
-------------------

The installation script is tested on Ubuntu and MacOS X. Installation
is likely to work well on other Linux platforms, although the script
may stop to ask you to install some dependency packages. Installation
on other Unix platforms may work but is untested. Installation on
Windows is very unlikely to work.

Supported python distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On Ubuntu, the system installed python 2.7 is supported and tested.
On Mac OS, the homebrew_ installed python 2.7 is supported and tested.

.. warning::

   The installation script *does not work* with anaconda_ based python
   installations, because it uses virtualenvs which anaconda does not
   support.

Upgrade
-------

The install script will install an upgrade script in
`firedrake/bin/firedrake-update`. Running this script will update
Firedrake and all its dependencies.

Cleaning disk caches after upgrade
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After upgrading, you may need to clear any disk caches that Firedrake
maintains to ensure that your problem does not pick up any out of date
compiled modules. To do this run::

  firedrake-clean

If you installed to a virtualenv, you will need to activate the
virtualenv first.

Recovering from a broken installation script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you find yourself in the unfortunate position that
`firedrake-update` won't run because of a bug, and the bug has been
fixed in Firedrake master, then the following procedure will rebuild
`firedrake-update` using the latest version.

From the top directory of your Firedrake install,
type::

  cd src/firedrake
  git pull
  ./scripts/firedrake-install --rebuild-script

You should also pass any of the other options to `firedrake-install`
which you wish the rebuilt script to apply (for example `--user` or
`--disable-ssh`). You should now be able to run `firedrake-update`.


Visualisation software
----------------------

Firedrake can output data in VTK format, suitable for viewing in
Paraview_.  On Ubuntu and similar systems, you can obtain Paraview by
installing the ``paraview`` package.  On Mac OS, the easiest approach
is to download a binary from the `paraview website <Paraview_>`_.

Installing from individual components
=====================================

.. warning::

  Installation from individual components is an unsupported route.  If
  you tried the installation script and it failed, please let us know
  by filing a `bug report
  <https://github.com/firedrakeproject/firedrake/issues>`__, we
  *absolutely* want to fix it.  This section of the documentation
  provides a bare-bones description of the required dependencies.

Firedrake itself is a Python package available on `github
<https://github.com/firedrakeproject/firedrake>`__, however, you will
need a number of additional libraries to use it.

Mac OS
------

We list here the required homebrew_ packages:

- openmpi (or mpich)
- python
- cmake
- spatialindex

Ubuntu
------

On Ubuntu, the following apt packages are required:

- build-essential
- cmake
- gfortran
- git-core
- libblas-dev
- liblapack-dev
- libopenmpi-dev
- libspatialindex-dev
- mercurial
- openmpi-bin
- python-dev
- python-pip

Common dependencies
-------------------

PETSc
~~~~~

We maintain branches of PETSc_ and petsc4py_ that are known to work
with Firedrake.  Use the ``firedrake`` branch for both:

- https://bitbucket.org/mapdes/petsc
- https://bitbucket.org/mapdes/petsc4py

PETSc must be built with (at least) support for:

- HDF5
- CHACO
- Triangle
- Ctetgen

We also recommend that you build PETSc with shared libraries.

h5py
~~~~

Firedrake uses h5py_ to write checkpoint files.  It is critical that
h5py_ is linked against the same version of the HDF5 library that
PETSc was built with.  This is unfortunately not possible to specify
when using ``pip``.  Instead, please follow the instructions for a
`custom installation`_.  If PETSc was linked against a system HDF5
library, use that library when building h5py.  If the PETSc
installation was used to build HDF5 (via ``--download-hdf5``) then the
appropriate HDF5 library is in the PETSc install directory.  If
installed with ``pip``, this can be obtained using::

  python -c "import petsc; print petsc.get_petsc_dir()"

Otherwise, use the appropriate values of ``PETSC_DIR`` and ``PETSC_ARCH``.

.. note::

   It is not necessary that h5py be built with MPI support, although
   Firedrake supports both options.

Further dependencies
~~~~~~~~~~~~~~~~~~~~

Firedrake depends on the Python packages PyOP2_, FFC_, FIAT_ and UFL_.

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

For performance reasons, there are various levels of caching with
eviction policies.  To support these, you will need to install the
python packages::

- cachetools
- psutil

Documentation dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

Building the documention requires Sphinx_ (including the Youtube and
Bibtex plugins) and wget_.  In addition the Sphinx Youtube and bibtex
plugins are required.  The former is available from the
`sphinx-contrib repository
<https://bitbucket.org/birkenfeld/sphinx-contrib>`__, the latter is
the python package ``sphinxcontrib-bibtex``.

.. _petsc4py: https://bitbucket.org/mapdes/petsc4py
.. _PETSc: http://www.mcs.anl.gov/petsc/
.. _PyOP2: http://op2.github.io/PyOP2
.. _FFC: https://bitbucket.org/mapdes/ffc
.. _FIAT: https://bitbucket.org/mapdes/fiat
.. _UFL: https://bitbucket.org/mapdes/ufl
.. _Paraview: http://www.paraview.org
.. _Sphinx: http://www.sphinx-doc.org
.. _wget: http://www.gnu.org/software/wget/
.. _virtualenv: https://virtualenv.pypa.io/
.. _pytest: http://pytest.org/latest/
.. _libspatialindex: https://libspatialindex.github.io/
.. _h5py: http://www.h5py.org/
.. _custom installation: http://docs.h5py.org/en/latest/build.html#via-setup-py
.. _homebrew: http://brew.sh
.. _anaconda: https://www.continuum.io/downloads
