Obtaining Firedrake
===================

Firedrake is installed using its install script::

  curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
  python3 firedrake-install

Running ``firedrake-install`` with no arguments will install firedrake in
a python venv_ created in a ``firedrake`` subdirectory of the
current directory. Run::

  python3 firedrake-install --help

for a full list of install options.  In particular, you may
wish to customise the set of options used to build PETSc.  To do so,
set the environment variable ``PETSC_CONFIGURE_OPTIONS`` before
running ``firedrake-install``.  You can see the set of options passed
to PETSc by providing the flag ``--show-petsc-configure-options``.

You will need to activate the virtualenv in each shell from which you
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
virtualenv_ as above and then run::

  cd firedrake/src/firedrake
  make alltest


System requirements
-------------------

The installation script is tested on Ubuntu and MacOS X. Installation
is likely to work well on other Linux platforms, although the script
may stop to ask you to install some dependency packages. Installation
on other Unix platforms may work but is untested. Installation on
Windows is very unlikely to work.

Supported Python distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Firedrake requires Python 3.5 or later.

On Ubuntu (16.04 or later), the system installed Python 3 is supported and tested.
On Mac OS, the homebrew_ installed Python 3 is supported and tested.

.. warning::

   The installation script *does not work* with anaconda_ based python
   installations. This is due to venv issues in anaconda.

Upgrade
-------

The install script will install an upgrade script in
`firedrake/bin/firedrake-update`. Running this script will update
Firedrake and all its dependencies.

.. note::

   You should activate the venv_ before running
   `firedrake-update`.


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


Documentation dependencies
--------------------------

Building the documention requires Sphinx_ (including the Youtube and
Bibtex plugins) and wget_.  In addition the Sphinx Youtube and bibtex
plugins are required.  The former is available from the
`sphinx-contrib repository
<https://bitbucket.org/birkenfeld/sphinx-contrib>`__, the latter is
the python package ``sphinxcontrib-bibtex``.

.. _Paraview: http://www.paraview.org
.. _Sphinx: http://www.sphinx-doc.org/
.. _wget: http://www.gnu.org/software/wget/
.. _venv: https://docs.python.org/3/tutorial/venv.html
.. _homebrew: https://brew.sh/
.. _anaconda: https://www.continuum.io/downloads
