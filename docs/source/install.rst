.. raw:: latex

   \clearpage

.. contents::

====================
Installing Firedrake
====================


.. _supported_systems:

Supported systems
=================

A :ref:`native installation<pip_install_firedrake>` of Firedrake is officially
supported on Ubuntu and ARM Macs (Intel Macs are no longer supported) though
it should be installable on any Linux distribution. Windows users are encouraged
to use WSL_ or one of Firedrake's
:ref:`alternative installation mechanisms<alternative_install>`.

If installing on an HPC system most of the following steps should remain
applicable, though care will have to be taken to make sure that the right system
packages are used. A community-maintained collection of instructions for how to install
Firedrake onto a number of different HPC systems may be found
`here <https://github.com/firedrakeproject/firedrake/wiki/HPC-installation>`__.
If install on an HPC system not included in the wiki, please consider contributing a
page describing the installation on that system.

.. _pip_install_firedrake:

Installing Firedrake using pip
==============================

A native installation of Firedrake is accomplished in 3 steps:

#. :ref:`Install system dependencies<install_system_dependencies>`
#. :ref:`Install PETSc<install_petsc>`
#. :ref:`Install Firedrake<install_firedrake>`

If you encounter any problems then please refer to our list of
:ref:`common installation issues<common_issues>` or consider
:doc:`getting in touch</contact>`.

.. _prerequisites:

Prerequisites
-------------

On Linux the only prerequisite needed to install Firedrake is a suitable version
of Python (3.10 or greater). On macOS it is important that homebrew_ and Xcode_
are installed and up to date and that the homebrew-installed Python is used
instead of the system one.


.. _firedrake_configure:

firedrake-configure
-------------------

To simplify the installation process, Firedrake provides a utility script called
``firedrake-configure``. This script can be downloaded by executing::

  $ curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/main/scripts/firedrake-configure

Note that ``firedrake-configure`` **does not install Firedrake for you**. It
is simply a helper script that emits the configuration options that Firedrake
needs for the various steps needed during installation.

To improve robustness, ``firedrake-configure`` is intentionally kept extremely
minimal and simple. This means that if you want to install Firedrake in a
non-standard way (for instance with a custom installation of PETSc, HDF5 or MPI)
then it is your responsibility to modify the output from ``firedrake-configure``
as necessary. This is described in more detail in :ref:`customising`.


.. _install_system_dependencies:

Installing system dependencies
------------------------------

If on Ubuntu or macOS, system dependencies can be installed with
``firedrake-configure``. On Ubuntu run::

  $ sudo apt install $(python3 firedrake-configure --show-system-packages)

which will install the following packages:

.. literalinclude:: apt_deps.txt
   :language: text

If on macOS you should instead run::

  $ brew install $(python3 firedrake-configure --show-system-packages)

which will install the following packages:

.. literalinclude:: homebrew_deps.txt
   :language: text

The packages installed here are a combination of system dependencies,
like a C compiler, BLAS, and MPI,  and 'external packages' that are used by PETSc, like
MUMPS and HDF5.

If you are not installing onto Ubuntu or macOS then it is your responsibility to
ensure that these system dependencies are in place. Some of the dependencies
(e.g. a C compiler) must come from your system whereas others, if desired, may be
downloaded by PETSc ``configure`` by passing additional flags like
``--download-mpich`` or ``--download-openblas`` (run ``./configure --help | less`` to
see what is available). To give you a guide as to what system dependencies are
needed, on Ubuntu they are:

.. literalinclude:: minimal_apt_deps.txt
   :language: text

.. _install_petsc:

Installing PETSc
----------------

For Firedrake to work as expected, it is important that a specific version of PETSc_
is installed with a specific set of external packages. To install PETSc you need to
do the following steps:

#. Clone the PETSc repository, checking out the right version::

   $ git clone --branch $(python3 firedrake-configure --show-petsc-version) https://gitlab.com/petsc/petsc.git
   $ cd petsc

#. Run PETSc ``configure``, passing in the flags generated by ``firedrake-configure``::

   $ python3 ../firedrake-configure --show-petsc-configure-options | xargs -L1 ./configure

#. Compile PETSc by running the ``make`` command prompted by ``configure``. This
   will look something like:

   .. code-block:: text

      make PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-firedrake-default all

#. Test the installation (optional) and return to the parent directory::

   $ make check
   $ cd ..

If you are using one of the
:ref:`officially supported distributions<supported_systems>` then these configure
options will include paths to system packages so PETSc can correctly find and
link against them. If you are not then you should pass the ``--no-package-manager``
flag to obtain a set of configure options where ``firedrake-configure``
pessimistically assumes that no external packages are available, and hence need
to be downloaded and compiled from source::

   $ python3 ../firedrake-configure --no-package-manager --show-petsc-configure-options | xargs -L1 ./configure

For the default build, running ``firedrake-configure`` with
``--no-package-manager`` will produce the flags:

.. literalinclude:: petsc_configure_options.txt
   :language: text


.. _install_firedrake:

Installing Firedrake
--------------------

Now that the right system packages are installed and PETSc is built we can now
install Firedrake. To do this perform the following steps:

#. Create a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_::

      $ python3 -m venv venv-firedrake
      $ . venv-firedrake/bin/activate

   This is optional but strongly recommended to avoid polluting your system Python
   environment.

#. Purge the pip cache::

      $ pip cache purge

   This is also optional but strongly recommended as some cached pip packages
   may be linked against old or missing libraries and hence will break your
   installation. For a lighter-weight alternative you could run some or all
   of the following::

      $ pip cache remove mpi4py
      $ pip cache remove petsc4py
      $ pip cache remove h5py
      $ pip cache remove slepc4py
      $ pip cache remove libsupermesh
      $ pip cache remove firedrake

   Noting that this list may not be exhaustive.


#. Set any necessary environment variables. This can be achieved using
   ``firedrake-configure``::

     $ export $(python3 firedrake-configure --show-env)

   Which at a minimum will set the following variables:

   .. code-block:: text

      CC=mpicc CXX=mpicxx PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-firedrake-{default,complex} HDF5_MPI=ON

   .. note::
      This command will only work if you have the right starting directory.
      Specifically it is assumed that PETSc was cloned into a *subdirectory
      of the current working directory* (i.e. ``<cwd>/petsc``). If
      you have exactly followed the instructions up to this point this should
      already be the case.

#. Install Firedrake::

      $ pip install --no-binary h5py 'firedrake[check]'

   .. note::
      Though not strictly necessary to install Firedrake's optional
      dependencies with ``[check]`` it is recommended because it allows you
      to check that the install was successful (see
      :ref:`below<firedrake_check>`).

#. Firedrake is now installed and ready for use!


.. _firedrake_check:

Checking the installation
~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend that you run some simple tests after installation to check
that Firedrake is fully functional. To do this, after the installation run::

  $ firedrake-check

This command will run a few of the unit tests, which exercise a good
chunk of the functionality of the library. These tests should take a
minute or less. If they fail to run for any reason, please check out
our list of :ref:`commonly encountered installation issues<common_issues>`
or consider :doc:`getting in touch</contact>`.

Note that for you to be able to run the tests you need to have installed
Firedrake with its optional test dependencies by specifying the ``[check]``
dependency group as shown :ref:`above<install_firedrake>`.


Updating Firedrake
------------------

Updating Firedrake involves following the same steps as above when
:ref:`installing Firedrake<install_firedrake>`. First, use ``firedrake-configure``
to set the right environment variables and then run::

     $ pip install --upgrade firedrake

Previously generated code may not be compatible with a newer
Firedrake installation, and may crash with cryptic messages.
We recommend removing any cached code after updating by running::

     $ firedrake-clean

Updating PETSc
~~~~~~~~~~~~~~

To update PETSc you should:

#. Re-download ``firedrake-configure``.

#. Run::

   $ cd /path/to/petsc
   $ git fetch
   $ git checkout -b $(python3 /path/to/firedrake-configure --show-petsc-version)
   $ make

Note that this will only recompile PETSc's source code, not that of the external
packages, and so should be relatively quick. If your PETSc is sufficiently
out-of-date you may also need to rebuild the external packages by running::

   $ make reconfigure

.. _common_issues:

Common installation issues
--------------------------

No such file or directory: '/tmp/.../petsc/conf/petscvariables'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter the error:

.. code-block:: text

   FileNotFoundError: [Errno 2] No such file or directory: '/tmp/.../petsc/conf/petscvariables'

when running the ``pip install`` instruction this is usually a sign that either:
you are using a cached version of petsc4py that is linked incorrectly, or that
the environment variables ``PETSC_DIR`` or ``PETSC_ARCH`` are not set correctly.
To fix this we suggest purging the pip cache before re-exporting the environment
variables::

   $ pip cache purge
   $ export $(python3 firedrake-configure --show-env)

You can check that ``PETSC_DIR`` and ``PETSC_ARCH`` are set correctly by making
sure that you can run the following command without error::

   $ ls $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/petscvariables 

Missing symbols post install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the installation completes but then you get errors regarding missing symbols
when you import Firedrake this is usually a sign that one of the Python bindings
packages used by Firedrake (h5py, mpi4py, petsc4py, slepc4py), or Firedrake
itself, is linked against the wrong compiled library. This is usually caused
by issues with caching.

To resolve the problem we recommend removing your virtual environment, purging
the cache and then :ref:`attempting another installation<install_firedrake>`:

.. code-block:: bash

   $ deactivate  # needed if venv-firedrake is currently activated
   $ rm -r venv-firedrake
   $ pip cache purge
   $ python3 -m venv venv-firedrake
   # etc

Unable to configure PETSc on macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are running on macOS and encounter error messages during PETSc
``configure`` like the following:

.. code-block:: text

   *********************************************************************************************
              UNABLE to CONFIGURE with GIVEN OPTIONS (see configure.log for details):
   ---------------------------------------------------------------------------------------------
            Cannot use scalapack without Fortran, make sure you do NOT have --with-fc=0
   *********************************************************************************************

then this is usually a sign that your Homebrew and/or Xcode are too old. We
recommend making sure that they are both up-to-date before trying again.

For Homebrew it is sometimes useful to run the command::

   $ brew doctor

as this can flag issues with your system that should be resolved before
installing Firedrake.

.. _customising:

Customising Firedrake
=====================


.. _firedrake_archs:

Prepared configurations
-----------------------

``firedrake-configure`` provides a number of different possible configurations
(termed 'ARCHs') that specify how PETSc is configured and which external
packages are built. The currently supported ARCHs are:

* ``default``: the default installation, suitable for most users
* ``complex``: an installation where PETSc is configured using complex numbers

The different configurations can be selected by passing the flag ``--arch`` to
``firedrake-configure``. For example::

   $ python3 firedrake-configure --show-petsc-configure-options --arch complex


Optional dependencies
---------------------

SLEPc
~~~~~

To install Firedrake with SLEPc support you should:

#. Pass ``--download-slepc`` when running PETSc ``configure`` (see :ref:`Installing PETSc<install_petsc>`)::

   $ python3 ../firedrake-configure --show-petsc-configure-options | xargs -L1 ./configure --download-slepc

#. Set ``SLEPC_DIR``::

   $ export SLEPC_DIR=$PETSC_DIR/$PETSC_ARCH

#. Continue with the installation as normal but install Firedrake with the
   ``slepc`` optional dependency. For example::

   $ pip install --no-binary h5py 'firedrake[check,slepc]'

VTK
~~~

To install Firedrake with VTK, it should be installed using the ``vtk`` optional
dependency. For example::

   $ pip install --no-binary h5py 'firedrake[check,vtk]'

At present VTK wheels are not available for ARM Linux machines. Depending on your
Python version you may be able to work around this by downloading and pip installing
the appropriate ``.whl`` file from
`here <https://github.com/scientificcomputing/vtk-aarch64/releases>`__.


PyTorch
~~~~~~~

To install Firedrake with `PyTorch <https://pytorch.org/>`_, it should be installed
using the ``torch`` optional dependency. For example::

   $ pip install --no-binary h5py 'firedrake[check,torch]' --extra-index-url https://download.pytorch.org/whl/cpu

Observe that, in addition to specifying ``torch``, an additional
argument (``--extra-index-url``) is needed. More information on installing
PyTorch can be found `here <https://pytorch.org/get-started/locally/>`__.


JAX
~~~

To install Firedrake with JAX, it should be installed using the ``jax`` optional
dependency. For example::

   $ pip install --no-binary h5py 'firedrake[check,jax]'


Netgen
~~~~~~

To install Firedrake with `Netgen <https://ngsolve.org/>`_ support, it should be
installed with the ``netgen`` optional dependency. For example::

   $ pip install --no-binary h5py 'firedrake[check,netgen]'


Customising PETSc
-----------------

Since ``firedrake-configure`` only outputs a string of options it is straightforward
to customise the options that are passed to PETSc ``configure``. You can either:

* Append additional options when ``configure`` is invoked. For example, to
  build PETSc with support for 64-bit indices you should run::

   $ python3 ../firedrake-configure --show-petsc-configure-options | xargs -L1 ./configure --with-64-bit-indices

* Write the output of ``firedrake-configure`` to a file than can be modified. For example::

   $ python3 ../firedrake-configure --show-petsc-configure-options > my_configure_options.txt
   <edit my_configure_options.txt>
   $ cat my_configure_options.txt | xargs -L1 ./configure

.. note::
   If additional options are passed to ``configure`` then care must be taken when
   using externally-installed system packages (i.e. ``--with-package=...`` or
   ``--with-package-{include,lib}=...`` are in the ``configure`` options) as they
   may no longer be suitable for the new configuration. It is your responsibility
   to either ensure that the configuration is suitable, or replace the
   ``configure`` option with ``--download-package`` so that PETSc will download
   and install the right thing.

Reconfiguring an existing PETSc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If rebuilding an existing PETSc installation, rather than removing everything and
starting from scratch, it can be useful to modify and run the ``reconfigure-ARCH.py``
Python script that PETSc generates. This can be found in
``$PETSC_DIR/$PETSC_ARCH/lib/petsc/conf``. Other example scripts can be found in
``$PETSC_DIR/config/examples`` directory.

.. _alternative_install:

Alternative installation methods
================================

If for some reason you are unable to install Firedrake natively using pip,
Firedrake has a number of alternative mechanisms that you can use to obtain
an environment to run your Firedrake code.

Docker
------

Firedrake provides a number of different
`Docker <https://www.docker.com/>`_ images that can be found
`here <https://hub.docker.com/u/firedrakeproject>`__. The main images best
suited for users are:

* `firedrake-vanilla-default <https://hub.docker.com/repository/docker/firedrakeproject/firedrake-vanilla-default>`__: a complete Firedrake installation with ARCH ``default``
* `firedrake-vanilla-complex <https://hub.docker.com/repository/docker/firedrakeproject/firedrake-vanilla-complex>`__: a complete Firedrake installation with ARCH ``complex``
* `firedrake <https://hub.docker.com/repository/docker/firedrakeproject/firedrake>`__: the firedrake-vanilla-default image with extra downstream packages installed

To use one of the containers you should run::

   $ docker pull firedrakeproject/<image name>:latest

to download the most recent image (replacing ``<image name>`` with the desired
image). Then you can run::

   $ docker run -it firedrakeproject/<image name>:latest

to start and enter a container.

.. note::

   The 'full-fat' ``firedrakeproject/firedrake`` image only exists for x86
   architectures because some external packages do not provide ARM wheels.
   If you are using an ARM Mac (i.e. M1, M2, etc) then you are encouraged to
   use the ``firedrakeproject/firedrake-vanilla-default`` or
   ``firedrakeproject/firedrake-vanilla-complex`` images instead.

It is possible to use `Microsoft VSCode <https://code.visualstudio.com/>`__
inside a running container. Instructions for how to do this may be found
`here <https://github.com/firedrakeproject/firedrake/wiki/Writing-Firedrake-code-with-VSCode-inside-a-Docker-container>`__.

.. warning::

   The Docker daemon runs with superuser privileges and so has the potential to 
   damage your system, in particular if volumes are mounted between the container
   and host. We therefore strongly advise you to take care when using Docker.
   More information can be found
   `here <https://docs.docker.com/engine/security/#docker-daemon-attack-surface>`__.

Google Colab
------------

Firedrake can also be used inside the brower using Jupyter notebooks and
`Google Colab <https://colab.research.google.com/>`_. For more information
please see :doc:`here</notebooks>`.

.. _dev_install:

Developer install
=================

.. only:: release

   .. warning::
      You are currently looking at the documentation for the current stable
      release of Firedrake. For the most recent developer documentation you
      should follow the instructions `here <https://firedrakeproject.org/firedrake/install>`__.

In order to install a development version of Firedrake the following steps
should be followed:

#. Install system dependencies :ref:`as before<install_system_dependencies>`

#. Clone and build the *default branch* of PETSc:

   .. code-block:: text

      $ git clone https://gitlab.com/petsc/petsc.git
      $ cd petsc
      $ python3 ../firedrake-configure --show-petsc-configure-options | xargs -L1 ./configure
      $ make PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-firedrake-default all
      $ make check
      $ cd ..

#. Clone Firedrake::

   $ git clone <firedrake url>

   where ``<firedrake url>`` is ``https://github.com/firedrakeproject/firedrake.git``
   or ``git@github.com:firedrakeproject/firedrake.git`` as preferred.

#. Set the necessary environment variables::

   $ export $(python3 firedrake-configure --show-env)

#. Create and activate a virtual environment::

   $ python3 -m venv venv-firedrake
   $ . venv-firedrake/bin/activate

#. Install petsc4py and Firedrake's other build dependencies:

   .. code-block:: text

      $ pip cache purge
      $ pip install $PETSC_DIR/src/binding/petsc4py
      $ pip install -r ./firedrake/requirements-build.txt

#. Install Firedrake in editable mode without build isolation along with
   any developer dependencies::

   $ pip install --no-build-isolation --no-binary h5py --editable './firedrake' --group ./firedrake/pyproject.toml:dev

   .. note::
      Installing the developer dependencies requires pip to be version 25.1
      or greater. You may need to run ``pip install -U pip`` first.

Editing subpackages
-------------------

Firedrake dependencies can be cloned and installed in editable mode in an
identical way to Firedrake. For example, to install
`FIAT <https://github.com/firedrakeproject/fiat.git>`_ in editable mode you
should run::

   $ git clone <fiat url>
   $ pip install --editable ./fiat

For most packages it should not be necessary to pass ``--no-build-isolation``.

It is important to note that these packages **must be installed after Firedrake**.
This is because otherwise installing Firedrake will overwrite the just-installed
package.

.. _discussion: https://github.com/firedrakeproject/firedrake/discussions
.. _issue: https://github.com/firedrakeproject/firedrake/issues
.. _homebrew: https://brew.sh/
.. _Xcode: https://developer.apple.com/xcode/
.. _PETSc: https://petsc.org/
.. _petsc4py: https://petsc.org/release/petsc4py/reference/petsc4py.html
.. _venv: https://docs.python.org/3/tutorial/venv.html
.. _WSL: https://github.com/firedrakeproject/firedrake/wiki/Installing-on-Windows-Subsystem-for-Linux
