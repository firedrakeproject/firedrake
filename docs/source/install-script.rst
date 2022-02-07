:orphan: true

.. role:: bash(code)
   :language: bash

=================
Firedrake Install
=================

The structure of the Firedrake install process consists of the following steps:

1. Installing system dependencies
2. Building and configuring PETSc
3. Installing Python dependencies  

These steps are only supported for for Linux and MacOS distributions. 
Further we will take the above steps and discuss them in detail. 

Installing system dependencies
-------------------------------

Linux
~~~~~

For Linux distributions the system dependencies are installed via the :bash:`apt-get` package manager. The necessary dependencies for running default firedrake are:

- build-essential
- autoconf
- automake
- bison
- flex
- cmake
- gfortran
- git
- libtool
- python3-dev
- python3-pip
- python3-tk
- python3-venv
- zlib1g-dev
- libboost-dev
- libopenblas-dev (if no configuration is provided by the user)

As a small observation, packages like *bison* and *flex* are used for *PT-Scotch* (tool for efficient parallel graph ordering).

MacOS
~~~~~

For MacOs distributions (based on Darwin) the install script uses the :bash:`homebrew` package manager to install the following necessary dependencies:

- `gcc`
- `autoconf`
- `make`
- `automake`
- `cmake`
- `libtool`
- `boost`
- `openblas` (if no configuration is provided by the user)

Additionally the install script will also want to install xcode-command line tools.

After the installation of the system requirements is done, the installation script will then create a virtual python environment and prepare itself to install all its software dependencies there.

Building and configuring PETSc
-------------------------------

One important package on which Firedrake software is built is the `PETSc_ <https://petsc.org/release/>`_ library, which offers a toolkit when dealing with partial differential equations. 
The install script enables some customization for installing the PETSc, but also offers the user the option to use a previous installation of the library via the parameter :bash:`hounour_petsc`.
If no previous previous installation is provided, the installation script will then build the PETSc package from source in the *src*  directory within the virtual python environment.

The basic options for installing the library using the install argument :bash:`--minimal_petsc` are:

.. code-block:: python

   petsc_options = { "--with-fortran-bindings=0",
                     "--with-debugging=0",
                     "--with-shared-libraries=1",
                     "--with-c2html=0",
                     "--download-eigen=%s/src/eigen-3.3.3.tgz " % firedrake_env,
                     # File format
                     "--download-hdf5",
                     # AMG
                     "--download-hypre",
                     # Sparse direct solver
                     "--download-superlu_dist",
                     # Parallel mesh partitione[]()r
                     "--download-ptscotch",
                     # For superlu_dist amongst others.
                     "--with-cxx-dialect=C++11"}

This option is useful if we want to have a faster build and therefore is very useful for testing.
For a more detail explanation of the parameters please check the PETSc install documentation (page)[https://petsc.org/release/install/].

If no minimal version of PETSc is required (:bash:`--minimal_petsc false`):

- :bash:`--with-zlib`
- :bash:`--download-netcdf` (file format)
- :bash:`--download-pnetcdf` (file format) 
- :bash:`--download-suitesparse` (sparse direct solver)
- :bash:`--download-pastix` (sparse direct solver) 
- :bash:`--download-hwloc` (required by pastix package)
- :bash:`--download-metis` (serial mesh partitioner)

will be added as argument to the PETSc installation. 

Further the install script also provides the option for the user to decide between integer type for the PETsc (:bash:`--petsc-int-type`) and also choose if he want to use complex numbers via (:bash:`--complex`) argument.

Due to the fact that PETSc is a parallel program, the firedrake-install script offers the user the option to provide his *MPI wrappers* (:bash:`--mpiexec --mpicc --mpicxx --mpif90`) for the C, C++ and Fortran compiler that are going to be used to build the program. If none are provided, the script will take care to download the `mpich`.

Lastly, the script also provides the possibility via the :bash:`--with_blas` argument to  download or use an existing version of BLAS (Basic Linear Algebra Subprograms).

Installing Python dependencies
------------------------------

After the installation of PETSc, the install script finally starts installing `Firedrake <https://github.com/firedrakeproject/firedrake>`_ alongside it's Python dependencies (as they can be found in the *requirements-git.txt*):

- `ufl <https://github.com/firedrakeproject/ufl>`_
- `fiat <https://github.com/firedrakeproject/fiat>`_
- `FInAT <https://github.com/FInAT/FInAT>`_
- `tsfc <https://github.com/firedrakeproject/tsfc>`_
- `PyOP2 <https://github.com/OP2/PyOP2>`_
- `pyadoint <https://github.com/dolfin-adjoint/pyadjoint>`_
- petsc4py
- `COFFEE <https://github.com/coneoproject/COFFEE>`_
- `loopy <https://github.com/firedrakeproject/loopy>`_

When installing these packages we want to assure ABI compatibility within the packages and its dependencies, so we will enforce the pip package manager to not use wheel files, but rebuild the following packages:

- mpi4py
- randomgen
- islpy
- numpy

Finally in order to avoid other compatibility issues, the install script will also make sure that the following parallel packages will be installed using the same MPI wrappers as in for the PETSc installation:

- mpi4py
- h54py
- petsc4py
- PyOP2
- libersupermesh
- firedrake