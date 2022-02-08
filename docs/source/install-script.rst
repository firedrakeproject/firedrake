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

For Linux distributions the system dependencies are installed via the :bash:`apt-get` package manager. 
Also if your distribution doesn't have a :bash:`apt-get` manager, you could just use the field :bash:`--no-package-manager` with the install script, but in this case the user has the responsability to ensure that the following dependencies are installed.
The necessary dependencies for running default firedrake are:

.. include:: dependencies/system_requirements_linux.rst

As a small observation, packages like *bison* and *flex* are used for *PT-Scotch* (tool for efficient parallel graph ordering).

MacOS
~~~~~

For MacOs distributions (based on Darwin) the install script uses the :bash:`homebrew` package manager to install the following necessary dependencies:

.. include:: dependencies/system_requirements_mac.rst

Additionally the install script will also want to install xcode-command line tools.

After the installation of the system requirements is done, the installation script will then create a virtual python environment and prepare itself to install all its software dependencies there.

Building and configuring PETSc
-------------------------------

One important package on which Firedrake software is built is the `PETSc <https://petsc.org/release/>`_ library, which offers a toolkit when dealing with partial differential equations. 
The install script enables some customization for installing the PETSc, but also offers the user the option to use a previous installation of the library via the parameter :bash:`--honour-petsc-dir`.
If no previous previous installation is provided, the installation script will then build the PETSc package from source in the *src*  directory within the virtual python environment.

The basic options for installing the library using the install argument :bash:`--minimal-petsc` are:

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

If no minimal version of PETSc is required the following packages will be also installed:

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

Additionally, the script also provides the possibility via the :bash:`--with_blas` argument to  download or use an existing version of BLAS (Basic Linear Algebra Subprograms).

Finally, the Firedrake install script will also install the needed PETSc Python wrapper called *petsc4py*.


Installing Python dependencies
------------------------------

After the installation of PETSc, the install script finally starts installing `Firedrake <https://github.com/firedrakeproject/firedrake>`_ alongside it's Python dependencies (as they can be found in the *requirements-git.txt*):

.. include:: dependencies/firedrake_dependencies.rst

When installing these packages we want to assure ABI compatibility within the packages and their dependencies, so we will enforce the pip package manager to not use wheel files, but rebuild the following packages:

.. include:: dependencies/wheel_blacklist.rst

Finally in order to avoid other compatibility issues, the install script will also make sure that the following parallel packages will be installed using the same MPI wrappers as for the PETSc installation:

.. include:: dependencies/parallel_packages.rst