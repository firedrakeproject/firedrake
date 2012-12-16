# Installation

The main testing platform for PyOP2 is Ubuntu 12.04 64-bit with Python 2.7.3.
Other UNIX-like systems may or may not work. Microsoft Windows is not
supported.

## Dependencies

To install dependencies system-wide use `sudo -E pip install ...`, to install
to a user site use `pip install --user ...`. In the following we will use `pip
install ...` to mean either.

### Common
Common dependencies:
  * Cython >= 0.17
  * decorator
  * instant >= 1.0
  * numpy >= 1.6
  * [PETSc](https://bitbucket.org/fr710/petsc-3.3-omp) >= 3.2 with Fortran
    interface, C++ and OpenMP support
  * [PETSc4py](https://bitbucket.org/fr710/petsc4py) >= 3.3
  * PyYAML

Additional Python 2.6 dependencies:
  * argparse
  * ordereddict

Install dependencies via `pip`:
```
$ pip install Cython decorator instant numpy pyyaml
$ pip install argparse ordereddict # python < 2.7 only
```
PETSc and petsc4py require environment variables to be set:
```
PETSC_CONFIGURE_OPTIONS="--with-fortran-interfaces=1 --with-c++-support --with-openmp" \
                         pip install hg+https://bitbucket.org/fr710/petsc-3.3-omp
$ unset PETSC_DIR
$ unset PETSC_ARCH
$ pip install hg+https://bitbucket.org/fr710/petsc4py#egg=petsc4py
```
**Note:** When using PyOP2 with Fluidity it's crucial that both are built
against the same PETSc!

### CUDA backend:
Dependencies:
  * codepy >= 2012.1.2
  * Jinja2
  * mako
  * pycparser == 2.08 with [patch][1] applied
  * pycuda revision a6c9b40 or newer

The [cusp library](https://code.google.com/p/cusp-library/) headers need to be
in your (CUDA) include path.

Install via `pip`:
```
$ pip install codepy Jinja2 mako hg+https://bitbucket.org/fr710/pycparser#egg=pycparser-2.08
```

Above version of [pycparser](https://bitbucket.org/fr710/pycparser) includes a
[patch][1] to be able to use `switch`/`case` statements in your kernels.

pycuda: Make sure `nvcc` is in your `$PATH` and `libcuda.so` in your
`$LIBRARY_PATH` if in a non-standard location.
```
$ cd /tmp
$ git clone http://git.tiker.net/trees/pycuda.git
$ cd pycuda
$ git submodule init
$ git submodule update
# libcuda.so is in a non-standard location on Ubuntu systems
$ ./configure.py --no-use-shipped-boost \
  --cudadrv-lib-dir='/usr/lib/nvidia-current,${CUDA_ROOT}/lib,${CUDA_ROOT}/lib64'
$ python setup.py build
$ sudo python setup.py install
$ sudo cp siteconf.py /etc/aksetup-defaults.py
```

### OpenCL backend:
Dependencies:
  * Jinja2
  * mako
  * pycparser == 2.08 with [patch][1] applied
  * pyopencl >= 2012.1

Install via `pip`:
```
$ pip install Jinja2 mako pyopencl>=2012.1 \
    hg+https://bitbucket.org/fr710/pycparser#egg=pycparser-2.08
```

Above version of [pycparser](https://bitbucket.org/fr710/pycparser) includes a
[patch][1] to be able to use `switch`/`case` statements in your kernels.

Installing the Intel OpenCL toolkit (64bit systems only):

```
$ cd /tmp
# install alien to convert the rpm to a deb package
$ sudo apt-get install alien fakeroot
$ wget http://registrationcenter.intel.com/irc_nas/2563/intel_sdk_for_ocl_applications_2012_x64.tgz
$ tar xzf intel_sdk_for_ocl_applications_2012_x64.tgz
$ fakeroot alien *.rpm
$ sudo dpkg -i *.deb
```

Installing the [AMD OpenCL toolkit][2] (32bit and 64bit systems):

```
$ wget http://developer.amd.com/Downloads/AMD-APP-SDK-v2.7-lnx64.tgz
# on a 32bit system, instead
# wget http://developer.amd.com/Downloads/AMD-APP-SDK-v2.7-lnx32.tgz
$ tar xzf AMD-APP-SDK-v2.7-lnx*.tgz
# Install to /usr/local instead of /opt
$ sed -ie 's:/opt:/usr/local:g' default-install_lnx.pl
$ sudo ./Install-AMD-APP.sh
```

### HDF5
```
$ sudo apt-get install libhdf5-mpi-dev python-h5py
```

### FFC Interface

The easiest way to get all the dependencies for FFC is to install the FEniCS
toolchain from packages:

```
$ sudo apt-get install fenics
```

A branch of FFC is required, and it must be added to your `$PYTHONPATH`:

```
$ bzr branch lp:~mapdes/ffc/pyop2 $FFC_DIR
$ export PYTHONPATH=$FFC_DIR:$PYTHONPATH
```

This branch of FFC also requires the trunk version of UFL, also added to `$PYTHONPATH`:

```
$ bzr branch lp:ufl $UFL_DIR
$ export PYTHONPATH=$UFL_DIR:$PYTHONPATH
```

Alternatively, install FFC and all dependencies via pip:
```
pip install \
        bzr+ssh://bazaar.launchpad.net/~mapdes/ffc/pyop2#egg=ffc \
        bzr+ssh://bazaar.launchpad.net/~florian-rathgeber/ufc/python-setup#egg=ufc_utils \
        bzr+ssh://bazaar.launchpad.net/%2Bbranch/ufl#egg=ufl \
        bzr+ssh://bazaar.launchpad.net/%2Bbranch/fiat#egg=fiat \
        https://sourcesup.renater.fr/frs/download.php/2309/ScientificPython-2.8.tar.gz
```

[1]: http://code.google.com/p/pycparser/issues/detail?id=79
[2]: http://developer.amd.com/tools/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/
