# Installation

The main testing platform for PyOP2 is Ubuntu 12.04 64-bit with Python 2.7.3. Other UNIX-like systems may or may not work. Microsoft Windows is not supported.

## Dependencies

### Common
```
sudo pip install cython decorator pyyaml pytest
sudo pip install argparse # python < 2.7 only
```
petsc4py:
```
PETSC_CONFIGURE_OPTIONS='--with-fortran-interfaces=1' sudo -E pip install petsc
sudo pip install hg+https://bitbucket.org/fr710/petsc4py#egg=petsc4py
```
**Note:** When using PyOP2 with Fluidity it's crucial that both are built against the same PETSc!

### CUDA backend:
The [cusp library](https://code.google.com/p/cusp-library/) headers need to be in your (CUDA) include path.

```
sudo pip install codepy
```

You need a version of `pycuda` revision a6c9b40 or newer:

Make sure `nvcc` is in your `$PATH` and `libcuda.so` in your `$LIBRARY_PATH` if in a non-standard location.
```
cd /tmp
git clone http://git.tiker.net/trees/pycuda.git
cd pycuda
git submodule init
git submodule update
# libcuda.so is in a non-standard location on Ubuntu systems
./configure.py --no-use-shipped-boost \
  --cudadrv-lib-dir='/usr/lib/nvidia-current,${CUDA_ROOT}/lib,${CUDA_ROOT}/lib64'
python setup.py build
sudo python setup.py install
sudo cp siteconf.py /etc/aksetup-defaults.py
```

### OpenCL backend:
```
sudo pip install pyopencl pycparser ply jinja2 mako
```

If you want to be able to use `switch`/`case` statements in your kernels, you need to [apply a patch to your pycparser](http://code.google.com/p/pycparser/issues/detail?id=79).

Installing the Intel OpenCL toolkit (64bit systems only):

```
cd /tmp
# install alien to convert the rpm to a deb package
sudo apt-get install alien fakeroot
wget http://registrationcenter.intel.com/irc_nas/2563/intel_sdk_for_ocl_applications_2012_x64.tgz
tar xzf intel_sdk_for_ocl_applications_2012_x64.tgz
fakeroot alien *.rpm
sudo dpkg -i *.deb
```

Installing the AMD OpenCL toolkit (32bit and 64bit systems)

```
wget http://developer.amd.com/Downloads/AMD-APP-SDK-v2.7-lnx64.tgz
# on a 32bit system
# wget http://developer.amd.com/Downloads/AMD-APP-SDK-v2.7-lnx32.tgz
tar xzf AMD-APP-SDK-v2.7-lnx64.tgz
# Install to /usr/local instead of /opt
sed -ie 's:/opt:/usr/local:g'
```

### HDF5
```
wajig install libhdf5-mpi-dev python-h5py
```

### FFC Interface

The easiest way to get all the dependencies for FFC is to install the FEniCS toolchain from packages:

```
sudo apt-get install fenics
```

A branch of FFC is required, and it must be added to your `$PYTHONPATH`:

```
bzr branch lp:~mapdes/ffc/pyop2 $FFC_DIR
export PYTHONPATH=$FFC_DIR:$PYTHONPATH
```

This branch of FFC also requires the trunk version of UFL, also added to $PYTHONPATH:

```
bzr branch lp:ufl $UFL_DIR
export PYTHONPATH=$UFL_DIR:$PYTHONPATH
```
