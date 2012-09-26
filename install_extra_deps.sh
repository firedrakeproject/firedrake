#! /bin/bash

pip install hg+https://bitbucket.org/khinsen/scientificpython
pip install "codepy>=2013.1"
LDFLAGS=$TOX_LDFLAGS pip install git+git://github.com/inducer/pycuda.git#egg=pycuda
pip install "pyopencl>=2012.1"
C_INCLUDE_PATH=$TOX_C_INCLUDE_PATH pip install "h5py>=2.0.0"
PETSC_CONFIGURE_OPTIONS="--with-fortran-interfaces=1 --with-c++-support" pip install petsc
pip install hg+https://bitbucket.org/mapdes/petsc4py#egg=petsc4py
