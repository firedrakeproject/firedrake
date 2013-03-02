#! /bin/bash

PIP="pip install --user"
BASE_DIR=`pwd`
PATH=$HOME/.local/bin:$PATH

echo
echo "*** Preparing system ***"
echo

sudo apt-get update
sudo apt-get install -y build-essential python-dev bzr git-core mercurial \
  cmake cmake-curses-gui python-pip swig \
  libopenmpi-dev openmpi-bin libblas-dev liblapack-dev gfortran

echo
echo "*** Installing OP2-Common ***"
echo

git clone git://github.com/OP2/OP2-Common.git
cd OP2-Common/op2/c
./cmake.local -DOP2_WITH_CUDA=0 -DOP2_WITH_HDF5=0 -DOP2_WITH_MPI=0 -DOP2_WITH_OPENMP=0
cd ..
export OP2_DIR=`pwd`

cd $BASE_DIR

echo
echo "*** Installing dependencies ***"
echo

${PIP} Cython decorator instant numpy pyyaml
PETSC_CONFIGURE_OPTIONS="--with-fortran --with-fortran-interfaces --with-c++-support --with-openmp" \
  ${PIP} hg+https://bitbucket.org/ggorman/petsc-3.3-omp#egg=petsc-3.3
${PIP} hg+https://bitbucket.org/mapdes/petsc4py#egg=petsc4py

echo
echo "*** Installing PyOP2 ***"
echo

cd $BASE_DIR

git clone git://github.com/OP2/PyOP2.git
cd PyOP2
make ext
export PYOP2_DIR=`pwd`
export PYTHONPATH=`pwd`:$PYTHONPATH

if [ ! -f .env ]; then
  cat > .env <<EOF
export PYOP2_DIR=${PYOP2_DIR}
export OP2_DIR=${OP2_DIR}
export PYTHONPATH=`pwd`:\$PYTHONPATH
EOF
fi

echo "
Congratulations! PyOP2 installed successfully!

To use PyOP2, make sure the following environment variables are set:
export PYOP2_DIR=${PYOP2_DIR}
export OP2_DIR=${OP2_DIR}
export PYTHONPATH=`pwd`:\$PYTHONPATH

or source the '.env' script with '. .env'
"

echo
echo "*** Installing PyOP2 testing dependencies ***"
echo

${PIP} pytest
sudo apt-get install -y gmsh unzip

if [ ! -x triangle ]; then
  mkdir -p /tmp/triangle
  cd /tmp/triangle
  wget http://www.netlib.org/voronoi/triangle.zip
  unzip triangle.zip
  make triangle
  cp triangle $HOME/.local/bin
fi

echo
echo "*** Testing PyOP2 ***"
echo

cd $PYOP2_DIR

make test BACKENDS="sequential openmp mpi_sequential"

if [ $? -ne 0 ]; then
  echo "PyOP2 testing failed" 1>&2
  exit 1
fi

echo
echo "Congratulations! PyOP2 tests finished successfully!"
