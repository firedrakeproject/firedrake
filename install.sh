#! /bin/bash

# PyOP2 quick installation script. Installs PyOP2 and dependencies.
#
# Usage: install.sh [user name]
#
#   When run with superuser privileges, user name is used for commands to be
#   run unprivileged if given. Otherwise $USERNAME is queried, which works
#   when calling this script with sudo but not when calling from a root shell.

BASE_DIR=`pwd`
PYOP2_DIR=$BASE_DIR/PyOP2
TEMP_DIR=/tmp
if [ -d $PYOP2_DIR ]; then
  LOGFILE=$PYOP2_DIR/pyop2_install.log
else
  LOGFILE=$BASE_DIR/pyop2_install.log
fi

if [ -f $LOGFILE ]; then
  mv $LOGFILE $LOGFILE.old
fi

echo "PyOP2 installation started at `date`" | tee -a $LOGFILE
echo "  on `uname -a`" | tee -a $LOGFILE
echo | tee -a $LOGFILE

if (( EUID != 0 )); then
  echo "*** Unprivileged installation ***" | tee -a $LOGFILE
  echo | tee -a $LOGFILE
  PIP="pip install --user"
  PREFIX=$HOME/.local
  PATH=$PREFIX/bin:$PATH
  ASUSER=""
else
  echo "*** Privileged installation ***" | tee -a $LOGFILE
  echo "  Running unprivileged commands as ${SUDO_USER}" | tee -a $LOGFILE
  echo | tee -a $LOGFILE
  PIP="pip install"
  PREFIX=/usr/local
  HOME=$(getent passwd $SUDO_USER | cut -d: -f6)
  ASUSER="sudo -u ${SUDO_USER} -E HOME=${HOME} "
fi

echo "*** Preparing system ***" | tee -a $LOGFILE
echo | tee -a $LOGFILE

if (( EUID != 0 )); then
  echo "PyOP2 requires the following packages to be installed:
  build-essential python-dev git-core mercurial cmake cmake-curses-gui libmed1
  gmsh python-pip libhdf5-openmpi-dev libopenmpi-dev openmpi-bin libblas-dev
  liblapack-dev gfortran libbost-dev"
else
  apt-get update >> $LOGFILE 2>&1
  apt-get install -y python-software-properties >> $LOGFILE 2>&1
  apt-get install -y build-essential python-dev git-core mercurial cmake \
    cmake-curses-gui libmed1 gmsh python-pip libhdf5-openmpi-dev \
    libopenmpi-dev openmpi-bin libblas-dev liblapack-dev gfortran \
    >> $LOGFILE 2>&1
fi

echo "*** Installing dependencies ***" | tee -a $LOGFILE
echo | tee -a $LOGFILE

# Install Cython so we can build PyOP2 from source
${PIP} Cython decorator numpy >> $LOGFILE 2>&1

echo "*** Installing PETSc ***" | tee -a $LOGFILE
echo | tee -a $LOGFILE

PETSC_CONFIGURE_OPTIONS="--download-ctetgen --download-triangle --download-chaco"
${PIP} git+https://bitbucket.org/petsc/petsc.git#egg=petsc >> $LOGFILE 2>&1
${PIP} git+https://bitbucket.org/petsc/petsc4py.git#egg=petsc4py >> $LOGFILE 2>&1

echo "*** Installing PyOP2 ***" | tee -a $LOGFILE
echo | tee -a $LOGFILE

if [ ! -d PyOP2/.git ]; then
  ${ASUSER}git clone git://github.com/OP2/PyOP2.git >> $LOGFILE 2>&1
fi
cd $PYOP2_DIR
${ASUSER}python setup.py develop --user >> $LOGFILE 2>&1

python -c 'from pyop2 import op2'
if [ $? != 0 ]; then
  echo "PyOP2 installation failed" 1>&2
  echo "  See ${LOGFILE} for details" 1>&2
  exit 1
fi

echo "
Congratulations! PyOP2 installed successfully!
"

echo "*** Installing PyOP2 testing dependencies ***" | tee -a $LOGFILE
echo | tee -a $LOGFILE

${PIP} pytest flake8 >> $LOGFILE 2>&1
if (( EUID != 0 )); then
  echo "PyOP2 tests require the following packages to be installed:"
  echo "  gmsh triangle-bin unzip"
else
  apt-get install -y gmsh triangle-bin unzip >> $LOGFILE 2>&1
fi

echo "*** Testing PyOP2 ***" | tee -a $LOGFILE
echo | tee -a $LOGFILE

cd $PYOP2_DIR

${ASUSER}make test BACKENDS="sequential openmp" >> $LOGFILE 2>&1

if [ $? -ne 0 ]; then
  echo "PyOP2 testing failed" 1>&2
  echo "  See ${LOGFILE} for details" 1>&2
  exit 1
fi

echo "Congratulations! PyOP2 tests finished successfully!"

echo | tee -a $LOGFILE
echo "PyOP2 installation finished at `date`" | tee -a $LOGFILE
