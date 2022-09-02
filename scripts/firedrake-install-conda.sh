#!/usr/bin/env bash

# For some odd reason, on MacOS we need to tell it where to find `ar`,
# otherwise it might try to find an `llvm-ar` that doesn't necessarily exist.
if [[ $(uname -s) == Darwin ]]; then
    export AR=/usr/bin/ar
fi

# Make an environment variable for the directory where the conda environment
# will live; this will depend on whether or not you're using `pyenv`.
if [[ $(command -v pyenv) ]]; then
    export DIRECTORY=$(pyenv root)/versions/$(pyenv version-name)/envs/firedrake
else
    export DIRECTORY=${HOME}/.conda/envs/firedrake
fi

# Make a conda environment that excludes HDF5 & mpich (PETSc will build these)
conda list --explicit | grep -v mpich | grep -v hdf5 | grep -v h5py | grep -v asn1crypto > spec-file.txt
conda create --prefix ${DIRECTORY} --file spec-file.txt

# Install Firedrake into a subdirectory of the conda env
cd ${DIRECTORY}
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
source activate firedrake
python firedrake-install --no-package-manager --disable-ssh --venv-name=firedrake-venv

# Make an activate script
mkdir -p etc/conda/activate.d
cp firedrake-venv/bin/activate etc/conda/activate.d/01_firedrake-venv.sh

# Make a custom deactivate script
mkdir -p etc/conda/deactivate.d
echo deactivate > etc/conda/deactivate.d/01_firedrake-venv.sh
